"""Part 2 (27/10, request: "test the best Part 1 models on Part 2"): exports as developed prior
a Part 1 model variant (without the final credibility on claims history, which worsens results in the emerging market: bench4), with the same
selection of common variables as transfer.export_source. Variants:
  v2      = fs2_v2 (property with relative premium; motor = fs2_all_ll because mileage is missing in the emerging market)   [sub_26]
  v2f     = v2 + per-peril frequency from grid (FREQ_CHOSEN)                                                    [sub_30]
  v2fs    = v2f + per-peril motor severity (SEV_CHOSEN: constant bodily injury)                                  [sub_38/39]
  v2fsllr = v2fs + large losses allocated by risk                                                                [sub_45]
  gbm73   = Qwen GBM (qwen_gbm73), for the blend                                                                 [sub_25]
  v2prem  = freq×sev with the relative premium in frequency and severity in motor too (28/10)
Output: <new results>/source_pred_<line>_p1<variant>.parquet (mu_src, per-peril parts if the model produces them).
Usage: IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_export_p1.py <line> <variant> configs/parte2_llc.json [fraction=1]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M, market_config as MC  # noqa: E402

line, var, new_cfg = sys.argv[1], sys.argv[2], MC.load(sys.argv[3]); FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
t0 = time.time()
src = TR.harmonize(P.load(line))
import os
_RANGE = os.environ.get("P1_RANGE")   # 28/10: "lo-hi" = policies with hash % 100 in [lo, hi) (disjoint halves for bagging)
if _RANGE:
    lo, hi = map(int, _RANGE.split("-")); hh = pd.util.hash_array(src["policy_id"].to_numpy() + "_c") % 100
    src = src[(hh >= lo) & (hh < hi)].reset_index(drop=True); FRAC = (hi - lo) / 100
elif FRAC < 1:
    src = src[pd.util.hash_array(src["policy_id"].to_numpy() + "_c") % 100 < FRAC * 100].reset_index(drop=True)
new_tr = pd.read_parquet(Path(new_cfg["model_dir"]) / f"model_{line}.parquet")
new_te = pd.read_parquet(Path(new_cfg["model_dir"]) / f"test_{line}.parquet")
common = [c for c in M._features(src, line) if c in new_tr.columns and c in new_te.columns]


def overlap(c):
    if pd.api.types.is_numeric_dtype(src[c]) and pd.api.types.is_numeric_dtype(new_te[c]):
        return 1.0
    seen = set(src[c].dropna().astype(str).unique()); v = new_te[c].dropna().astype(str)
    return float(v.isin(seen).mean()) if len(v) else 1.0


common = [c for c in common if overlap(c) >= 0.5]
M.FEATURE_WHITELIST = set(common) | {"prem_rel"}
import os
if os.environ.get("P1_THREADS"):   # 28/10: more threads to stay within the time limit (results nearly identical, not bit-for-bit)
    for d_ in (M.FREQ_PARAMS, M.SEV_PARAMS):
        d_["num_threads"] = int(os.environ["P1_THREADS"])
both = pd.concat([new_tr, new_te], ignore_index=True)
fn = {"v2": M.fs2_v2, "gbm73": M.qwen_gbm73}
def v2f(tr, va, l):
    M._FREQ_OVERRIDE = M.FREQ_CHOSEN
    try:
        return M.fs2_v2(tr, va, l)
    finally:
        M._FREQ_OVERRIDE = {}
def v2fs(tr, va, l):
    M._SEV_PERIL = M.SEV_CHOSEN
    try:
        return v2f(tr, va, l)
    finally:
        M._SEV_PERIL = {}
def v2fsllr(tr, va, l):
    M._LL_RISK = True
    try:
        return v2fs(tr, va, l)
    finally:
        M._LL_RISK = False
def v2prem(tr, va, l):   # 28/10: freq×sev with the relative premium in motor too (in fs2_v2 it was property only)
    M.SEV_DRIVERS[l] = M.SEV_DRIVERS[l] + ["prem_rel"]
    try:
        return M.fs2_all_ll(M._prem(tr), M._prem(va), l)
    finally:
        M.SEV_DRIVERS[l] = [c for c in M.SEV_DRIVERS[l] if c != "prem_rel"]
fn.update(v2f=v2f, v2fs=v2fs, v2fsllr=v2fsllr, v2prem=v2prem)
M._fs2.last_parts = None
pred = np.asarray(fn[var](src, both, line), float)
out = pd.DataFrame({"row_id": both["row_id"].to_numpy(), "mu_src": pred})
lp = getattr(M._fs2, "last_parts", None)
gb = pd.read_parquet(Path(new_cfg["results"]) / f"source_pred_{line}.parquet").set_index("row_id").reindex(out.row_id)
if lp and var != "gbm73":
    for per, (f_va, s_va) in lp["perils"].items():
        out[f"f_{per}"] = f_va; out[f"s_{per}"] = s_va
    out["ll_src"] = lp["ll"]; out["c_src"] = lp["c"]
else:   # per-peril parts taken from the current prior (needed only for peril levels and expected counts)
    for c in gb.columns:
        if c != "mu_src":
            out[c] = gb[c].to_numpy()
out.to_parquet(Path(new_cfg["results"]) / f"source_pred_{line}_p1{var}{('_r' + _RANGE.replace('-', '_')) if _RANGE else ('' if FRAC == 1 else '_s' + str(int(FRAC * 100)))}.parquet", index=False)
print(f"{line} {var}: log corr with the current prior {np.corrcoef(np.log(pred), np.log(gb.mu_src.to_numpy()))[0, 1]:.4f}, "
      f"ratio of totals {pred.sum() / gb.mu_src.sum():.3f} ({time.time() - t0:.0f}s, {len(src)} rows)", flush=True)
