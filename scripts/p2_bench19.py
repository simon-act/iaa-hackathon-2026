"""Part 2 (27/10): the best Part 1 models as prior (scripts/p2_export_p1.py), with the full p2_sub_19 pipeline on top.
Compared with the reference prior (same developed sample). Rule ≥ 5/7 shape AND own.
Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench19.py <line> <reference suffix> <suffix1,suffix2,...>
(suffix "" = current prior; e.g. _p1v2_s60)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402
l, ref = sys.argv[1], sys.argv[2].replace("none", ""); alts = [s.replace("none", "") for s in sys.argv[3].split(",")]


def glm(tr, d, v, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, v, ov
    try:
        return np.asarray(TR.src_glm(tr, d, l), float) if v else np.asarray(TR.src_level(tr, d, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def sub19(tr, va, suf):
    TR.PRIOR_SUFFIX = suf
    try:
        if l == "motor":
            a = lambda d: 0.5 * glm(tr, d, ["new_business", "gender"]) + 0.5 * glm(tr, d, ["new_business", "gender", "fuel_type", "cover_type"])
            return a(va) * TR.gbm_freq_correction(tr, va, l, a(tr)) * TR.freq_correction(tr, va, l, ["prem_rel"])
        if l == "buildings":
            ov = TR.peril_own_override(tr, va, l, ["storm"])[0]
            b = glm(tr, va, None, ov); TR.PRIOR_SUFFIX = ""
            sc = np.asarray(TR.scratch(tr, va, l), float)
            return 0.9 * b + 0.1 * sc * b.sum() / sc.sum()
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
        return glm(tr, va, ["gender"], ov) * TR.freq_correction(tr, va, l, ["works_from_home"])
    finally:
        TR.PRIOR_SUFFIX = ""


rows = []
df = P.load(l); df = M._prem(df) if l == "motor" else df
for fname, tr, va in folds(df):
    y = va[P.TARGET].to_numpy()
    for k in [ref] + alts:
        p = sub19(tr, va, k)
        rows.append(dict(fold=fname, variante=k or "attuale", forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    print(fname, flush=True)
r = pd.DataFrame(rows); b = r[r.variante == (ref or "attuale")].set_index("fold")
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[r.fold, c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / f"p2_bench19_{l}{ref}.csv", index=False)
print(l, "ref", ref or "attuale"); print(r[r.variante != (ref or "attuale")].groupby("variante").agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
      pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/7"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/7")).round(2).to_string())
print(r[r.variante != (ref or "attuale")].pivot_table(index="variante", columns="fold", values="d_forma").round(1).to_string())
