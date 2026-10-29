"""Part 2 (27/10, idea: restart from the best model on the Part 1 private score, sub_26 = fs2_v2_cred, which uses the relative
premium prem_rel = log premium before discount − median of the year). The current prior (fs2_all_ll) does not use it. Here prem_rel is
a correction on the emerging data on top of p2_sub_18: on frequency (TR.freq_correction) and, in motor, also in the cost GLM.
Rule ≥ 5/7 shape AND own. Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench17.py <line>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

l = sys.argv[1]


def glm(tr, d, v, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, v, ov
    try:
        return np.asarray(TR.src_glm(tr, d, l), float) if v else np.asarray(TR.src_level(tr, d, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def preds(tr, va):
    out = {}
    if l == "motor":
        def run(extra):
            a = lambda d: 0.5 * glm(tr, d, ["new_business", "gender"] + extra) + 0.5 * glm(tr, d, ["new_business", "gender", "fuel_type", "cover_type"] + extra)
            return a(va) * TR.gbm_freq_correction(tr, va, l, a(tr))
        out["sub_18"] = b = run([])
        out["prem freq"] = b * TR.freq_correction(tr, va, l, ["prem_rel"])
        out["prem costo"] = run(["prem_rel"])
    elif l == "buildings":
        ov = TR.peril_own_override(tr, va, l, ["storm"])[0]
        b = glm(tr, va, None, ov); sc = np.asarray(TR.scratch(tr, va, l), float)
        fin = lambda p: 0.9 * p + 0.1 * sc * p.sum() / sc.sum()
        out["sub_18"] = fin(b); out["prem freq"] = fin(b * TR.freq_correction(tr, va, l, ["prem_rel"]))
    else:
        p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
        ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
        b = glm(tr, va, ["gender"], ov) * TR.freq_correction(tr, va, l, ["works_from_home"])
        out["sub_18"] = b; out["prem freq"] = b * TR.freq_correction(tr, va, l, ["prem_rel"])
    return out


rows = []
df = M._prem(P.load(l))
for fname, tr, va in folds(df):
    y = va[P.TARGET].to_numpy()
    for k, p in preds(tr, va).items():
        rows.append(dict(fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    print(fname, flush=True)
r = pd.DataFrame(rows); b = r[r.variante == "sub_18"].set_index("fold")
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[r.fold, c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / f"p2_bench17_{l}.csv", index=False)
print(r[r.variante != "sub_18"].groupby("variante").agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
      pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/7"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/7")).round(2).to_string())
print(r[r.variante != "sub_18"].pivot_table(index="variante", columns="fold", values="d_forma").round(1).to_string())
