"""Part 2 (28/10): two variants of p2_sub_20 that change motor only (buildings and contents = sub_20):
  premcost : relative premium in the cost GLM (both halves) and in the GBM on claim count, without frequency correction
             on the premium (bench17 'prem costo': +35.0 / +16.9, 5/7 · 5/7 vs sub_18)
  v2prem60 : sub_20 pipeline on top of the motor prior with relative premium fitted on 60% of the developed market (bench22: 7/7 · 5/7 vs the
             base on the same sample, 3/7 vs the current prior)
Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_sub_varianti_auto.py <premcost|v2prem60> <output name>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]; l = "motor"; var, name = sys.argv[1], sys.argv[2]
tr, te = M._prem(P.load(l)), M._prem(P.load(l, "test"))
def glm(d, v):
    old, TR.CORRECTION_VARS = TR.CORRECTION_VARS, v
    try:
        return np.asarray(TR.src_glm(tr, d, l), float)
    finally:
        TR.CORRECTION_VARS = old
if var == "premcost":
    a = lambda d: 0.5 * glm(d, ["new_business", "gender", "prem_rel"]) + 0.5 * glm(d, ["new_business", "gender", "fuel_type", "cover_type", "prem_rel"])
    p = a(te) * TR.gbm_freq_correction(tr, te, l, a(tr))
else:
    TR.PRIOR_SUFFIX = "_p1v2prem_s60"
    a = lambda d: 0.5 * glm(d, ["new_business", "gender"]) + 0.5 * glm(d, ["new_business", "gender", "fuel_type", "cover_type"])
    p = a(te) * TR.gbm_freq_correction(tr, te, l, a(tr)) * TR.freq_correction(tr, te, l, ["prem_rel"]); TR.PRIOR_SUFFIX = ""
s = pd.read_csv(ROOT / "submissions_p2" / "p2_sub_20_prior_p1v2_casa.csv")
m = pd.Series(np.round(p, 3), index=te.row_id.to_numpy()); k = s.row_id.isin(m.index)
r0 = s.prediction.copy(); s.loc[k, "prediction"] = s.loc[k, "row_id"].map(m)
assert s.prediction.notna().all() and (s.prediction >= 0).all()
s.to_csv(ROOT / "submissions_p2" / f"{name}.csv", index=False)
r = (s.prediction / r0)[k]
print(var, "motor vs sub_20 p5/50/95", np.percentile(r, [5, 50, 95]).round(3), "total %.4f" % (s.prediction[k].sum() / r0[k].sum()))
