"""Part 2 (27/10): grafting the GLM curve onto the motor GBM prior, only on the experience variables where the two diverge and the
emerging data side with the GLM (scripts/p2_glm_vs_gbm.py: licence years, NCB, age; GBM too steep: new drivers A/E 0.84,
high NCB 1.13–1.14). prior' = GBM prior × Π_v r_v(class)^w, r_v = (weighted) mean of GLM/GBM within the class of v, computed only
from the two priors (no target) on all emerging rows. w = 1 and 0.5 (decision: w = 1). Then the full sub_18 pipeline
(50/50 GLM fuel+cover and GBM on claim count). Vs sub_18; rule ≥ 5/7 shape AND own.
Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench16.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402
from p2_resid_util import binned  # noqa: E402

l = "motor"; res = Path(P.RESULTS); VARS = ["years_licensed", "ncb_years", "driver_age_band"]
g = pd.read_parquet(res / f"source_pred_{l}_glm.parquet")
tr_all, te_all = P.load(l), P.load(l, "test"); allx = pd.concat([tr_all, te_all], ignore_index=True)
gg = g.set_index("row_id").reindex(allx.row_id)
lr = np.log(gg.mu_src.to_numpy() / gg.mu_src.sum() * gg.mu_src_gbm.sum() / gg.mu_src_gbm.to_numpy()); wgt = gg.mu_src_gbm.to_numpy()
fac = np.zeros(len(allx))
for v in VARS:
    b = binned(allx[v]).to_numpy()
    m = pd.DataFrame({"b": b, "x": lr * wgt, "w": wgt}).groupby("b").sum(); f = (m.x / m.w)
    fac += pd.Series(b).map(f).to_numpy()
    print(v, f.round(3).to_dict())
# 27/10 (after the first round): joint estimate of the three factors (WLS of the log ratio on the three variables together), because summing
# three one-variable means counts the same effect several times (the variables are correlated)
Xj = pd.get_dummies(pd.DataFrame({v: binned(allx[v]).to_numpy() for v in VARS}), drop_first=True).astype(float).to_numpy()
Xj = np.c_[np.ones(len(Xj)), Xj]; sw = np.sqrt(wgt)
beta = np.linalg.lstsq(Xj * sw[:, None], lr * sw, rcond=None)[0]; fac_j = Xj @ beta - beta[0]
for w, suf, ff in ((1.0, "_graft", fac), (0.5, "_graft05", fac), (1.0, "_graftj", fac_j)):
    o = g.drop(columns=["mu_src_gbm"]).set_index("row_id").reindex(allx.row_id).reset_index()
    o["mu_src"] = gg.mu_src_gbm.to_numpy() * np.exp(w * ff); o.to_parquet(res / f"source_pred_{l}{suf}.parquet", index=False)


def sub18(tr, va, suf):
    TR.PRIOR_SUFFIX = suf
    try:
        def glm(d, v):
            old, TR.CORRECTION_VARS = TR.CORRECTION_VARS, v
            try:
                return np.asarray(TR.src_glm(tr, d, l), float)
            finally:
                TR.CORRECTION_VARS = old
        a = lambda d: 0.5 * glm(d, ["new_business", "gender"]) + 0.5 * glm(d, ["new_business", "gender", "fuel_type", "cover_type"])
        return a(va) * TR.gbm_freq_correction(tr, va, l, a(tr))
    finally:
        TR.PRIOR_SUFFIX = ""


rows = []
for fname, tr, va in folds(tr_all):
    y = va[P.TARGET].to_numpy()
    for k, suf in (("sub_18", ""), ("innesto w=1", "_graft"), ("innesto w=0,5", "_graft05"), ("innesto congiunto", "_graftj")):
        p = sub18(tr, va, suf)
        rows.append(dict(fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    print(fname, flush=True)
r = pd.DataFrame(rows); b = r[r.variante == "sub_18"].set_index("fold")
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[r.fold, c].to_numpy()) * 1e4
r.to_csv(res / "p2_bench16.csv", index=False)
print(r[r.variante != "sub_18"].groupby("variante").agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
      pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/7"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/7")).round(2).to_string())
print(r[r.variante != "sub_18"].pivot_table(index="variante", columns="fold", values="d_forma").round(1).to_string())
