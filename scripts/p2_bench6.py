"""Part 2 (25/10): GLM corrections (gender, new business) estimated on a per-row capped target, so that they are not driven by a few
large claims (the +20% of motor new business comes from 5 bodily-injury claims of 37k–137k). The level stays that of the full target:
the prediction is rescaled to the uncapped train total. 7-check bench (folds from p2_bench2), rule ≥ 5/7 on shape and own.
Usage: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench6.py → results_p2/p2_bench6.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

SPEC = {"motor": (["gender", "new_business"], [10000, 25000, 50000]), "contents": (["gender"], [5000, 10000, 20000])}
rows, coefs = [], []
for l, (vars_, caps) in SPEC.items():
    df = P.load(l); TR.CORRECTION_VARS = vars_
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        preds = {"sub_07": TR.src_glm(tr, va, l)}
        coefs.append(dict(linea=l, fold=fname, variante="sub_07", **TR.src_glm.last))
        for cap in caps:
            trc = tr.copy(); trc[P.TARGET] = trc[P.TARGET].clip(upper=cap)
            p = np.asarray(TR.src_glm(trc, va, l), float) * tr[P.TARGET].sum() / trc[P.TARGET].sum()
            preds[f"tagliato {cap // 1000}k"] = p
            coefs.append(dict(linea=l, fold=fname, variante=f"tagliato {cap // 1000}k", **TR.src_glm.last))
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()),
                             own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "done", flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "sub_07"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench6.csv", index=False); pd.DataFrame(coefs).to_csv(Path(P.RESULTS) / "p2_bench6_coef.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "sub_07"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 vs sub_07"); print(s.round(1).to_string())
c = pd.DataFrame(coefs); print(c[c.fold == "K1"][["linea", "variante", "alpha", "coef"]].to_string(index=False))
