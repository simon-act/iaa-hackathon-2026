"""Part 2 (25/10): gender × age band interaction (the developed market has no gender, so the prior cannot contain it; in motor
young men are the classic case). 7-check bench (folds from p2_bench2), rule ≥ 5/7 on shape and own.
Motor: gender+nb (sub_07) vs gender×age+nb and gender+age+nb; contents: gender vs gender×age.
Usage: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench7.py → results_p2/p2_bench7.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

AGE = {"motor": "driver_age_band", "contents": "policyholder_age_band"}
SPEC = {"motor": {"sub_07": ["gender", "new_business"], "genere×età+nb": ["g_age", "new_business"], "genere+età+nb": ["gender", "age", "new_business"]},
        "contents": {"sub_07": ["gender"], "genere×età": ["g_age"], "genere+età": ["gender", "age"]}}
rows, coefs = [], []
for l, spec in SPEC.items():
    df = P.load(l).copy()
    df["age"] = df[AGE[l]].astype(str); df["g_age"] = df["gender"].astype(str) + "_" + df["age"]
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy(); preds = {}
        for k, v in spec.items():
            TR.CORRECTION_VARS = v; preds[k] = TR.src_glm(tr, va, l)
            coefs.append(dict(linea=l, fold=fname, variante=k, **TR.src_glm.last))
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()),
                             own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "done", flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "sub_07"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench7.csv", index=False); pd.DataFrame(coefs).to_csv(Path(P.RESULTS) / "p2_bench7_coef.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 300)
s = r[r.variante != "sub_07"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 vs sub_07"); print(s.round(1).to_string())
c = pd.DataFrame(coefs); print(c[(c.fold == "K1") & (c.variante != "sub_07")][["linea", "variante", "alpha", "coef"]].to_string(index=False))
