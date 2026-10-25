"""Parte 2 (25/10): penalità dei GLM di correzione sul banco a 7 verifiche (stessi fold di p2_bench.py).
Oggi la penalità è scelta su un 10% di polizze (auto genere+nb: 0,01; contenuto genere: 1,0). Griglia fissa, confronto con la
scelta automatica (= sub_07). Regola: una penalità fissa entra solo se batte l'automatica in ≥ 5/7 verifiche su forma e su own.
Uso: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench3.py → results_p2/p2_bench3.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

SPEC = {"motor": (["gender", "new_business"], [0.001, 0.003, 0.03, 0.1, 0.3, 1.0]),
        "contents": (["gender"], [0.03, 0.1, 0.3, 3.0, 10.0])}
rows, coefs = [], []
for l, (vars_, grid) in SPEC.items():
    df = P.load(l)
    TR.CORRECTION_VARS = vars_
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        preds = {"auto": TR.src_glm(tr, va, l)}
        coefs.append(dict(linea=l, fold=fname, variante="auto", **TR.src_glm.last))
        for a in grid:
            preds[f"alpha={a}"] = TR.src_glm(tr, va, l, alpha=a)
            coefs.append(dict(linea=l, fold=fname, variante=f"alpha={a}", **TR.src_glm.last))
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()),
                             own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "fatto", flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "auto"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench3.csv", index=False)
pd.DataFrame(coefs).to_csv(Path(P.RESULTS) / "p2_bench3_coef.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "auto"].groupby(["linea", "variante"]).agg(
    forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 rispetto alla penalità automatica (sub_07)"); print(s.round(1).to_string())
c = pd.DataFrame(coefs); print(c[c.fold == "F2024"][["linea", "variante", "alpha", "coef"]].to_string(index=False))
