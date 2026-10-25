"""Parte 2 (25/10 sera): correzioni larghe sul banco a 7 verifiche (prima viste solo sul fold F2024): GBM di correzione con il prior
(× genere/nb di sub_07) come init_score, 50 / 150 alberi; GLM su 19 variabili (CORRECTION_VARS2). Motivo: il primo in classifica è
+0,007 sopra e le sonde dicono che il livello auto spiega al massimo +0,0007 → se c'è qualcosa, è nella forma.
Regola ≥ 5/7 su forma e own. Uso: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench8.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402
VARS = {"motor": ["gender", "new_business"], "contents": ["gender"], "buildings": None}
rows = []
for l in P.LINES:
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        if VARS[l]:
            TR.CORRECTION_VARS = VARS[l]; base_va = np.asarray(TR.src_glm(tr, va, l), float); base_tr = np.asarray(TR.src_glm(tr, tr, l), float)
        else:
            base_va = np.asarray(TR.src_level(tr, va, l), float); base_tr = np.asarray(TR.src_level(tr, tr, l), float)
        preds = {"sub_07": base_va}
        ov = pd.DataFrame({"mu_src": np.r_[base_tr, base_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
        TR._PRIOR_OVERRIDE = ov
        for rnd in (50, 150):
            preds[f"gbm {rnd}"] = np.asarray(TR.src_gbm(tr, va, l, rounds=rnd), float)
        TR.CORRECTION_VARS = TR.CORRECTION_VARS2
        preds["glm 19 var"] = np.asarray(TR.src_glm(tr, va, l), float)
        TR._PRIOR_OVERRIDE = None
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "fatto", flush=True)
r = pd.DataFrame(rows); b = r[r.variante == "sub_07"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench8.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "sub_07"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 contro sub_07"); print(s.round(1).to_string())
