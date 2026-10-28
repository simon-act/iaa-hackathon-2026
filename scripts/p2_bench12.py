"""Parte 2 (27/10): ensemble di sub_10 / sub_14 con le varianti quasi passate (richiesta di Simone, punto 5).
Fabbricato: media 50/50 di storm (sub_14) e storm+subsidence (bench10, 4/7); auto: media 50/50 di sub_10 e +fuel+cover (bench11,
4/7 forma, 6/7 own). Banco 7 verifiche contro sub_10; regola ≥ 5/7 forma E own.
Uso: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench12.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402


def glm(tr, va, l, vars_, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, vars_, ov
    try:
        return np.asarray(TR.src_glm(tr, va, l), float) if vars_ else np.asarray(TR.src_level(tr, va, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


rows = []
for l in ("buildings", "motor"):
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        if l == "buildings":
            sc = np.asarray(TR.scratch(tr, va, l), float)
            fin = lambda b: 0.9 * b + 0.1 * sc * b.sum() / sc.sum()
            base = fin(glm(tr, va, l, None))
            st = fin(glm(tr, va, l, None, TR.peril_own_override(tr, va, l, ["storm"])[0]))
            ss = fin(glm(tr, va, l, None, TR.peril_own_override(tr, va, l, ["storm", "subsidence"])[0]))
            preds = {"sub_10": base, "sub_14 storm": st, "50/50 storm, storm+subs": 0.5 * st + 0.5 * ss}
        else:
            base = glm(tr, va, l, ["new_business", "gender"])
            fc = glm(tr, va, l, ["new_business", "gender", "fuel_type", "cover_type"])
            preds = {"sub_10": base, "50/50 sub_10, +fuel+cover": 0.5 * base + 0.5 * fc}
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
        print(l, fname, flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "sub_10"].set_index(["linea", "fold"])
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench12.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "sub_10"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 contro sub_10"); print(s.round(2).to_string())
print(r[r.variante != "sub_10"].pivot_table(index=["linea", "variante"], columns="fold", values="d_forma").round(1).to_string())
