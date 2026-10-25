"""Parte 2 — banco interno più robusto del solo fold 2023 → 2024 (24/10, richiesta di Simone: niente overfitting sulla classifica).
Fold: F2024 (stima 2023 → valuta 2024), F2023 (stima 2024 → valuta 2023, all'indietro), K1–K5 (5 gruppi di polizze sui due anni:
gli effetti trasversali come genere e nuovo business si vedono senza confonderli con l'anno).
Varianti, tutte sul prior del developed: livello; + genere; + nuovo business; + genere + nuovo business; + regione;
0,9 prior + 0,1 scratch. Misure: D² a livello riallineato (forma), senza spalmatura, grandi sinistri veri; Δ e·4 contro "livello".
Uso: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench.py [linee] ; output results_p2/p2_bench.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402

VARS = {"genere": ["gender"], "nuovo_business": ["new_business"], "genere+nb": ["gender", "new_business"], "regione": ["region"]}


def folds(df):
    yield "F2024", df[df.year == 2023], df[df.year == 2024]
    yield "F2023", df[df.year == 2024], df[df.year == 2023]
    g = pd.util.hash_array((df["policy_id"] + "_k").to_numpy()) % 5   # hash diverso da quello della penalità del GLM
    for k in range(5):
        yield f"K{k + 1}", df[g != k], df[g == k]


lines = sys.argv[1].split(",") if len(sys.argv) > 1 else P.LINES
rows = []
for l in lines:
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        base = TR.src_level(tr, va, l)
        preds = {"livello": base}
        for k, v in VARS.items():
            TR.CORRECTION_VARS, old = v, TR.CORRECTION_VARS
            try:
                preds[k] = TR.src_glm(tr, va, l)
            finally:
                TR.CORRECTION_VARS = old
        sc = np.asarray(TR.scratch(tr, va, l), float)
        preds["mix_scratch_0.1"] = 0.9 * base + 0.1 * sc * base.sum() / sc.sum()
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()),
                             own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, "fatto", flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "livello"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
out = Path(P.RESULTS) / "p2_bench.csv"
r.to_csv(out, mode="a", header=not out.exists(), index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "livello"].groupby(["linea", "variante"]).agg(
    forma_media=("d_forma", "mean"), own_media=("d_own", "mean"), unc_media=("d_unc", "mean"),
    fold_positivi_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), fold_positivi_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print(s.round(1).to_string())
print(r.pivot_table(index=["linea", "variante"], columns="fold", values="d_forma").round(1).to_string())
