"""Parte 2 (27/10): bench17c = come bench17b, più: il premio è solo un indicatore dell'età? (fascia di premio più cara = 97% 17–24 anni).\nControllo pulito di bench17 in auto. In bench17 prem_rel entrava anche tra le variabili del GBM sul numero di
sinistri (anche nel riferimento). Qui, contro p2_sub_18 esatta (senza prem_rel): 'sub_18 + prem nel GBM' e 'sub_18 + prem nel GBM
+ correzione di frequenza prem_rel' (= la variante 'prem freq' di bench17). Regola ≥ 5/7 forma E own.
Uso: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench17b.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402
l = "motor"


def run(tr, va):
    def glm(d, v):
        old, TR.CORRECTION_VARS = TR.CORRECTION_VARS, v
        try:
            return np.asarray(TR.src_glm(tr, d, l), float)
        finally:
            TR.CORRECTION_VARS = old
    a = lambda d: 0.5 * glm(d, ["new_business", "gender"]) + 0.5 * glm(d, ["new_business", "gender", "fuel_type", "cover_type"])
    return a(va) * TR.gbm_freq_correction(tr, va, l, a(tr))


rows = []
df = P.load(l)
for fname, tr, va in folds(df):
    y = va[P.TARGET].to_numpy()
    trp, vap = M._prem(tr), M._prem(va)
    b = run(tr, va); bp = run(trp, vap)
    for k, p in (("sub_18", b), ("prem nel GBM + freq prem", bp * TR.freq_correction(trp, vap, l, ["prem_rel"])),
                 ("freq età", b * TR.freq_correction(tr, va, l, ["driver_age_band"])),
                 ("freq età+patente+ncb", b * TR.freq_correction(tr, va, l, ["driver_age_band", "years_licensed", "ncb_years"])),
                 ("prem nel GBM + freq prem+età", bp * TR.freq_correction(trp, vap, l, ["prem_rel", "driver_age_band"]))):
        rows.append(dict(fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    print(fname, flush=True)
r = pd.DataFrame(rows); b = r[r.variante == "sub_18"].set_index("fold")
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[r.fold, c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench17c.csv", index=False)
print(r[r.variante != "sub_18"].groupby("variante").agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
      pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/7"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/7")).round(2).to_string())
print(r[r.variante != "sub_18"].pivot_table(index="variante", columns="fold", values="d_forma").round(1).to_string())
