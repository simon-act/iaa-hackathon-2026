"""Parte 2: piccole varianti del trasferimento sul fold F2024 (forma = D² ufficiale a livello riallineato). Uso: IAA_MARKET=configs/parte2.json python3 scripts/p2_sweep.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402
rows = []
for l in P.LINES:
    df = P.load(l); tr, va = df[df.year == 2023], df[df.year == 2024]
    y = va[P.TARGET].to_numpy()
    sc = pd.read_parquet(P.RESULTS / f"oof_parts/p2_scratch_{l}_F2024.parquet").set_index("row_id")["pred"].reindex(va.row_id).to_numpy()
    base = TR.src_level(tr, va, l)
    res = {"src_level": base}
    for w in (0.1, 0.2, 0.3):
        res[f"mix_scratch_{w}"] = (1 - w) * base + w * sc * base.sum() / sc.sum()
    for vars_ in (["region"], ["region", "gender"], ["gender"]):
        TR.CORRECTION_VARS, old = vars_, TR.CORRECTION_VARS
        try:
            res["glm_" + "+".join(vars_)] = TR.src_glm(tr, va, l)
        finally:
            TR.CORRECTION_VARS = old
    for r in (10, 25):
        res[f"peril_gbm{r}"] = TR.src_peril(tr, va, l, rounds=r)
    for k, p in res.items():
        rows.append(dict(linea=l, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
    print(l, "fatto", flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "src_level"].set_index("linea")
r["d_forma_e4"] = (r.forma - r.linea.map(b.forma)) * 1e4; r["d_own_e4"] = (r.own - r.linea.map(b.own)) * 1e4
print(r.round(4).to_string(index=False)); r.to_csv(P.RESULTS / "p2_sweep.csv", index=False)
