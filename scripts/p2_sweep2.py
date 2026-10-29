"""Part 2: combinations (gender correction × blend with scratch) on fold F2024. Usage: IAA_MARKET=configs/parte2.json python3 scripts/p2_sweep2.py"""
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
    TR.CORRECTION_VARS, old = ["gender"], TR.CORRECTION_VARS
    g = TR.src_glm(tr, va, l); TR.CORRECTION_VARS = old
    res = {"src_level": base, "glm_gender": g}
    for w in (0.1, 0.2):
        res[f"gender+scratch_{w}"] = (1 - w) * g + w * sc * g.sum() / sc.sum()
        res[f"level+scratch_{w}"] = (1 - w) * base + w * sc * base.sum() / sc.sum()
    for k, p in res.items():
        rows.append(dict(linea=l, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
    print(l, "gender coef", TR.src_glm.last, flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "src_level"].set_index("linea")
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c] - r.linea.map(b[c])) * 1e4
print(r.round(4).to_string(index=False)); r.to_csv(P.RESULTS / "p2_sweep2.csv", index=False)
