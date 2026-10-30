"""Second Kaggle selection for Part 2 (30/10, a hedge, not the model): p2_sub_21 with motor multiplied by tw05_nbx / sub_21
(exact Tweedie offset + new-business terms, scripts/p2_bench33.py). It fails both benches; see docs/MODEL_CARD.md.
Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_sub_tw05_nbx.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.argv = [sys.argv[0], "27", "none"]
exec((Path(__file__).resolve().parent / "p2_bench33.py").read_text().split("df = add_nb(")[0])
ROOT = Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "submissions_p2" / "p2_sub_21_auto_prem_costo.csv").set_index("row_id")["prediction"]
tr = add_nb(derive(M._prem(P.load(l)))); te = add_nb(derive(M._prem(P.load(l, "test"))))
base = np.asarray(motor(tr, te), float); nbx = np.asarray(motor_nbx(tr, te, 0.5), float)
r = base / s.reindex(te.row_id).to_numpy(); print("sub_21 rebuild ratio min/max", round(r.min(), 6), round(r.max(), 6))
q = nbx / base; print("nbx/base p5/50/95", np.round(np.percentile(q, [5, 50, 95]), 3), "total", round(nbx.sum() / base.sum(), 4))
nb = te["_isnb"].to_numpy(); print("new business: mean ratio", round(nbx[nb].sum() / base[nb].sum(), 4), "renewals", round(nbx[~nb].sum() / base[~nb].sum(), 4))
out = s.copy(); out.loc[te.row_id.to_numpy()] = np.round(s.reindex(te.row_id).to_numpy() * q, 3)
assert out.notna().all() and (out >= 0).all() and len(out) == len(s)
out.reset_index().to_csv(ROOT / "submissions_p2" / "p2_sonda_tw05_nbx.csv", index=False); print("written", len(out))
