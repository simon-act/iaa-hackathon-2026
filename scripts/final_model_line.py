"""Stima finale di UN modello su tutte le polizze 2021-24 per UNA linea → previsione del test 2025.
Come final.py submit --all, a pezzi (container 8 GB). Colonne caricate solo quelle che servono (memoria).
Uso: IAA_MARKET=configs/parte1_s3.json python3 -I scripts/final_model_line.py <modello> <linea>"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
name, line = sys.argv[1], sys.argv[2]
if name.startswith("qwen"):          # il GBM usa solo feature ammesse + target: caricamento ridotto
    names = pq.ParquetFile(P.MODEL_DIR / f"model_{line}.parquet").schema.names
    keep = [c for c in names if c in ("row_id", "policy_id", "year", "exposure", P.TARGET, "pol_prev_n")
            or c.startswith(("pi_", "n_")) or (c in P.allowed_features(pd.DataFrame(columns=names)) and not c.startswith("addr_"))]

    def _slim(l, kind="train"):
        f = P.MODEL_DIR / f"{'model' if kind == 'train' else 'test'}_{l}.parquet"
        nm = pq.ParquetFile(f).schema.names
        return pd.read_parquet(f, columns=[c for c in keep if c in nm])
    P.load = _slim
import models_extra as M  # noqa: E402
t0 = time.time()
df = P.load(line); te = P.load(line, "test")
pred = np.asarray(getattr(M, name)(df, te, line), float)
d = P.RESULTS / "final_parts"; d.mkdir(exist_ok=True)
pd.DataFrame({"row_id": te["row_id"].to_numpy(), "line": line, "exposure": te["exposure"].to_numpy(), f"p_{name}": pred}
             ).to_parquet(d / f"{name}_{line}.parquet", index=False)
print(name, line, f"{time.time() - t0:.0f}s rate {pred.sum() / te['exposure'].sum():.1f}", flush=True)
