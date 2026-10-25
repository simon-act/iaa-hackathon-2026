"""Prepara i dati del loop Parte 2 (una volta): righe emerging 2023–24 con variabili ammesse, prior del developed (totale e per
perilo, results_p2/source_pred_<linea>.parquet), colonne del giudice. Dati di gara: data_p2.parquet NON va nel repo.
Uso: IAA_MARKET=configs/parte2.json python3 autoresearch_p2/build_data.py"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import prepare as P  # noqa: E402
J = pd.read_parquet(P.RESULTS / "judge_own.parquet")[["row_id", "target_rep", "pi_eff", "ll_expo"]]
U = pd.read_parquet(P.RESULTS / "judge_unc.parquet")
out = []
for l in P.LINES:
    df = P.load(l)
    feats = [c for c in P.allowed_features(df) if c not in ("exposure",)]
    s = pd.read_parquet(P.RESULTS / f"source_pred_{l}.parquet")
    s = s.rename(columns={c: f"prior_{c}" for c in s.columns if c != "row_id"})
    d = df[["row_id", "policy_id", "year", "exposure", P.TARGET] + feats].merge(s, on="row_id").merge(J, on="row_id").merge(U, on="row_id", how="left")
    d["line"] = l
    out.append(d)
o = pd.concat(out, ignore_index=True)
o.to_parquet(HERE / "data_p2.parquet", index=False)
print(o.shape, o.groupby("line").size().to_dict())
