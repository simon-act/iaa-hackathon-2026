"""Test submission (pipeline check): prediction = 2025 line level × exposure.
No risk relativities: it only checks format and level.
Usage: python -I baseline_submission.py <model_dir> <interim_dir> <raw_dir> <out_csv>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from metrics import d2_tweedie

model, interim, raw, out = map(Path, sys.argv[1:5])
lvl = pd.read_parquet(interim / "level_2025.parquet")["bc_2025_total"]
subs, chk = [], []
for line in ("motor", "buildings", "contents"):
    te = pd.read_parquet(model / f"test_{line}.parquet", columns=["row_id", "exposure"])
    subs.append(pd.DataFrame({"row_id": te["row_id"], "prediction": lvl[line] * te["exposure"]}))
    # local check on 2024: y = ultimate per row (not trended to 2025) + actual excess amounts
    m = pd.read_parquet(model / f"model_{line}.parquet")
    m = m[m["year"] == 2024]
    exc = m[[c for c in m.columns if c.startswith("excess_")]].sum(axis=1)
    y = m["ult_capped"] + exc
    rate = y.sum() / m["exposure"].sum()
    chk.append(dict(line=line, d2_const_per_row=d2_tweedie(y, np.full(len(y), y.mean())),
                    d2_rate_x_expo=d2_tweedie(y, rate * m["exposure"])))
sub = pd.concat(subs, ignore_index=True)
ss = pd.read_parquet(raw / "sample_submission.parquet")
assert set(sub.row_id) == set(ss.row_id) and len(sub) == len(ss), "row_id do not match sample_submission"
assert (sub.prediction >= 0).all()
sub = ss[["row_id"]].merge(sub, on="row_id", how="left")
sub.to_csv(out, index=False)
print(pd.DataFrame(chk).round(4)); print(sub.describe()); print("sample constant:", ss.prediction.iloc[0])
