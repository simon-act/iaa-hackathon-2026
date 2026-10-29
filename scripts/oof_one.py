"""A single line × fold of oof.py (the container restarts on long jobs). Then: oof_one.py merge <name>.
Usage: IAA_MARKET=... python3 -I scripts/oof_one.py <name> <line> <fold>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
import oof as O  # noqa: E402

name = sys.argv[1]
d = P.RESULTS / "oof_parts"; d.mkdir(parents=True, exist_ok=True)
if sys.argv[2] == "merge":
    o = pd.concat([pd.read_parquet(d / f"{name}_{l}_{f}.parquet") for l in P.LINES for f in P.FOLDS], ignore_index=True)
    o.to_parquet(P.RESULTS / "oof" / f"{name}.parquet", index=False); print(O.score(o))
    sys.exit()
line, fname = sys.argv[2], sys.argv[3]
fn = O.get_fn(name)
df = P.load(line); df = df[~P.is_holdout(df["policy_id"])]
tr_years, va_year = P.FOLDS[fname]
tr, va = df[df["year"].isin(tr_years)], df[df["year"] == va_year]
pred = np.asarray(fn(tr, va, line), float)
pd.DataFrame({"row_id": va["row_id"].to_numpy(), "line": line, "fold": fname, "year": va_year,
              "exposure": va["exposure"].to_numpy(), "target_b": va[P.TARGET].to_numpy(), "pred": pred}
             ).to_parquet(d / f"{name}_{line}_{fname}.parquet", index=False)
print(name, line, fname, "D2", round(P.d2(va[P.TARGET].to_numpy(), pred), 5), "own", round(P.d2_own(va["row_id"], pred), 5), flush=True)
