"""Third judge (20/10): with the actual large losses instead of the spread loading.
target_unc = target_rep (reported attritional, developed) + excess over the large-loss cap of the row's claims
(on-level). prediction = (pred − LL) × pi_eff + LL_row × pi_eff, level realigned by line × fold.
Used to assess HOW the large-loss loading is allocated (the other two judges see it as flat or not at all).
Very noisy (few large losses per year): to be read together with the other two.
Writes <results>/judge_unc.parquet (row_id, excess_onl)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, LINES, TARGET_YEAR  # noqa: E402
INTERIM = Path(CFG["interim"]); MODEL = Path(CFG["model_dir"]); RES = Path(CFG["results"])
dev = pd.read_parquet(INTERIM / "development_onlevel.parquet")
out = []
for line in LINES:
    f = MODEL / f"model_{line}.parquet"
    ex = [c for c in pq.ParquetFile(f).schema.names if c.startswith("excess_")]
    df = pd.read_parquet(f, columns=["row_id", "year"] + ex)
    dl = dev[dev["line"] == line]; yrs = df["year"].to_numpy(); tot = np.zeros(len(df))
    for c in ex:
        p = c[len("excess_"):]
        r = dl[dl["peril"] == p].set_index("ay").reindex(yrs)
        onl = ((1 + r["freq_trend"].to_numpy()) * (1 + r["sev_trend"].to_numpy())) ** (TARGET_YEAR - yrs)
        tot += df[c].to_numpy() * np.nan_to_num(onl, nan=1.0)
    out.append(pd.DataFrame({"row_id": df["row_id"].to_numpy(), "excess_onl": tot}))
    print(line, "rows with excess", int((tot > 0).sum()), "total excess M", round(tot.sum() / 1e6, 1))
pd.concat(out).to_parquet(RES / "judge_unc.parquet", index=False)
