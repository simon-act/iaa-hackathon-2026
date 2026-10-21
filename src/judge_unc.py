"""Terzo giudice (20/10): con i grandi sinistri veri invece del caricamento spalmato.
target_unc = target_rep (attritivo denunciato sviluppato) + eccesso oltre il taglio dei grandi sinistri della riga
(on-level). previsione = (pred − LL) × pi_eff + LL_riga × pi_eff, livello riallineato per linea × fold.
Serve a valutare COME si ripartisce il caricamento large loss (gli altri due giudici lo vedono piatto o non lo vedono).
Molto rumoroso (pochi grandi sinistri per anno): si legge insieme agli altri due.
Scrive <results>/judge_unc.parquet (row_id, excess_onl)."""
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
    print(line, "righe con eccesso", int((tot > 0).sum()), "eccesso tot M", round(tot.sum() / 1e6, 1))
pd.concat(out).to_parquet(RES / "judge_unc.parquet", index=False)
