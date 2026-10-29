"""Judge without smoothing (19/10 evening, from the external review).

target_b = Σ_p [reported cost × IBNER + expected IBNR spread ∝ exposure×(1−π)] × on-level × rec_adj + LL × exposure.
The two spread components (IBNR and LL) are equal for rows with the same exposure and completeness, so a D² on
target_b rewards predictions flatter than the truth. Here they are separated:
  target_rep = Σ_p cost_p × IBNER_p(AY) × onlevel_p × rec_adj     (reported claims only, developed)
  pi_eff     = Σ_p s_p(line, AY) × π_p,row                         (expected share of cost already reported)
              with s_p = peril share of the line×AY ultimate cost (development table)
"Own" judge: D²( target_rep , (pred − LL×exposure) × pi_eff × c ), with c realigning the level by
line×fold (only the shape is judged; the level is calibrated separately). This is the "completeness offset" approach.
Writes <results>/judge_own.parquet (row_id, line, year, exposure, target_b, target_rep, pi_eff, ll_expo).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, LINES, TARGET_YEAR  # noqa: E402

INTERIM = Path(CFG["interim"]); MODEL = Path(CFG["model_dir"]); RES = Path(CFG["results"])


def build():
    dev = pd.read_parquet(INTERIM / "development_onlevel.parquet")
    lvl = pd.read_parquet(INTERIM / "level_2025.parquet")
    out = []
    for line in LINES:
        dl = dev[dev["line"] == line]
        perils = sorted(dl["peril"].unique())
        cols = ["row_id", "year", "exposure", "target_b", "ult25_capped", "rec_adj"] + \
               [f"cost_{p}" for p in perils] + [f"pi_{p}" for p in perils]
        df = pd.read_parquet(MODEL / f"model_{line}.parquet", columns=cols)
        share = dl.pivot(index="ay", columns="peril", values="cost_ult")
        share = share.div(share.sum(axis=1), axis=0)
        rep = np.zeros(len(df)); pie = np.zeros(len(df))
        yrs = df["year"].to_numpy()
        for p in perils:
            r = dl[dl["peril"] == p].set_index("ay").reindex(yrs)
            onl = ((1 + r["freq_trend"].to_numpy()) * (1 + r["sev_trend"].to_numpy())) ** (TARGET_YEAR - yrs)
            rep += df[f"cost_{p}"].to_numpy() * r["ibner_factor"].to_numpy() * onl
            pie += share[p].reindex(yrs).to_numpy() * df[f"pi_{p}"].to_numpy()
        rep *= df["rec_adj"].to_numpy()
        ll = float(lvl.loc[line, "ll_load_2025"])
        o = pd.DataFrame({"row_id": df["row_id"].to_numpy(), "line": line, "year": yrs,
                          "exposure": df["exposure"].to_numpy(), "target_b": df["target_b"].to_numpy(),
                          "target_rep": rep, "pi_eff": pie, "ll_expo": ll * df["exposure"].to_numpy()})
        # check: target_b − LL − target_rep = spread IBNR ≥ 0
        o["ibnr"] = o["target_b"] - o["ll_expo"] - o["target_rep"]
        out.append(o)
        g = o.groupby("year")[["target_b", "target_rep", "ibnr", "ll_expo", "exposure"]].sum()
        g["ibnr_share"] = g["ibnr"] / g["target_b"]; g["ll_share"] = g["ll_expo"] / g["target_b"]
        g["flat_per_expo"] = (g["ibnr"] + g["ll_expo"]) / g["exposure"]
        print(line, "min ibnr", round(float(o["ibnr"].min()), 4)); print(g[["ibnr_share", "ll_share", "flat_per_expo"]].round(3))
        del df
    j = pd.concat(out, ignore_index=True)
    RES.mkdir(parents=True, exist_ok=True)
    j.to_parquet(RES / "judge_own.parquet", index=False)
    return j


if __name__ == "__main__":
    build()
