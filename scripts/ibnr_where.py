"""21/10 (Simone's question): should IBNR be placed only on rows that already have a claim, or spread across all rows?
Emulation: valuation at 31/12 of the accident year (AY 2021–2023). 'Late' claims = occurred in the AY and
reported after 31/12. For each row: does it already have a claim reported by 31/12 (same line)?
- share of late claims falling on rows with an already-reported claim, vs the exposure share of those rows;
- lift = late claims per unit of expectation on rows with a claim / on the others. Expectation ∝ exposure, by period-end
  quarter (which determines how much is still to be reported); for AY 2023 also ∝ the OOF blend prediction (removes the
  risk already explained by the variables).
Output results_s3/ibnr_where.csv"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
c = pd.read_parquet(ROOT / "data/interim_s3/claims_clean.parquet", columns=["row_id", "line", "loss_date", "report_date", "ay"])
O = pd.read_parquet(ROOT / "results_s3/oof/blend_opt.parquet", columns=["row_id", "fold", "pred"]).set_index("row_id")
out = []
for line in ["motor", "buildings", "contents"]:
    d = pd.read_parquet(ROOT / f"data/model_s3/model_{line}.parquet", columns=["row_id", "year", "exposure", "period_end"])
    for ay in (2021, 2022, 2023):
        r = d[d.year == ay].copy()
        cut = pd.Timestamp(f"{ay}-12-31")
        x = c[(c.line == line) & (c.ay == ay)]
        early = x[x.report_date <= cut].groupby("row_id").size()
        late = x[x.report_date > cut].groupby("row_id").size()
        r["early"] = r.row_id.map(early).fillna(0) > 0
        r["late"] = r.row_id.map(late).fillna(0)
        r["q"] = pd.to_datetime(r.period_end).dt.quarter.astype(str) + pd.to_datetime(r.period_end).dt.year.astype(str)
        for base in (["esposizione"] + (["previsione OOF"] if ay == 2023 else [])):
            w = r.exposure.to_numpy() if base == "esposizione" else O.reindex(r.row_id)["pred"].to_numpy()
            r["w"] = w
            # expected late claims per row: weight × late-claim rate of the period-end quarter
            rate = r.groupby("q").apply(lambda z: z.late.sum() / z.w.sum())
            r["E"] = r.w * r.q.map(rate)
            g = r.groupby("early")[["late", "E", "exposure"]].sum()
            lift = (g.loc[True, "late"] / g.loc[True, "E"]) / (g.loc[False, "late"] / g.loc[False, "E"])
            out.append(dict(line=line, ay=ay, atteso=base, quota_righe_con_sinistro=round(r.early.mean(), 4),
                            quota_esposizione=round(g.loc[True, "exposure"] / g.exposure.sum(), 4),
                            quota_tardivi=round(g.loc[True, "late"] / g.late.sum(), 4), tardivi=int(g.late.sum()),
                            lift=round(lift, 2), lift_se=round(lift / np.sqrt(max(g.loc[True, "late"], 1)), 2)))
res = pd.DataFrame(out); res.to_csv(ROOT / "results_s3/ibnr_where.csv", index=False); print(res.to_string())
