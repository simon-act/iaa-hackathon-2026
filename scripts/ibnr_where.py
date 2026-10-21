"""21/10 (domanda di Simone): l'IBNR va messo solo sulle righe che hanno già un sinistro, o spalmato su tutte?
Emulazione: valutazione al 31/12 dell'anno di accadimento (AY 2021–2023). Sinistri 'tardivi' = accaduti nell'AY e
denunciati dopo il 31/12. Per ogni riga: ha già un sinistro denunciato entro il 31/12 (stessa linea)?
- quota dei tardivi che cade su righe con sinistro già denunciato, contro la quota di esposizione di quelle righe;
- lift = tardivi per unità di atteso sulle righe con sinistro / sulle altre. Atteso ∝ esposizione, per trimestre di fine
  periodo (che decide quanta parte è ancora da denunciare); per AY 2023 anche ∝ previsione OOF del blend (toglie il
  rischio già spiegato dalle variabili).
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
            # atteso dei tardivi per riga: peso × tasso dei tardivi del trimestre di fine periodo
            rate = r.groupby("q").apply(lambda z: z.late.sum() / z.w.sum())
            r["E"] = r.w * r.q.map(rate)
            g = r.groupby("early")[["late", "E", "exposure"]].sum()
            lift = (g.loc[True, "late"] / g.loc[True, "E"]) / (g.loc[False, "late"] / g.loc[False, "E"])
            out.append(dict(line=line, ay=ay, atteso=base, quota_righe_con_sinistro=round(r.early.mean(), 4),
                            quota_esposizione=round(g.loc[True, "exposure"] / g.exposure.sum(), 4),
                            quota_tardivi=round(g.loc[True, "late"] / g.late.sum(), 4), tardivi=int(g.late.sum()),
                            lift=round(lift, 2), lift_se=round(lift / np.sqrt(max(g.loc[True, "late"], 1)), 2)))
res = pd.DataFrame(out); res.to_csv(ROOT / "results_s3/ibnr_where.csv", index=False); print(res.to_string())
