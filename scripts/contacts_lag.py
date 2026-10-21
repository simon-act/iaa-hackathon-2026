"""21/10 (domanda di Simone): customer_service_contacts e renewal_decision possono servire come segnale?
1) Distribuzione nel test 2025 contro training per numero di sinistri denunciati (fatto a parte).
2) Test onesto: contatti dell'anno PRIMA (noti al momento del prezzo) contro l'A/E del blend OOF nell'anno dopo,
   solo per righe dell'anno prima senza sinistri denunciati (contatti 'di base', non generati dai sinistri).
3) renewal_decision dell'anno prima contro l'A/E dell'anno dopo.
A/E sul giudice senza spalmatura (come spie_scan): A = target_rep, E = (pred - ll_expo) * pi_eff."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
O = pd.read_parquet(ROOT / "results_s3/oof/blend_opt.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"]
J = pd.read_parquet(ROOT / "results_s3/judge_own.parquet", columns=["row_id", "target_rep", "pi_eff", "ll_expo"]).set_index("row_id")
raw = {l: pd.read_parquet(ROOT / f"data/raw/policies_{l}.parquet", columns=["policy_id", "period_start", "renewal_decision"]) for l in ["motor", "buildings", "contents"]}
out = []
for l in ["motor", "buildings", "contents"]:
    nm = pq.ParquetFile(ROOT / f"data/model_s3/model_{l}.parquet").schema.names
    nc = [c for c in nm if c.startswith("n_") and c != "n_total"]
    d = pd.read_parquet(ROOT / f"data/model_s3/model_{l}.parquet", columns=["row_id", "policy_id", "year", "customer_service_contacts"] + nc)
    d["N"] = d[nc].sum(axis=1)
    r = raw[l].assign(year=pd.to_datetime(raw[l].period_start).dt.year).drop_duplicates(["policy_id", "year"])[["policy_id", "year", "renewal_decision"]]
    d = d.merge(r, on=["policy_id", "year"], how="left")
    prev = d[["policy_id", "year", "customer_service_contacts", "N", "renewal_decision"]].copy(); prev["year"] += 1
    prev = prev.drop_duplicates(["policy_id", "year"]).rename(columns={"customer_service_contacts": "c_prev", "N": "N_prev", "renewal_decision": "rd_prev"})
    cur = d[d.year >= 2023].merge(prev, on=["policy_id", "year"], how="inner")
    p = O.reindex(cur.row_id).to_numpy(); j = J.reindex(cur.row_id)
    ok = np.isfinite(p)
    cur = cur[ok]; j = j[ok]; p = p[ok]
    cur["A"] = j["target_rep"].to_numpy(); cur["E"] = np.clip(p - j["ll_expo"].to_numpy(), 1e-3, None) * j["pi_eff"].to_numpy()
    base = cur[cur.N_prev == 0]
    for col, df in (("contatti_anno_prima (anno prima senza sinistri)", base.assign(k=base.c_prev.clip(upper=4).astype(str))),
                    ("renewal_decision anno prima", cur.assign(k=cur.rd_prev.astype(str)))):
        for fold in (2023, 2024):
            g = df[df.year == fold].groupby("k").agg(righe=("A", "size"), A=("A", "sum"), E=("E", "sum"))
            g["AE"] = (g.A / g.E).round(3)
            # errore standard approssimato dell'A/E: da sinistri con sev. variabile -> uso varianza empirica di A
            sd = df[df.year == fold].groupby("k")["A"].agg(lambda a: np.sqrt((a ** 2).sum()))
            g["se"] = (sd / g.E).round(3)
            for k, row in g.iterrows():
                out.append(dict(line=l, var=col, fold=fold, valore=k, righe=int(row.righe), AE=row.AE, se=row.se))
res = pd.DataFrame(out)
res.to_csv(ROOT / "results_s3/contacts_lag.csv", index=False)
print(res.pivot_table(index=["line", "var", "valore"], columns="fold", values=["righe", "AE", "se"]).to_string())
