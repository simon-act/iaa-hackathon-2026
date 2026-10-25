"""22/10 notte, T1: i contatti col servizio clienti segnalano i sinistri denunciati DOPO la fine dell'anno?
Simulazione: per le righe dell'anno Y (2022, 2023) il cutoff è il 31/12/Y; un sinistro con loss_date nel periodo della riga
e report_date > cutoff è "tardivo" (al momento del prezzo dell'anno dopo non si vede). Domanda: fra le righe senza
sinistri denunciati entro il cutoff, i contatti predicono i tardivi? Uso: python3 scripts/contacts_late.py"""
import pandas as pd, numpy as np
cl = pd.read_parquet("data/raw/public/claims.parquet")
cl["loss_date"] = pd.to_datetime(cl.loss_date); cl["report_date"] = pd.to_datetime(cl.report_date)
out = []
for line in ["motor", "buildings", "contents"]:
    m = pd.read_parquet(f"data/model/model_{line}.parquet", columns=["row_id", "policy_id", "year", "period_start", "period_end", "exposure", "customer_service_contacts", "n_total"])
    m["period_start"] = pd.to_datetime(m.period_start); m["period_end"] = pd.to_datetime(m.period_end)
    c = cl[cl.line == line].merge(m[["row_id", "policy_id", "period_start", "period_end", "year"]], on="policy_id")
    c = c[(c.loss_date >= c.period_start) & (c.loss_date <= c.period_end)]
    for Y in [2022, 2023, 2024]:
        cut = pd.Timestamp(f"{Y}-12-31")
        cy = c[c.year == Y]
        rep = cy[cy.report_date <= cut].groupby("row_id").size().rename("n_rep")
        late = cy[cy.report_date > cut].groupby("row_id").size().rename("n_late")
        r = m[m.year == Y].set_index("row_id").join(rep).join(late).fillna({"n_rep": 0, "n_late": 0})
        r["cs"] = r.customer_service_contacts.clip(upper=4)
        z = r[r.n_rep == 0]
        g = z.groupby("cs").agg(righe=("n_late", "size"), tardivi=("n_late", "sum"), tasso=("n_late", "mean"))
        g["quota_tardivi_tot"] = g.tardivi / r.n_late.sum()
        g = g.assign(linea=line, anno=Y, n_late_tot=r.n_late.sum(), n_rep_tot=r.n_rep.sum())
        out.append(g.reset_index())
        # i contatti delle righe con tardivi vs righe senza niente
        print(line, Y, "tardivi/denunciati:", round(r.n_late.sum() / max(r.n_rep.sum(), 1), 4),
              "contatti medi: 0 sinistri", round(z[z.n_late == 0].customer_service_contacts.mean(), 3),
              "| solo tardivi", round(z[z.n_late > 0].customer_service_contacts.mean(), 3),
              "| denunciati", round(r[r.n_rep > 0].customer_service_contacts.mean(), 3))
o = pd.concat(out); o.to_csv("results_s3/contacts_late.csv", index=False)
print(o.pivot_table(index=["linea", "cs"], columns="anno", values="tasso").round(4))
print(o.pivot_table(index=["linea", "cs"], columns="anno", values="quota_tardivi_tot").round(3))
