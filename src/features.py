"""Passo 2a — feature di modellazione per linea (training 2021-24 e test 2025).

Aggiunge a data/model/{model,test}_<line>.parquet:
  - esperienza sinistri della polizza negli anni PRECEDENTI (solo AY < anno della riga):
    numero, costo, numero nell'anno prima, anni osservati;
  - esperienza del cliente sulle ALTRE linee negli anni precedenti;
  - numero di linee che il cliente ha nell'anno (motor/buildings/contents);
  - variabili derivate (età veicolo/edificio, mese di inizio, frazione d'anno, ecc.).
Le colonne che non entrano nel modello sono elencate in DROP con il motivo.
Uso: python -I features.py <interim> <model_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from market_config import LINES  # noqa: E402

# escluse dal modello, con motivo (governance / leakage)
DROP = {
    "renewal_decision": "solo nel training: informazione futura",
    "written_premium": "prezzo attuale dell'assicuratore: modello circolare (testato a parte)",
    "exposure_raw": "esposizione originale con errori, sostituita",
    "construction_type_raw": "codice originale, sostituito da construction_type",
    "email_domain": "nessun legame plausibile con il rischio; possibile proxy",
    "date_of_birth": "sostituita dall'età",
    "customer_since_year": "sostituita da customer_tenure",
    "year_built": "sostituita da building_age",
}
ID_COLS = ["row_id", "policy_id", "customer_id", "address_id"]


def experience(claims: pd.DataFrame, keys: pd.DataFrame, by: str, line_filter, prefix: str) -> pd.DataFrame:
    """Per ogni (by, year) in keys: sinistri con AY < year (e opzionalmente di certe linee)."""
    c = claims[line_filter(claims)][[by, "ay", "net_capped"]]
    agg = c.groupby([by, "ay"]).agg(n=("net_capped", "size"), cost=("net_capped", "sum")).reset_index()
    out = []
    for y in sorted(keys["year"].unique()):
        past = agg[agg["ay"] < y]
        s = past.groupby(by).agg(n=("n", "sum"), cost=("cost", "sum"))
        last = past[past["ay"] == y - 1].groupby(by)["n"].sum().rename("n_last")
        s = s.join(last, how="outer").fillna(0)
        s["year"] = y
        out.append(s.reset_index())
    e = pd.concat(out, ignore_index=True)
    e = e.rename(columns={"n": f"{prefix}_n", "cost": f"{prefix}_cost", "n_last": f"{prefix}_n_last"})
    return e


def main(interim: str, model_dir: str) -> None:
    global FIRST_YEAR
    FIRST_YEAR = int(pd.concat([pd.read_parquet(Path(interim) / f"train_{l}.parquet", columns=["year"]) for l in LINES])["year"].min())
    interim, model_dir = Path(interim), Path(model_dir)
    claims = pd.read_parquet(interim / "claims_clean.parquet")
    pol_line = claims["policy_id"].str[0].map({"M": "motor", "B": "buildings", "C": "contents"})
    claims = claims.assign(line=pol_line)
    # customer_id dei sinistri: dalle righe abbinate
    rows = []
    for line in LINES:
        p = pd.read_parquet(interim / f"train_{line}.parquet", columns=["row_id", "customer_id", "address_id"])
        rows.append(p)
    rc = pd.concat(rows)
    claims = claims.merge(rc, on="row_id", how="left")
    # storia dell'indirizzo: tutti i sinistri (ogni linea, ogni cliente) sull'indirizzo della riga, solo anni precedenti
    addr_agg = claims.groupby(["address_id", "ay"]).agg(n=("net_capped", "size"), cost=("net_capped", "sum")).reset_index()

    # clienti per indirizzo per anno (training + test, tutte le linee)
    occ = []
    for line in LINES:
        for kind in ("train", "test"):
            p = pd.read_parquet(interim / f"{kind}_{line}.parquet", columns=["customer_id", "address_id", "period_start"])
            p["year"] = pd.to_datetime(p["period_start"]).dt.year
            occ.append(p[["customer_id", "address_id", "year"]])
    occ = pd.concat(occ).drop_duplicates()
    addr_cust = occ.groupby(["address_id", "year"])["customer_id"].nunique().rename("addr_n_customers").reset_index()

    # linee detenute dal cliente per anno (training + test)
    hold = []
    for line in LINES:
        for kind in ("train", "test"):
            p = pd.read_parquet(interim / f"{kind}_{line}.parquet", columns=["customer_id", "period_start"])
            p["year"] = pd.to_datetime(p["period_start"]).dt.year
            hold.append(p[["customer_id", "year"]].drop_duplicates().assign(line=line))
    hold = pd.concat(hold).drop_duplicates()
    n_lines = hold.groupby(["customer_id", "year"]).size().rename("cust_n_lines").reset_index()
    holds = hold.assign(v=1).pivot_table(index=["customer_id", "year"], columns="line", values="v", fill_value=0)
    holds.columns = [f"cust_has_{c}" for c in holds.columns]
    holds = holds.reset_index()

    rep = {}
    for line in LINES:
        for kind, fname in (("train", f"model_{line}"), ("test", f"test_{line}")):
            df = pd.read_parquet(model_dir / f"{fname}.parquet")
            # idempotente: rimuove feature di un run precedente
            old = [c for c in df.columns if c.startswith(("pol_prev", "cust_other_prev", "cust_has_", "pol_years", "addr_"))
                   or c in ("cust_n_lines", "start_month", "starts_jan1", "ends_dec", "customer_age",
                            "licence_age", "mileage_k", "si_per_bedroom")]
            df = df.drop(columns=old)
            keys = df[["policy_id", "customer_id", "year"]]
            ep = experience(claims, keys, "policy_id", lambda c, L=line: c["line"] == L, "pol_prev")
            ec = experience(claims, keys, "customer_id", lambda c, L=line: c["line"] != L, "cust_other_prev")
            df = df.merge(ep, on=["policy_id", "year"], how="left").merge(ec, on=["customer_id", "year"], how="left")
            for c in [x for x in df.columns if x.startswith(("pol_prev", "cust_other_prev"))]:
                df[c] = df[c].fillna(0)
            # anni in cui la polizza è stata osservata prima (dentro 2021..year-1)
            first = pd.read_parquet(interim / f"train_{line}.parquet", columns=["policy_id", "year"]).groupby("policy_id")["year"].min()
            df["pol_years_seen"] = (df["year"] - df["policy_id"].map(first)).clip(lower=0).fillna(0)
            # tassi per anno osservato: confrontabili tra training (1,5 anni di storia in media) e test (fino a 4)
            yrs = df["pol_years_seen"].clip(lower=1)
            df["pol_prev_rate"] = df["pol_prev_n"] / yrs
            df["pol_prev_cost_rate"] = df["pol_prev_cost"] / yrs
            df["cust_other_prev_rate"] = df["cust_other_prev_n"] / (df["year"] - FIRST_YEAR).clip(lower=1)
            df["pol_years_seen"] = df["pol_years_seen"].clip(upper=3)
            # storia indirizzo: n e costo sinistri con AY < anno; tasso per anno di storia disponibile (NaN nel 2021)
            ad = []
            for y in sorted(df["year"].unique()):
                past = addr_agg[addr_agg["ay"] < y].groupby("address_id")[["n", "cost"]].sum()
                past["year"] = y
                ad.append(past.reset_index())
            ad = pd.concat(ad).rename(columns={"n": "addr_prev_n", "cost": "addr_prev_cost"})
            df = df.merge(ad, on=["address_id", "year"], how="left")
            df[["addr_prev_n", "addr_prev_cost"]] = df[["addr_prev_n", "addr_prev_cost"]].fillna(0)
            hist = (df["year"] - FIRST_YEAR).astype(float).where(df["year"] > FIRST_YEAR)
            df["addr_prev_rate"] = df["addr_prev_n"] / hist
            df["addr_prev_cost_rate"] = df["addr_prev_cost"] / hist
            df = df.merge(addr_cust, on=["address_id", "year"], how="left")
            df = df.merge(n_lines, on=["customer_id", "year"], how="left").merge(holds, on=["customer_id", "year"], how="left")
            ps = pd.to_datetime(df["period_start"]); pe = pd.to_datetime(df["period_end"])
            df["start_month"] = ps.dt.month
            df["starts_jan1"] = (ps.dt.month == 1) & (ps.dt.day == 1)
            df["ends_dec"] = pe.dt.month == 12
            # colonne derivate solo se le sorgenti esistono (revisione Qwen 19/10: Parte 2 con colonne rinominate/assenti)
            if "birth_year" in df:
                df["customer_age"] = df["year"] - df["birth_year"]
            if line == "motor" and {"driver_age", "years_licensed"} <= set(df.columns):
                df["licence_age"] = df["driver_age"] - df["years_licensed"]
            if line == "motor" and "annual_mileage" in df:
                df["mileage_k"] = df["annual_mileage"] / 1000
            if "sum_insured" in df and "bedrooms" in df:
                df["si_per_bedroom"] = df["sum_insured"] / df["bedrooms"].clip(lower=1)
            df.to_parquet(model_dir / f"{fname}.parquet", index=False)
            rep[f"{kind}_{line}"] = {"rows": len(df),
                                     "pol_prev_n_mean": round(float(df["pol_prev_n"].mean()), 4),
                                     "cust_other_prev_n_mean": round(float(df["cust_other_prev_n"].mean()), 4),
                                     "cust_n_lines_dist": df["cust_n_lines"].value_counts().sort_index().to_dict()}
    (model_dir / "report_features.json").write_text(json.dumps({"drop": DROP, "stats": rep}, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:3])
