"""Passo 1e — dataset di modellazione.

(a) model_<line>.parquet: una riga per riga di polizza 2021-24 con feature (polizza + cliente +
    indirizzo + meteo dell'anno), esposizione, per ogni perilo: n sinistri denunciati, costo netto
    tagliato, completezza attesa pi (offset di frequenza = log(exposure * pi)).
(b) stessa tabella con il target "ultimate per riga":
      ult_capped = sum_perili [ costo denunciato * fattore IBNER(AY)
                                + IBNR atteso = freq_ult(AY) * exposure * (1 - pi_riga) * sev_ult * rel_ibnr ]
    e la versione riportata al 2025 (trend) + caricamento large loss della linea.
    Controllo: somma per linea × AY = ultimate della tabella di sviluppo.
test_<line>.parquet: stesse feature per il 2025 (meteo 2025).
Uso: python -I build_datasets.py <raw_dir> <interim_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from market_config import LINES, TARGET_YEAR  # noqa: E402


def add_context(df, customers, addresses, weather):
    df = df.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))
    df = df.merge(addresses, on="address_id", how="left", suffixes=("", "_addr"))
    w = weather.rename(columns={"year": "year"})
    df = df.merge(w, on=["address_id", "year"], how="left")
    df["customer_tenure"] = df["year"] - df["customer_since_year"]
    df["building_age"] = df["year"] - df["year_built"]
    return df


def main(raw, interim, out):
    raw, interim, out = Path(raw), Path(interim), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    customers = pd.read_parquet(raw / "customers.parquet").drop(columns=["surname"])
    customers["birth_year"] = pd.to_datetime(customers["date_of_birth"]).dt.year
    customers = customers.drop(columns=["date_of_birth"])
    addresses = pd.read_parquet(raw / "addresses.parquet")
    weather = pd.read_parquet(raw / "weather.parquet")
    claims = pd.read_parquet(interim / "claims_clean.parquet")
    dev = pd.read_parquet(interim / "development_onlevel.parquet")
    lvl = pd.read_parquet(interim / "level_2025.parquet")
    rep = {}

    for line in LINES:
        pol = pd.read_parquet(interim / f"train_{line}.parquet")
        comp = pd.read_parquet(interim / f"completeness_{line}.parquet")
        perils = [c for c in comp.columns if c != "row_id"]
        cl = claims[claims["line"] == line]
        agg = (cl.groupby(["row_id", "peril"])
               .agg(n=("claim_id", "size"), cost=("net_capped", "sum"), excess=("excess_over_cap", "sum"))
               .unstack("peril").fillna(0))
        agg.columns = [f"{a}_{p}" for a, p in agg.columns]
        df = pol.merge(agg, left_on="row_id", right_index=True, how="left")
        df = df.merge(comp.add_prefix("pi_").rename(columns={"pi_row_id": "row_id"}), on="row_id", how="left")
        for p in perils:
            for a in ("n", "cost", "excess"):
                col = f"{a}_{p}"
                if col not in df:
                    df[col] = 0.0
                df[col] = df[col].fillna(0.0)

        # (b) ultimate per riga
        dl = dev[dev["line"] == line].set_index(["peril", "ay"])
        ult = np.zeros(len(df)); ult25 = np.zeros(len(df))
        for p in perils:
            r = dl.xs(p, level="peril").reindex(df["year"].to_numpy())
            ibner = r["ibner_factor"].to_numpy()
            ibnr_n = r["freq_ult"].to_numpy() * df["exposure"].to_numpy() * (1 - df[f"pi_{p}"].to_numpy())
            ibnr_cost = ibnr_n * r["sev_rep_ult"].to_numpy() * r["rel_ibnr"].to_numpy()
            u = df[f"cost_{p}"].to_numpy() * ibner + ibnr_cost
            # recuperi futuri sugli aperti sono già nel fattore? no: li applico a livello AY
            # tramite il rapporto cost_rep/net_capped (vedi develop.py) -> correzione per AY
            df[f"ult_{p}"] = u
            onl = ((1 + r["freq_trend"].to_numpy()) * (1 + r["sev_trend"].to_numpy())) ** (TARGET_YEAR - df["year"].to_numpy())
            df[f"ult25_{p}"] = u * onl
            ult += u; ult25 += df[f"ult25_{p}"].to_numpy()
        df["ult_capped"] = ult
        ll = float(lvl.loc[line, "ll_load_2025"])
        df["ult25_capped"] = ult25
        df["target_b"] = ult25 + ll * df["exposure"]       # target (b) al livello 2025
        df["n_total"] = df[[f"n_{p}" for p in perils]].sum(axis=1)
        df["cost_total"] = df[[f"cost_{p}" for p in perils]].sum(axis=1)

        # riconciliazione con la tabella di sviluppo (prima della correzione recuperi)
        chk = df.groupby("year")["ult_capped"].sum()
        devsum = dev[dev["line"] == line].groupby("ay")["cost_ult"].sum()
        # la tabella di sviluppo include i recuperi futuri attesi sugli aperti: riallineo per AY
        adj = (devsum / chk).rename("rec_adj")
        df = df.merge(adj, left_on="year", right_index=True)
        for col in ["ult_capped", "ult25_capped"] + [f"ult_{p}" for p in perils] + [f"ult25_{p}" for p in perils]:
            df[col] = df[col] * df["rec_adj"]
        df["target_b"] = df["ult25_capped"] + ll * df["exposure"]
        chk2 = df.groupby("year")["ult_capped"].sum()
        rep[line] = {"rows": len(df), "perils": perils,
                     "rec_adj_by_ay": adj.round(5).to_dict(),
                     "reconciliation_max_abs_diff": float((chk2 - devsum).abs().max()),
                     "claims_matched": int(df["n_total"].sum()),
                     "claims_in_clean_table": int(len(cl)),
                     "ll_load_2025": ll,
                     "target_b_per_expo_2025": float(df["target_b"].sum() / df["exposure"].sum())}

        df = add_context(df, customers, addresses, weather)
        df.to_parquet(out / f"model_{line}.parquet", index=False)

        te = pd.read_parquet(interim / f"test_{line}.parquet")
        te["year"] = TARGET_YEAR
        te = add_context(te, customers, addresses, weather)
        te.to_parquet(out / f"test_{line}.parquet", index=False)
        rep[line]["test_rows"] = len(te)
        rep[line]["test_missing_weather"] = int(te["hail_days"].isna().sum())
        rep[line]["test_missing_customer"] = int(te["credit_band"].isna().sum())

    (out / "report_datasets.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
