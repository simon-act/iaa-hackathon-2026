"""Mercato finto per provare la catena della Parte 2 PRIMA che escano i dati veri.

Prende i dati grezzi della Parte 1, tiene il 15% delle polizze e introduce differenze "realistiche" di un mercato
emergente, tutte note, così possiamo verificare che l'audit le trovi e che l'adattamento le gestisca:
  - storia più corta: solo 2022–2024 (il test resta il 2025 delle polizze tenute)
  - frequenza diversa per regione (×0,6 … ×1,6) e severità ×1,35 (inflazione / valuta)
  - sum_insured in migliaia (÷1000) per tutte le righe casa (unità diversa)
  - motor: telematics_score assente; ncb_years rinominata no_claims_years; colonna nuova vehicle_segment
  - construction_type con un livello nuovo "adobe" (10% delle righe)
  - sinistri: stato "declined" al posto di "repudiated"; 2% di sinistri duplicati in più
Uso: python -I scripts/make_fake_market.py data/raw data/fake_market [quota_polizze=0.15]
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LINES = ["motor", "buildings", "contents"]
REGION_FACTOR = {}


def keep_mask(ids: pd.Series, share: float) -> np.ndarray:
    h = ids.map(lambda s: int(hashlib.md5(("fake" + s).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return (h < share).to_numpy()


def main(raw, out, share=0.15):
    raw, out = Path(raw), Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    addresses = pd.read_parquet(raw / "addresses.parquet")
    regions = sorted(addresses["region"].dropna().unique())
    fac = dict(zip(regions, np.round(rng.uniform(0.6, 1.6, len(regions)), 2)))
    keep_pol = {}
    for line in LINES:
        p = pd.read_parquet(raw / f"policies_{line}.parquet")
        p = p[keep_mask(p["policy_id"], share) & (pd.to_datetime(p["period_start"]).dt.year >= 2022)].copy()
        keep_pol[line] = set(p["policy_id"])
        if "sum_insured" in p.columns:
            p["sum_insured"] = p["sum_insured"] / 1000.0
        if line == "motor":
            p = p.drop(columns=["telematics_score"]).rename(columns={"ncb_years": "no_claims_years"})
            p["vehicle_segment"] = rng.choice(["A", "B", "C", "SUV"], len(p), p=[.3, .35, .2, .15])
        if "construction_type" in p.columns:
            m = rng.random(len(p)) < 0.10
            p.loc[m, "construction_type"] = "adobe"
        p.to_parquet(out / f"policies_{line}.parquet", index=False)
        t = pd.read_parquet(raw / f"test_{line}.parquet")
        t = t[t["policy_id"].isin(keep_pol[line])].copy()
        if "sum_insured" in t.columns:
            t["sum_insured"] = t["sum_insured"] / 1000.0
        if line == "motor":
            t = t.drop(columns=["telematics_score"]).rename(columns={"ncb_years": "no_claims_years"})
            t["vehicle_segment"] = rng.choice(["A", "B", "C", "SUV"], len(t), p=[.3, .35, .2, .15])
        if "construction_type" in t.columns:
            m = rng.random(len(t)) < 0.10
            t.loc[m, "construction_type"] = "adobe"
        t.to_parquet(out / f"test_{line}.parquet", index=False)
    allpol = set().union(*keep_pol.values())
    c = pd.read_parquet(raw / "claims.parquet")
    c = c[c["policy_id"].isin(allpol) & (pd.to_datetime(c["loss_date"]).dt.year >= 2022)].copy()
    # frequenza per regione: tengo ogni sinistro con probabilità fac/1.6 (così il max è 1), poi severità ×1,35
    pol_addr = pd.concat([pd.read_parquet(raw / f"policies_{l}.parquet", columns=["policy_id", "address_id"]) for l in LINES]).drop_duplicates("policy_id")
    reg = pol_addr.merge(addresses[["address_id", "region"]], on="address_id", how="left").set_index("policy_id")["region"]
    pk = c["policy_id"].map(reg).map(fac).fillna(1.0) / 1.6
    c = c[rng.random(len(c)) < pk.to_numpy()].copy()
    for col in ["paid_to_date", "incurred_to_date", "recoveries"]:
        c[col] = (c[col] * 1.35).round(2)
    c["status"] = c["status"].replace({"repudiated": "declined"})
    dup = c.sample(frac=0.02, random_state=1).copy(); dup["claim_id"] = dup["claim_id"] + "D"
    c = pd.concat([c, dup], ignore_index=True)
    c.to_parquet(out / "claims.parquet", index=False)
    cust = pd.read_parquet(raw / "customers.parquet")
    cids = set().union(*[set(pd.read_parquet(out / f"policies_{l}.parquet", columns=["customer_id"])["customer_id"]) for l in LINES])
    cust[cust["customer_id"].isin(cids)].to_parquet(out / "customers.parquet", index=False)
    aids = set().union(*[set(pd.read_parquet(out / f"policies_{l}.parquet", columns=["address_id"])["address_id"]) for l in LINES])
    addresses[addresses["address_id"].isin(aids)].to_parquet(out / "addresses.parquet", index=False)
    w = pd.read_parquet(raw / "weather.parquet"); w[w["address_id"].isin(aids)].to_parquet(out / "weather.parquet", index=False)
    ss = pd.concat([pd.read_parquet(out / f"test_{l}.parquet", columns=["row_id"]) for l in LINES]); ss["prediction"] = 0.0
    ss.to_parquet(out / "sample_submission.parquet", index=False)
    pd.Series(fac).to_json(out / "_fattori_regione_veri.json")
    print("mercato finto scritto in", out, "| fattori regione:", fac)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.15)
