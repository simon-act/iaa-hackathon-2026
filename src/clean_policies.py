"""Passo 1a — pulizia delle tabelle di polizza (training e test).

Regole deterministiche; ogni correzione lascia un flag nella riga e un contatore
nel report JSON, così si può riconciliare e spiegare.
Uso: python -I clean_policies.py <raw_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, LINES  # noqa: E402

FIX = CFG["fixes"]
# W e T non documentati nella Parte 1: nel test esiste solo W con quota = W+T del training e "timber" non compare mai
CONSTRUCTION_MAP = FIX["construction_map"]
SI_FACTOR = 1000.0


def clean_one(df: pd.DataFrame, line: str, addresses: pd.DataFrame, rep: dict) -> pd.DataFrame:
    df = df.copy()
    if CFG["rename"]:
        df = df.rename(columns=CFG["rename"])
    for c, k in FIX.get("scale", {}).items():      # unità diverse dal mercato di partenza (dall'audit)
        if c in df.columns:
            df[c] = df[c] * float(k); rep[f"scaled_{c}"] = float(k)
    # customer_id motor: ~10% ha perso gli zeri iniziali (letto come numero); gli ID sono a 8 cifre (Parte 1)
    if FIX["customer_id_zfill"]:
        short = df["customer_id"].astype(str).str.len() < FIX["customer_id_zfill"]
        rep["customer_id_zero_padded"] = int(short.sum())
        df["customer_id"] = df["customer_id"].astype(str).str.zfill(FIX["customer_id_zfill"])
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    days = (df["period_end"] - df["period_start"]).dt.days + 1
    expo_clean = days / 365.0
    diff = (df["exposure"] - expo_clean).abs()
    df["flag_exposure_fixed"] = diff > 0.5 / 365  # più di mezzo giorno di scarto
    rep["exposure_fixed"] = int(df["flag_exposure_fixed"].sum())
    rep["exposure_fixed_days_coded"] = int((df["exposure"] > 1.5).sum())
    rep["exposure_fixed_365_360"] = int(df["exposure"].between(1.013, 1.015).sum())
    rep["exposure_sum_original"] = float(df["exposure"].sum())
    df["exposure_raw"] = df["exposure"]
    if FIX["exposure_from_dates"]:
        df["exposure"] = expo_clean
    rep["exposure_sum_clean"] = float(df["exposure"].sum())
    df["year"] = df["period_start"].dt.year

    if "sum_insured" in df.columns:
        a = addresses[["address_id", "bedrooms"]]
        tmp = df[["address_id", "sum_insured"]].merge(a, on="address_id", how="left")
        per_bed = tmp["sum_insured"] / tmp["bedrooms"].clip(lower=1)
        med = per_bed.median()
        thr = FIX["sum_insured_div_if_per_bedroom_over"]
        big = (per_bed > thr * med).to_numpy() if thr else np.zeros(len(df), bool)
        df["flag_si_div1000"] = big
        df.loc[big, "sum_insured"] = df.loc[big, "sum_insured"] / SI_FACTOR
        rep["sum_insured_div1000"] = int(big.sum())

    if "construction_type" in df.columns:
        raw = df["construction_type"].astype("string").str.strip().str.lower()
        df["construction_type_raw"] = df["construction_type"]
        df["construction_type"] = raw.map(CONSTRUCTION_MAP).fillna("missing")
        rep["construction_levels"] = df["construction_type"].value_counts().to_dict()

    if "annual_mileage" in df.columns:
        z = df["annual_mileage"] == 0
        df["flag_mileage_zero"] = z
        df.loc[z, "annual_mileage"] = np.nan
        rep["mileage_zero_to_missing"] = int(z.sum())

    if line == "motor" and {"ncb_years", "years_licensed", "driver_age"} <= set(df.columns):
        df["flag_ncb_gt_licensed"] = df["ncb_years"] > df["years_licensed"]
        df["flag_licensed_gt_age"] = df["years_licensed"] > (df["driver_age"] - 16)
        rep["ncb_gt_licensed"] = int(df["flag_ncb_gt_licensed"].sum())
        rep["licensed_gt_age"] = int(df["flag_licensed_gt_age"].sum())

    # missing espliciti per le categoriche (i numerici restano NaN: li gestirà il modello)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) in ("string", "str"):
            if c in ("row_id", "policy_id", "customer_id", "address_id", "renewal_decision"):
                continue
            n = df[c].isna().sum()
            if n:
                df[c] = df[c].astype("string").fillna("missing")
    return df


def main(raw: str, out: str) -> None:
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    addresses = pd.read_parquet(raw / "addresses.parquet")
    report = {}
    for line in LINES:
        for kind, fname in (("train", f"policies_{line}"), ("test", f"test_{line}")):
            df = pd.read_parquet(raw / f"{fname}.parquet")
            rep = {"rows": len(df)}
            cleaned = clean_one(df, line, addresses, rep)
            cleaned.to_parquet(out / f"{kind}_{line}.parquet", index=False)
            report[f"{kind}_{line}"] = rep
    (out / "report_clean_policies.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
