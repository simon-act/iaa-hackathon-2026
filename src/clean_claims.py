"""Passo 1b — pulizia sinistri, abbinamento alla riga di polizza, taglio large loss.

Uso: python -I clean_claims.py <raw_dir> <interim_dir>
Output in <interim_dir>: claims_clean.parquet, claims_excluded.parquet, report_clean_claims.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, LINES  # noqa: E402

VALUATION = None   # fissata in main: ultimo giorno dell'ultimo anno di polizza
# soglia large loss per linea (~ P99,9 del costo netto per sinistro)
CAP = CFG["cap"]


def main(raw: str, interim: str) -> None:
    raw, interim = Path(raw), Path(interim)
    c = pd.read_parquet(raw / "claims.parquet")
    c["loss_date"] = pd.to_datetime(c["loss_date"])
    c["report_date"] = pd.to_datetime(c["report_date"])
    rep = {"claims_raw": len(c), "steps": []}
    excluded = []

    def drop(mask, reason):
        n = int(mask.sum())
        excluded.append(c[mask].assign(exclusion_reason=reason))
        rep["steps"].append({"step": reason, "removed": n,
                             "removed_incurred": round(float(c.loc[mask, "incurred_to_date"].sum()), 2)})
        return c[~mask].copy()

    c = drop(c["status"].astype(str).str.lower().isin(CFG["declined_status"]), "repudiated/declined (rifiutato)")
    key = ["policy_id", "loss_date", "peril", "incurred_to_date", "report_date", "paid_to_date", "status"]
    dup = c.sort_values("claim_id").duplicated(subset=key, keep="first").reindex(c.index)
    c = drop(dup, "duplicato esatto (stessa polizza, data, perilo, importi, denuncia)")

    # abbinamento alla riga: policy_id + loss_date dentro il periodo
    rows = []
    for line in LINES:
        p = pd.read_parquet(interim / f"train_{line}.parquet",
                            columns=["row_id", "policy_id", "period_start", "period_end"])
        rows.append(p)
    p = pd.concat(rows, ignore_index=True)
    m = c[["claim_id", "policy_id", "loss_date"]].merge(p, on="policy_id", how="left")
    m = m[(m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])]
    nmatch = m.groupby("claim_id").size()
    rep["multi_match"] = int((nmatch > 1).sum())
    c = c.merge(m[["claim_id", "row_id"]], on="claim_id", how="left")
    first_year = int(pd.to_datetime(p["period_start"]).dt.year.min())
    global VALUATION
    VALUATION = pd.Timestamp(year=int(pd.to_datetime(p["period_end"]).dt.year.max()), month=12, day=31)
    c = drop(c["row_id"].isna() & (c["loss_date"].dt.year < first_year), f"accaduto prima del {first_year} (nessuna polizza attiva)")
    c = drop(c["row_id"].isna(), "accaduto prima della decorrenza della polizza")

    # importi
    c["net_incurred"] = c["incurred_to_date"] - c["recoveries"]
    c["net_paid"] = c["paid_to_date"] - c["recoveries"]
    c["cap"] = c["line"].map(CAP)
    c["net_capped"] = c["net_incurred"].clip(upper=c["cap"])
    c["excess_over_cap"] = c["net_incurred"] - c["net_capped"]
    c["is_large"] = c["excess_over_cap"] > 0
    c["ay"] = c["loss_date"].dt.year
    c["report_lag_days"] = (c["report_date"] - c["loss_date"]).dt.days
    c["days_to_valuation"] = (VALUATION - c["loss_date"]).dt.days
    c["is_open"] = c["status"] == "open"

    rep["claims_clean"] = len(c)
    rep["large_losses"] = (c[c["is_large"]].groupby("line")
                           .agg(n=("claim_id", "size"), excess=("excess_over_cap", "sum"))
                           .round(0).to_dict(orient="index"))
    rep["by_line"] = (c.groupby("line").agg(n=("claim_id", "size"), open=("is_open", "sum"),
                                            net_incurred=("net_incurred", "sum"),
                                            net_capped=("net_capped", "sum"))
                      .round(0).to_dict(orient="index"))
    # riconciliazione: raw - esclusi = puliti
    ex = pd.concat(excluded, ignore_index=True)
    rep["reconciliation_ok"] = bool(rep["claims_raw"] - len(ex) == len(c))
    c.drop(columns=["cap"]).to_parquet(interim / "claims_clean.parquet", index=False)
    ex.to_parquet(interim / "claims_excluded.parquet", index=False)
    (interim / "report_clean_claims.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
