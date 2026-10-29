"""Shared functions for p2_resid.py (27/10)."""
import numpy as np, pandas as pd

DROP = {"row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure", "exposure_raw",
        "customer_service_contacts", "renewal_decision", "written_premium", "email_domain", "construction_type_raw",
        "flag_exposure_fixed", "target_b", "cost_total", "rec_adj", "ult_capped", "ult25_capped", "year"}


def binned(s):
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
        e = np.unique(np.nanquantile(s.astype(float), np.linspace(0, 1, 7)[1:-1]))
        return pd.Series(np.where(s.isna(), "na", np.searchsorted(e, s.astype(float).fillna(0)).astype(str)), index=s.index)
    s = s.astype(object).where(s.notna(), "na").astype(str)
    vc = s.value_counts()
    return s.where(s.isin(vc[vc >= 200].index), "rare")


