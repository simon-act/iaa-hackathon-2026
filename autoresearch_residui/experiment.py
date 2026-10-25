"""Residual model on top of the current best model (baseline for the residual loop, 19/10).

fit_predict(train, valid, line):
  train: feature columns + exposure + base_pred (prediction of the best model) + target_b (real target)
  valid: feature columns + exposure + base_pred
  returns the new prediction for valid (same units as base_pred).
Idea of the baseline: a small, heavily regularised LightGBM Tweedie that starts from log(base_pred) (init_score)
and learns only a correction; the total level is kept equal to base_pred.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

DECOYS = {"engine_size_cc", "body_type", "seats", "driver_assist", "fuel_type", "vehicle_colour", "transmission",
          "telematics_opt_in", "marital_status", "solar_panels", "document_preference", "customer_tenure",
          "listed_building", "marketing_consent", "cust_other_prev_rate", "cust_other_prev_n_last", "mortgage",
          "starts_jan1", "mileage_k", "licence_age", "ends_dec", "smoke_alarms", "lodgers", "bicycle_cover",
          "cust_has_buildings", "cust_has_contents", "cust_has_motor", "pol_years_seen", "start_month",
          "previous_insurer", "campaign_code", "branch_id", "marketing_segment", "email_domain"}
SKIP = {"exposure", "base_pred", "target_b", "row_id", "policy_id", "customer_id", "address_id",
        "period_start", "period_end", "year"}

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.02, num_leaves=4,
              min_data_in_leaf=3000, lambda_l2=50.0, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              max_bin=63, cat_smooth=100, min_data_per_group=2000, verbose=-1, num_threads=4, seed=1)
ROUNDS = 0          # 0 = baseline: base_pred unchanged. 150 rounds of boosting on all features scored -0.0079.


def _frame(df, cols, cats=None):
    X = df[cols].copy()
    out_cats = {}
    for c in cols:
        if X[c].dtype == object or str(X[c].dtype) in ("category", "string", "str", "bool"):
            v = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
            levels = cats[c] if cats else sorted(v.unique())
            X[c] = pd.Categorical(v, categories=levels)
            out_cats[c] = levels
    return X, out_cats


def power_calibration(train, valid):
    """pred^b per line, b fitted on train by Tweedie deviance, total level of valid unchanged.
    Found 19/10: b ~0.92-0.97 motor, 1.03-1.09 buildings, 0.91-0.94 contents; +0.00035 in both directions."""
    from scipy.optimize import minimize_scalar
    y = train["target_b"].to_numpy(float)
    lb = np.log(np.clip(train["base_pred"].to_numpy(float), 1e-6, None)); lb = lb - lb.mean()

    def dev(b):
        mu = np.exp(b * lb); mu = mu * y.sum() / mu.sum()
        return float((y * mu ** -0.5 + mu ** 0.5).sum())   # Tweedie p=1.5 deviance up to constants
    b = minimize_scalar(dev, bounds=(0.7, 1.3), method="bounded").x
    bv = valid["base_pred"].to_numpy(float)
    p = np.exp(b * (np.log(np.clip(bv, 1e-6, None)) - np.log(np.clip(bv, 1e-6, None)).mean()))
    return p * bv.sum() / p.sum()


def fit_predict(train, valid, line):
    if ROUNDS == 0:
        base_valid = valid["base_pred"].to_numpy(float)
        
        # Check if we are in motor line and have the specific feature
        if line == "motor" and "vehicle_value_group" in train.columns and "vehicle_value_group" in valid.columns:
            # Define bands for vehicle_value_group (assuming it's an ordinal/int or categorical)
            # We will use quantiles on the training set to create stable bands
            try:
                # Create 5 bands based on quantiles of the training data
                q = train["vehicle_value_group"].quantile([0.2, 0.4, 0.6, 0.8])
                cuts = [train["vehicle_value_group"].min() - 1, q[0.2], q[0.4], q[0.6], q[0.8], train["vehicle_value_group"].max() + 1]
                
                train["vvg_band"] = pd.cut(train["vehicle_value_group"], bins=cuts, labels=False, include_lowest=True)
                valid["vvg_band"] = pd.cut(valid["vehicle_value_group"], bins=cuts, labels=False, include_lowest=True)
                
                # Calculate A/E for each band
                # A = Actual Cost, E = Expected Cost (base_pred * exposure? No, base_pred is the prediction for the row)
                # The target_b is the "real ultimate cost of the row".
                # base_pred is the prediction for the row.
                # So A/E = target_b / base_pred.
                
                agg = train.groupby("vvg_band").agg(
                    actual=("target_b", "sum"),
                    expected=("base_pred", "sum"),
                    count=("target_b", "count")
                )
                
                agg["ae"] = agg["actual"] / agg["expected"]
                
                # Credibility weighting
                # Z = n / (n + k)
                # We want to shrink towards 1 (A/E = 1)
                # We need to estimate the variance or use a fixed k.
                # A common heuristic for claims frequency/severity is k ~ 100-500 depending on granularity.
                # Let's use a fixed k=200 for the band credibility, assuming ~200 claims per band is the "fully credible" level.
                k = 200
                agg["z"] = agg["count"] / (agg["count"] + k)
                
                # Smoother factor
                # If A/E is 1.2 and Z is 0.5, factor = 0.5*1.2 + 0.5*1.0 = 1.1
                agg["factor"] = agg["z"] * agg["ae"] + (1 - agg["z"]) * 1.0
                
                # Map factors back to valid
                # Reindex to ensure all valid bands exist (fill NaNs with 1.0)
                valid["vvg_factor"] = valid["vvg_band"].map(agg["factor"])
                valid["vvg_factor"] = valid["vvg_factor"].fillna(1.0)
                
                # Apply correction
                corrected = base_valid * valid["vvg_factor"].to_numpy(float)
                
                # Rescale to keep the total level unchanged (sum of predictions = sum of base_pred)
                if corrected.sum() > 0:
                    corrected = corrected * (base_valid.sum() / corrected.sum())
                
                return corrected
            except Exception:
                pass
        
        return base_valid
    cols = [c for c in train.columns if c not in SKIP and c not in DECOYS]
    Xtr, cats = _frame(train, cols)
    Xva, _ = _frame(valid, cols, cats)
    init_tr = np.log(np.clip(train["base_pred"].to_numpy(float), 1e-6, None))
    init_va = np.log(np.clip(valid["base_pred"].to_numpy(float), 1e-6, None))
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, label=train["target_b"].to_numpy(float), init_score=init_tr),
                  num_boost_round=ROUNDS)
    pred = np.exp(m.predict(Xva, raw_score=True) + init_va)
    return pred * valid["base_pred"].sum() / pred.sum()
