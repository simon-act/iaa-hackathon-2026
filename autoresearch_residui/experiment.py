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
        return valid["base_pred"].to_numpy(float)   # baseline = best model unchanged (gain 0)
    cols = [c for c in train.columns if c not in SKIP and c not in DECOYS]
    Xtr, cats = _frame(train, cols)
    Xva, _ = _frame(valid, cols, cats)
    init_tr = np.log(np.clip(train["base_pred"].to_numpy(float), 1e-6, None))
    init_va = np.log(np.clip(valid["base_pred"].to_numpy(float), 1e-6, None))
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, label=train["target_b"].to_numpy(float), init_score=init_tr),
                  num_boost_round=ROUNDS)
    pred = np.exp(m.predict(Xva, raw_score=True) + init_va)
    return pred * valid["base_pred"].sum() / pred.sum()
