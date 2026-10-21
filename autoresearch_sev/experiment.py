"""Severity model — starting point of the severity loop (19/10).

fit_predict(train, valid, line, peril): train has the policy-row features + `sev` (claim cost, > 0); valid has the
features only. Return the predicted mean severity for each valid row (> 0).
Baseline = what the team's main model does today: a Gamma LightGBM with 1-split trees (additive in log space) on a
short list of plausible drivers per line, strong L2, starting from log(mean severity). Perils with few claims fall
back to the mean.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

DRIVERS = {"motor": ["make_model", "vehicle_age", "vehicle_value_group", "urban_density", "region"],
           "buildings": ["sum_insured", "si_per_bedroom", "property_type", "construction_type", "bedrooms", "region"],
           "contents": ["sum_insured", "si_per_bedroom", "high_value_items", "tenure", "region"]}
PARAMS = dict(objective="gamma", num_leaves=2, max_depth=1, learning_rate=0.05, feature_fraction=1.0,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=50.0, max_bin=31, max_cat_to_onehot=8,
              cat_smooth=100, min_data_per_group=1000, verbose=-1, num_threads=4, seed=12)
ROUNDS = 150
MIN_CLAIMS = 1500


def _frame(df, cols, cats=None):
    X = df[cols].copy()
    out = {}
    for c in cols:
        if X[c].dtype == object or str(X[c].dtype) in ("category", "string", "str", "bool"):
            v = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
            levels = cats[c] if cats else sorted(v.unique())
            X[c] = pd.Categorical(v, categories=levels)
            out[c] = levels
    return X, out


def fit_predict(train, valid, line, peril):
    y = train["sev"].to_numpy(float)
    mean = float(y.mean())
    if len(train) < MIN_CLAIMS:
        return np.full(len(valid), mean)
    cols = [c for c in DRIVERS[line] if c in train.columns]
    Xtr, cats = _frame(train, cols); Xva, _ = _frame(valid, cols, cats)
    params = dict(PARAMS, min_data_in_leaf=max(200, len(train) // 50))
    m = lgb.train(params, lgb.Dataset(Xtr, label=y, init_score=np.full(len(y), np.log(mean))), num_boost_round=ROUNDS)
    return np.exp(m.predict(Xva, raw_score=True) + np.log(mean))
