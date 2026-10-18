"""EXPERIMENT FILE — the only file the research agent may change.

Contract (checked by eval_experiment.py, do not change the signature):
    fit_predict(train: DataFrame, valid: DataFrame, line: str) -> np.ndarray
      - train: allowed feature columns + "exposure" + "target_b" (cost of the row, 2025 level, CU)
      - valid: allowed feature columns + "exposure" (NO target)
      - line : "motor" | "buildings" | "contents"
      - return: expected claim cost per valid row in CU (>= 0, finite, same length as valid)
Score = Tweedie deviance D² (power 1.5) on temporal folds, higher is better.

Starting point (16/10): LightGBM Tweedie, offset log(exposure), start from the mean rate,
heavy regularisation, rebalanced so that sum(pred) = sum(target) on train.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

THREADS = int(os.environ.get("N_THREADS", "0"))  # 0 = all cores

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.02, num_leaves=12,
              min_data_in_leaf=3500, min_sum_hessian_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=100.0, max_cat_to_onehot=8, cat_smooth=200, cat_l2=100,
              min_data_per_group=3000, max_cat_threshold=16, max_bin=63, verbose=-1)
ROUNDS = {"motor": 300, "buildings": 250, "contents": 200}

# High-cardinality categoricals to drop (noisy, low signal)
DROP_COLS = {"previous_insurer", "campaign_code", "branch_id", "make_model"}

# Seeds for bagging
SEEDS = [7, 13, 42]


def add_features(df: pd.DataFrame, line: str) -> pd.DataFrame:
    """Feature engineering from allowed columns only. Must give the same columns for train and valid."""
    df = df.copy()
    
    if line == "motor":
        # Driver risk interaction
        df["driver_age_x_lic"] = df["driver_age"] * df["years_licensed"]
        # Vehicle risk interaction
        df["veh_val_x_age"] = df["vehicle_value_group"] * df["vehicle_age"]
        # Subsidence risk (clay soil + dry summer)
        if "soil_type" in df.columns and "dry_summer_index" in df.columns:
            is_clay = (df["soil_type"] == "clay").astype(float)
            df["subsidence_risk"] = is_clay * df["dry_summer_index"]
        # Flood risk interaction
        if "flood_zone" in df.columns and "distance_to_river_km" in df.columns:
            df["flood_severity"] = df["flood_zone"] * df["distance_to_river_km"]
        # Telematics missingness indicator
        if "telematics_score" in df.columns:
            df["telematics_missing"] = df["telematics_score"].isna().astype(float)
            
    elif line in ("buildings", "contents"):
        # Sum insured density per bedroom
        if "sum_insured" in df.columns and "bedrooms" in df.columns:
            df["si_per_bedroom"] = df["sum_insured"] / (df["bedrooms"] + 1e-6)
        # Subsidence risk (clay soil + dry summer)
        if "soil_type" in df.columns and "dry_summer_index" in df.columns:
            is_clay = (df["soil_type"] == "clay").astype(float)
            df["subsidence_risk"] = is_clay * df["dry_summer_index"]
        # Flood risk interaction
        if "flood_zone" in df.columns and "distance_to_river_km" in df.columns:
            df["flood_severity"] = df["flood_zone"] * df["distance_to_river_km"]
            
    return df


def to_frame(df, cols, cats=None):
    X = df[cols].copy()
    cats = {} if cats is None else cats
    for c in cols:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            v = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
            if c not in cats:
                cats[c] = sorted(v.unique().tolist())
            X[c] = pd.Categorical(v, categories=cats[c])
    return X, cats


def fit_predict(train, valid, line):
    train = add_features(train, line)
    valid = add_features(valid, line)
    
    # Exclude target, exposure, and high-cardinality noisy categoricals
    exclude = {"exposure", "target_b"} | DROP_COLS
    cols = [c for c in train.columns if c not in exclude and train[c].nunique(dropna=False) > 1]
    
    Xtr, cats = to_frame(train, cols)
    Xva, _ = to_frame(valid, cols, cats)
    y = train["target_b"].to_numpy()
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    base = np.log(y.sum() / e_tr.sum())
    
    # Bag multiple seeds
    valid_preds = np.zeros(len(valid))
    train_preds = np.zeros(len(train))
    
    for seed in SEEDS:
        params = dict(PARAMS, num_threads=THREADS, seed=seed)
        m = lgb.train(params, lgb.Dataset(Xtr, label=y, init_score=np.log(e_tr) + base), num_boost_round=ROUNDS[line])
        valid_preds += np.exp(m.predict(Xva, raw_score=True) + np.log(e_va) + base)
        train_preds += np.exp(m.predict(Xtr, raw_score=True) + np.log(e_tr) + base)
    
    valid_preds /= len(SEEDS)
    train_preds /= len(SEEDS)
    
    # Rebalance so that sum(pred) = sum(target) on train
    scale = y.sum() / train_preds.sum()
    return valid_preds * scale
