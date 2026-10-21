"""GBM loop (20/10) — the ONLY file the research agent changes.
Starting point: the team GBM (Qwen iteration 73) with one correction by Claude: no monotone constraint on vehicle_age
(the data show a hump: cost 0.87 -> 1.09 at 5-8 years -> 0.97 after 12).
Contract: fit_predict(train, valid, line) -> expected cost per valid row.
  train: allowed features + exposure + target (cost at 2025 level); valid: features + exposure.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

THREADS = int(os.environ.get("N_THREADS", "8"))  # 0 = all cores

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.015, num_leaves=8,
              min_data_in_leaf=3000, min_sum_hessian_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.6,
              bagging_freq=1, lambda_l2=80.0, max_cat_to_onehot=8, cat_smooth=300, cat_l2=150,
              min_data_per_group=2500, max_cat_threshold=16, max_bin=63, verbose=-1)
ROUNDS = {"motor": 800, "buildings": 700, "contents": 600}

# High-cardinality categoricals to drop (noisy, low signal)
DROP_COLS = {"previous_insurer", "campaign_code", "branch_id", "make_model"}

# Decoy columns with no effect (verified)
DECOYS = {
    "engine_size_cc", "body_type", "seats", "driver_assist", "fuel_type", "vehicle_colour",
    "transmission", "telematics_opt_in", "marital_status", "solar_panels", "document_preference",
    "customer_tenure", "listed_building", "marketing_consent", "mortgage", "smoke_alarms",
    "lodgers", "bicycle_cover", "previous_insurer", "campaign_code", "branch_id",
    "marketing_segment", "start_month", "starts_jan1", "ends_dec", "mileage_k", "licence_age",
    "pol_years_seen", "cust_has_buildings", "cust_has_contents", "cust_has_motor"
}

# Seeds for bagging
SEEDS = [7, 13, 42, 99, 123]


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
    
    # Exclude target, exposure, decoys, and high-cardinality noisy categoricals
    exclude = {"exposure", "target"} | DROP_COLS | DECOYS
    cols = [c for c in train.columns if c not in exclude and train[c].nunique(dropna=False) > 1]
    
    Xtr, cats = to_frame(train, cols)
    Xva, _ = to_frame(valid, cols, cats)
    y = train["target"].to_numpy()
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    base = np.log(y.sum() / e_tr.sum())
    
    # Build monotone constraints: +1 for certain-direction drivers
    # flood_zone: higher zone => higher risk (positive)
    # building_age: older building => higher risk (positive) 
    # vehicle_age: older vehicle => higher risk (positive)
    # sum_insured: higher sum => higher cost (positive)
    monotone_constraints = [0] * len(cols)
    for i, c in enumerate(cols):
        if c in ["flood_zone", "building_age", "sum_insured"]:   # 20/10 Claude: tolto vehicle_age (effetto a campana: 0,87 → 1,09 a 5–8 anni → 0,97 oltre 12)
            monotone_constraints[i] = 1
    
    # Bag multiple seeds
    valid_preds = np.zeros(len(valid))
    train_preds = np.zeros(len(train))
    
    for seed in SEEDS:
        params = dict(PARAMS, num_threads=THREADS, seed=seed, monotone_constraints=monotone_constraints)
        m = lgb.train(params, lgb.Dataset(Xtr, label=y, init_score=np.log(e_tr) + base), num_boost_round=ROUNDS[line])
        valid_preds += np.exp(m.predict(Xva, raw_score=True) + np.log(e_va) + base)
        train_preds += np.exp(m.predict(Xtr, raw_score=True) + np.log(e_tr) + base)
    
    valid_preds /= len(SEEDS)
    train_preds /= len(SEEDS)
    
    # Rebalance so that sum(pred) = sum(target) on train
    scale = y.sum() / train_preds.sum()
    return valid_preds * scale
