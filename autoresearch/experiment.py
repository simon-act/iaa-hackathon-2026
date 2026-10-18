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

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.03, num_leaves=15,
              min_data_in_leaf=2000, min_sum_hessian_in_leaf=50, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=50.0, max_cat_to_onehot=8, cat_smooth=100, cat_l2=50,
              min_data_per_group=2000, max_cat_threshold=16, max_bin=63, verbose=-1, seed=7)
ROUNDS = {"motor": 199, "buildings": 153, "contents": 116}


def add_features(df: pd.DataFrame, line: str) -> pd.DataFrame:
    """Feature engineering from allowed columns only. Must give the same columns for train and valid."""
    df = df.copy()
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
    cols = [c for c in train.columns if c not in ("exposure", "target_b") and train[c].nunique(dropna=False) > 1]
    Xtr, cats = to_frame(train, cols)
    Xva, _ = to_frame(valid, cols, cats)
    y = train["target_b"].to_numpy()
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    base = np.log(y.sum() / e_tr.sum())
    params = dict(PARAMS, num_threads=THREADS)
    m = lgb.train(params, lgb.Dataset(Xtr, label=y, init_score=np.log(e_tr) + base), num_boost_round=ROUNDS[line])
    pred = np.exp(m.predict(Xva, raw_score=True) + np.log(e_va) + base)
    pin = np.exp(m.predict(Xtr, raw_score=True) + np.log(e_tr) + base)
    return pred * (y.sum() / pin.sum())
