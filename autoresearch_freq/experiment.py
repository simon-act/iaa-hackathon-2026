"""Frequency loop (20/10) — the ONLY file the research agent changes.
Starting point = the team's frequency model: per peril, LightGBM Poisson with 1-split trees (additive in log space),
strong L2, starting from the offset; make_model (motor) enters as a credibility target encoding; subsidence gets 3-leaf
trees (soil × drought interaction). Number of trees grows with the number of claims.
Contract: fit_predict(train, valid, line, peril) -> expected number of claims per valid row.
  train: features + exposure + year + offset + n (claims); valid: features + exposure + year + offset.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

THREADS = int(os.environ.get("N_THREADS", "8"))
PARAMS = dict(objective="poisson", num_leaves=2, max_depth=1, learning_rate=0.1, min_data_in_leaf=1000,
              feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=63,
              max_cat_to_onehot=8, cat_smooth=50, min_data_per_group=500, verbose=-1, seed=11)
MULT = 4


def te_make_model(train, valid, y, e_exp, k=30.0, folds=5, seed=0):
    mm_tr = train["make_model"].astype(str).to_numpy(); mm_va = valid["make_model"].astype(str).to_numpy()
    def enc(idx, keys):
        s = pd.DataFrame({"k": mm_tr[idx], "a": y[idx], "e": e_exp[idx]}).groupby("k")[["a", "e"]].sum()
        z = s["e"] / (s["e"] + k)
        return pd.Series(keys).map(np.log(z * (s["a"] / s["e"].clip(lower=1e-9)) + (1 - z))).fillna(0.0).to_numpy()
    out = np.zeros(len(train)); f = np.random.default_rng(seed).integers(0, folds, len(train))
    for i in range(folds):
        out[f == i] = enc(np.where(f != i)[0], mm_tr[f == i])
    return out, enc(np.arange(len(train)), mm_va)


def frame(df, cols, cats=None):
    X = df[cols].copy(); out = {}
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool:
            v = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
            lv = cats[c] if cats else sorted(v.unique()); X[c] = pd.Categorical(v, categories=lv); out[c] = lv
    return X, out


def fit_predict(train, valid, line, peril):
    n = train["n"].to_numpy(float); off_tr = train["offset"].to_numpy(); off_va = valid["offset"].to_numpy()
    base = np.log(max(n.sum(), 1) / np.exp(off_tr).sum())
    cols = [c for c in train.columns if c not in ("n", "offset", "exposure", "year", "make_model")]
    Xtr, cats = frame(train, cols); Xva, _ = frame(valid, cols, cats)
    if "make_model" in train.columns:
        a, b = te_make_model(train, valid, n, np.exp(off_tr + base))
        Xtr = Xtr.assign(mm_freq_te=a); Xva = Xva.assign(mm_freq_te=b)
    params = dict(PARAMS, num_threads=THREADS)
    if peril == "subsidence":
        params.update(num_leaves=3, max_depth=2)
    rounds = (300 if n.sum() > 5000 else 150 if n.sum() > 1000 else 80) * MULT
    m = lgb.train(params, lgb.Dataset(Xtr, label=n, init_score=off_tr + base), num_boost_round=rounds)
    return np.exp(m.predict(Xva, raw_score=True) + off_va + base)
