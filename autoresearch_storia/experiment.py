"""Experience rating on top of the best model (baseline for the history loop, 21/10).

fit_predict(train, valid, line):
  train: features + history columns + exposure + base_pred + base_nocred + judge columns (target_b, target_rep,
         pi_eff, ll_expo) for ONE year;
  valid: same columns without the judge columns, for the OTHER year.
  return the new prediction for valid (same units as base_pred). The judge rescales the total to base_pred.
Baseline: the current experience rating, i.e. base_pred unchanged (pooled_mean = 0).
"""
import numpy as np
import pandas as pd


def fit_predict(train, valid, line):
    return valid["base_pred"].to_numpy(float)
