"""FROZEN JUDGE for the GBM loop (20/10). Do not modify.

Data: autoresearch_gbm/data/<line>_<year>.parquet — one row per policy row (holdout policies removed), with the allowed
features, `exposure`, and the judge columns (never given to the experiment): target_b, target_gross, target_rep, pi_eff,
ll_expo.
Training target given to the experiment as `target`: target_gross = developed reported cost / expected completeness of
the row (IBNR allocated to the rows that reported claims, not spread by exposure), at 2025 level, + large-loss load.
Folds: fit 2021–22 → score 2023; fit 2021–23 → score 2024.
Score per line and fold = Tweedie D² (p=1.5) WITHOUT spread: y = target_rep (reported, developed), prediction =
(pred − ll_expo) × pi_eff, rescaled to the total of y (shape only; the level is set elsewhere). Also reported: the official
D² on target_b (biased towards flat predictions: it contains IBNR and large-loss load spread by exposure).
Aggregation: lines weighted like Kaggle; `pooled_mean` = WORSE of the two folds (an idea must help in both years);
`score` = mean of the two folds.
Contract: fit_predict(train, valid, line) -> expected cost per valid row (finite, >= 0).
Usage: python -I eval_experiment.py <experiment.py> [--columns-out columns.json]
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = {"motor": 254986, "buildings": 184155, "contents": 187498}
JUDGE = ["row_id", "year", "target_b", "target_gross", "target_rep", "pi_eff", "ll_expo"]


def dev(y, mu, p=1.5):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    return float(2 * np.sum(np.where(y > 0, y ** (2 - p) / ((1 - p) * (2 - p)), 0) - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)))


def d2(y, mu):
    return 1 - dev(y, mu) / dev(y, np.full(len(y), np.mean(y)))


def main():
    path = sys.argv[1]
    cols_out = sys.argv[sys.argv.index("--columns-out") + 1] if "--columns-out" in sys.argv else None
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("experiment_under_test", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    by, by_off, columns = {}, {}, {}
    for line in W:
        d = {y: pd.read_parquet(HERE / "data" / f"{line}_{y}.parquet") for y in (2021, 2022, 2023, 2024)}
        feats = [c for c in d[2021].columns if c not in JUDGE and c != "exposure"]
        columns[line] = {c: str(d[2021][c].dtype) for c in feats}
        for last in (2022, 2023):
            tr = pd.concat([d[y] for y in range(2021, last + 1)], ignore_index=True)
            va = d[last + 1]
            train = tr[feats + ["exposure"]].assign(target=tr["target_gross"].to_numpy())
            valid = va[feats + ["exposure"]].copy()
            pred = np.asarray(mod.fit_predict(train, valid, line), float)
            if pred.shape != (len(va),) or not np.isfinite(pred).all() or (pred < 0).any():
                raise RuntimeError("predictions must be finite, >= 0 and one per valid row")
            y = va["target_rep"].to_numpy(float)
            att = np.clip(pred - va["ll_expo"].to_numpy(float), 1e-3, None) * va["pi_eff"].to_numpy(float)
            by[f"{line}|{last + 1}"] = round(d2(y, att * y.sum() / att.sum()), 5)
            by_off[f"{line}|{last + 1}"] = round(d2(va["target_b"].to_numpy(float), pred), 5)
    folds = {y: sum(by[f"{l}|{y}"] * W[l] for l in W) / sum(W.values()) for y in (2023, 2024)}
    off = {y: sum(by_off[f"{l}|{y}"] * W[l] for l in W) / sum(W.values()) for y in (2023, 2024)}
    res = {"by": by, "pooled_mean": round(min(folds.values()), 5), "score": round(float(np.mean(list(folds.values()))), 5),
           "folds": {str(k): round(v, 5) for k, v in folds.items()},
           "official_folds": {str(k): round(v, 5) for k, v in off.items()}, "by_official": by_off,
           "seconds": round(time.time() - t0, 1)}
    if cols_out:
        Path(cols_out).write_text(json.dumps(columns, indent=1))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
