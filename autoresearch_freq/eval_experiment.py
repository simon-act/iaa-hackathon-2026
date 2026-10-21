"""FROZEN JUDGE for the frequency loop (20/10). Do not modify.

Task: predict the expected NUMBER of claims of one peril for each policy row of the next year (frequency part of the
team's per-peril frequency × severity model). Data: autoresearch_freq/data/<line>_<year>.parquet (holdout removed):
rating features, `exposure`, and per peril `n_<peril>` (reported claims) and `pi_<peril>` (expected reporting
completeness of the row). meta.json: perils, features, frequency trend per peril, cost share of each peril.
The experiment receives `offset` = log(exposure × completeness) − (2025 − year) × log(1 + trend): the expected count is
exp(offset + f(features)). Train rows include `n` (the target); valid rows do not.
Folds: fit 2021–22 → score 2023; fit 2021–23 → score 2024. Score per line, peril, fold = Poisson D² of the reported
counts, prediction rescaled to the observed total (shape only). Per line: perils weighted by cost share; lines weighted
like Kaggle. `pooled_mean` = WORSE fold; `score` = mean of the two folds.
Contract: fit_predict(train, valid, line, peril) -> expected count per valid row (finite, > 0).
Usage: python -I eval_experiment.py <experiment.py> [--columns-out columns.json]
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
W = {"motor": 254986, "buildings": 184155, "contents": 187498}
META = json.loads((HERE / "data" / "meta.json").read_text())


def pd2(y, mu):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-12, None)
    dev = lambda m: 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / m), 0) - (y - m))
    return 1 - dev(mu) / dev(np.full(len(y), y.mean()))


def main():
    path = sys.argv[1]; t0 = time.time()
    cols_out = sys.argv[sys.argv.index("--columns-out") + 1] if "--columns-out" in sys.argv else None
    spec = importlib.util.spec_from_file_location("experiment_under_test", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    by, columns = {}, {}
    for line in W:
        m = META[line]; feats = m["features"]
        d = {y: pd.read_parquet(HERE / "data" / f"{line}_{y}.parquet") for y in (2021, 2022, 2023, 2024)}
        columns[line] = {c: str(d[2021][c].dtype) for c in feats}
        for last in (2022, 2023):
            tr = pd.concat([d[y] for y in range(2021, last + 1)], ignore_index=True); va = d[last + 1]
            num = den = 0.0
            for p in m["perils"]:
                lt = np.log1p(m["freq_trend"][p])
                off = lambda x: np.log(x["exposure"].to_numpy() * np.clip(x[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - x["year"].to_numpy()) * lt
                train = tr[feats + ["exposure", "year"]].assign(offset=off(tr), n=tr[f"n_{p}"].to_numpy(float))
                valid = va[feats + ["exposure", "year"]].assign(offset=off(va))
                mu = np.asarray(mod.fit_predict(train, valid, line, p), float)
                if mu.shape != (len(va),) or not np.isfinite(mu).all() or (mu <= 0).any():
                    raise RuntimeError("predictions must be finite, > 0, one per valid row")
                y = va[f"n_{p}"].to_numpy(float); mu = mu * y.sum() / mu.sum()
                s = pd2(y, mu); by[f"{line}|{p}|{last + 1}"] = round(s, 5)
                w = m["cost_share"][p]; num += s * w; den += w
            by[f"{line}|{last + 1}"] = round(num / den, 5)
    folds = {y: sum(by[f"{l}|{y}"] * W[l] for l in W) / sum(W.values()) for y in (2023, 2024)}
    res = {"by": by, "pooled_mean": round(min(folds.values()), 5), "score": round(float(np.mean(list(folds.values()))), 5),
           "folds": {str(k): round(v, 5) for k, v in folds.items()}, "seconds": round(time.time() - t0, 1)}
    if cols_out:
        Path(cols_out).write_text(json.dumps(columns, indent=1))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
