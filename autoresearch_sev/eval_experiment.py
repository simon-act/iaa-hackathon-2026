"""FROZEN JUDGE for the severity autoresearch loop (19/10). Do not modify.

Task: predict the mean cost of a claim (severity, capped, developed to ultimate, at 2025 price level) from the
policy-row features, per line and peril. This is the severity half of the team's frequency x severity model.
Data: autoresearch_sev/claims_sev.parquet — one row per claim with the allowed features of its policy row,
`line`, `peril`, `ay` (accident year), `sev` (target, > 0).

Temporal folds like the main judge: fit on ay <= 2022 -> score 2023; fit on ay <= 2023 -> score 2024.
Metric per (line, peril, fold): Gamma D² = 1 - dev(pred) / dev(mean of the fit years), on the scored year.
Aggregation: per line, perils weighted by their total cost in the scored year; lines weighted like Kaggle
(motor 254986, buildings 184155, contents 187498 test rows). `pooled_mean` = min over the two folds (an idea
must help in both years); `score` = mean of the two folds.
Contract: fit_predict(train, valid, line, peril) -> predicted severity for each row of valid (finite, > 0).
train: features + `sev`; valid: features only. No access to valid targets, files, network or processes.
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
SKIP = {"row_id", "line", "peril", "ay", "sev", "exposure", "policy_id", "customer_id", "address_id",
        "period_start", "period_end", "year"}


def gamma_dev(y, mu):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    return float(2 * np.sum(-np.log(y / mu) + (y - mu) / mu))


def load_experiment(path):
    spec = importlib.util.spec_from_file_location("experiment_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "fit_predict"):
        raise RuntimeError("experiment has no fit_predict")
    return mod


def main():
    path = sys.argv[1]
    cols_out = sys.argv[sys.argv.index("--columns-out") + 1] if "--columns-out" in sys.argv else None
    t0 = time.time()
    mod = load_experiment(path)
    c = pd.read_parquet(HERE / "claims_sev.parquet")
    feats = [x for x in c.columns if x not in SKIP]
    columns, by, line_fold = {}, {}, {}
    for line in W:
        cl = c[c["line"] == line]
        columns[line] = {x: str(cl[x].dtype) for x in feats if cl[x].notna().any()}
        for last in (2022, 2023):
            num = den = 0.0
            for p, g in cl.groupby("peril"):
                tr = g[g["ay"] <= last]; va = g[g["ay"] == last + 1]
                if len(va) < 20 or len(tr) < 50:
                    continue
                ftr = [x for x in feats if g[x].notna().any()]
                pred = np.asarray(mod.fit_predict(tr[ftr + ["sev"]].reset_index(drop=True).copy(),
                                                  va[ftr].reset_index(drop=True).copy(), line, p), dtype=float)
                if pred.shape != (len(va),):
                    raise RuntimeError(f"wrong prediction shape {pred.shape} for {len(va)} rows ({line}/{p})")
                if not np.isfinite(pred).all() or (pred <= 0).any():
                    raise RuntimeError("predictions must be finite and > 0")
                d2 = 1 - gamma_dev(va["sev"], pred) / gamma_dev(va["sev"], np.full(len(va), tr["sev"].mean()))
                by[f"{line}|{p}|{last + 1}"] = round(float(d2), 5)
                w = float(va["sev"].sum()); num += d2 * w; den += w
            line_fold[(line, last + 1)] = num / den if den else 0.0
    tot = sum(W.values())
    folds = {f"F{y}": sum(line_fold[(l, y)] * W[l] for l in W) / tot for y in (2023, 2024)}
    res = {"by": by, "by_line": {f"{l}|{y}": round(v, 5) for (l, y), v in line_fold.items()},
           "pooled": {k: round(v, 5) for k, v in folds.items()},
           "pooled_mean": round(min(folds.values()), 5), "score": round(float(np.mean(list(folds.values()))), 5),
           "seconds": round(time.time() - t0, 1)}
    if cols_out:
        Path(cols_out).write_text(json.dumps(columns, indent=1))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
