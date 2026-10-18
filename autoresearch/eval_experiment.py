"""FROZEN JUDGE for the autoresearch loop. Do not modify.

Runs an experiment file on the temporal folds of src/prepare.py and prints one line:
    RESULT {"score": ..., "pooled_mean": ..., "by": {...}, "seconds": ...}
Guardrails:
  - the experiment only receives allowed feature columns (prepare.allowed_features) plus exposure;
    the target is given for train rows only, never for valid rows;
  - the closed holdout (10% of policies) is never passed to the experiment;
  - predictions must be finite, >= 0 and of the right length.
Usage: python -I eval_experiment.py <experiment.py> [--columns-out columns.json]
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare as P  # noqa: E402


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
    columns = {}

    def wrapped(train, valid, line):
        feats = P.allowed_features(train)
        columns[line] = {c: str(train[c].dtype) for c in feats}
        tr = train[feats + ["exposure", P.TARGET]].reset_index(drop=True)
        va = valid[feats + ["exposure"]].reset_index(drop=True)
        pred = np.asarray(mod.fit_predict(tr, va, line), dtype=float)
        if pred.shape != (len(va),):
            raise RuntimeError(f"wrong prediction shape {pred.shape} for {len(va)} rows")
        if not np.isfinite(pred).all() or (pred < 0).any():
            raise RuntimeError("predictions must be finite and >= 0")
        return pred

    res = P.evaluate(wrapped, verbose=False)
    res["seconds"] = round(time.time() - t0, 1)
    if cols_out:
        Path(cols_out).write_text(json.dumps(columns, indent=1))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
