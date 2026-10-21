"""FROZEN JUDGE for the residual autoresearch loop (19/10). Do not modify.

Task: improve the CURRENT BEST MODEL (blend submitted as sub_18) by learning what it still gets wrong.
Data: out-of-fold predictions of the best model (`base_pred`) on the validation years 2023 and 2024,
with the real target (`target_b`) and all allowed feature columns.

Temporal cross-fit (honest: the residual model never sees the year it is scored on):
  - fit on the 2023 rows -> predict the 2024 rows   (gain_2024)
  - fit on the 2024 rows -> predict the 2023 rows   (gain_2023)
Gain = D2(new prediction) - D2(base_pred), Tweedie p=1.5, per line, weighted by test rows like Kaggle.
The loop optimises `pooled_mean` = min(gain_2023, gain_2024): an idea must help in BOTH directions.
`score` = mean of the two gains.

Guardrails: the experiment gets the feature columns + exposure + base_pred; target_b only for the fitting year.
Predictions must be finite, >= 0, right length.
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import prepare as P  # noqa: E402

W = {"motor": 254986, "buildings": 184155, "contents": 187498}


def d2(y, mu, p=1.5):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)

    def dev(m):
        return 2 * (np.power(np.clip(y, 0, None), 2 - p) / ((1 - p) * (2 - p)) - y * np.power(m, 1 - p) / (1 - p)
                    + np.power(m, 2 - p) / (2 - p))
    return 1 - dev(mu).sum() / dev(np.full_like(y, y.mean())).sum()


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
    base = pd.read_parquet(HERE / "base_oof.parquet")
    by, columns = {}, {}
    for line in ["motor", "buildings", "contents"]:
        feat = P.load(line)
        feats = [c for c in P.allowed_features(feat) if c not in ("exposure",)]
        columns[line] = {c: str(feat[c].dtype) for c in feats}
        b = base[base["line"] == line].drop(columns=["exposure"]).merge(feat[["row_id", "exposure"] + feats], on="row_id")
        for fit_fold, score_fold in (("F2023", "F2024"), ("F2024", "F2023")):
            tr = b[b["fold"] == fit_fold].reset_index(drop=True)
            va = b[b["fold"] == score_fold].reset_index(drop=True)
            pred = np.asarray(mod.fit_predict(tr[feats + ["exposure", "base_pred", "target_b"]].copy(),
                                              va[feats + ["exposure", "base_pred"]].copy(), line), dtype=float)
            if pred.shape != (len(va),):
                raise RuntimeError(f"wrong prediction shape {pred.shape} for {len(va)} rows")
            if not np.isfinite(pred).all() or (pred < 0).any():
                raise RuntimeError("predictions must be finite and >= 0")
            by[f"{line}|{score_fold}"] = round(float(d2(va["target_b"], pred) - d2(va["target_b"], va["base_pred"])), 6)
    tot = sum(W.values())
    g = {f: sum(by[f"{l}|{f}"] * W[l] for l in W) / tot for f in ("F2023", "F2024")}
    res = {"by": by, "pooled": {k: round(v, 6) for k, v in g.items()},
           "pooled_mean": round(min(g.values()), 6), "score": round(float(np.mean(list(g.values()))), 6),
           "seconds": round(time.time() - t0, 1)}
    if cols_out:
        Path(cols_out).write_text(json.dumps(columns, indent=1))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
