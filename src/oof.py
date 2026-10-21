"""Previsioni out-of-fold salvate (fold F2023 e F2024, holdout escluso) per diagnostica e blend.

Uso: python -I oof.py <nome_modello> [<nome_modello> ...]
  nomi = funzioni di train.py (glm_b, gbm_b, ...) o di models_extra.py (se presente)
Output: results/oof/<modello>.parquet con row_id, line, fold, year, exposure, target_b, pred
        + riga in results/results.tsv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402
import train as T  # noqa: E402

OUT = P.RESULTS / "oof"


def get_fn(name):
    if hasattr(T, name):
        return getattr(T, name)
    import models_extra as M  # noqa
    return getattr(M, name)


def run(name):
    if T.STATE.exists():
        T.NUM_ROUNDS.update(json.loads(T.STATE.read_text()).get("num_rounds", {}))
    fn = get_fn(name)
    OUT.mkdir(parents=True, exist_ok=True)
    parts, t0 = [], time.time()
    for line in P.LINES:
        df = P.load(line)
        df = df[~P.is_holdout(df["policy_id"])]
        for fname, (tr_years, va_year) in P.FOLDS.items():
            tr, va = df[df["year"].isin(tr_years)], df[df["year"] == va_year]
            pred = np.asarray(fn(tr, va, line), float)
            parts.append(pd.DataFrame({"row_id": va["row_id"].to_numpy(), "line": line, "fold": fname,
                                       "year": va_year, "exposure": va["exposure"].to_numpy(),
                                       "target_b": va[P.TARGET].to_numpy(), "pred": pred}))
            print(f"  {name} {line} {fname}: D2={P.d2(va[P.TARGET].to_numpy(), pred):.4f}", flush=True)
    o = pd.concat(parts, ignore_index=True)
    o.to_parquet(OUT / f"{name}.parquet", index=False)
    r = score(o)
    print(name, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
    with open(OUT.parent / "results.tsv", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{name}\t{r['score']}\t{r['pooled_mean']}\t{json.dumps(r['by'])}\n")
    return o


def score(o):
    """Stesso punteggio di prepare.evaluate a partire da previsioni salvate."""
    w = {"motor": 254986, "buildings": 184155, "contents": 187498}
    by = {f"{l}|{f}": P.d2(g["target_b"], g["pred"]) for (l, f), g in o.groupby(["line", "fold"])}
    sc = sum(by[f"{l}|{f}"] * w[l] for l in P.LINES for f in P.FOLDS) / (sum(w.values()) * len(P.FOLDS))
    pool = {f: P.d2(g["target_b"], g["pred"]) for f, g in o.groupby("fold")}
    out = {"by": {k: round(v, 5) for k, v in by.items()}, "score": round(sc, 5),
           "pooled": {k: round(v, 5) for k, v in pool.items()}, "pooled_mean": round(float(np.mean(list(pool.values()))), 5)}
    own = {f"{l}|{f}": P.d2_own(g["row_id"], g["pred"]) for (l, f), g in o.groupby(["line", "fold"])}
    if all(v is not None for v in own.values()):   # giudice secondario senza spalmatura (19/10)
        out["by_own"] = {k: round(v, 5) for k, v in own.items()}
        out["score_own"] = round(sum(own[f"{l}|{f}"] * w[l] for l in P.LINES for f in P.FOLDS) / (sum(w.values()) * len(P.FOLDS)), 5)
    unc = {f"{l}|{f}": P.d2_unc(g["row_id"], g["pred"]) for (l, f), g in o.groupby(["line", "fold"])}
    if all(v is not None for v in unc.values()):   # terzo giudice con i grandi sinistri veri (20/10)
        out["by_unc"] = {k: round(v, 5) for k, v in unc.items()}
        out["score_unc"] = round(sum(unc[f"{l}|{f}"] * w[l] for l in P.LINES for f in P.FOLDS) / (sum(w.values()) * len(P.FOLDS)), 5)
    return out


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
