"""Pesi della combinazione di modelli per linea, scelti sulle previsioni out-of-fold.

Combinazione geometrica o aritmetica? Aritmetica (media pesata delle previsioni), pesi ≥ 0 che sommano a 1.
Onestà: i pesi stimati su un fold vengono valutati sull'altro (cross-fold); i pesi finali usano entrambi i fold.
Uso: python -I blend.py modello1 modello2 [...]
Output: results/blend_weights.json e una riga in results/results.tsv (blend_opt)
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402
from oof import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OOF = P.RESULTS / "oof"


def simplex_grid(k, step=0.05):
    pts = np.arange(0, 1 + 1e-9, step)
    for w in product(pts, repeat=k - 1):
        if sum(w) <= 1 + 1e-9:
            yield np.array(list(w) + [1 - sum(w)])


def best_weights(Y, preds):
    best, bw = -1e9, None
    for w in simplex_grid(len(preds), 0.05 if len(preds) <= 3 else 0.1):
        s = P.d2(Y, sum(wi * p for wi, p in zip(w, preds)))
        if s > best:
            best, bw = s, w
    return bw, best


def main():
    names = sys.argv[1:]
    os_ = [pd.read_parquet(OOF / f"{n}.parquet") for n in names]
    for o in os_[1:]:
        assert (o["row_id"].to_numpy() == os_[0]["row_id"].to_numpy()).all()
    base = os_[0][["row_id", "line", "fold", "year", "exposure", "target_b"]].copy()
    W, cross = {}, {}
    out = base.copy(); out["pred"] = 0.0
    for line in P.LINES:
        m = (base["line"] == line).to_numpy()
        f23 = m & (base["fold"] == "F2023").to_numpy(); f24 = m & (base["fold"] == "F2024").to_numpy()
        y = base["target_b"].to_numpy()
        P_ = [o["pred"].to_numpy() for o in os_]
        w23, _ = best_weights(y[f23], [p[f23] for p in P_])
        w24, _ = best_weights(y[f24], [p[f24] for p in P_])
        cross[line] = {"w_from_F2023_scored_on_F2024": round(P.d2(y[f24], sum(w * p[f24] for w, p in zip(w23, P_))), 5),
                       "w_from_F2024_scored_on_F2023": round(P.d2(y[f23], sum(w * p[f23] for w, p in zip(w24, P_))), 5),
                       "w23": w23.round(2).tolist(), "w24": w24.round(2).tolist()}
        wall, _ = best_weights(y[m], [p[m] for p in P_])
        W[line] = dict(zip(names, wall.round(3).tolist()))
        out.loc[m, "pred"] = sum(w * p[m] for w, p in zip(wall, P_))
    r = score(out)
    # punteggio onesto: per ogni fold, pesi stimati sull'altro fold
    honest = base.copy(); honest["pred"] = 0.0
    for line in P.LINES:
        for f_eval, wkey in (("F2024", "w23"), ("F2023", "w24")):
            mm = ((base["line"] == line) & (base["fold"] == f_eval)).to_numpy()
            w = cross[line][wkey]
            honest.loc[mm, "pred"] = sum(wi * o["pred"].to_numpy()[mm] for wi, o in zip(w, os_))
    rh = score(honest)
    res = {"models": names, "weights": W, "in_sample": r, "cross_fold": rh, "cross_detail": cross}
    (P.RESULTS / "blend_weights.json").write_text(json.dumps(res, indent=1))
    out.to_parquet(OOF / "blend_opt.parquet", index=False)
    with open(P.RESULTS / "results.tsv", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M')}\tblend_opt[{'+'.join(names)}]\t{rh['score']}\t{rh['pooled_mean']}\t{json.dumps(W)}\n")
    print(json.dumps({"weights": W, "pooled_in_sample": r["pooled_mean"], "pooled_cross_fold": rh["pooled_mean"],
                      "score_cross_fold": rh["score"]}, indent=1))


if __name__ == "__main__":
    main()
