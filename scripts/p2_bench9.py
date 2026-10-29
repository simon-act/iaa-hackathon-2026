"""Part 2 (25/10 evening): single model on both markets with a "market" flag instead of prior + corrections.
GBM Tweedie (p=1.5) on cost/exposure with weight = exposure, trained on ALL of the developed market + the emerging train of the fold,
common variables (banded age, normalised occupation, region dropped) + gender (missing in the developed market) + market flag.
Variants: 'additivo' (1-split trees: the market effect is level-only per variable, i.e. a level) and 'interazioni'
(15 leaves: the model can learn where the markets differ). Prediction on the emerging valid set, rescaled to the level of the fold's
emerging train. Compared with sub_07 on the 7-check bench (folds from p2_bench2), rule ≥ 5/7 on shape and own.
Usage: IAA_MARKET=configs/parte2.json python3 scripts/p2_bench9.py [lines]"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M, train as T  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
VARS = {"motor": ["gender", "new_business"], "contents": ["gender"], "buildings": None}
PAR = {"additivo": dict(num_leaves=2, max_depth=1, learning_rate=0.1, rounds=800),
       "interazioni": dict(num_leaves=15, learning_rate=0.05, rounds=500)}


def dev_frame(line):
    d = TR.harmonize(pd.read_parquet(ROOT / f"data/model_s3/model_{line}.parquet"))
    d["row_id"] = "D" + d["row_id"].astype(str)
    return d


lines = sys.argv[1].split(",") if len(sys.argv) > 1 else P.LINES
rows = []
for l in lines:
    em = P.load(l)
    dv = dev_frame(l)
    feats = [c for c in M._features(dv, l) if c in em.columns and c != "region"]
    feats = sorted(set(feats) | {"gender"} - {"region"})
    dv["gender"] = np.nan if "gender" not in dv else dv["gender"]
    dv["market"] = 0.0; em = em.assign(market=1.0)
    cols = feats + ["market"]
    print(l, "variables", len(cols), flush=True)
    for fname, tr, va in folds(em):
        y = va[P.TARGET].to_numpy()
        if VARS[l]:
            TR.CORRECTION_VARS = VARS[l]; base = np.asarray(TR.src_glm(tr, va, l), float)
        else:
            b = np.asarray(TR.src_level(tr, va, l), float); sc = np.asarray(TR.scratch(tr, va, l), float); base = 0.9 * b + 0.1 * sc * b.sum() / sc.sum()
        preds = {"sub_07": base}
        pool = pd.concat([dv[cols + [P.TARGET, "exposure"]], tr[cols + [P.TARGET, "exposure"]]], ignore_index=True)
        Xp, cats = T.to_frame(pool, cols); Xv, _ = T.to_frame(va, cols, cats); Xt, _ = T.to_frame(tr, cols, cats)
        e = pool["exposure"].clip(lower=1e-3).to_numpy()
        for name, pp in PAR.items():
            prm = dict(T.GBM_PARAMS, num_threads=4, **{k: v for k, v in pp.items() if k != "rounds"})
            m = lgb.train(prm, lgb.Dataset(Xp, label=pool[P.TARGET].to_numpy() / e, weight=e), num_boost_round=pp["rounds"])
            pv = m.predict(Xv) * va["exposure"].to_numpy(); pt = m.predict(Xt) * tr["exposure"].to_numpy()
            preds[f"unico {name}"] = pv * tr[P.TARGET].sum() / pt.sum()
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
        print(l, fname, {k: round(P.d2(y, p * y.sum() / p.sum()), 4) for k, p in preds.items()}, flush=True)
r = pd.DataFrame(rows); bb = r[r.variante == "sub_07"].set_index(["linea", "fold"])
for c in ("forma", "own", "unc"):
    r[f"d_{c}"] = (r[c].to_numpy() - bb.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
r.to_csv(Path(P.RESULTS) / "p2_bench9.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "sub_07"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"), unc=("d_unc", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 vs sub_07"); print(s.round(1).to_string())
