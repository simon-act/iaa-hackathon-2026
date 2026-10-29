"""Part 2 (27/10): correction GBM redone properly (in bench13 it stopped at 1–35 trees more or less at random, because
early stopping on a single COST holdout follows large-claim noise). Vs p2_sub_17 (spec v10):
- 'gbm freq cv': Poisson on claim COUNT, init = log(expected count from the prior × correction already in sub_17), trees chosen
  with lgb.cv over 5 policy groups (lr 0.03, 4 leaves, min 500 rows); factor applied to cost, mean 1 on train.
- 'gbm costo cv': Tweedie 1.5 on cost, init = log(sub_17), trees with lgb.cv over 5 groups.
- 'gbm freq N' (N = 25/50/100/200, lr 0.03): curve diagnostics only, not a decision.
Rule: ≥ 5/7 shape AND own. Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench14.py [lines]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

COMMON = dict(learning_rate=0.03, num_leaves=4, min_data_in_leaf=500, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, max_bin=63, max_cat_to_onehot=8, cat_smooth=50, min_data_per_group=300, verbose=-1, num_threads=4, seed=7)
PF = dict(COMMON, objective="poisson"); PT = dict(COMMON, objective="tweedie", tweedie_variance_power=1.5)


def glm(tr, va, l, vars_, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, vars_, ov
    try:
        return np.asarray(TR.src_glm(tr, va, l), float) if vars_ else np.asarray(TR.src_level(tr, va, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def base17(tr, va, l):
    """sub_17 per row on tr (in-sample) and va, without the scratch part; plus the final function and the raw prior."""
    if l == "motor":
        a = lambda d: 0.5 * glm(tr, d, l, ["new_business", "gender"]) + 0.5 * glm(tr, d, l, ["new_business", "gender", "fuel_type", "cover_type"])
        return a(tr), a(va), (lambda p: p)
    if l == "buildings":
        ov = TR.peril_own_override(tr, va, l, ["storm"])[0]
        sc = np.asarray(TR.scratch(tr, va, l), float)
        return glm(tr, tr, l, None, ov), glm(tr, va, l, None, ov), (lambda p: 0.9 * p + 0.1 * sc * p.sum() / sc.sum())
    p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
    p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
    ov = pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])
    f_tr = TR.freq_correction(tr, tr, l, ["works_from_home"]); f_va = TR.freq_correction(tr, va, l, ["works_from_home"])
    return glm(tr, tr, l, ["gender"], ov) * f_tr, glm(tr, va, l, ["gender"], ov) * f_va, (lambda p: p)


def frames(tr, va, l):
    feats = [c for c in M._features(tr, l) if c in va.columns] + (["fuel_type"] if l == "motor" else [])
    Xtr, cats = M.to_frame(tr, feats); Xva, _ = M.to_frame(va, feats, cats)
    return Xtr, Xva


def cv_rounds(prm, X, y, init, groups):
    folds_ = [(np.where(groups != k)[0], np.where(groups == k)[0]) for k in range(5)]
    r = lgb.cv(prm, lgb.Dataset(X, label=y, init_score=init), num_boost_round=1500, folds=folds_,
               callbacks=[lgb.early_stopping(150, verbose=False)], stratified=False)
    return len(next(iter(r.values())))


rows, logs = [], []
LINES = sys.argv[1].split(",") if len(sys.argv) > 1 else P.LINES
for l in LINES:
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        b_tr, b_va, fin = base17(tr, va, l)
        preds = {"sub_17": fin(b_va)}
        Xtr, Xva = frames(tr, va, l)
        grp = pd.util.hash_array(tr["policy_id"].to_numpy()) % 5
        # frequency: expected count from the prior × relative sub_17 correction (b / prior)
        n, e = TR._exp_count(tr, l)
        rel = b_tr / TR._prior(tr, l) if l != "contents" else b_tr / np.clip(np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float), 1e-6, None)
        ee = np.clip(e * rel, 1e-6, None); ee *= n.sum() / ee.sum(); init = np.log(ee)
        nr = cv_rounds(PF, Xtr, n, init, grp)
        for N, tag in [(nr, "gbm freq cv")] + [(k, f"gbm freq {k}") for k in (25, 50, 100, 200)]:
            m = lgb.train(PF, lgb.Dataset(Xtr, label=n, init_score=init), num_boost_round=N)
            c_tr = np.exp(m.predict(Xtr, raw_score=True)); c_va = np.exp(m.predict(Xva, raw_score=True))
            preds[tag] = fin(b_va * c_va / np.average(c_tr, weights=ee))
        yt = tr[P.TARGET].to_numpy(); it = np.log(np.clip(b_tr, 1e-6, None))
        nc = cv_rounds(PT, Xtr, yt, it, grp)
        m = lgb.train(PT, lgb.Dataset(Xtr, label=yt, init_score=it), num_boost_round=nc)
        c_tr = np.exp(m.predict(Xtr, raw_score=True)); c_va = np.exp(m.predict(Xva, raw_score=True))
        preds["gbm costo cv"] = fin(b_va * c_va / np.average(c_tr, weights=b_tr))
        logs.append(dict(linea=l, fold=fname, alberi_freq_cv=nr, alberi_costo_cv=nc))
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
        print(l, fname, "freq trees", nr, "cost trees", nc, flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "sub_17"].set_index(["linea", "fold"])
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
tag = "_".join(LINES)
out = Path(P.RESULTS); r.to_csv(out / f"p2_bench14_{tag}.csv", index=False); pd.DataFrame(logs).to_csv(out / f"p2_bench14_{tag}_alberi.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "sub_17"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 vs sub_17"); print(s.round(2).to_string())
