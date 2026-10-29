"""Part 2 (27/10): two new ideas vs p2_sub_16 (spec v9), 7-check bench, rule ≥ 5/7 shape AND own.
1. Correction GBM à la Dutang–Spedicato–Guibert (Variance 2025, transfer with init_score): Tweedie 1.5 on the target with
   init_score = log(sub_16 prediction on train), lr 0.02, 4 leaves, min 1000 rows, early stopping on 20% of the train
   policies; prediction = sub_16 × exp(correction), fold level unchanged. Buildings: on the prior part (0.9), then + 0.1 scratch.
2. FREQUENCY correction: Poisson GLM (glum, ridge, alpha chosen as in src_glm) on claim counts with offset = expected
   count from the prior (Σ f_p × pi_p × deflator), factors applied to cost and rescaled to mean 1 on train.
   Base: motor with the sub_10 cost correction (nb, gender); contents as sub_10.
Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_bench13.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_bench2 import folds  # noqa: E402

FREQV = {"motor": {"freq fuel+cover": ["fuel_type", "cover_type"],
                   "freq fuel+cover+vage+ylic": ["fuel_type", "cover_type", "vehicle_age", "years_licensed"]},
         "contents": {"freq wfh": ["works_from_home"], "freq wfh+region": ["works_from_home", "region"]}}
GBM = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.02, num_leaves=4, min_data_in_leaf=1000,
           lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, max_bin=63, max_cat_to_onehot=8,
           cat_smooth=50, min_data_per_group=500, verbose=-1, num_threads=4, seed=7)


def glm(tr, va, l, vars_, ov=None):
    old, TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = TR.CORRECTION_VARS, vars_, ov
    try:
        return np.asarray(TR.src_glm(tr, va, l), float) if vars_ else np.asarray(TR.src_level(tr, va, l), float)
    finally:
        TR.CORRECTION_VARS, TR._PRIOR_OVERRIDE = old, None


def peril_ov(tr, va, l):
    p_tr = np.asarray(TR.src_peril(tr, tr, l, correct=False, k_freq=100.0, k_sev=200.0), float)
    p_va = np.asarray(TR.src_peril(tr, va, l, correct=False, k_freq=100.0, k_sev=200.0), float)
    return pd.DataFrame({"mu_src": np.r_[p_tr, p_va]}, index=np.r_[tr.row_id.to_numpy(), va.row_id.to_numpy()])


def base16(tr, va, l):
    """(sub_16 prediction on va, on the part without scratch for tr and va, final function)"""
    if l == "motor":
        a = lambda d: 0.5 * glm(tr, d, l, ["new_business", "gender"]) + 0.5 * glm(tr, d, l, ["new_business", "gender", "fuel_type", "cover_type"])
        return a(tr), a(va), (lambda p: p)
    if l == "buildings":
        ov = TR.peril_own_override(tr, va, l, ["storm"])[0]
        ptr, pva = glm(tr, tr, l, None, ov), glm(tr, va, l, None, ov)
        sc = np.asarray(TR.scratch(tr, va, l), float)
        return ptr, pva, (lambda p: 0.9 * p + 0.1 * sc * p.sum() / sc.sum())
    ov = peril_ov(tr, va, l)
    return glm(tr, tr, l, ["gender"], ov), glm(tr, va, l, ["gender"], ov), (lambda p: p)


def gbm_corr(tr, va, l, b_tr, b_va):
    feats = [c for c in M._features(tr, l) if c in va.columns] + [c for c in ("fuel_type",) if c in tr.columns and l == "motor"]
    Xtr, cats = M.to_frame(tr, feats); Xva, _ = M.to_frame(va, feats, cats)
    h = (pd.util.hash_array(tr["policy_id"].to_numpy()) % 5 == 0)
    y = tr[P.TARGET].to_numpy(); init = np.log(np.clip(b_tr, 1e-6, None))
    dtr = lgb.Dataset(Xtr[~h], label=y[~h], init_score=init[~h]); dho = lgb.Dataset(Xtr[h], label=y[h], init_score=init[h], reference=dtr)
    m = lgb.train(GBM, dtr, num_boost_round=3000, valid_sets=[dho], callbacks=[lgb.early_stopping(100, verbose=False)])
    it = m.best_iteration or 0
    if it == 0:
        return b_va, 0
    c_tr = np.exp(m.predict(Xtr, raw_score=True, num_iteration=it)); c_va = np.exp(m.predict(Xva, raw_score=True, num_iteration=it))
    return b_va * c_va / np.average(c_tr, weights=b_tr), it


def freq_corr(tr, va, l, vars_, b_va):
    f = TR.freq_correction(tr, va, l, vars_)
    return b_va * f, TR.freq_correction.last


rows, logs = [], []
for l in P.LINES:
    df = P.load(l)
    for fname, tr, va in folds(df):
        y = va[P.TARGET].to_numpy()
        b_tr, b_va, fin = base16(tr, va, l)
        preds = {"sub_16": fin(b_va)}
        g, it = gbm_corr(tr, va, l, b_tr, b_va); preds["gbm corr (early stop)"] = fin(g); logs.append(dict(linea=l, fold=fname, prova="gbm", alberi=it))
        if l in FREQV:
            b10 = glm(tr, va, l, ["new_business", "gender"]) if l == "motor" else b_va
            for k, vs in FREQV[l].items():
                p, info = freq_corr(tr, va, l, vs, b10); preds[k] = p; logs.append(dict(linea=l, fold=fname, prova=k, **{kk: str(v) for kk, v in info.items()}))
        for k, p in preds.items():
            rows.append(dict(linea=l, fold=fname, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p)))
        print(l, fname, "trees", it, flush=True)
r = pd.DataFrame(rows)
b = r[r.variante == "sub_16"].set_index(["linea", "fold"])
for c in ("forma", "own"):
    r[f"d_{c}"] = (r[c].to_numpy() - b.loc[list(zip(r.linea, r.fold)), c].to_numpy()) * 1e4
out = Path(P.RESULTS); r.to_csv(out / "p2_bench13.csv", index=False); pd.DataFrame(logs).to_csv(out / "p2_bench13_log.csv", index=False)
pd.set_option("display.width", 220)
s = r[r.variante != "sub_16"].groupby(["linea", "variante"]).agg(forma=("d_forma", "mean"), own=("d_own", "mean"),
    pos_forma=("d_forma", lambda x: f"{(x > 0).sum()}/{len(x)}"), pos_own=("d_own", lambda x: f"{(x > 0).sum()}/{len(x)}"))
print("Δ e·4 vs sub_16"); print(s.round(2).to_string())
print(r[r.variante != "sub_16"].pivot_table(index=["linea", "variante"], columns="fold", values="d_forma").round(1).to_string())
