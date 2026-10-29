"""Test bench for the per-peril SEVERITY of the freq×sev (20/10): distributions and parameters.
Data: autoresearch_sev/claims_sev.parquet (one claim per row, sev = capped, developed cost at 2025 level).
Folds: fit AY ≤ 2022 → evaluate 2023; AY ≤ 2023 → 2024. Metric: Gamma D² per peril (Gamma deviance: relative error
counts equally for small and large claims), then per-line mean weighted by peril cost.
"base" = the fs2 severity: 1-split Gamma GBM on SEV_DRIVERS (+ make_model encoding for motor) if > 1500 claims,
Gamma GLM on log SI for the small home perils, otherwise the mean.
Usage: python3 -I scripts/sev_bench.py <config1,config2,...>  (output results_s3/sev_bench.csv)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import GammaRegressor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os; os.environ.setdefault("IAA_MARKET", "configs/parte1_s3.json")
import models_extra as M  # noqa: E402

C = pd.read_parquet(ROOT / "autoresearch_sev/claims_sev.parquet")
DRIV = M.SEV_DRIVERS
# 22/10 evening: deductible relative to insured value (fixed compulsory deductible: 250 motor, 350 home)
C["xs_rel"] = (C["voluntary_excess"].astype(float) + np.where(C["line"] == "motor", 250, 350)) / np.where(
    C["line"] == "motor", C["vehicle_value_group"].astype(float), C["sum_insured"].astype(float).clip(lower=1))
ALLX = [c for c in C.columns if c not in ("row_id", "line", "peril", "ay", "is_open", "exposure", "sev")]
CONFIGS = {
    "base": {}, "tweedie1.8": {"objective": "tweedie", "tweedie_variance_power": 1.8},
    "tweedie1.95": {"objective": "tweedie", "tweedie_variance_power": 1.95}, "lognormale": {"_log": "l2"},
    "huber_log": {"_log": "huber"}, "l2_10": {"lambda_l2": 10.0}, "l2_200": {"lambda_l2": 200.0},
    "giri_75": {"_rounds": 75}, "giri_300": {"_rounds": 300}, "foglie3": {"num_leaves": 3, "max_depth": 2},
    "min_sinistri_750": {"_min": 750}, "min_sinistri_3000": {"_min": 3000}, "piu_franchigia": {"_extra": ["voluntary_excess", "compulsory_excess"]},
    "xsrel": {"_extra": ["voluntary_excess", "xs_rel"]},
    "tutte_variabili": {"_all": True}, "glm_logsi_casa_tutti": {"_glm_all": True}, "media": {"_mean": True},
}


def gdev(y, mu):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    return 2 * np.sum(-np.log(y / mu) + (y - mu) / mu)


def frame(df, cols, cats=None):
    X = df[cols].copy(); out = {}
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool:
            v = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
            lv = cats[c] if cats else sorted(v.unique()); X[c] = pd.Categorical(v, categories=lv); out[c] = lv
    return X, out


def fit_predict(tr, va, line, p, cfg):
    cfg = dict(cfg); y = tr["sev"].to_numpy(float); mean = float(y.mean())
    if cfg.pop("_mean", False):
        return np.full(len(va), mean)
    mn = cfg.pop("_min", 1500); rounds = cfg.pop("_rounds", M._SEV_ROUNDS); lg = cfg.pop("_log", None)
    extra = cfg.pop("_extra", []); allx = cfg.pop("_all", False); glm_all = cfg.pop("_glm_all", False)
    small_glm = (line in M.SMALL_SEV_SI and p in M.SMALL_SEV_SI[line]) or (glm_all and line != "motor")
    if len(tr) <= mn or (glm_all and line != "motor"):
        if small_glm and len(tr) > 150:
            ls = np.log(tr["sum_insured"].clip(lower=1).to_numpy()); mu_, sd = ls.mean(), ls.std() + 1e-9
            g = GammaRegressor(alpha=1.0, max_iter=500).fit(((ls - mu_) / sd).reshape(-1, 1), y)
            return g.predict(((np.log(va["sum_insured"].clip(lower=1).to_numpy()) - mu_) / sd).reshape(-1, 1))
        return np.full(len(va), mean)
    cols = [c for c in (ALLX if allx else DRIV[line] + extra) if c in tr.columns and c != "make_model"]
    Xtr, cats = frame(tr, cols); Xva, _ = frame(va, cols, cats)
    if line == "motor" and "make_model" in tr.columns and not allx:
        te_tr, te_va = M._te_make_model(tr, va, y, np.full(len(tr), mean), k=20.0 * mean)
        Xtr = Xtr.assign(mm_sev_te=te_tr); Xva = Xva.assign(mm_sev_te=te_va)
    params = dict(M.SEV_PARAMS, min_data_in_leaf=max(200, len(tr) // 50), num_threads=2); params.update(cfg)
    if lg:
        ly = np.log(y); params.update(objective="huber" if lg == "huber" else "regression", alpha=1.0)
        m = lgb.train(params, lgb.Dataset(Xtr, label=ly, init_score=np.full(len(y), ly.mean())), num_boost_round=rounds)
        r = ly - (m.predict(Xtr, raw_score=True) + ly.mean())
        return np.exp(m.predict(Xva, raw_score=True) + ly.mean()) * np.mean(np.exp(r))   # Duan smearing
    m = lgb.train(params, lgb.Dataset(Xtr, label=y, init_score=np.full(len(y), np.log(mean))), num_boost_round=rounds)
    return np.exp(m.predict(Xva, raw_score=True) + np.log(mean))


rows = []
for cname in sys.argv[1].split(","):
    for line in ("motor", "buildings", "contents"):
        cl = C[C.line == line]
        for last in (2022, 2023):
            num = den = 0.0
            for p, g in cl.groupby("peril"):
                tr = g[g.ay <= last]; va = g[g.ay == last + 1]
                pred = fit_predict(tr, va, line, p, CONFIGS[cname])
                d2 = 1 - gdev(va.sev, pred) / gdev(va.sev, np.full(len(va), tr.sev.mean()))
                w = float(va.sev.sum()); num += d2 * w; den += w
                rows.append({"config": cname, "linea": line, "perilo": p, "fold": f"F{last + 1}", "d2": d2, "peso": w})
            print(cname, line, f"F{last + 1}", round(num / den, 5), flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results_s3" / f"sev_bench_{sys.argv[1].split(',')[0]}.csv", index=False)
