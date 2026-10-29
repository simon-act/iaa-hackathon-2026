"""Part 2 (28/10 evening): importance weights "how much the 2023–24 row resembles the 2025 test" (adversarial validation).
Binary LightGBM train(2023–24)=0 / test(2025)=1 on the model's rating variables (excluding year, weather of the year and claims history, which
separate by construction), out-of-fold probabilities over 5 policy groups; weight = p/(1−p) × n_train/n_test, capped at the 1st–99th pct.
Output: results_p2_llc/p2_shift_weights_<line>.parquet (row_id, w). Usage: IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_shift_weights.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, models_extra as M  # noqa: E402
from sklearn.metrics import roc_auc_score
DROP = {"customer_since_year", "year", "days_quote_to_inception", "renewal_count", "tenure", "years_at_address", "prior_claims_5y",
        "hail_days", "dry_summer_index", "freezing_days", "rainfall_mm", "mean_temperature_c", "wildfire_index", "wind_exposure_index"}
DROP_PREFIX = ("addr_prev", "pol_prev", "cust_")   # weather of the year and claims history: change with the year by construction
for l in P.LINES:
    tr, te = P.load(l), P.load(l, "test")
    if l == "motor":
        tr, te = M._prem(tr), M._prem(te)
    feats = [c for c in M._features(tr, l) if c in te.columns and c not in DROP and not c.startswith(DROP_PREFIX)]
    both = pd.concat([tr.assign(_y=0), te.assign(_y=1)], ignore_index=True)
    X, _ = M.to_frame(both, feats); y = both["_y"].to_numpy()
    grp = pd.util.hash_array(both["policy_id"].astype(str).to_numpy()) % 5
    p = np.zeros(len(both))
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=3)
    for k in range(5):
        m = lgb.train(prm, lgb.Dataset(X[grp != k], label=y[grp != k]), num_boost_round=300)
        p[grp == k] = m.predict(X[grp == k])
    auc = roc_auc_score(y, p)
    ptr = np.clip(p[y == 0], 1e-4, 1 - 1e-4); w = ptr / (1 - ptr) * (y == 0).sum() / (y == 1).sum()
    lo, hi = np.percentile(w, [1, 99]); w = np.clip(w, lo, hi); w /= w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum()
    imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False).head(6)
    pd.DataFrame({"row_id": tr["row_id"].to_numpy(), "w": w}).to_parquet(Path(P.RESULTS) / f"p2_shift_weights_{l}.parquet", index=False)
    nb = tr["new_business"].astype(bool).to_numpy()
    print(f"{l}: AUC {auc:.3f} | mean weight NB {w[nb].mean():.2f} renewals {w[~nb].mean():.2f} | effective rows {ess:.0f} of {len(w)} |",
          "variables:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.items()), flush=True)
