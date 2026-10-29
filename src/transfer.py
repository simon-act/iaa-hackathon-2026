"""Part 2 — adaptation of the Part 1 model to a new market (actuarial "transfer").

Idea: the source market has a lot of data and a structure (relativities by peril) that we know well; the new
market has little data, a different level and possibly different relativities. We proceed as with credibility:
    prediction = Part 1 relativities (on the common variables)  ×  correction estimated on the new data,
where the correction is wider the more the new data support it.

Three steps, one command each:

1. export-source  (runs with the config of the SOURCE market, e.g. IAA_MARKET=configs/parte1.json)
       python -I transfer.py export-source configs/<new_market>.json
   Fits the Part 1 freq×sev using only the columns that also exist in the new market, and writes its
   prediction for each new row (training and test) to <new results>/source_pred_<line>.parquet.
   This is the actuarial "prior": what this policy would cost if the new market behaved like Part 1.

2. new-market models (run with the config of the NEW market), available to oof.py / blend.py / final.py:
     - scratch      : freq×sev fitted on the new data only (benchmark: "ignore Part 1")
     - src_level    : Part 1 prior × a single level factor per line (benchmark: "ignore the new data")
     - src_glm      : prior as offset + Tweedie GLM with ridge penalty on the variables flagged by the audit
                      (default: region, year, unobservable peril mix → region + building/vehicle age)
     - src_gbm      : prior as init_score + heavily regularised Tweedie GBM (few iterations, large leaves)
                      that learns only robust deviations
   Policy-level credibility (experience.py) is applied on top as in Part 1 (*_cred wrappers in models_extra).

3. blend.py on the models above: the weights say HOW FAR the new market departs from Part 1 — this is the number to
   present to the judges ("adaptation").

Guardrail check: the 'scratch' model must beat 'src_level' on the new market's folds; if it does not,
the new data are too few or too different and we stay close to the prior.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402
import models_extra as M  # noqa: E402
import market_config as MC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORRECTION_VARS = ["region", "urban_density", "building_age", "vehicle_age", "driver_age", "policyholder_age",
                   "sum_insured", "cover_type", "property_type", "tenure"]


# ----------------------------------------------------------------- 1. export of the prior from Part 1
AGE_BANDS = [(17, 24, "17-24"), (25, 34, "25-34"), (35, 49, "35-49"), (50, 64, "50-64"), (65, 200, "65+")]


def _band(age):
    a = pd.Series(age, dtype=float)
    out = pd.Series(pd.NA, index=a.index, dtype="object")
    for lo, hi, lab in AGE_BANDS:
        out[(a >= lo) & (a <= hi)] = lab
    out[a < 17] = "17-24"
    return out


def harmonize(df):
    """Part 2 (23/10): source-market columns made comparable with the emerging market: age -> emerging-market bands
    (driver_age_band, policyholder_age_band), occupation in lower case without spaces."""
    df = df.copy()
    if "driver_age" in df.columns and "driver_age_band" not in df.columns:
        df["driver_age_band"] = _band(df["driver_age"]).to_numpy()
    if "policyholder_age" in df.columns and "policyholder_age_band" not in df.columns:
        df["policyholder_age_band"] = _band(df["policyholder_age"]).to_numpy()
    if "occupation" in df.columns:
        df["occupation"] = df["occupation"].astype("string").str.strip().str.lower()
    return df


def export_source(new_cfg_path: str):
    new_cfg = MC.load(new_cfg_path)
    out = Path(new_cfg["results"]); out.mkdir(parents=True, exist_ok=True)
    for line in P.LINES:
        src = harmonize(P.load(line))                        # source market (current config)
        new_tr = pd.read_parquet(Path(new_cfg["model_dir"]) / f"model_{line}.parquet")
        new_te = pd.read_parquet(Path(new_cfg["model_dir"]) / f"test_{line}.parquet")
        common = [c for c in M._features(src, line) if c in new_tr.columns and c in new_te.columns]
        # 23/10: a categorical whose new-market levels do not exist in the source (region R01–R12 vs E01–E06)
        # carries no information in the prior: drop it, and let the correction on the new data estimate it
        def _overlap(c):
            if pd.api.types.is_numeric_dtype(src[c]) and pd.api.types.is_numeric_dtype(new_te[c]):
                return 1.0
            seen = set(src[c].dropna().astype(str).unique())
            v = new_te[c].dropna().astype(str)
            return float(v.isin(seen).mean()) if len(v) else 1.0
        dropped = [c for c in common if _overlap(c) < 0.5]
        common = [c for c in common if c not in dropped]
        if dropped:
            print(f"{line}: variables dropped from the prior (new levels): {dropped}", flush=True)
        M.FEATURE_WHITELIST = set(common)
        import os
        if os.environ.get("P2_NB_OFF"):   # 24/10: probe "the broker portfolio is not organic new business"
            new_te = new_te.assign(new_business=False)
        both = pd.concat([new_tr, new_te], ignore_index=True)
        # technical columns the freq×sev reads from the "valid" frame: exposure and, for the LL loading, sum_insured
        pred = np.asarray(M._fs2(src, both, line, ll_si=True), float)
        outdf = pd.DataFrame({"row_id": both["row_id"].to_numpy(), "mu_src": pred})
        # 23/10: also expected frequency (count, 2025 level) and severity by peril of the source market, for the
        # peril-by-peril transfer (new-market levels by peril instead of a single factor per line)
        lp = getattr(M._fs2, "last_parts", None)
        if lp:
            for per, (f_va, s_va) in lp["perils"].items():
                outdf[f"f_{per}"] = f_va; outdf[f"s_{per}"] = s_va
            outdf["ll_src"] = lp["ll"]; outdf["c_src"] = lp["c"]
        outdf.to_parquet(out / (f"source_pred_{line}_nonb.parquet" if os.environ.get("P2_NB_OFF") else f"source_pred_{line}.parquet"), index=False)
        M.FEATURE_WHITELIST = None
        print(f"{line}: Part 1 prior on {len(common)} common variables (out of {len(M._features(src, line))}) -> {out / f'source_pred_{line}.parquet'}",
              f"| prior level {pred[:len(new_tr)].sum() / new_tr['exposure'].sum():.1f} vs new target {new_tr[P.TARGET].sum() / new_tr['exposure'].sum():.1f}", flush=True)
    (out / "source_export.json").write_text(json.dumps({"source": MC.CFG.get("_path", "parte1 (default)"), "new": new_cfg_path}, indent=1))


# ----------------------------------------------------------------- 2. new-market models
_PRIOR_OVERRIDE = None   # 24/10: DataFrame indexed by row_id with mu_src (and components) replacing the file (experiments)
GLM_WPOW = 1.0           # 28/10 (test D1): GLM weight = prior**GLM_WPOW; 1 = as in sub_21, 0.5 = exact offset for Tweedie p=1.5
PRIOR_SUFFIX = ""        # 24/10: e.g. "_meteo" -> source_pred_<line>_meteo.parquet


def _prior(df, line):
    if _PRIOR_OVERRIDE is not None:
        return np.clip(_PRIOR_OVERRIDE["mu_src"].reindex(df["row_id"]).to_numpy(), 1e-3, None)
    f = P.RESULTS / f"source_pred_{line}{PRIOR_SUFFIX}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"missing {f}: first run  IAA_MARKET=<source config> python -I transfer.py export-source <new config>")
    s = pd.read_parquet(f).set_index("row_id")["mu_src"]
    mu = s.reindex(df["row_id"]).to_numpy()
    if np.isnan(mu).any():
        raise ValueError(f"{int(np.isnan(mu).sum())} rows without prior in {line}: rerun export-source")
    return np.clip(mu, 1e-3, None)


def scratch(train, valid, line):
    return M.fs2_all_ll(train, valid, line)


def src_level(train, valid, line):
    """prior × a line level factor (estimated on the new data)."""
    p_tr, p_va = _prior(train, line), _prior(valid, line)
    k = train[P.TARGET].sum() / p_tr.sum()
    src_level.last_k = float(k)
    return p_va * k


def _bins(train, valid, cols, q=6):
    Xtr, Xva = {}, {}
    for c in cols:
        a, b = train[c], valid[c]
        if pd.api.types.is_numeric_dtype(a) and a.nunique() > q:
            e = np.unique(np.nanquantile(a.astype(float), np.linspace(0, 1, q + 1)[1:-1]))
            fa = np.where(a.isna(), "na", np.searchsorted(e, a.astype(float).fillna(0)).astype(str))
            fb = np.where(b.isna(), "na", np.searchsorted(e, b.astype(float).fillna(0)).astype(str))
        else:
            fa = a.astype(object).where(a.notna(), "na").astype(str).to_numpy(); fb = b.astype(object).where(b.notna(), "na").astype(str).to_numpy()
        Xtr[c], Xva[c] = fa, fb
    Xtr, Xva = pd.DataFrame(Xtr), pd.DataFrame(Xva)
    for c in Xtr.columns:
        vc = Xtr[c].value_counts(); rare = set(vc[vc < 500].index)
        if rare:
            Xtr[c] = Xtr[c].where(~Xtr[c].isin(rare), "rare"); Xva[c] = Xva[c].where(~Xva[c].isin(rare), "rare")
        cats = sorted(set(Xtr[c])); Xtr[c] = pd.Categorical(Xtr[c], categories=cats)
        Xva[c] = pd.Categorical(Xva[c].where(Xva[c].isin(cats), cats[0]), categories=cats)
    return Xtr, Xva


def src_glm(train, valid, line, alpha=None):
    """prior as offset + ridge Tweedie GLM on a few variables: the readable 'actuarial' correction."""
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    p_tr, p_va = _prior(train, line), _prior(valid, line)
    cols = [c for c in CORRECTION_VARS if c in train.columns and c in valid.columns and train[c].nunique() > 1]
    Xtr, Xva = _bins(train, valid, cols)
    y = train[P.TARGET].to_numpy(); e = train["exposure"].to_numpy()
    if alpha is None:   # penalty chosen on 10% of the new training policies
        h = pd.util.hash_array(train["policy_id"].to_numpy()) % 10 == 3
        best = (-1e9, None)
        for a in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
            g = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=a, l1_ratio=0.0, max_iter=300)
            g.fit(Xtr[~h], y[~h] / p_tr[~h], sample_weight=p_tr[~h] ** GLM_WPOW)     # response = y / prior, weight = prior: ≡ log-prior offset only for Poisson; with Tweedie p=1.5 the exact offset would need weight prior^0.5 (known approximation, kept because it is the form measured on the bench; independent review 28/10)
            sc = P.d2(y[h], g.predict(Xtr[h]) * p_tr[h])
            if sc > best[0]:
                best = (sc, a)
        alpha = best[1]
    g = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=alpha, l1_ratio=0.0, max_iter=300)
    g.fit(Xtr, y / p_tr, sample_weight=p_tr ** GLM_WPOW)
    src_glm.last = {"alpha": alpha, "coef": dict(zip(g.feature_names_, np.round(g.coef_, 3))), "intercept": round(float(g.intercept_), 3)}
    out = g.predict(Xva) * p_va
    return out * (y.sum() / (g.predict(Xtr) * p_tr).sum())


def src_gbm(train, valid, line, rounds=150):
    """prior as init_score + regularised Tweedie GBM: learns only robust deviations from the prior."""
    import lightgbm as lgb
    import train as T
    p_tr, p_va = _prior(train, line), _prior(valid, line)
    cols = [c for c in M._features(train, line)]
    Xtr, cats = T.to_frame(train, cols); Xva, _ = T.to_frame(valid, cols, cats)
    y = train[P.TARGET].to_numpy()
    base = np.log(y.sum() / p_tr.sum())
    params = dict(T.GBM_PARAMS, learning_rate=0.03, num_leaves=7, min_data_in_leaf=max(2000, len(train) // 100),
                  lambda_l2=50.0, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, num_threads=2)
    m = lgb.train(params, lgb.Dataset(Xtr, label=y, init_score=np.log(p_tr) + base), num_boost_round=rounds)
    pred = np.exp(m.predict(Xva, raw_score=True) + np.log(p_va) + base)
    pin = np.exp(m.predict(Xtr, raw_score=True) + np.log(p_tr) + base)
    return pred * (y.sum() / pin.sum())


# ----------------------------------------------------------------- 3. dress rehearsal
def check(lines=None):
    """guardrail: on each new-market line, comparison of the four models on the folds (frozen judge)."""
    res = {}
    for name, fn in [("src_level", src_level), ("scratch", scratch), ("src_glm", src_glm), ("src_gbm", src_gbm)]:
        r = P.evaluate(fn, lines=lines or P.LINES, verbose=False)
        res[name] = r; print(name, r["by"], "insieme", r.get("pooled_mean"), flush=True)
    (P.RESULTS / "transfer_check.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export-source":
        export_source(sys.argv[2])
    elif cmd == "check":
        check(sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    else:
        print(__doc__)


# ----------------------------------------------------------------- 23/10: peril-by-peril transfer
def _prior_parts(df, line):
    s = (_PRIOR_OVERRIDE if _PRIOR_OVERRIDE is not None else
         pd.read_parquet(P.RESULTS / f"source_pred_{line}{PRIOR_SUFFIX}.parquet").set_index("row_id")).reindex(df["row_id"])
    if s["mu_src"].isna().any():
        raise ValueError("rows without prior: rerun export-source")
    return s


def src_peril(train, valid, line, correct=True, k_freq=100.0, k_sev=200.0, rounds=60):
    """Source-market prior BY PERIL (expected frequency and severity at the 2025 level) with:
    1. new-market frequency and severity level by peril, with credibility towards the line level
       (k_freq / k_sev = claims at which the peril gets 50% weight);
    2. (correct=True) frequency correction with a very short 1-split Poisson GBM on the new data, with the prior
       as init_score: learns region, gender, new business… only where the new data support it.
    Severity: prior × peril level (new-market severity relativities are too noisy with 2 years)."""
    import lightgbm as lgb
    dev, claims, lvl = M._tables()
    dl = dev[dev["line"] == line]
    st, sv = _prior_parts(train, line), _prior_parts(valid, line)
    perils = [c[2:] for c in st.columns if c.startswith("f_")]
    yrs = train["year"].to_numpy()
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    cl = claims[(claims["line"] == line) & claims["row_id"].isin(train["row_id"])]
    # observed count and cost by peril, cost developed (IBNER) and trended to 2025 with the peril trend
    obs = {}
    for p in perils:
        d = dl[dl["peril"] == p].set_index("ay")
        if d.empty:
            continue
        ft, stt = float(d["freq_trend"].iloc[0]), float(d["sev_trend"].iloc[0])
        n = train[f"n_{p}"].to_numpy(float) if f"n_{p}" in train else np.zeros(len(train))
        pi = np.clip(train[f"pi_{p}"].to_numpy(), 1e-3, None) if f"pi_{p}" in train else np.ones(len(train))
        defl = (1 + ft) ** (-(2025 - yrs))
        e_cnt = st[f"f_{p}"].to_numpy() * pi * defl                       # expected count from the prior on the new data
        c = cl[cl["peril"] == p]
        cost = (c["net_capped"] * c["ay"].map(d["ibner_factor"]) * (1 + stt) ** (2025 - c["ay"])).sum()
        obs[p] = dict(n=n, pi=pi, defl=defl, e_cnt=e_cnt, cost=float(cost), n_sum=float(n.sum()),
                      e_sum=float(e_cnt.sum()), s_prior_sum=float((n * st[f"s_{p}"].to_numpy()).sum()))
    a_line = sum(o["n_sum"] for o in obs.values()) / max(sum(o["e_sum"] for o in obs.values()), 1e-9)
    b_line = sum(o["cost"] for o in obs.values()) / max(sum(o["s_prior_sum"] for o in obs.values()), 1e-9)
    tot_va, tot_tr, log = np.zeros(len(valid)), np.zeros(len(train)), {}
    feats = [c for c in M._features(train, line) if c in valid.columns]
    Xtr, cats = M.to_frame(train, feats); Xva, _ = M.to_frame(valid, feats, cats)
    for p, o in obs.items():
        z_f = o["n_sum"] / (o["n_sum"] + k_freq)
        a = z_f * (o["n_sum"] / max(o["e_sum"], 1e-9)) + (1 - z_f) * a_line
        z_s = o["n_sum"] / (o["n_sum"] + k_sev)
        b = z_s * (o["cost"] / max(o["s_prior_sum"], 1e-9)) + (1 - z_s) * b_line
        f_tr = st[f"f_{p}"].to_numpy() * a; f_va = sv[f"f_{p}"].to_numpy() * a
        if correct and o["n_sum"] > 200:
            init = np.log(np.clip(o["e_cnt"] * a, 1e-9, None))
            prm = dict(M.FREQ_PARAMS, learning_rate=0.05, min_data_in_leaf=max(1000, len(train) // 50), lambda_l2=50.0, num_threads=4)
            m = lgb.train(prm, lgb.Dataset(Xtr, label=o["n"], init_score=init), num_boost_round=rounds)
            f_va = f_va * np.exp(m.predict(Xva, raw_score=True)); f_tr = f_tr * np.exp(m.predict(Xtr, raw_score=True))
        s_tr = st[f"s_{p}"].to_numpy() * b; s_va = sv[f"s_{p}"].to_numpy() * b
        tot_va += f_va * s_va; tot_tr += f_tr * s_tr
        log[p] = dict(a=round(a, 3), b=round(b, 3), n=int(o["n_sum"]), z_f=round(z_f, 2), z_s=round(z_s, 2))
    ll = float(lvl.loc[line, "ll_load_2025"])
    tot_va += ll * e_va; tot_tr += ll * e_tr
    c = train[P.TARGET].sum() / tot_tr.sum()
    src_peril.last_log = dict(perils=log, a_line=round(a_line, 3), b_line=round(b_line, 3), c=round(float(c), 3))
    return tot_va * c


def src_peril_nocorr(train, valid, line):
    return src_peril(train, valid, line, correct=False)


# 23/10: GLM correction also using the new or changed emerging-market variables (gender, age bands, new business,
# construction codes, channel) in addition to the Part 2 ones prepared earlier
CORRECTION_VARS2 = CORRECTION_VARS + ["gender", "driver_age_band", "policyholder_age_band", "new_business", "construction_type",
                                      "channel", "ncb_years", "vehicle_value_group", "flood_zone"]


def src_glm2(train, valid, line):
    global CORRECTION_VARS
    old = CORRECTION_VARS
    CORRECTION_VARS = CORRECTION_VARS2
    try:
        return src_glm(train, valid, line)
    finally:
        CORRECTION_VARS = old


# ----------------------------------------------------------------- 26/10: own level only for some perils
def peril_own_override(train, valid, line, own, k_freq=100.0, k_sev=200.0):
    """Prior × (1 + Σ_{p in own} share_p × (r_p − 1)), r_p = (a_p·b_p)/(a_line·b_line) from src_peril (credibility k towards the
    line, estimated on train only). The other perils stay at the line level. Returns (override for _PRIOR_OVERRIDE, r).
    Bench: scripts/p2_bench10.py (buildings storm k=100/200 passes 5/7, motor glass+TPP does not)."""
    src_peril(train, train, line, correct=False, k_freq=k_freq, k_sev=k_sev)
    lg = src_peril.last_log
    r = {p: v["a"] * v["b"] / (lg["a_line"] * lg["b_line"]) for p, v in lg["perils"].items()}

    def adj(df):
        s = _prior_parts(df, line)
        m = np.ones(len(df))
        for p in own:
            m += s[f"f_{p}"].to_numpy() * s[f"s_{p}"].to_numpy() / s["mu_src"].to_numpy() * (r[p] - 1)
        return s["mu_src"].to_numpy() * m
    ids = np.r_[train.row_id.to_numpy(), valid.row_id.to_numpy()]
    return pd.DataFrame({"mu_src": np.r_[adj(train), adj(valid)]}, index=ids), r, lg


# ----------------------------------------------------------------- 27/10: correction estimated on frequency
def _exp_count(df, line):
    dev, _, _ = M._tables(); dl = dev[dev["line"] == line]
    s = _prior_parts(df, line); yrs = df["year"].to_numpy(); e = np.zeros(len(df)); n = np.zeros(len(df))
    for p in [c[2:] for c in s.columns if c.startswith("f_")]:
        d = dl[dl["peril"] == p]
        if d.empty or f"n_{p}" not in df:
            continue
        ft = float(d["freq_trend"].iloc[0])
        e += s[f"f_{p}"].to_numpy() * np.clip(df[f"pi_{p}"].to_numpy(), 1e-3, None) * (1 + ft) ** (-(2025 - yrs))
        n += df[f"n_{p}"].to_numpy(float)
    return n, e


def freq_correction(train, valid, line, vars_):
    """Poisson GLM (ridge) on claim count with offset = expected count from the prior (Σ f_p × pi_p × deflator); returns
    the factor for the valid rows, with mean 1 (weighted by expected count) on train. To be applied to cost. Bench: p2_bench13."""
    from glum import GeneralizedLinearRegressor
    n, e = _exp_count(train, line); e = np.clip(e, 1e-6, None)
    Xtr, Xva = _bins(train, valid, vars_)
    h = pd.util.hash_array(train["policy_id"].to_numpy()) % 10 == 3
    best = (-np.inf, None)
    for a in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        g = GeneralizedLinearRegressor(family="poisson", link="log", alpha=a, l1_ratio=0.0, max_iter=300)
        g.fit(Xtr[~h], n[~h] / e[~h], sample_weight=e[~h])
        mu = g.predict(Xtr[h]) * e[h]
        ll = float(np.sum(n[h] * np.log(np.clip(mu, 1e-12, None)) - mu))
        if ll > best[0]:
            best = (ll, a)
    g = GeneralizedLinearRegressor(family="poisson", link="log", alpha=best[1], l1_ratio=0.0, max_iter=300).fit(Xtr, n / e, sample_weight=e)
    freq_correction.last = dict(alpha=best[1], coef=dict(zip(g.feature_names_, np.round(g.coef_, 3))))
    return g.predict(Xva) / np.average(g.predict(Xtr), weights=e)


# ----------------------------------------------------------------- 27/10: correction GBM on claim COUNT (bench p2_bench14)
GBM_FREQ_PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=4, min_data_in_leaf=500, lambda_l2=10.0,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, max_bin=63, max_cat_to_onehot=8, cat_smooth=50,
                       min_data_per_group=300, verbose=-1, num_threads=4, seed=7)


def gbm_freq_correction(train, valid, line, b_tr):
    """Poisson on claim count with init = log(expected count from the prior × relative correction already applied, b_tr / prior),
    trees chosen with lgb.cv over 5 policy groups (early stopping 150). Returns the factor for valid (mean 1 on train)."""
    import lightgbm as lgb
    feats = [c for c in M._features(train, line) if c in valid.columns] + (["fuel_type"] if line == "motor" else [])
    Xtr, cats = M.to_frame(train, feats); Xva, _ = M.to_frame(valid, feats, cats)
    grp = pd.util.hash_array(train["policy_id"].to_numpy()) % 5
    n, e = _exp_count(train, line)
    ee = np.clip(e * b_tr / _prior(train, line), 1e-6, None); ee *= n.sum() / ee.sum(); init = np.log(ee)
    folds_ = [(np.where(grp != k)[0], np.where(grp == k)[0]) for k in range(5)]
    r = lgb.cv(GBM_FREQ_PARAMS, lgb.Dataset(Xtr, label=n, init_score=init), num_boost_round=1500, folds=folds_,
               callbacks=[lgb.early_stopping(150, verbose=False)], stratified=False)
    nr = len(next(iter(r.values())))
    m = lgb.train(GBM_FREQ_PARAMS, lgb.Dataset(Xtr, label=n, init_score=init), num_boost_round=nr)
    gbm_freq_correction.last = dict(alberi=nr)
    return np.exp(m.predict(Xva, raw_score=True)) / np.average(np.exp(m.predict(Xtr, raw_score=True)), weights=ee)
