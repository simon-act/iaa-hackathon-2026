"""Modelli strutturali del passo 2b (dopo la ricostruzione del generatore).

fs_peril   frequenza × severità per perilo:
           - frequenza: LightGBM Poisson ADDITIVO (alberi a 1 split = log-lineare con curve libere),
             offset log(esposizione × completezza) − (2025−anno)·log(1+trend freq) → frequenza ultimate al 2025
           - severità: media per perilo dei sinistri sviluppati (IBNER) e riportati al 2025, opzionalmente
             con un GBM Gamma additivo molto regolarizzato sui perili con molti sinistri
           - costo = Σ_perili freq × sev, + caricamento large loss; ribilanciato sul training
fs_peril_const  come sopra ma severità costante per perilo (test dell'ipotesi "severità = rumore")
glm2       GLM Tweedie migliorato (convergenza piena, più variabili, interazioni attuariali binnate)
Tutte con la firma fit_predict(train, valid, line) di prepare.evaluate.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402
from train import to_frame, _bin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_DEV = None
_CLAIMS = None
_LVL = None
DROP_NOISE = {"previous_insurer", "campaign_code", "branch_id", "marketing_segment", "vehicle_colour",
              "document_preference", "marketing_consent"}


def _tables():
    global _DEV, _CLAIMS, _LVL
    if _DEV is None:
        _DEV = pd.read_parquet(ROOT / "data" / "interim" / "development_onlevel.parquet")
        _CLAIMS = pd.read_parquet(ROOT / "data" / "interim" / "claims_clean.parquet",
                                  columns=["row_id", "line", "peril", "ay", "net_capped"])
        _LVL = pd.read_parquet(ROOT / "data" / "interim" / "level_2025.parquet")
    return _DEV, _CLAIMS, _LVL


# driver di severità plausibili (dalla ricostruzione del generatore); il resto è rumore
SEV_DRIVERS = {"motor": ["make_model", "vehicle_age", "vehicle_value_group", "urban_density", "region"],
               "buildings": ["sum_insured", "si_per_bedroom", "property_type", "construction_type", "bedrooms", "region"],
               "contents": ["sum_insured", "si_per_bedroom", "high_value_items", "tenure", "region"]}
FREQ_PARAMS = dict(objective="poisson", num_leaves=2, max_depth=1, learning_rate=0.1, min_data_in_leaf=1000,
                   feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=63,
                   max_cat_to_onehot=8, cat_smooth=50, min_data_per_group=500, verbose=-1, num_threads=2, seed=11)
SEV_PARAMS = dict(objective="gamma", num_leaves=2, max_depth=1, learning_rate=0.05, feature_fraction=1.0,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=50.0, max_bin=31, max_cat_to_onehot=8,
                  cat_smooth=100, min_data_per_group=1000, verbose=-1, num_threads=2, seed=12)


FEATURE_WHITELIST = None   # Parte 2: se impostato, solo queste colonne (intersezione tra i mercati)


def _features(train, line, drop_noise=True):
    cols = P.allowed_features(train)
    if drop_noise:
        cols = [c for c in cols if c not in DROP_NOISE and c not in DECOYS]
    if FEATURE_WHITELIST is not None:
        cols = [c for c in cols if c in FEATURE_WHITELIST]
    return [c for c in cols if train[c].nunique(dropna=False) > 1]


def _fs(train, valid, line, sev_model=True, freq_rounds=None, freq_rounds_mult=1):
    dev, claims, lvl = _tables()
    dl = dev[dev["line"] == line]
    perils = sorted(dl["peril"].unique())
    cols = _features(train, line)
    Xtr, cats = to_frame(train, cols)
    Xva, _ = to_frame(valid, cols, cats)
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    yrs = train["year"].to_numpy()
    total_va = np.zeros(len(valid)); total_tr = np.zeros(len(train))
    for p in perils:
        d = dl[dl["peril"] == p].set_index("ay")
        ft = float(d["freq_trend"].iloc[0]); st = float(d["sev_trend"].iloc[0])
        n = train[f"n_{p}"].to_numpy(float)
        pi = np.clip(train[f"pi_{p}"].to_numpy(), 1e-3, None)
        off = np.log(e_tr * pi) - (2025 - yrs) * np.log1p(ft)
        base = np.log(max(n.sum(), 1) / np.exp(off).sum())
        rounds = freq_rounds or (300 if n.sum() > 5000 else 150 if n.sum() > 1000 else 80) * freq_rounds_mult
        m = lgb.train(FREQ_PARAMS, lgb.Dataset(Xtr, label=n, init_score=off + base), num_boost_round=rounds)
        f_va = np.exp(m.predict(Xva, raw_score=True) + base) * e_va          # sinistri ultimate attesi al 2025
        f_tr = np.exp(m.predict(Xtr, raw_score=True) + base) * e_tr
        # severità: sinistri della riga, sviluppati e riportati al 2025
        cl = claims[(claims["line"] == line) & (claims["peril"] == p) & claims["row_id"].isin(train["row_id"])]
        sev_c = cl["net_capped"] * cl["ay"].map(d["ibner_factor"]) * (1 + st) ** (2025 - cl["ay"])
        pos_ = (sev_c > 0).to_numpy()
        cl, sev_c = cl[pos_], sev_c[pos_]
        rel_ibnr = float(np.average(d["rel_ibnr"], weights=d["n_ult"] - d["n_rep"] + 1e-9))
        mean_sev = float(sev_c.mean()) if len(sev_c) else 0.0
        s_va = np.full(len(valid), mean_sev); s_tr = np.full(len(train), mean_sev)
        if sev_model and len(cl) > 1500:
            agg = pd.DataFrame({"row_id": cl["row_id"], "s": sev_c}).groupby("row_id")["s"].agg(["mean", "size"])
            mask = train["row_id"].isin(agg.index).to_numpy()
            a = agg.loc[train.loc[mask, "row_id"]]
            scols = [c for c in SEV_DRIVERS[line] if c in cols]
            ms = lgb.train(dict(SEV_PARAMS, min_data_in_leaf=max(200, int(mask.sum() / 50))),
                           lgb.Dataset(Xtr.loc[mask, scols], label=a["mean"].to_numpy(), weight=a["size"].to_numpy(float),
                                       init_score=np.full(mask.sum(), np.log(mean_sev))), num_boost_round=150)
            s_va = np.exp(ms.predict(Xva[scols], raw_score=True) + np.log(mean_sev))
            s_tr = np.exp(ms.predict(Xtr[scols], raw_score=True) + np.log(mean_sev))
        total_va += f_va * s_va; total_tr += f_tr * s_tr
    ll = float(lvl.loc[line, "ll_load_2025"])
    pred_va = total_va + ll * e_va
    pred_tr = total_tr + ll * e_tr
    return pred_va * (train[P.TARGET].sum() / pred_tr.sum())


def fs_peril(train, valid, line):
    return _fs(train, valid, line, sev_model=True)


def fs_peril_const(train, valid, line):
    return _fs(train, valid, line, sev_model=False)


GLM2_EXTRA = {
    "motor": ["named_young_driver", "telematics_opt_in", "telematics_score", "occupation", "marital_status",
              "household_size", "overnight_parking", "transmission", "driver_assist", "cust_n_lines", "pol_prev_n_last",
              "customer_tenure", "new_business", "renewal_count", "seats", "make_model"],
    "buildings": ["energy_rating", "solar_panels", "bedrooms", "smoke_alarms", "mortgage", "hail_days",
                  "freezing_days", "wildfire_index", "cust_n_lines", "new_business", "occupancy", "si_per_bedroom"],
    "contents": ["energy_rating", "household_size", "occupation", "marital_status", "cust_n_lines", "new_business",
                 "hail_days", "rainfall_mm", "distance_to_river_km", "elevation_m", "trees_within_5m", "si_per_bedroom"],
}
GLM2_INTER = {
    "motor": [("driver_age", "named_young_driver"), ("cover_type", "vehicle_value_group")],
    "buildings": [("soil_type", "dry_summer_index"), ("flood_zone", "distance_to_river_km"), ("soil_type", "trees_within_5m")],
    "contents": [("flood_zone", "distance_to_river_km"), ("tenure", "security_level")],
}


def glm2(train, valid, line):
    from sklearn.linear_model import TweedieRegressor
    from sklearn.preprocessing import OneHotEncoder
    from train import GLM_FEATURES
    cols = [c for c in GLM_FEATURES[line] + GLM2_EXTRA[line] if c in train.columns]
    cols = list(dict.fromkeys(cols))
    P.check_features(cols, train)
    Btr, Bva = {}, {}
    for c in cols:
        a, (b,) = _bin(train[c], [valid[c]], q=10)
        Btr[c], Bva[c] = a, b
    for a_, b_ in GLM2_INTER[line]:
        if a_ in Btr and b_ in Btr:
            ka, (va_,) = _bin(train[a_], [valid[a_]], q=4); kb, (vb_,) = _bin(train[b_], [valid[b_]], q=4)
            Btr[f"{a_}*{b_}"] = ka + "|" + kb; Bva[f"{a_}*{b_}"] = va_ + "|" + vb_
    Btr, Bva = pd.DataFrame(Btr), pd.DataFrame(Bva)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=500, drop="first", sparse_output=True)
    Xtr = enc.fit_transform(Btr); Xva = enc.transform(Bva)
    e = train["exposure"].to_numpy()
    m = TweedieRegressor(power=1.5, link="log", alpha=3e-4, max_iter=2000, tol=1e-6, solver="newton-cholesky")
    m.fit(Xtr, train[P.TARGET].to_numpy() / e, sample_weight=e)
    pred = m.predict(Xva) * valid["exposure"].to_numpy()
    pin = m.predict(Xtr) * e
    return pred * (train[P.TARGET].sum() / pin.sum())


# variabili esca dalla ricostruzione del generatore (importanza ~0 in frequenza su tutti i perili, o rumore in severità)
DECOYS = {"engine_size_cc", "body_type", "seats", "driver_assist", "fuel_type", "vehicle_colour", "transmission",
          "telematics_opt_in", "marital_status", "solar_panels", "document_preference", "customer_tenure",
          "listed_building", "marketing_consent", "cust_other_prev_rate", "cust_other_prev_n_last", "mortgage",
          "starts_jan1", "mileage_k", "licence_age", "ends_dec", "smoke_alarms", "lodgers", "bicycle_cover",
          "cust_has_buildings", "cust_has_contents", "cust_has_motor", "pol_years_seen", "start_month",
          "previous_insurer", "campaign_code", "branch_id", "marketing_segment", "email_domain"}


def gbm_clean(train, valid, line):
    """GBM del passo 2 senza le variabili esca."""
    import train as T
    orig = T.features_for
    T.features_for = lambda df, l: [c for c in orig(df, l) if c not in DECOYS]
    try:
        return T.gbm_b(train, valid, line)
    finally:
        T.features_for = orig


# GLM guidato dal generatore: solo i driver emersi (importanza > 1,5% in almeno un perilo) + driver di severità
GLM_DGP = {
    "motor": ["cover_type", "make_model", "ncb_years", "customer_age", "years_licensed", "overnight_parking",
              "voluntary_excess", "hail_days", "driver_age", "urban_density", "annual_mileage", "region",
              "pol_prev_rate", "prior_claims_5y", "vehicle_value_group", "vehicle_age", "credit_band", "addr_prev_rate"],
    "buildings": ["accidental_damage_cover", "flood_zone", "building_age", "elevation_m", "soil_type", "dry_summer_index",
                  "property_type", "wildfire_index", "region", "freezing_days", "mean_temperature_c", "days_unoccupied",
                  "wind_exposure_index", "prior_claims_5y", "construction_type", "trees_within_5m", "voluntary_excess",
                  "sum_insured", "roof_type", "occupancy", "credit_band", "pol_prev_rate", "addr_prev_rate"],
    "contents": ["accidental_damage_cover", "flood_zone", "elevation_m", "urban_density", "freezing_days", "region",
                 "security_level", "voluntary_excess", "prior_claims_5y", "credit_band", "sum_insured", "building_age",
                 "policyholder_age", "works_from_home", "property_type", "high_value_items", "tenure", "pol_prev_rate",
                 "addr_prev_rate"],
}
GLM_DGP_INTER = {"motor": [], "buildings": [("soil_type", "dry_summer_index")], "contents": []}


def glm_dgp(train, valid, line):
    from sklearn.linear_model import TweedieRegressor
    from sklearn.preprocessing import OneHotEncoder
    cols = [c for c in GLM_DGP[line] if c in train.columns]
    P.check_features(cols, train)
    Btr, Bva = {}, {}
    for c in cols:
        a, (b,) = _bin(train[c], [valid[c]], q=8)
        Btr[c], Bva[c] = a, b
    for a_, b_ in GLM_DGP_INTER[line]:
        ka, (va_,) = _bin(train[a_], [valid[a_]], q=4); kb, (vb_,) = _bin(train[b_], [valid[b_]], q=4)
        Btr[f"{a_}*{b_}"] = ka + "|" + kb; Bva[f"{a_}*{b_}"] = va_ + "|" + vb_
    Btr, Bva = pd.DataFrame(Btr), pd.DataFrame(Bva)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=300, drop="first", sparse_output=True)
    Xtr = enc.fit_transform(Btr); Xva = enc.transform(Bva)
    e = train["exposure"].to_numpy()
    m = TweedieRegressor(power=1.5, link="log", alpha=1e-4, max_iter=1000, solver="lbfgs")
    m.fit(Xtr, train[P.TARGET].to_numpy() / e, sample_weight=e)
    pin = m.predict(Xtr) * e
    return m.predict(Xva) * valid["exposure"].to_numpy() * (train[P.TARGET].sum() / pin.sum())


def fs_peril_s(train, valid, line):
    """freq×sev con più shrinkage sulla frequenza (i residui mostravano relatività troppo spinte)."""
    old = dict(FREQ_PARAMS)
    FREQ_PARAMS.update(learning_rate=0.05)
    try:
        return _fs(train, valid, line, sev_model=True)
    finally:
        FREQ_PARAMS.clear(); FREQ_PARAMS.update(old)


def fs_peril_r2(train, valid, line):
    """freq×sev con il doppio dei round sulla frequenza (fs_peril_s con più shrinkage era peggio)."""
    return _fs(train, valid, line, sev_model=True, freq_rounds_mult=2)


def fs_peril_r4(train, valid, line):
    return _fs(train, valid, line, sev_model=True, freq_rounds_mult=4)


def fs_peril_r8(train, valid, line):
    return _fs(train, valid, line, sev_model=True, freq_rounds_mult=8)


# ---------------------------------------------------------------- esplorazione 3
def _te_make_model(train, valid, y, e_exp, k, folds=5, seed=0):
    """Encoding con credibilità di make_model: log( Z·(A/E) + (1−Z) ), Z = E/(E+k).
    y = osservato (sinistri o costo), e_exp = atteso di riferimento. Out-of-fold sulle righe di training."""
    mm_tr = train["make_model"].astype(str).to_numpy(); mm_va = valid["make_model"].astype(str).to_numpy()
    def enc(idx_fit, keys):
        s = pd.DataFrame({"k": mm_tr[idx_fit], "a": y[idx_fit], "e": e_exp[idx_fit]}).groupby("k")[["a", "e"]].sum()
        z = s["e"] / (s["e"] + k)
        r = np.log(z * (s["a"] / s["e"].clip(lower=1e-9)) + (1 - z))
        return pd.Series(keys).map(r).fillna(0.0).to_numpy()
    out_tr = np.zeros(len(train))
    rng = np.random.default_rng(seed); fold = rng.integers(0, folds, len(train))
    for f in range(folds):
        out_tr[fold == f] = enc(np.where(fold != f)[0], mm_tr[fold == f])
    out_va = enc(np.arange(len(train)), mm_va)
    return out_tr, out_va


def _rounds_es(Xtr, n, init, yrs, params, max_rounds=4000):
    """numero di alberi scelto con early stopping: stima sugli anni precedenti, valida sull'ultimo anno del training."""
    last = yrs.max(); a = yrs < last; b = yrs == last
    if a.sum() == 0 or n[b].sum() < 30:
        return None
    dtr = lgb.Dataset(Xtr[a], label=n[a], init_score=init[a])
    dva = lgb.Dataset(Xtr[b], label=n[b], init_score=init[b], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    return max(50, int(m.best_iteration * 1.15))


SMALL_SEV_SI = {"buildings": ["fire", "flood", "subsidence"], "contents": ["flood"]}


def _ll_weights(train, valid, line):
    """Caricamento large loss per casa ripartito per fascia di somma assicurata (i sinistri oltre soglia sono quasi
    tutti nel quartile più alto di SI), con credibilità 50% verso il caricamento uniforme. Motor: uniforme."""
    if line == "motor":
        return np.ones(len(train)), np.ones(len(valid))
    ex = [c for c in train.columns if c.startswith("excess_")]
    exc = train[ex].sum(axis=1).to_numpy(); e = train["exposure"].to_numpy()
    edges = np.unique(np.quantile(train["sum_insured"], [0, .25, .5, .75, 1.0]))
    btr = np.clip(np.searchsorted(edges, train["sum_insured"], side="right") - 1, 0, len(edges) - 2)
    bva = np.clip(np.searchsorted(edges, valid["sum_insured"], side="right") - 1, 0, len(edges) - 2)
    rate = pd.Series(exc).groupby(btr).sum() / pd.Series(e).groupby(btr).sum()
    rel = (rate / (exc.sum() / e.sum())).reindex(range(len(edges) - 1)).fillna(1.0)
    rel = 0.5 * rel + 0.5
    norm = np.sum(rel.to_numpy()[btr] * e) / e.sum()
    return rel.to_numpy()[btr] / norm, rel.to_numpy()[bva] / norm


HIST = ("pol_prev", "cust_other_prev", "addr_prev")


def _fs2(train, valid, line, mm_te=True, small_sev=True, subs_inter=True, es_rounds=False, mult=4, ll_si=False,
         fparams=None, freq_glm=False, drop_hist=False, zero_keep=None, drop_cust=False):
    dev, claims, lvl = _tables()
    dl = dev[dev["line"] == line]
    perils = sorted(dl["peril"].unique())
    cols = _features(train, line)
    if drop_hist:
        cols = [c for c in cols if not c.startswith(HIST)]
    if drop_cust:
        cols = [c for c in cols if not c.startswith("cust_other_prev")]
    has_mm = mm_te and "make_model" in cols
    base_cols = [c for c in cols if not (has_mm and c == "make_model")]
    Xtr0, cats = to_frame(train, base_cols); Xva0, _ = to_frame(valid, base_cols, cats)
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    yrs = train["year"].to_numpy()
    total_va = np.zeros(len(valid)); total_tr = np.zeros(len(train)); log = {}
    parts = {}
    for p in perils:
        d = dl[dl["peril"] == p].set_index("ay")
        ft = float(d["freq_trend"].iloc[0]); st = float(d["sev_trend"].iloc[0])
        n = train[f"n_{p}"].to_numpy(float)
        pi = np.clip(train[f"pi_{p}"].to_numpy(), 1e-3, None)
        off = np.log(e_tr * pi) - (2025 - yrs) * np.log1p(ft)
        base = np.log(max(n.sum(), 1) / np.exp(off).sum())
        Xtr, Xva = Xtr0, Xva0
        if has_mm:
            te_tr, te_va = _te_make_model(train, valid, n, np.exp(off + base), k=30.0)
            Xtr = Xtr0.assign(mm_freq_te=te_tr); Xva = Xva0.assign(mm_freq_te=te_va)
        params = dict(FREQ_PARAMS, **(fparams or {}))
        if subs_inter and p == "subsidence" and params.get("num_leaves", 2) <= 3:
            params.update(num_leaves=3, max_depth=2)
        default = (300 if n.sum() > 5000 else 150 if n.sum() > 1000 else 80) * mult
        rounds = (_rounds_es(Xtr, n, off + base, yrs, params) if es_rounds else None) or default
        log[p] = rounds
        if freq_glm:
            f_tr, f_va = _freq_glm(train, valid, line, p, n, np.exp(off), e_tr, e_va, Xtr, Xva)
        else:
            if zero_keep:
                # subsampling senza peso delle righe senza sinistri di questo perilo, poi livello riportato ai conteggi
                sel = (n > 0) | (np.random.default_rng(7).random(len(n)) < zero_keep)
                m = lgb.train(params, lgb.Dataset(Xtr[sel], label=n[sel], init_score=(off + base)[sel]), num_boost_round=rounds)
            else:
                m = lgb.train(params, lgb.Dataset(Xtr, label=n, init_score=off + base), num_boost_round=rounds)
            raw_tr = m.predict(Xtr, raw_score=True)
            f_va = np.exp(m.predict(Xva, raw_score=True) + base) * e_va
            f_tr = np.exp(raw_tr + base) * e_tr
            if zero_keep:
                c = n.sum() / np.exp(raw_tr + base + off).sum()
                f_va, f_tr = f_va * c, f_tr * c
        cl = claims[(claims["line"] == line) & (claims["peril"] == p) & claims["row_id"].isin(train["row_id"])]
        sev_c = cl["net_capped"] * cl["ay"].map(d["ibner_factor"]) * (1 + st) ** (2025 - cl["ay"])
        pos_ = (sev_c > 0).to_numpy(); cl, sev_c = cl[pos_], sev_c[pos_]
        mean_sev = float(sev_c.mean()) if len(sev_c) else 0.0
        s_va = np.full(len(valid), mean_sev); s_tr = np.full(len(train), mean_sev)
        agg = pd.DataFrame({"row_id": cl["row_id"].to_numpy(), "s": sev_c.to_numpy()}).groupby("row_id")["s"].agg(["mean", "size"])
        mask = train["row_id"].isin(agg.index).to_numpy()
        if len(cl) > 1500:
            a = agg.loc[train.loc[mask, "row_id"]]
            scols = [c for c in SEV_DRIVERS[line] if c in base_cols]
            Str, Sva = Xtr0[scols], Xva0[scols]
            if has_mm:
                # encoding di make_model sulla severità (solo righe con sinistri), credibilità k=20 sinistri
                sev_row = np.zeros(len(train)); cnt = np.zeros(len(train))
                sev_row[mask] = a["mean"].to_numpy() * a["size"].to_numpy(); cnt[mask] = a["size"].to_numpy()
                ts_tr, ts_va = _te_make_model(train, valid, sev_row, cnt * mean_sev, k=20.0 * mean_sev, seed=1)
                Str = Str.assign(mm_sev_te=ts_tr); Sva = Sva.assign(mm_sev_te=ts_va)
            ms = lgb.train(dict(SEV_PARAMS, min_data_in_leaf=max(200, int(mask.sum() / 50))),
                           lgb.Dataset(Str[mask], label=a["mean"].to_numpy(), weight=a["size"].to_numpy(float),
                                       init_score=np.full(mask.sum(), np.log(mean_sev))), num_boost_round=150)
            s_va = np.exp(ms.predict(Sva, raw_score=True) + np.log(mean_sev))
            s_tr = np.exp(ms.predict(Str, raw_score=True) + np.log(mean_sev))
        elif small_sev and p in SMALL_SEV_SI.get(line, []) and len(cl) > 150:
            # severità ~ somma assicurata (legge di potenza), GLM Gamma con log-link su log(SI) standardizzato
            from sklearn.linear_model import GammaRegressor
            a = agg.loc[train.loc[mask, "row_id"]]
            lsi = np.log(train["sum_insured"].clip(lower=1).to_numpy())
            mu, sd = lsi[mask].mean(), lsi[mask].std() + 1e-9
            g = GammaRegressor(alpha=1.0, max_iter=500)
            g.fit(((lsi[mask] - mu) / sd).reshape(-1, 1), a["mean"].to_numpy(), sample_weight=a["size"].to_numpy(float))
            lsv = np.log(valid["sum_insured"].clip(lower=1).to_numpy())
            s_tr = g.predict(((lsi - mu) / sd).reshape(-1, 1)); s_va = g.predict(((lsv - mu) / sd).reshape(-1, 1))
            log[p + "_sev_slope"] = round(float(g.coef_[0] / sd), 3)
        total_va += f_va * s_va; total_tr += f_tr * s_tr
        parts[p] = (np.asarray(f_va, float), np.broadcast_to(np.asarray(s_va, float), np.shape(f_va)).copy())
    _fs2.last_log = log
    ll = float(lvl.loc[line, "ll_load_2025"])
    w_tr, w_va = _ll_weights(train, valid, line) if ll_si else (np.ones(len(train)), np.ones(len(valid)))
    c = train[P.TARGET].sum() / (total_tr + ll * e_tr * w_tr).sum()
    _fs2.last_parts = {"perils": parts, "ll": ll * e_va * w_va, "c": c}   # numero atteso e severità per perilo (prima di c)
    return (total_va + ll * e_va * w_va) * c


def fs2_mm(train, valid, line):
    return _fs2(train, valid, line, mm_te=True, small_sev=False, subs_inter=False)


def fs2_all(train, valid, line):
    return _fs2(train, valid, line, mm_te=True, small_sev=True, subs_inter=True)


def fs2_all_es(train, valid, line):
    return _fs2(train, valid, line, mm_te=True, small_sev=True, subs_inter=True, es_rounds=True)


def fs2_all_ll(train, valid, line):
    return _fs2(train, valid, line, mm_te=True, small_sev=True, subs_inter=True, ll_si=True)


def qwen_gbm(train, valid, line):
    """Miglior experiment.py trovato dal loop Qwen (copiato in qwen/experiment_best.py), con le stesse colonne
    che gli passa il giudice del loop."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen_best", str(ROOT / "qwen" / "experiment_best.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    feats = [c for c in P.allowed_features(train) if not c.startswith("addr_")]   # come sul Mac (senza storia indirizzo)
    tr = train[feats + ["exposure", P.TARGET]].reset_index(drop=True)
    va = valid[feats + ["exposure"]].reset_index(drop=True)
    return np.asarray(mod.fit_predict(tr, va, line), float)


def gbm_te(train, valid, line):
    """GBM senza esche con make_model sostituito dal suo indice di rischio lisciato (sul costo, credibilità)."""
    import train as T
    if "make_model" not in train.columns:
        return gbm_clean(train, valid, line)
    y = train[P.TARGET].to_numpy(); e = train["exposure"].to_numpy()
    rate = y.sum() / e.sum()
    te_tr, te_va = _te_make_model(train, valid, y, e * rate, k=40.0 * rate * 3, seed=5)
    tr = train.assign(mm_cost_te=te_tr); va = valid.assign(mm_cost_te=te_va)
    orig = T.features_for
    T.features_for = lambda df, l: [c for c in orig(df, l) if c not in DECOYS and c not in ("make_model", "mm_cost_te")] + ["mm_cost_te"]
    try:
        return T.gbm_b(tr, va, line)
    finally:
        T.features_for = orig


def _freq_glm(train, valid, line, peril, n, expo_adj, e_tr, e_va, Xtr, Xva):
    """frequenza per perilo con GLM Poisson classico: variabili del generatore binnate (8 classi) + indice make_model,
    tasso = n / (esposizione × completezza × trend), pesi = denominatore. Previsione = frequenza 2025 × esposizione."""
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import OneHotEncoder
    cols = [c for c in GLM_DGP[line] if c in train.columns and c != "make_model"]
    Btr, Bva = {}, {}
    for c in cols:
        a, (b,) = _bin(train[c], [valid[c]], q=8)
        Btr[c], Bva[c] = a, b
    if "mm_freq_te" in Xtr.columns:
        a, (b,) = _bin(Xtr["mm_freq_te"], [Xva["mm_freq_te"]], q=10)
        Btr["mm"], Bva["mm"] = a, b
    if peril == "subsidence" and "soil_type" in train.columns:
        ka, (va_,) = _bin(train["soil_type"], [valid["soil_type"]]); kb, (vb_,) = _bin(train["dry_summer_index"], [valid["dry_summer_index"]], q=4)
        Btr["soil*dry"] = ka + "|" + kb; Bva["soil*dry"] = va_ + "|" + vb_
    Btr, Bva = pd.DataFrame(Btr), pd.DataFrame(Bva)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=300, drop="first", sparse_output=True)
    A = enc.fit_transform(Btr); B = enc.transform(Bva)
    g = PoissonRegressor(alpha=1e-4 if n.sum() > 3000 else 1e-3, max_iter=500, solver="newton-cholesky")
    g.fit(A, n / expo_adj, sample_weight=expo_adj)
    return g.predict(A) * e_tr, g.predict(B) * e_va


def fs2_bag(train, valid, line):
    """bagging del freq×sev: 3 semi, campionamento di righe e colonne diverso."""
    ps = []
    for sd in (101, 202, 303):
        ps.append(_fs2(train, valid, line, ll_si=True, fparams=dict(seed=sd, bagging_fraction=0.7, feature_fraction=0.8)))
    return np.mean(ps, axis=0)


def glm_peril(train, valid, line):
    """freq GLM Poisson per perilo × severità come fs2 (stessa struttura, stima diversa)."""
    return _fs2(train, valid, line, ll_si=True, freq_glm=True)


def gbm_peril_deep(train, valid, line):
    """freq per perilo con alberi profondi (interazioni libere) × severità come fs2."""
    return _fs2(train, valid, line, ll_si=True, mult=1,
                fparams=dict(num_leaves=15, max_depth=-1, learning_rate=0.05, min_data_in_leaf=3000, lambda_l2=50.0))


def fs2_nohist(train, valid, line):
    """freq×sev senza variabili di storia sinistri (polizza, cliente, indirizzo): base per la credibilità."""
    return _fs2(train, valid, line, ll_si=True, drop_hist=True)


# --- experience rating con credibilità sopra i modelli (src/experience.py); k scelto su entrambi i fold ---
def _cred(fn, k):
    def f(train, valid, line):
        import experience as X
        return X.apply(fn(train, valid, line), valid, line, k)
    f.__name__ = fn.__name__ + "_cred"
    return f


fs2_all_ll_cred = _cred(fs2_all_ll, 8)
fs2_nohist_cred = _cred(fs2_nohist, 4)
qwen_gbm_cred = _cred(qwen_gbm, 8)


def fs2_zk30(train, valid, line):
    """fs2_all_ll con subsampling senza peso (30%) delle righe senza sinistri nella frequenza."""
    return _fs2(train, valid, line, ll_si=True, zero_keep=0.3)


def fs2_nocust(train, valid, line):
    """fs2_all_ll senza la storia sinistri del cliente sulle altre linee (A/E 0,72 dove è alta: sovrappesata)."""
    return _fs2(train, valid, line, ll_si=True, drop_cust=True)


def qwen_gbm73(train, valid, line):
    """Miglior experiment.py del loop Qwen all'iterazione 73 (5 semi, monotonia, esche tolte): qwen/experiment_best_v73.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen_best73", str(ROOT / "qwen" / "experiment_best_v73.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.THREADS = 2
    feats = [c for c in P.allowed_features(train) if not c.startswith("addr_")]
    tr = train[feats + ["exposure", P.TARGET]].reset_index(drop=True)
    va = valid[feats + ["exposure"]].reset_index(drop=True)
    return np.asarray(mod.fit_predict(tr, va, line), float)


def glm_enet(train, valid, line):
    """GLM Tweedie elastic net (glmnet) — src/models_nn.py."""
    import models_nn as N
    return N.glm_enet(train, valid, line)


def cann(train, valid, line):
    """CANN: freq×sev come base + rete neurale sulla correzione — src/models_nn.py."""
    import models_nn as N
    return N.cann(train, valid, line)


def _gbm_clean_w(train, valid, line, kind):
    """gbm_clean con pesi di stima: 'pi' = completezza media della riga (meno peso agli anni con molto IBNR),
    'recent' = anni recenti pesano di più (×0,8 per ogni anno indietro)."""
    import lightgbm as lgb
    import train as T
    cols = [c for c in T.features_for(train, line) if c not in DECOYS]
    Xtr, cats = T.to_frame(train, cols); Xva, _ = T.to_frame(valid, cols, cats)
    if kind == "pi":
        per = [c[3:] for c in train.columns if c.startswith("pi_")]
        nn = train[[f"n_{p}" for p in per]].sum().to_numpy(); w = nn / nn.sum()
        wt = np.clip(train[[f"pi_{p}" for p in per]].to_numpy() @ w, 0.05, 1)
    else:
        wt = 0.8 ** (train["year"].max() - train["year"]).to_numpy()
    y, e = train[P.TARGET].to_numpy(), train["exposure"].to_numpy()
    base = np.log((wt * y).sum() / (wt * e).sum())
    m = lgb.train(T.GBM_PARAMS, lgb.Dataset(Xtr, label=y, weight=wt, init_score=np.log(e) + base), num_boost_round=T.NUM_ROUNDS[line])
    pred = np.exp(m.predict(Xva, raw_score=True) + np.log(valid["exposure"].to_numpy()) + base)
    pin = np.exp(m.predict(Xtr, raw_score=True) + np.log(e) + base)
    return pred * (y.sum() / pin.sum())


def gbm_clean_pi(train, valid, line):
    return _gbm_clean_w(train, valid, line, "pi")


def gbm_clean_recent(train, valid, line):
    return _gbm_clean_w(train, valid, line, "recent")


qwen_gbm73_cred = _cred(qwen_gbm73, 8)


# --- Parte 2: modelli di trasferimento (src/transfer.py); richiedono results/<mercato>/source_pred_<linea>.parquet
def tr_scratch(train, valid, line):
    import transfer as TR
    return TR.scratch(train, valid, line)


def tr_src_level(train, valid, line):
    import transfer as TR
    return TR.src_level(train, valid, line)


def tr_src_glm(train, valid, line):
    import transfer as TR
    return TR.src_glm(train, valid, line)


def tr_src_gbm(train, valid, line):
    import transfer as TR
    return TR.src_gbm(train, valid, line)


tr_scratch_cred = _cred(tr_scratch, 8)
tr_src_glm_cred = _cred(tr_src_glm, 8)
tr_src_gbm_cred = _cred(tr_src_gbm, 8)
