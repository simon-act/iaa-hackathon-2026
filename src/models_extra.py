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
        inter = Path(P.CFG["interim"])   # 18/10: prima era fisso a data/interim (sbagliato per un mercato con config propria)
        _DEV = pd.read_parquet(inter / "development_onlevel.parquet")
        _CLAIMS = pd.read_parquet(inter / "claims_clean.parquet",
                                  columns=["row_id", "line", "peril", "ay", "net_capped"])
        _LVL = pd.read_parquet(inter / "level_2025.parquet")
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
_DROP_CUST_LINES = {}   # 20/10: linee in cui togliere cust_other_prev_* (variante fs2_v2_nocust_cred)
_YEAR_DECAY = None      # 21/10: peso degli anni di stima = decay^(ultimo anno − anno) in frequenza e severità (None = pesi uguali)
_SEV_PERIL = {}         # 20/10: {linea: {perilo: {"_all": True} (severità su tutte le variabili del modello) o {"_mean": True}}}
_FREQ_2PH = {}          # 22/10 sera: {linea: perili} con la franchigia applicata dopo (due fasi)
_FREQ_XS = {}           # 22/10 sera: {linea: perili} con la franchigia relativa (xs_rel) nella frequenza
_FREQ_OVERRIDE = {}     # 20/10: {linea: {perilo o "*": parametri LightGBM della frequenza, "_mult" = moltiplicatore dei giri}}


def _fs2(train, valid, line, mm_te=True, small_sev=True, subs_inter=True, es_rounds=False, mult=4, ll_si=False,
         fparams=None, freq_glm=False, drop_hist=False, zero_keep=None, drop_cust=False):
    dev, claims, lvl = _tables()
    dl = dev[dev["line"] == line]
    perils = sorted(dl["peril"].unique())
    cols = _features(train, line)
    if drop_hist:
        cols = [c for c in cols if not c.startswith(HIST)]
    if drop_cust or _DROP_CUST_LINES.get(line):   # 20/10: effetto tra linee sovrastimato (A/E 0,88–0,94 auto e fabbricato)
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
        es_tr, es_va = (_season_expo(train, _season_weights(train, line, p)), _season_expo(valid, _season_weights(train, line, p))) \
            if _SEASON_ON else (e_tr, e_va)
        off = np.log(es_tr * pi) - (2025 - yrs) * np.log1p(ft)
        base = np.log(max(n.sum(), 1) / np.exp(off).sum())
        Xtr, Xva = Xtr0, Xva0
        if has_mm:
            te_tr, te_va = _te_make_model(train, valid, n, np.exp(off + base), k=30.0)
            Xtr = Xtr0.assign(mm_freq_te=te_tr); Xva = Xva0.assign(mm_freq_te=te_va)
        two_ph = p in _FREQ_2PH.get(line, ())
        if two_ph:   # 22/10 sera, "lordo poi netto": frequenza senza franchigia, fattore franchigia × classe di valore applicato dopo
            Xtr = Xtr.drop(columns=["voluntary_excess"], errors="ignore"); Xva = Xva.drop(columns=["voluntary_excess"], errors="ignore")
        if p in _FREQ_XS.get(line, ()):   # 22/10 sera: franchigia relativa al valore del bene nella frequenza di questo perilo
            Xtr = Xtr.assign(xs_rel=_add_xs(train, line)["xs_rel"].to_numpy())
            Xva = Xva.assign(xs_rel=_add_xs(valid, line)["xs_rel"].to_numpy())
        params = dict(FREQ_PARAMS, **(fparams or {}))
        ov = dict(_FREQ_OVERRIDE.get(line, {}).get("*", {}), **_FREQ_OVERRIDE.get(line, {}).get(p, {}))
        mult_p = ov.pop("_mult", mult)
        params.update(ov)
        if subs_inter and p == "subsidence" and params.get("num_leaves", 2) <= 3:
            params.update(num_leaves=3, max_depth=2)
        default = (300 if n.sum() > 5000 else 150 if n.sum() > 1000 else 80) * mult_p
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
                wy = (_YEAR_DECAY ** (yrs.max() - yrs)).astype(float) if _YEAR_DECAY else None
                m = lgb.train(params, lgb.Dataset(Xtr, label=n, init_score=off + base, weight=wy), num_boost_round=rounds)
            raw_tr = m.predict(Xtr, raw_score=True)
            f_va = np.exp(m.predict(Xva, raw_score=True) + base) * es_va
            f_tr = np.exp(raw_tr + base) * es_tr
            if two_ph:
                fac_tr, fac_va = _xs_factor(train, valid, line, n, np.exp(raw_tr + base + off))
                f_tr, f_va = f_tr * fac_tr, f_va * fac_va
            if _PERIL_CRED_K:
                f_va = _peril_cred(f_va, es_va, valid, line, p, ft, _PERIL_CRED_K)
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
        sp = _SEV_PERIL.get(line, {}).get(p, {})
        if len(cl) > 1500 and not sp.get("_mean"):
            a = agg.loc[train.loc[mask, "row_id"]]
            scols = list(base_cols) if sp.get("_all") else [c for c in SEV_DRIVERS[line] if c in base_cols]
            Str, Sva = Xtr0[scols], Xva0[scols]
            if has_mm and not sp.get("_all"):
                # encoding di make_model sulla severità (solo righe con sinistri), credibilità k=20 sinistri
                sev_row = np.zeros(len(train)); cnt = np.zeros(len(train))
                sev_row[mask] = a["mean"].to_numpy() * a["size"].to_numpy(); cnt[mask] = a["size"].to_numpy()
                ts_tr, ts_va = _te_make_model(train, valid, sev_row, cnt * mean_sev, k=20.0 * mean_sev, seed=1)
                Str = Str.assign(mm_sev_te=ts_tr); Sva = Sva.assign(mm_sev_te=ts_va)
            ms = lgb.train(dict(SEV_PARAMS, min_data_in_leaf=max(200, int(mask.sum() / 50)), **(_SEV_OVERRIDE or {})),
                           lgb.Dataset(Str[mask], label=a["mean"].to_numpy(),
                                       weight=a["size"].to_numpy(float) * ((_YEAR_DECAY ** (yrs.max() - yrs[mask])) if _YEAR_DECAY else 1.0),
                                       init_score=np.full(mask.sum(), np.log(mean_sev))), num_boost_round=_SEV_ROUNDS)
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
        if sp.get("_mix_glm") and len(cl) > 150:   # 21/10: media geometrica con un GLM Gamma sui driver (glm_sev_bench.py)
            a = agg.loc[train.loc[mask, "row_id"]]
            g_tr, g_va = _glm_sev(train, valid, mask, a, [c for c in SEV_DRIVERS[line] if c in train.columns])
            s_tr, s_va = np.sqrt(s_tr * g_tr), np.sqrt(s_va * g_va)
        total_va += f_va * s_va; total_tr += f_tr * s_tr
        parts[p] = (np.asarray(f_va, float), np.broadcast_to(np.asarray(s_va, float), np.shape(f_va)).copy())
        _fs2.parts_tr = dict(getattr(_fs2, "parts_tr", {}) if p != perils[0] else {}, **{p: np.asarray(f_tr, float)})
    _fs2.last_log = log
    ll = float(lvl.loc[line, "ll_load_2025"])
    w_tr, w_va = _ll_weights(train, valid, line) if ll_si else (np.ones(len(train)), np.ones(len(valid)))
    if _LL_RISK:   # 20/10: caricamento large loss ripartito sul rischio previsto (Poisson dei grandi sinistri)
        if _LL_RISK == "bi" and line == "motor":   # 22/10: grandi sinistri auto ripartiti sulla frequenza lesioni prevista
            w_tr, w_va = _ll_risk_weights(train, valid, line, _fs2.parts_tr["bodily_injury"] * 1.0, parts["bodily_injury"][0] * 1.0, e_tr, e_va)
        else:
            w_tr, w_va = _ll_risk_weights(train, valid, line, total_tr, total_va, e_tr, e_va)
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


def _glm_sev(train, valid, mask, a, cols):
    """GLM Gamma (glum) della severità media per riga sui driver: numeriche con > 10 valori in 10 classi per quantili,
    categorie con < 50 righe raggruppate; peso = numero di sinistri; ridge scelta su un 20% delle righe con sinistri.
    Come scripts/glm_sev_bench.py (che lavora per sinistro). Restituisce la severità su train e valid."""
    from glum import GeneralizedLinearRegressor
    A, B = {}, {}
    trm = train.loc[mask]
    for c in cols:
        s_all, s_m, t = train[c], trm[c], valid[c]
        if pd.api.types.is_numeric_dtype(s_m) and s_m.dtype != bool and s_m.nunique() > 10:
            e = np.unique(np.nanquantile(s_m.astype(float), np.linspace(0, 1, 11))); e[0], e[-1] = -np.inf, np.inf
            f = lambda x: pd.cut(x.astype(float), e, labels=False).astype(float).fillna(-1).astype(int).astype(str)
            a_all, b = f(s_all), f(t)
        else:
            f = lambda x: x.astype(object).where(x.notna(), "NA").astype(str)
            a_all, b = f(s_all), f(t)
            vc = a_all[mask].value_counts(); keep = set(vc.index[vc >= 50])
            a_all = a_all.where(a_all.isin(keep), "altro"); b = b.where(b.isin(keep), "altro")
        lv = sorted(set(a_all[mask]))
        A[c] = pd.Categorical(a_all.where(a_all.isin(lv), lv[0]), categories=lv)
        B[c] = pd.Categorical(b.where(b.isin(lv), lv[0]), categories=lv)
    A, B = pd.DataFrame(A), pd.DataFrame(B)
    y = a["mean"].to_numpy(); w = a["size"].to_numpy(float); Am = A[mask].reset_index(drop=True)
    inner = pd.util.hash_array(trm["row_id"].to_numpy()) % 5 == 2
    best = (np.inf, None)
    for al in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        g = GeneralizedLinearRegressor(family="gamma", alpha=al, l1_ratio=0.0, drop_first=False, max_iter=200)
        g.fit(Am[~inner], y[~inner], sample_weight=w[~inner])
        mu = g.predict(Am[inner]); d = float(np.sum(w[inner] * 2 * (-np.log(y[inner] / mu) + (y[inner] - mu) / mu)))
        if d < best[0]:
            best = (d, al)
    g = GeneralizedLinearRegressor(family="gamma", alpha=best[1], l1_ratio=0.0, drop_first=False, max_iter=200).fit(Am, y, sample_weight=w)
    return g.predict(A), g.predict(B)


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


# --- chilometraggio zero: dato vero (basso) invece che mancante (18/10, passo 2 pulizia) ---
def _mil0(df):
    if "flag_mileage_zero" in df.columns and "annual_mileage" in df.columns:
        df = df.copy()
        df.loc[df["flag_mileage_zero"].astype(bool), "annual_mileage"] = 0.0
    return df


def _reuse_oof(name):
    import pandas as _pd
    o = _pd.read_parquet(P.RESULTS / "oof" / f"{name}.parquet")
    return o.set_index("row_id")["pred"]


def fs2_mil0_cred(train, valid, line):
    """fs2_all_ll_cred con annual_mileage = 0 lasciato come valore (era → mancante). Le linee casa riusano l'OOF."""
    if line != "motor":
        return _reuse_oof("fs2_all_ll_cred").reindex(valid["row_id"]).to_numpy()
    return fs2_all_ll_cred(_mil0(train), _mil0(valid), line)


# --- premio dell'assicuratore come variabile (18/10, richiesta di Simone) ---
def _prem(df):
    """log del premio annuo lordo di sconto, centrato sulla mediana dell'anno: toglie l'adeguamento tariffario annuo."""
    df = df.copy()
    g = np.log(df["written_premium"].clip(lower=1) / (1 - df["discount_applied"].fillna(0)))
    df["prem_rel"] = g - g.groupby(df["year"]).transform("median")
    return df


def fs2_prem_cred(train, valid, line):
    """fs2_all_ll_cred + prem_rel in frequenza e severità."""
    SEV_DRIVERS[line] = SEV_DRIVERS[line] + ["prem_rel"]
    try:
        return fs2_all_ll_cred(_prem(train), _prem(valid), line)
    finally:
        SEV_DRIVERS[line] = [c for c in SEV_DRIVERS[line] if c != "prem_rel"]


def fs2_v2_cred(train, valid, line):
    """Pulizia rivista (18/10): motor con chilometraggio 0 come valore; casa con il premio relativo come variabile."""
    return fs2_mil0_cred(train, valid, line) if line == "motor" else fs2_prem_cred(train, valid, line)


glm_enet_cred = _cred(glm_enet, 8)   # stessa costruzione delle OOF glm_enet_cred (verificato: scarto 0)


# --- ablazioni della pulizia (18/10, richiesta di Simone: "rivaluta tutti gli interventi") ---
# Ogni funzione toglie UNA correzione dalle variabili (train e valid) e rifà fs2_all_ll_cred; le linee non toccate
# riusano le OOF. Il target resta quello pulito: misura quanto il modello perde se la correzione non ci fosse.
_CUST_COLS = None


def _undo_si(df):
    df = df.copy()
    if "flag_si_div1000" in df:
        f = df["flag_si_div1000"].astype(bool)
        for c in ("sum_insured", "si_per_bedroom"):
            if c in df:
                df.loc[f, c] = df.loc[f, c] * 1000.0
    return df


def abl_si_cred(train, valid, line):
    if line == "motor":
        return _reuse_oof("fs2_all_ll_cred").reindex(valid["row_id"]).to_numpy()
    return fs2_all_ll_cred(_undo_si(train), _undo_si(valid), line)


def _broken_cust_rows():
    raw = Path(P.CFG["raw"])
    ids = []
    for f in ("policies_motor", "test_motor"):
        r = pd.read_parquet(raw / f"{f}.parquet", columns=["row_id", "customer_id"])
        ids.append(r.loc[r["customer_id"].astype(str).str.len() < 8, "row_id"])
    return set(pd.concat(ids))


def _undo_cust(df):
    global _CUST_COLS
    if _CUST_COLS is None:
        cc = [c for c in pd.read_parquet(Path(P.CFG["raw"]) / "customers.parquet").columns if c not in ("customer_id", "surname")]
        _CUST_COLS = (cc + ["birth_year", "customer_age", "customer_tenure", "cust_n_lines"], _broken_cust_rows())
    cols, bad = _CUST_COLS
    df = df.copy(); m = df["row_id"].isin(bad).to_numpy()
    for c in [c for c in df.columns if c in cols or c.startswith(("cust_other_prev", "cust_has_"))]:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("string"); df.loc[m, c] = "missing"
        else:
            df[c] = df[c].astype(float); df.loc[m, c] = np.nan
    return df


def abl_cust_cred(train, valid, line):
    if line != "motor":
        return _reuse_oof("fs2_all_ll_cred").reindex(valid["row_id"]).to_numpy()
    return fs2_all_ll_cred(_undo_cust(train), _undo_cust(valid), line)


def abl_expo_cred(train, valid, line):
    """esposizione dichiarata (sbagliata) nel training; la validazione resta pulita come il test."""
    tr = train.copy(); tr["exposure"] = tr["exposure_raw"].to_numpy()
    return fs2_all_ll_cred(tr, valid, line)


def fs2_v2_m6_cred(train, valid, line):
    """Punto 5 (18/10): come fs2_v2_cred ma con più iterazioni nella frequenza auto (mult 6 invece di 4)."""
    if line != "motor":
        return _reuse_oof("fs2_v2_cred").reindex(valid["row_id"]).to_numpy()
    import experience as X
    tr, va = _mil0(train), _mil0(valid)
    return X.apply(_fs2(tr, va, line, mm_te=True, small_sev=True, subs_inter=True, ll_si=True, mult=6), va, line, 8)



# --- stagionalità dei sinistri (19/10): tempesta e acqua condotta hanno dic–feb a 1,4–1,6 volte la media ---
# L'esposizione di una riga parziale non vale "giorni/365" per tutti i perili: per i perili stagionali vale la quota
# dei sinistri annui che cade nei mesi coperti. Pesi mensili stimati sui sinistri delle righe annuali del training,
# con credibilità verso l'uniforme (k = 300 sinistri).
_SEASON_ON = False
_PERIL_CRED_K = None   # 19/10: credibilità per perilo sulla storia della polizza (sinistri dello stesso perilo)
_PH_CACHE = {}


def _peril_hist(line):
    """Per ogni riga (train e test): N_p = sinistri passati della polizza nel perilo p (AY < anno della riga),
    H_p = Σ sulle righe precedenti di esposizione × completezza_p × (1+trend_p)^(anno − 2025) (= attesi per unità di
    frequenza 2025). Usa claims_clean (perilo noto per sinistro)."""
    if line in _PH_CACHE:
        return _PH_CACHE[line]
    dev, claims, _ = _tables()
    tr = P.load(line); te = P.load(line, "test")
    perils = sorted(dev[dev["line"] == line]["peril"].unique())
    ft = dev[dev["line"] == line].groupby("peril")["freq_trend"].first()
    cl = pd.read_parquet(Path(P.CFG["interim"]) / "claims_clean.parquet", columns=["line", "policy_id", "ay", "peril"])
    cl = cl[cl["line"] == line].groupby(["policy_id", "ay", "peril"]).size().rename("n").reset_index()
    rows = pd.concat([tr[["row_id", "policy_id", "year", "exposure"] + [f"pi_{p}" for p in perils]],
                     te[["row_id", "policy_id", "year", "exposure"]].assign(**{f"pi_{p}": np.nan for p in perils})])
    rows = rows.sort_values(["policy_id", "year"]).reset_index(drop=True)
    out = pd.DataFrame({"row_id": rows["row_id"]})
    for p in perils:
        eh = rows["exposure"] * rows[f"pi_{p}"] * (1 + float(ft[p])) ** (rows["year"] - 2025)
        out[f"H_{p}"] = eh.groupby(rows["policy_id"]).transform(lambda s: s.shift().fillna(0).cumsum()).to_numpy()
        n = cl[cl["peril"] == p].pivot_table(index="policy_id", columns="ay", values="n", aggfunc="sum").fillna(0)
        n = n.reindex(columns=sorted(n.columns)).cumsum(axis=1)          # cumulato fino all'AY incluso
        prev = rows["year"] - 1
        # N fino all'anno precedente: cumulato all'ultimo AY <= anno − 1
        out[f"N_{p}"] = 0.0
        for ay in sorted(n.columns):
            sel = (prev >= ay).to_numpy()
            v = rows.loc[sel, "policy_id"].map(n[ay]).fillna(0).to_numpy()
            out.loc[sel, f"N_{p}"] = v
    _PH_CACHE[line] = out.set_index("row_id")
    return _PH_CACHE[line]


def _peril_cred(f_va, e_va, valid, line, p, ft, k):
    h = _peril_hist(line).reindex(valid["row_id"].to_numpy())
    N = h[f"N_{p}"].to_numpy(float); H = h[f"H_{p}"].to_numpy(float)
    lam = f_va / np.maximum(np.asarray(e_va, float), 1e-9)      # frequenza 2025 della riga
    M = (N + k) / (H * lam + k)
    out = f_va * M
    return out * f_va.sum() / out.sum()
_SEV_OVERRIDE = None   # 19/10: prova di severità meno regolarizzata (A/E cresce con vehicle_value_group: 0,85 → 1,2)
_SEV_ROUNDS = 150
_LOSSM = None
_SW_CACHE = {}


def _season_weights(train, line, p, k=300.0):
    global _LOSSM
    key = (line, p, len(train), int(train["year"].max()))
    if key in _SW_CACHE:
        return _SW_CACHE[key]
    if _LOSSM is None:
        c = pd.read_parquet(Path(P.CFG["interim"]) / "claims_clean.parquet", columns=["row_id", "line", "peril", "loss_date"])
        c["m"] = pd.to_datetime(c["loss_date"]).dt.month
        _LOSSM = c[["row_id", "line", "peril", "m"]]
    full = train.loc[train["exposure"] > 0.99, "row_id"]
    c = _LOSSM[(_LOSSM["line"] == line) & (_LOSSM["peril"] == p) & _LOSSM["row_id"].isin(full)]
    cnt = c["m"].value_counts().reindex(range(1, 13), fill_value=0).to_numpy(float)
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)
    rate = cnt / days                                   # sinistri per giorno nel mese
    w = rate / (cnt.sum() / 365.0) if cnt.sum() else np.ones(12)
    z = cnt.sum() / (cnt.sum() + k)
    w = 1 + z * (w - 1)
    _SW_CACHE[key] = w
    return w


def _season_expo(df, w):
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    daily = np.repeat(w, days) / 365.0
    daily = daily / daily.sum()
    C = np.concatenate([[0.0], np.cumsum(daily)])
    ps = pd.to_datetime(df["period_start"]); pe = pd.to_datetime(df["period_end"])
    a = np.minimum(ps.dt.dayofyear.to_numpy(), 365) - 1
    b = np.minimum(pe.dt.dayofyear.to_numpy(), 365)
    e = C[b] - C[a]
    return np.clip(e, 1e-4, None) * (df["exposure"].to_numpy() / np.clip((b - a) / 365.0, 1e-4, None))


def fs2_season_cred(train, valid, line):
    """fs2_v2_cred con esposizione stagionale per perilo nella frequenza."""
    global _SEASON_ON
    _SEASON_ON = True
    try:
        return fs2_v2_cred(train, valid, line)
    finally:
        _SEASON_ON = False
_PERIL_CRED_K = None   # 19/10: credibilità per perilo sulla storia della polizza (sinistri dello stesso perilo)
_PH_CACHE = {}


def _peril_hist(line):
    """Per ogni riga (train e test): N_p = sinistri passati della polizza nel perilo p (AY < anno della riga),
    H_p = Σ sulle righe precedenti di esposizione × completezza_p × (1+trend_p)^(anno − 2025) (= attesi per unità di
    frequenza 2025). Usa claims_clean (perilo noto per sinistro)."""
    if line in _PH_CACHE:
        return _PH_CACHE[line]
    dev, claims, _ = _tables()
    tr = P.load(line); te = P.load(line, "test")
    perils = sorted(dev[dev["line"] == line]["peril"].unique())
    ft = dev[dev["line"] == line].groupby("peril")["freq_trend"].first()
    cl = pd.read_parquet(Path(P.CFG["interim"]) / "claims_clean.parquet", columns=["line", "policy_id", "ay", "peril"])
    cl = cl[cl["line"] == line].groupby(["policy_id", "ay", "peril"]).size().rename("n").reset_index()
    rows = pd.concat([tr[["row_id", "policy_id", "year", "exposure"] + [f"pi_{p}" for p in perils]],
                     te[["row_id", "policy_id", "year", "exposure"]].assign(**{f"pi_{p}": np.nan for p in perils})])
    rows = rows.sort_values(["policy_id", "year"]).reset_index(drop=True)
    out = pd.DataFrame({"row_id": rows["row_id"]})
    for p in perils:
        eh = rows["exposure"] * rows[f"pi_{p}"] * (1 + float(ft[p])) ** (rows["year"] - 2025)
        out[f"H_{p}"] = eh.groupby(rows["policy_id"]).transform(lambda s: s.shift().fillna(0).cumsum()).to_numpy()
        n = cl[cl["peril"] == p].pivot_table(index="policy_id", columns="ay", values="n", aggfunc="sum").fillna(0)
        n = n.reindex(columns=sorted(n.columns)).cumsum(axis=1)          # cumulato fino all'AY incluso
        prev = rows["year"] - 1
        # N fino all'anno precedente: cumulato all'ultimo AY <= anno − 1
        out[f"N_{p}"] = 0.0
        for ay in sorted(n.columns):
            sel = (prev >= ay).to_numpy()
            v = rows.loc[sel, "policy_id"].map(n[ay]).fillna(0).to_numpy()
            out.loc[sel, f"N_{p}"] = v
    _PH_CACHE[line] = out.set_index("row_id")
    return _PH_CACHE[line]


def _peril_cred(f_va, e_va, valid, line, p, ft, k):
    h = _peril_hist(line).reindex(valid["row_id"].to_numpy())
    N = h[f"N_{p}"].to_numpy(float); H = h[f"H_{p}"].to_numpy(float)
    lam = f_va / np.maximum(np.asarray(e_va, float), 1e-9)      # frequenza 2025 della riga
    M = (N + k) / (H * lam + k)
    out = f_va * M
    return out * f_va.sum() / out.sum()
_SEV_OVERRIDE = None   # 19/10: prova di severità meno regolarizzata (A/E cresce con vehicle_value_group: 0,85 → 1,2)
_SEV_ROUNDS = 150


# --- ipotesi riviste (19/10): variabili di calendario e non proporzionalità all'esposizione ---
# Residui OOF del blend: fabbricato ends_early A/E 0,78 / 0,74 (fold 2023 / 2024, ~540 sinistri per fold);
# auto esposizione < 0,35 A/E 1,14 / 1,07. Nel test il 17-20% delle righe finisce prima di dicembre.
_TIMING_EXTRA = ()


def _with_timing(fn, extra):
    def f(train, valid, line):
        global DECOYS, _features
        old_dec, old_feat = DECOYS, _features
        DECOYS = DECOYS - {"ends_dec", "starts_jan1", "start_month"}
        if "exposure" in extra:
            def _features(tr, l, drop_noise=True, _o=old_feat):
                return _o(tr, l, drop_noise) + ["exposure"]
        try:
            return fn(train, valid, line)
        finally:
            DECOYS, _features = old_dec, old_feat
    return f


def fs2_timing_cred(train, valid, line):
    return _with_timing(fs2_v2_cred, ("exposure",))(train, valid, line)


def _sev_variant(over, rounds):
    def f(train, valid, line):
        global _SEV_OVERRIDE, _SEV_ROUNDS
        if line != "motor":
            return _reuse_oof("fs2_v2_cred").reindex(valid["row_id"]).to_numpy()
        _SEV_OVERRIDE, _SEV_ROUNDS = over, rounds
        try:
            return fs2_v2_cred(train, valid, line)
        finally:
            _SEV_OVERRIDE, _SEV_ROUNDS = None, 150
    return f


fs2_sev1_cred = _sev_variant({"lambda_l2": 5.0}, 300)            # meno shrinkage, più giri
fs2_sev2_cred = _sev_variant({"lambda_l2": 5.0, "num_leaves": 4, "max_depth": 2}, 300)   # + interazioni di severità


def _pc_variant(k, total_k=None):
    def f(train, valid, line):
        global _PERIL_CRED_K
        _PERIL_CRED_K = k
        try:
            base = fs2_v2 if total_k is None else _cred(fs2_v2, total_k)
            return base(train, valid, line)
        finally:
            _PERIL_CRED_K = None
    return f


def fs2_v2(train, valid, line):
    """fs2_v2_cred senza la credibilità finale sul totale."""
    if line == "motor":
        return fs2_all_ll(_mil0(train), _mil0(valid), line)
    SEV_DRIVERS[line] = SEV_DRIVERS[line] + ["prem_rel"]
    try:
        return fs2_all_ll(_prem(train), _prem(valid), line)
    finally:
        SEV_DRIVERS[line] = [c for c in SEV_DRIVERS[line] if c != "prem_rel"]


fs2_pc2 = _pc_variant(2.0)                 # credibilità per perilo k=2, senza quella sul totale
fs2_pc4 = _pc_variant(4.0)
fs2_pc4_cred = _pc_variant(4.0, 8)         # per perilo + totale (k=8)


# --- 19/10 sera (review esterna): GBM di Qwen stimato su un target senza IBNR spalmato per esposizione ---
# target_gross = costo denunciato sviluppato / completezza attesa della riga (IBNR ripartito come il rischio osservato,
# non come l'esposizione), riportato al totale per anno di target_b − LL, + LL × esposizione.
_JG = None


def _gross_target(df):
    global _JG
    if _JG is None:
        _JG = pd.read_parquet(P.RESULTS / "judge_own.parquet").set_index("row_id")
    j = _JG.reindex(df["row_id"].to_numpy())
    g = j["target_rep"].to_numpy() / np.clip(j["pi_eff"].to_numpy(), 0.3, None)
    base = df[P.TARGET].to_numpy() - j["ll_expo"].to_numpy()
    out = np.zeros(len(df))
    for y in np.unique(df["year"].to_numpy()):
        m = df["year"].to_numpy() == y
        out[m] = g[m] * base[m].sum() / g[m].sum()
    return out + j["ll_expo"].to_numpy()


def qwen_gbm73_gross(train, valid, line):
    tr = train.copy()
    tr[P.TARGET] = _gross_target(train)
    return qwen_gbm73(tr, valid, line)


qwen_gbm73_gross_cred = _cred(qwen_gbm73_gross, 8)


# --- 20/10 notte: caricamento large loss proporzionale al rischio (verifica: elasticità dei grandi sinistri al rischio
# previsto 1,22 (se 0,30) auto, 1,29 + 3,3·log SI fabbricato, 2,3 + 3,2·log SI contenuto: il riparto piatto/per quartili
# è troppo piatto). Poisson sul numero di grandi sinistri per riga (stimata sulle righe di stima):
#   log E[n_grandi] = log(esposizione) + a + b·log(costo attritivo previsto / esposizione) [+ c·log SI per la casa],
# coefficienti con credibilità n/(n+K) verso 0 (= riparto piatto); totale del caricamento invariato.
_LL_RISK = False
_LL_K = 20.0
_LARGE = None


def _large_counts():
    global _LARGE
    if _LARGE is None:
        c = pd.read_parquet(Path(P.CFG["interim"]) / "claims_clean.parquet", columns=["row_id", "is_large"])
        _LARGE = c[c["is_large"]].groupby("row_id").size()
    return _LARGE


def _poisson_irls(X, y, off, iters=30):
    b = np.zeros(X.shape[1]); b[0] = np.log(max(y.sum(), 0.5) / np.exp(off).sum())
    for _ in range(iters):
        eta = X @ b + off; mu = np.exp(eta)
        g = X.T @ (y - mu); H = (X * mu[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def _ll_risk_weights(train, valid, line, tot_tr, tot_va, e_tr, e_va):
    n = train["row_id"].map(_large_counts()).fillna(0).to_numpy(float)
    x_tr = np.log(np.clip(tot_tr / e_tr, 1e-6, None)); x_va = np.log(np.clip(tot_va / e_va, 1e-6, None))
    mx = np.average(x_tr, weights=e_tr)
    cols_tr, cols_va = [x_tr - mx], [x_va - mx]
    if line != "motor" and "sum_insured" in train:
        s_tr = np.log(train["sum_insured"].clip(lower=1).to_numpy()); s_va = np.log(valid["sum_insured"].clip(lower=1).to_numpy())
        ms = np.average(s_tr, weights=e_tr); cols_tr.append(s_tr - ms); cols_va.append(s_va - ms)
    X = np.column_stack([np.ones(len(train))] + cols_tr); Xv = np.column_stack([np.ones(len(valid))] + cols_va)
    b = _poisson_irls(X, n, np.log(e_tr))
    z = n.sum() / (n.sum() + _LL_K); b[1:] *= z
    w_tr = np.exp(X[:, 1:] @ b[1:]); w_va = np.exp(Xv[:, 1:] @ b[1:])
    norm = np.sum(w_tr * e_tr) / e_tr.sum()
    _fs2.last_log = dict(getattr(_fs2, "last_log", {}), ll_beta=[round(float(v), 3) for v in b[1:]], ll_Z=round(z, 3), ll_n=int(n.sum()))
    return w_tr / norm, w_va / norm


def fs2_v2_llr_cred(train, valid, line):
    """fs2_v2_cred con il caricamento large loss ripartito sul rischio previsto."""
    global _LL_RISK
    _LL_RISK = True
    try:
        return fs2_v2_cred(train, valid, line)
    finally:
        _LL_RISK = False


# --- 20/10: GBM di Qwen corretto da Claude (verifica indipendente): senza il vincolo monotono crescente su vehicle_age,
# che nei dati non è monotono (costo relativo 0,87 → 1,09 a 5–8 anni → 0,97 oltre 12 anni)
def qwen_gbm73fix(train, valid, line):
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen_fix73", str(ROOT / "qwen" / "experiment_v73_fix.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.THREADS = 2
    feats = [c for c in P.allowed_features(train) if not c.startswith("addr_")]
    tr = train[feats + ["exposure", P.TARGET]].reset_index(drop=True)
    va = valid[feats + ["exposure"]].reset_index(drop=True)
    return np.asarray(mod.fit_predict(tr, va, line), float)


qwen_gbm73fix_cred = _cred(qwen_gbm73fix, 8)


def qwen_gbm73fix_gross(train, valid, line):
    tr = train.copy()
    tr[P.TARGET] = _gross_target(train)
    return qwen_gbm73fix(tr, valid, line)


qwen_gbm73fix_gross_cred = _cred(qwen_gbm73fix_gross, 8)


def fs2_v2_nocust_cred(train, valid, line):
    """20/10: fs2_v2_cred senza le variabili dei sinistri del cliente su altre linee in auto e fabbricato (nei residui chi ha
    sinistri solo su altre linee ha A/E 0,94/0,88 auto e 0,88/0,91 fabbricato nei due fold: effetto sovrastimato)."""
    global _DROP_CUST_LINES
    _DROP_CUST_LINES = {"motor": True, "buildings": True}
    try:
        if line == "motor":
            return fs2_all_ll_cred(_mil0(train), _mil0(valid), line)
        return fs2_prem_cred(train, valid, line)
    finally:
        _DROP_CUST_LINES = {}


# --- 20/10 pomeriggio: frequenza del freq×sev con i parametri scelti dalla griglia sul Mac (autoresearch_freq/grid.py) ---
# Regola docs/17: per ogni linea/perilo si cambia un parametro solo se migliora la D² Poisson in entrambi i fold.
# Scelte dalla griglia (results_s3/grid_freq_results.csv, 660 righe, D² Poisson per perilo, 2 fold):
#  auto: lesioni extra_trees (+4,4/+6,0 e·4), cristalli e danni propri foglie ≥300 (+4,5/+4,3; +5,1/+3,3); scelta stabile
#        anche con la selezione incrociata tra fold. Furto e RC cose: nessuna scelta stabile → invariati.
#  fabbricato: la scelta per perilo non regge la selezione incrociata → una sola modifica su tutti i perili, ff 0,7
#        (+4,7/+2,8 pesato col costo; già vista nel banco del container 20/10). Contenuto: nulla di stabile → invariato.
FREQ_CHOSEN = {"motor": {"bodily_injury": {"extra_trees": True}, "glass_hail": {"min_data_in_leaf": 300},
                         "own_damage": {"min_data_in_leaf": 300}},
               "buildings": {"*": {"feature_fraction": 0.7}}}


def fs2_v2f_cred(train, valid, line):
    """fs2_v2_cred (con credibilità senza cedimento, come sub_29) con la frequenza riparametrizzata per perilo (FREQ_CHOSEN)."""
    global _FREQ_OVERRIDE
    _FREQ_OVERRIDE = FREQ_CHOSEN
    try:
        return fs2_v2_cred(train, valid, line)
    finally:
        _FREQ_OVERRIDE = {}


def fs2_v2f_llr_cred(train, valid, line):
    """fs2_v2f_cred con il large loss ripartito sul rischio (come il fabbricato di sub_29)."""
    global _LL_RISK
    _LL_RISK = True
    try:
        return fs2_v2f_cred(train, valid, line)
    finally:
        _LL_RISK = False


# --- 20/10 pomeriggio: severità per perilo dal banco (results_s3/sev_bench_base.csv, D² Gamma, 2 fold, selezione incrociata) ---
# auto danni propri e RC cose: severità su tutte le variabili (+24/+19 e +8/+21 e·4, stabile anche scegliendo su un solo fold);
# auto lesioni: la media batte il GBM (+35/+15) → severità costante.
SEV_CHOSEN = {"motor": {"own_damage": {"_all": True}, "third_party_prop": {"_all": True}, "bodily_injury": {"_mean": True}}}


def fs2_v2fs_cred(train, valid, line):
    """fs2_v2f_cred (frequenza da griglia) + severità per perilo da SEV_CHOSEN."""
    global _SEV_PERIL
    _SEV_PERIL = SEV_CHOSEN
    try:
        return fs2_v2f_cred(train, valid, line)
    finally:
        _SEV_PERIL = {}


def fs2_v2fs_sevonly_cred(train, valid, line):
    """Solo la severità per perilo (frequenza attuale), per separare i due effetti."""
    global _SEV_PERIL
    _SEV_PERIL = SEV_CHOSEN
    try:
        return fs2_v2_cred(train, valid, line)
    finally:
        _SEV_PERIL = {}


def fs2_v2fs_nobi_cred(train, valid, line):
    """Controllo (20/10 sera): come fs2_v2fs_cred ma lesioni con la severità attuale (solo danni propri e RC cose cambiano)."""
    global _SEV_PERIL
    _SEV_PERIL = {"motor": {k: v for k, v in SEV_CHOSEN["motor"].items() if k != "bodily_injury"}}
    try:
        return fs2_v2f_cred(train, valid, line)
    finally:
        _SEV_PERIL = {}


def fs2_v2fb_cred(train, valid, line):
    """20/10 sera: fs2_v2f_cred + severità costante SOLO per le lesioni auto. La scomposizione di fs2_v2fs_cred mostra che il
    guadagno viene tutto dalle lesioni (own +2,0/+0,6 e·4); danni propri e RC cose su tutte le variabili peggiorano (−1,3/−0,4)."""
    global _SEV_PERIL
    _SEV_PERIL = {"motor": {"bodily_injury": {"_mean": True}}}
    try:
        return fs2_v2f_cred(train, valid, line)
    finally:
        _SEV_PERIL = {}



def _decay_variant(dec):
    def f(train, valid, line):
        """21/10 (idea di Simone: gli anni recenti somigliano di più al 2025): fs2_v2fb_cred con pesi per anno decay^(ultimo − anno)."""
        global _YEAR_DECAY
        _YEAR_DECAY = dec
        try:
            return fs2_v2fb_cred(train, valid, line) if line == "motor" else fs2_v2_cred(train, valid, line)
        finally:
            _YEAR_DECAY = None
    f.__name__ = f"fs2_dec{int(dec * 100)}_cred"
    return f


fs2_dec80_cred = _decay_variant(0.8)
fs2_dec60_cred = _decay_variant(0.6)


def fs2_v2_sevmix_cred(train, valid, line):
    """21/10 sera (domanda di Simone, GLM o GBM per perilo): fs2_v2_cred con la severità dell'alluvione del contenuto =
    media geometrica GBM × GLM Gamma (unico perilo che vince in entrambi gli anni nel banco glm_sev_bench)."""
    global _SEV_PERIL
    _SEV_PERIL = {"contents": {"flood": {"_mix_glm": True}}}
    try:
        return fs2_v2_cred(train, valid, line)
    finally:
        _SEV_PERIL = {}


def fs2_v2_llr_cust_cred(train, valid, line):
    """21/10 sera (hist_scan.py): nel fabbricato chi ha sinistri del cliente su altre linee ha A/E 0,73–0,80 anche nel
    blend di sub_41 (stesso evento meteo già contato dalla storia della polizza). Il freq×sev non vede la storia del cliente
    sulle altre linee (era tra le esche): qui la vede (cust_other_prev_rate, cust_other_prev_n_last). Resto = fs2_v2_llr_cred."""
    global DECOYS
    old = DECOYS
    DECOYS = DECOYS - {"cust_other_prev_rate", "cust_other_prev_n_last"}
    try:
        return fs2_v2_llr_cred(train, valid, line)
    finally:
        DECOYS = old


def fs2_v2_llr_lin_cred(train, valid, line):
    """22/10 (banco meteo, scripts/weather_bench.py): frequenza di alluvione e incendio del fabbricato con linear_tree
    (il meteo e le variabili continue pesano anche oltre il range visto): +22,6/+16,4 e +1,4/+9,4 e·4 di D² Poisson del perilo
    nei due fold. Resto = fs2_v2_llr_cred (sub_40)."""
    global _FREQ_OVERRIDE
    old = _FREQ_OVERRIDE
    lin = {"linear_tree": True, "linear_lambda": 1.0}
    _FREQ_OVERRIDE = {"buildings": {"flood": dict(lin), "fire": dict(lin)}}
    try:
        return fs2_v2_llr_cred(train, valid, line)
    finally:
        _FREQ_OVERRIDE = old


def fs2_v2_llr_linf_cred(train, valid, line):
    """22/10: come fs2_v2_llr_lin_cred ma linear_tree solo sull'alluvione del fabbricato (alluvione+incendio: +0,73/−0,38 e·4)."""
    global _FREQ_OVERRIDE
    old = _FREQ_OVERRIDE
    _FREQ_OVERRIDE = {"buildings": {"flood": {"linear_tree": True, "linear_lambda": 1.0}}}
    try:
        return fs2_v2_llr_cred(train, valid, line)
    finally:
        _FREQ_OVERRIDE = old



def _motor_ll_variant(mode):
    def f(train, valid, line):
        global _LL_RISK
        _LL_RISK = mode
        try:
            return fs2_v2fb_cred(train, valid, line)
        finally:
            _LL_RISK = False
    return f


# 22/10 (ricerca: rischio dei grandi sinistri legato al profilo, non uniforme per esposizione): carico LL auto (28 per
# anno-esposizione, ~5% del costo) ripartito sul rischio previsto totale ("tot") o sulla frequenza lesioni ("bi")
fs2_v2fb_llr_cred = _motor_ll_variant(True)
fs2_v2fb_llrbi_cred = _motor_ll_variant("bi")


# --- 22/10 sera: franchigia relativa al valore del bene nel GBM (domanda di Simone: "lordo e poi netto")
# xs_rel = (franchigia volontaria + obbligatoria) ÷ valore (classe di valore del veicolo / somma assicurata)
_XS_COMP = {"motor": 250, "buildings": 350, "contents": 350}


def _add_xs(df, line):
    d = df.copy()
    scale = d["vehicle_value_group"].astype(float) if line == "motor" else d["sum_insured"].astype(float).clip(lower=1)
    d["xs_rel"] = (d["voluntary_excess"].astype(float) + _XS_COMP[line]) / scale
    return d


def qwen_gbm73fix_xs(train, valid, line):
    return qwen_gbm73fix(_add_xs(train, line), _add_xs(valid, line), line)


qwen_gbm73fix_xs_cred = _cred(qwen_gbm73fix_xs, 8)


def qwen_gbm73_gross_xs(train, valid, line):
    return qwen_gbm73_gross(_add_xs(train, line), _add_xs(valid, line), line)


qwen_gbm73_gross_xs_cred = _cred(qwen_gbm73_gross_xs, 8)


# perili dove xs_rel nella frequenza vince nei due anni (scripts/excess_bench.py): danni propri e furto auto,
# danni accidentali fabbricato, acqua contenuto
_XS_PERILS = {"motor": ("own_damage", "theft"), "buildings": ("accidental_damage",), "contents": ("escape_of_water",)}


def _xs_variant(fn):
    def f(train, valid, line):
        global _FREQ_XS
        _FREQ_XS = _XS_PERILS
        try:
            return fn(train, valid, line)
        finally:
            _FREQ_XS = {}
    return f


fs2_v2fb_xs_cred = _xs_variant(fs2_v2fb_cred)
fs2_v2_llr_xs_cred = _xs_variant(fs2_v2_llr_cred)
fs2_v2_sevmix_xs_cred = _xs_variant(fs2_v2_sevmix_cred)


# controlli per il guadagno di qwen_gbm73_gross_xs nel fabbricato: è la franchigia o solo una colonna in più?
#  siinv = 1/somma assicurata senza franchigia (toglie l'effetto del vincolo monotono su sum_insured);
#  rnd   = colonna di rumore (misura quanto cambia il GBM solo per il campionamento delle colonne)
def _add_col(kind, seed=11):
    def g(df, line):
        d = df.copy()
        if kind == "siinv":
            d["si_inv"] = 1.0 / d["sum_insured"].astype(float).clip(lower=1)
        else:
            d["rnd"] = np.random.default_rng(seed).random(len(d))
        return d
    return g


def qwen_gbm73_gross_siinv(train, valid, line):
    g = _add_col("siinv"); return qwen_gbm73_gross(g(train, line), g(valid, line), line)


def qwen_gbm73_gross_rnd(train, valid, line):
    g = _add_col("rnd"); return qwen_gbm73_gross(g(train, line), g(valid, line), line)


qwen_gbm73_gross_siinv_cred = _cred(qwen_gbm73_gross_siinv, 8)
qwen_gbm73_gross_rnd_cred = _cred(qwen_gbm73_gross_rnd, 8)


def _xs_factor(train, valid, line, n, mu_tr, k=50.0):
    """Fattore franchigia volontaria × quintile di valore del bene: A/E sul train del modello senza franchigia, credibilità k sinistri."""
    sc = lambda d: (d["vehicle_value_group"] if line == "motor" else d["sum_insured"]).astype(float).to_numpy()
    edges = np.unique(np.quantile(sc(train), [0.2, 0.4, 0.6, 0.8]))
    q_tr, q_va = np.searchsorted(edges, sc(train), side="right"), np.searchsorted(edges, sc(valid), side="right")
    g = pd.DataFrame({"x": train["voluntary_excess"].to_numpy(), "q": q_tr, "n": n, "m": mu_tr}).groupby(["x", "q"]).sum()
    fac = (g.n + k) / (g.m + k)
    get = lambda d, q: fac.reindex(pd.MultiIndex.from_arrays([d["voluntary_excess"].to_numpy(), q])).fillna(1.0).to_numpy()
    return get(train, q_tr), get(valid, q_va)


# perili dove la versione a due fasi vince nei due anni (excess_bench.py, auto): danni propri +1,1/+2,9, furto +4,7/+5,1
_2PH_PERILS = {"motor": ("own_damage", "theft")}


def _2ph_variant(fn):
    def f(train, valid, line):
        global _FREQ_2PH
        _FREQ_2PH = _2PH_PERILS
        try:
            return fn(train, valid, line)
        finally:
            _FREQ_2PH = {}
    return f


fs2_v2fb_2ph_cred = _2ph_variant(fs2_v2fb_cred)


def _rnd_seed(seed):
    def f(train, valid, line):
        g = _add_col("rnd", seed); return qwen_gbm73_gross(g(train, line), g(valid, line), line)
    return _cred(f, 8)


qwen_gbm73_gross_rnd12_cred, qwen_gbm73_gross_rnd13_cred, qwen_gbm73_gross_rnd14_cred = _rnd_seed(12), _rnd_seed(13), _rnd_seed(14)


# --- 23/10 mattina: "lordo poi netto" anche nel GBM e nella tempesta del fabbricato (richiesta di Simone, punti 1 e 2) ---
# GBM a due fasi: stesso GBM senza franchigia volontaria (né xs_rel), poi fattore franchigia × quintile di valore del bene
# = (Σ target + K) / (Σ previsione + K) sul train (previsione in-sample), K = 2000 righe di costo medio; applicato alla validazione.
def _gbm_2ph(fn, gross):
    def f(train, valid, line):
        drop = [c for c in ("voluntary_excess", "xs_rel") if c in train.columns]
        tr, va = train.drop(columns=drop), valid.drop(columns=drop)
        both = pd.concat([tr, va], ignore_index=True)
        pr = np.asarray(fn(tr, both, line), float)
        p_tr, p_va = pr[:len(tr)], pr[len(tr):]
        y = _gross_target(train) if gross else train[P.TARGET].to_numpy(float)
        y = np.asarray(y, float)
        sc = lambda d: (d["vehicle_value_group"] if line == "motor" else d["sum_insured"]).astype(float).to_numpy()
        edges = np.unique(np.quantile(sc(train), [0.2, 0.4, 0.6, 0.8]))
        q_tr, q_va = np.searchsorted(edges, sc(train), side="right"), np.searchsorted(edges, sc(valid), side="right")
        K = y.sum() / len(y) * 2000.0
        g = pd.DataFrame({"x": train["voluntary_excess"].to_numpy(), "q": q_tr, "y": y, "p": p_tr}).groupby(["x", "q"]).sum()
        fac = (g.y + K) / (g.p + K)
        fv = fac.reindex(pd.MultiIndex.from_arrays([valid["voluntary_excess"].to_numpy(), q_va])).fillna(1.0).to_numpy()
        return p_va * fv
    return f


qwen_gbm73fix_2ph_cred = _cred(_gbm_2ph(qwen_gbm73fix, False), 8)          # auto (OOF come qwen_gbm73fix_cred)
qwen_gbm73fix_gross_2ph_cred = _cred(_gbm_2ph(qwen_gbm73fix_gross, True), 8)  # auto, stima finale
qwen_gbm73_gross_2ph_cred = _cred(_gbm_2ph(qwen_gbm73_gross, True), 8)        # fabbricato (sostituisce qwen_gbm73_gross_xs_cred)
qwen_gbm73_2ph_cred = _cred(_gbm_2ph(qwen_gbm73, False), 8)                   # contenuto (sostituisce qwen_gbm73_cred)


# tempesta del fabbricato a due fasi (excess_bench: +1,6 / +1,0 sul perilo)
def fs2_v2_llr_2phs_cred(train, valid, line):
    global _FREQ_2PH
    _FREQ_2PH = {"buildings": ("storm",)}
    try:
        return fs2_v2_llr_cred(train, valid, line)
    finally:
        _FREQ_2PH = {}


# --- 23/10 notte: Parte 2 (mercato emerging) — modelli di src/transfer.py richiamabili da oof_one.py / final_model_line.py
def _tr(name):
    def f(train, valid, line):
        import transfer as TR
        return getattr(TR, name)(train, valid, line)
    f.__name__ = f"p2_{name}"
    return f


p2_scratch, p2_src_level, p2_src_glm, p2_src_gbm = _tr("scratch"), _tr("src_level"), _tr("src_glm"), _tr("src_gbm")
p2_src_peril, p2_src_peril_nocorr = _tr("src_peril"), _tr("src_peril_nocorr")
p2_src_glm2 = _tr("src_glm2")
