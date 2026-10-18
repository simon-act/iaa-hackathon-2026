"""FILE SPERIMENTALE (passo 2): feature, modello, iperparametri. Il giudice è prepare.py.

Modelli disponibili:
  gbm_b  LightGBM Tweedie (p=1,5) sul target (b), offset log(esposizione)
  glm_b  GLM Tweedie (p=1,5) sul tasso target/esposizione, pesi = esposizione, poche variabili binnate
Uso:
  python -I train.py eval gbm_b          # punteggio sui fold temporali
  python -I train.py submit gbm_b out.csv # stima su 2021-24 (senza holdout) e previsione 2025
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prepare as P  # noqa: E402

# target molto rumoroso -> foglie grandi, regolarizzazione forte, categoriche smussate
GBM_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.03, num_leaves=15,
                  min_data_in_leaf=2000, min_sum_hessian_in_leaf=50, feature_fraction=0.6, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=50.0, max_cat_to_onehot=8, cat_smooth=100, cat_l2=50,
                  min_data_per_group=2000, max_cat_threshold=16, max_bin=63, verbose=-1, num_threads=2, seed=7)
NUM_ROUNDS = {"motor": 600, "buildings": 600, "contents": 600}   # aggiornati dopo early stopping su F2023
STATE = P.RESULTS / "train_state.json"


def to_frame(df: pd.DataFrame, cols: list, cats: dict | None = None):
    X = df[cols].copy()
    cats = cats if cats is not None else {}
    for c in cols:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            v = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
            if c not in cats:
                cats[c] = sorted(v.unique().tolist())
            X[c] = pd.Categorical(v, categories=cats[c])
    return X, cats


def features_for(df, line):
    cols = P.allowed_features(df)
    # costanti per linea o inutili
    cols = [c for c in cols if df[c].nunique(dropna=False) > 1]
    P.check_features(cols, df)
    return cols


def gbm_b(train, valid, line, num_rounds=None, es=False):
    cols = features_for(train, line)
    Xtr, cats = to_frame(train, cols)
    Xva, _ = to_frame(valid, cols, cats)
    # partenza dal tasso medio del training (con init_score LightGBM non fa boost_from_average:
    # partire da tasso 1 sottostimava il livello di ~7%)
    base = np.log(train[P.TARGET].sum() / train["exposure"].sum())
    dtr = lgb.Dataset(Xtr, label=train[P.TARGET].to_numpy(), init_score=np.log(train["exposure"].to_numpy()) + base,
                      free_raw_data=False)
    rounds = num_rounds or NUM_ROUNDS[line]
    callbacks, valid_sets = [], []
    if es:
        dva = lgb.Dataset(Xva, label=valid[P.TARGET].to_numpy(), init_score=np.log(valid["exposure"].to_numpy()) + base, reference=dtr)
        valid_sets = [dva]; callbacks = [lgb.early_stopping(50, verbose=False)]
        rounds = 2000
    m = lgb.train(GBM_PARAMS, dtr, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    pred = np.exp(m.predict(Xva, raw_score=True) + np.log(valid["exposure"].to_numpy()) + base)
    # ribilanciamento: con target asimmetrico e regolarizzazione forte il GBM sottostima la media
    # (foglie in scala log, effetto Jensen): scala = somma reale / somma prevista sul training
    pin = np.exp(m.predict(Xtr, raw_score=True) + np.log(train["exposure"].to_numpy()) + base)
    gbm_b.last_balance = float(train[P.TARGET].sum() / pin.sum())
    pred = pred * gbm_b.last_balance
    gbm_b.last_model, gbm_b.last_cols = m, cols
    return pred


GLM_FEATURES = {
    "motor": ["driver_age", "years_licensed", "ncb_years", "vehicle_value_group", "vehicle_age", "body_type",
              "fuel_type", "cover_type", "annual_mileage", "overnight_parking", "vehicle_use", "named_young_driver",
              "telematics_score", "voluntary_excess", "credit_band", "prior_claims_5y", "licence_points", "region",
              "urban_density", "pol_prev_rate", "channel", "payment_method", "missed_payments_12m", "engine_size_cc"],
    "buildings": ["policyholder_age", "sum_insured", "construction_type", "occupancy", "days_unoccupied",
                  "accidental_damage_cover", "voluntary_excess", "region", "urban_density", "flood_zone",
                  "distance_to_river_km", "elevation_m", "soil_type", "property_type", "roof_type", "basement",
                  "trees_within_5m", "building_age", "dry_summer_index", "wind_exposure_index", "rainfall_mm",
                  "credit_band", "pol_prev_rate", "listed_building"],
    "contents": ["policyholder_age", "sum_insured", "tenure", "security_level", "high_value_items",
                 "accidental_damage_cover", "works_from_home", "lodgers", "bicycle_cover", "voluntary_excess",
                 "region", "urban_density", "flood_zone", "credit_band", "pol_prev_rate", "property_type", "bedrooms"],
}


def _bin(train_col, cols_valid, q=8):
    """numerici -> classi per quantili (stimate sul training) + classe missing; categoriche come stringhe."""
    if pd.api.types.is_numeric_dtype(train_col) and train_col.dtype != bool and train_col.nunique() > 12:
        edges = np.unique(np.nanquantile(train_col.astype(float), np.linspace(0, 1, q + 1)))
        f = lambda x: pd.cut(x.astype(float), edges, include_lowest=True).astype(str).where(x.notna(), "missing")
        return f(train_col), [f(c) for c in cols_valid]
    g = lambda x: x.astype(object).where(x.notna(), "missing").astype(str)
    return g(train_col), [g(c) for c in cols_valid]


def glm_b(train, valid, line):
    from sklearn.linear_model import TweedieRegressor
    from sklearn.preprocessing import OneHotEncoder
    cols = GLM_FEATURES[line]
    P.check_features(cols, train)
    Btr, Bva = {}, {}
    for c in cols:
        a, (b,) = _bin(train[c], [valid[c]])
        Btr[c], Bva[c] = a, b
    Btr, Bva = pd.DataFrame(Btr), pd.DataFrame(Bva)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=200, drop="first", sparse_output=True)
    Xtr = enc.fit_transform(Btr); Xva = enc.transform(Bva)
    e_tr = train["exposure"].to_numpy()
    m = TweedieRegressor(power=1.5, link="log", alpha=1e-4, max_iter=300, solver="lbfgs")
    m.fit(Xtr, train[P.TARGET].to_numpy() / e_tr, sample_weight=e_tr)
    glm_b.last_model, glm_b.last_enc = m, enc
    return m.predict(Xva) * valid["exposure"].to_numpy()


def blend(train, valid, line, w=0.5):
    a = glm_b(train, valid, line); b = gbm_b(train, valid, line)
    return w * a + (1 - w) * b


GLM_GBM_ROUNDS = 200


def glm_gbm(train, valid, line):
    """GBM che parte dal GLM: init_score = log(previsione GLM), impara solo la correzione."""
    from sklearn.model_selection import KFold
    # previsione GLM out-of-fold sul training (per non dare al GBM un init troppo ottimista)
    oof = np.zeros(len(train))
    for k, (i, j) in enumerate(KFold(3, shuffle=True, random_state=1).split(train)):
        oof[j] = glm_b(train.iloc[i], train.iloc[j], line)
    g_va = glm_b(train, valid, line)
    cols = features_for(train, line)
    Xtr, cats = to_frame(train, cols); Xva, _ = to_frame(valid, cols, cats)
    params = dict(GBM_PARAMS, learning_rate=0.02)
    dtr = lgb.Dataset(Xtr, label=train[P.TARGET].to_numpy(), init_score=np.log(np.clip(oof, 1e-6, None)))
    m = lgb.train(params, dtr, num_boost_round=GLM_GBM_ROUNDS)
    gbm_b.last_model, gbm_b.last_cols = m, cols
    return np.exp(m.predict(Xva, raw_score=True) + np.log(np.clip(g_va, 1e-6, None)))


def const_line(train, valid, line):
    rate = train[P.TARGET].sum() / train["exposure"].sum()
    return rate * valid["exposure"].to_numpy()


def main():
    mode, model = sys.argv[1], sys.argv[2]
    STATE.parent.mkdir(exist_ok=True)
    t0 = time.time()
    if model == "gbm_b" and mode == "es":
        # early stopping sul fold F2023 per fissare il numero di alberi (poi fisso su F2024 e finale)
        best = {}
        for line in P.LINES:
            df = P.load(line); df = df[~P.is_holdout(df["policy_id"])]
            tr, va = df[df["year"].isin([2021, 2022])], df[df["year"] == 2023]
            gbm_b(tr, va, line, es=True)
            best[line] = int(gbm_b.last_model.best_iteration)
            print(line, "best_iter", best[line], flush=True)
        STATE.write_text(json.dumps({"num_rounds": best}))
        return
    if STATE.exists():
        NUM_ROUNDS.update(json.loads(STATE.read_text()).get("num_rounds", {}))
    fn = {"gbm_b": gbm_b, "glm_b": glm_b, "const_line": const_line, "blend": blend, "glm_gbm": glm_gbm}[model]
    if mode == "eval":
        r = P.evaluate(fn)
        print(json.dumps(r), f"{time.time() - t0:.0f}s")
        with open(STATE.parent / "results.tsv", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{model}\t{r['score']}\t{r['pooled_mean']}\t{json.dumps(r['by'])}\n")
    elif mode == "submit":
        out = sys.argv[3]
        subs = []
        for line in P.LINES:
            df = P.load(line); df = df[~P.is_holdout(df["policy_id"])]
            te = P.load(line, "test")
            pred = fn(df, te, line)
            subs.append(pd.DataFrame({"row_id": te["row_id"], "prediction": pred, "line": line,
                                      "exposure": te["exposure"]}))
            if model in ("gbm_b", "glm_gbm", "blend"):
                imp = pd.Series(gbm_b.last_model.feature_importance("gain"), index=gbm_b.last_cols).sort_values(ascending=False)
                imp.to_csv(STATE.parent / f"importance_{model}_{line}.csv")
        s = pd.concat(subs)
        s.to_parquet(STATE.parent / f"pred_{model}.parquet", index=False)
        s[["row_id", "prediction"]].to_csv(out, index=False)
        print(s.groupby("line").apply(lambda g: g.prediction.sum() / g.exposure.sum(), include_groups=False))


if __name__ == "__main__":
    main()
