"""Parte 2 — adattamento del modello della Parte 1 a un mercato nuovo ("transfer" attuariale).

Idea: il mercato di partenza ha molti dati e una struttura (relatività per perilo) che conosciamo bene; il mercato
nuovo ha pochi dati, un livello diverso e forse relatività diverse. Si fa come con la credibilità:
    previsione = relatività della Parte 1 (sulle variabili in comune)  ×  correzione stimata sui dati nuovi,
dove la correzione è tanto più ampia quanto più i dati nuovi la sostengono.

Tre passi, ognuno un comando:

1. export-source  (gira con la config del mercato di PARTENZA, es. IAA_MARKET=configs/parte1.json)
       python -I transfer.py export-source configs/mercato_nuovo.json
   Stima il freq×sev della Parte 1 usando solo le colonne che esistono anche nel mercato nuovo, e scrive la sua
   previsione per ogni riga nuova (training e test) in <results nuovo>/source_pred_<linea>.parquet.
   È il "prior" attuariale: cosa costerebbe questa polizza se il mercato nuovo si comportasse come la Parte 1.

2. modelli del mercato nuovo (girano con la config del mercato NUOVO), disponibili per oof.py / blend.py / final.py:
     - scratch      : freq×sev stimato solo sui dati nuovi (riferimento: "ignora la Parte 1")
     - src_level    : prior della Parte 1 × un solo fattore di livello per linea (riferimento: "ignora i dati nuovi")
     - src_glm      : prior come offset + GLM Tweedie con penalità ridge sulle variabili indicate dall'audit
                      (default: regione, anno, perilo-mix non osservabile → regione + età edificio/veicolo)
     - src_gbm      : prior come init_score + GBM Tweedie molto regolarizzato (poche iterazioni, foglie grandi)
                      che impara solo le deviazioni robuste
   La credibilità per polizza (experience.py) si applica sopra come nella Parte 1 (wrapper *_cred in models_extra).

3. blend.py sui modelli sopra: i pesi dicono QUANTO il mercato nuovo si allontana dalla Parte 1 — è il numero da
   raccontare ai giudici ("adaptation").

Controllo che fa da guardrail: il modello 'scratch' deve battere 'src_level' sui fold del mercato nuovo; se non lo fa,
i dati nuovi sono troppo pochi o troppo diversi e si resta vicini al prior.
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


# ----------------------------------------------------------------- 1. export del prior dalla Parte 1
AGE_BANDS = [(17, 24, "17-24"), (25, 34, "25-34"), (35, 49, "35-49"), (50, 64, "50-64"), (65, 200, "65+")]


def _band(age):
    a = pd.Series(age, dtype=float)
    out = pd.Series(pd.NA, index=a.index, dtype="object")
    for lo, hi, lab in AGE_BANDS:
        out[(a >= lo) & (a <= hi)] = lab
    out[a < 17] = "17-24"
    return out


def harmonize(df):
    """Parte 2 (23/10): colonne del mercato di partenza rese confrontabili con l'emerging: età -> fasce dell'emerging
    (driver_age_band, policyholder_age_band), professione in minuscolo senza spazi."""
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
        src = harmonize(P.load(line))                        # mercato di partenza (config corrente)
        new_tr = pd.read_parquet(Path(new_cfg["model_dir"]) / f"model_{line}.parquet")
        new_te = pd.read_parquet(Path(new_cfg["model_dir"]) / f"test_{line}.parquet")
        common = [c for c in M._features(src, line) if c in new_tr.columns and c in new_te.columns]
        # 23/10: una categorica i cui livelli del mercato nuovo non esistono in partenza (regione R01–R12 contro E01–E06)
        # non porta informazione nel prior: va tolta, la stima la correzione sui dati nuovi
        def _overlap(c):
            if pd.api.types.is_numeric_dtype(src[c]) and pd.api.types.is_numeric_dtype(new_te[c]):
                return 1.0
            seen = set(src[c].dropna().astype(str).unique())
            v = new_te[c].dropna().astype(str)
            return float(v.isin(seen).mean()) if len(v) else 1.0
        dropped = [c for c in common if _overlap(c) < 0.5]
        common = [c for c in common if c not in dropped]
        if dropped:
            print(f"{line}: variabili tolte dal prior (livelli nuovi): {dropped}", flush=True)
        M.FEATURE_WHITELIST = set(common)
        import os
        if os.environ.get("P2_NB_OFF"):   # 24/10: sonda "il portafoglio del broker non è nuovo business organico"
            new_te = new_te.assign(new_business=False)
        both = pd.concat([new_tr, new_te], ignore_index=True)
        # colonne tecniche che il freq×sev legge dal frame "valid": esposizione e, per il carico LL, sum_insured
        pred = np.asarray(M._fs2(src, both, line, ll_si=True), float)
        outdf = pd.DataFrame({"row_id": both["row_id"].to_numpy(), "mu_src": pred})
        # 23/10: anche frequenza attesa (numero, livello 2025) e severità per perilo del mercato di partenza, per il
        # trasferimento perilo per perilo (livelli del mercato nuovo per perilo invece di un solo fattore per linea)
        lp = getattr(M._fs2, "last_parts", None)
        if lp:
            for per, (f_va, s_va) in lp["perils"].items():
                outdf[f"f_{per}"] = f_va; outdf[f"s_{per}"] = s_va
            outdf["ll_src"] = lp["ll"]; outdf["c_src"] = lp["c"]
        outdf.to_parquet(out / (f"source_pred_{line}_nonb.parquet" if os.environ.get("P2_NB_OFF") else f"source_pred_{line}.parquet"), index=False)
        M.FEATURE_WHITELIST = None
        print(f"{line}: prior Parte 1 su {len(common)} variabili comuni (su {len(M._features(src, line))}) -> {out / f'source_pred_{line}.parquet'}",
              f"| livello prior {pred[:len(new_tr)].sum() / new_tr['exposure'].sum():.1f} vs target nuovo {new_tr[P.TARGET].sum() / new_tr['exposure'].sum():.1f}", flush=True)
    (out / "source_export.json").write_text(json.dumps({"source": MC.CFG.get("_path", "parte1 (default)"), "new": new_cfg_path}, indent=1))


# ----------------------------------------------------------------- 2. modelli del mercato nuovo
_PRIOR_OVERRIDE = None   # 24/10: DataFrame indicizzato per row_id con mu_src (e parti) che sostituisce il file (prove)
PRIOR_SUFFIX = ""        # 24/10: es. "_meteo" -> source_pred_<linea>_meteo.parquet


def _prior(df, line):
    if _PRIOR_OVERRIDE is not None:
        return np.clip(_PRIOR_OVERRIDE["mu_src"].reindex(df["row_id"]).to_numpy(), 1e-3, None)
    f = P.RESULTS / f"source_pred_{line}{PRIOR_SUFFIX}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"manca {f}: lanciare prima  IAA_MARKET=<config partenza> python -I transfer.py export-source <config nuovo>")
    s = pd.read_parquet(f).set_index("row_id")["mu_src"]
    mu = s.reindex(df["row_id"]).to_numpy()
    if np.isnan(mu).any():
        raise ValueError(f"{int(np.isnan(mu).sum())} righe senza prior in {line}: rilanciare export-source")
    return np.clip(mu, 1e-3, None)


def scratch(train, valid, line):
    return M.fs2_all_ll(train, valid, line)


def src_level(train, valid, line):
    """prior × un fattore di livello della linea (stimato sui dati nuovi)."""
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
    """prior come offset + GLM Tweedie ridge su poche variabili: la correzione 'da attuario', leggibile."""
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    p_tr, p_va = _prior(train, line), _prior(valid, line)
    cols = [c for c in CORRECTION_VARS if c in train.columns and c in valid.columns and train[c].nunique() > 1]
    Xtr, Xva = _bins(train, valid, cols)
    y = train[P.TARGET].to_numpy(); e = train["exposure"].to_numpy()
    if alpha is None:   # penalità scelta su un 10% di polizze del training nuovo
        h = pd.util.hash_array(train["policy_id"].to_numpy()) % 10 == 3
        best = (-1e9, None)
        for a in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
            g = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=a, l1_ratio=0.0, max_iter=300)
            g.fit(Xtr[~h], y[~h] / p_tr[~h], sample_weight=p_tr[~h])     # risposta = y / prior, peso = prior  (≡ offset log prior)
            sc = P.d2(y[h], g.predict(Xtr[h]) * p_tr[h])
            if sc > best[0]:
                best = (sc, a)
        alpha = best[1]
    g = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=alpha, l1_ratio=0.0, max_iter=300)
    g.fit(Xtr, y / p_tr, sample_weight=p_tr)
    src_glm.last = {"alpha": alpha, "coef": dict(zip(g.feature_names_, np.round(g.coef_, 3))), "intercept": round(float(g.intercept_), 3)}
    out = g.predict(Xva) * p_va
    return out * (y.sum() / (g.predict(Xtr) * p_tr).sum())


def src_gbm(train, valid, line, rounds=150):
    """prior come init_score + GBM Tweedie regolarizzato: impara solo deviazioni robuste dal prior."""
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


# ----------------------------------------------------------------- 3. prova generale
def check(lines=None):
    """guardrail: su ogni linea del mercato nuovo, confronto dei quattro modelli sui fold (giudice congelato)."""
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


# ----------------------------------------------------------------- 23/10: trasferimento perilo per perilo
def _prior_parts(df, line):
    s = (_PRIOR_OVERRIDE if _PRIOR_OVERRIDE is not None else
         pd.read_parquet(P.RESULTS / f"source_pred_{line}{PRIOR_SUFFIX}.parquet").set_index("row_id")).reindex(df["row_id"])
    if s["mu_src"].isna().any():
        raise ValueError("righe senza prior: rilanciare export-source")
    return s


def src_peril(train, valid, line, correct=True, k_freq=100.0, k_sev=200.0, rounds=60):
    """Prior del mercato di partenza PER PERILO (frequenza attesa e severità al livello 2025) con:
    1. livello di frequenza e di severità del mercato nuovo per perilo, con credibilità verso il livello della linea
       (k_freq / k_sev = sinistri a cui il perilo pesa il 50%);
    2. (correct=True) correzione della frequenza con un GBM Poisson a 1 split molto corto sui dati nuovi, con il prior
       come init_score: impara regione, genere, nuovo business… solo dove i dati nuovi lo sostengono.
    Severità: prior × livello del perilo (le relatività di severità del mercato nuovo sono troppo rumorose con 2 anni)."""
    import lightgbm as lgb
    dev, claims, lvl = M._tables()
    dl = dev[dev["line"] == line]
    st, sv = _prior_parts(train, line), _prior_parts(valid, line)
    perils = [c[2:] for c in st.columns if c.startswith("f_")]
    yrs = train["year"].to_numpy()
    e_tr, e_va = train["exposure"].to_numpy(), valid["exposure"].to_numpy()
    cl = claims[(claims["line"] == line) & claims["row_id"].isin(train["row_id"])]
    # numero e costo osservati per perilo, costo sviluppato (IBNER) e riportato al 2025 col trend del perilo
    obs = {}
    for p in perils:
        d = dl[dl["peril"] == p].set_index("ay")
        if d.empty:
            continue
        ft, stt = float(d["freq_trend"].iloc[0]), float(d["sev_trend"].iloc[0])
        n = train[f"n_{p}"].to_numpy(float) if f"n_{p}" in train else np.zeros(len(train))
        pi = np.clip(train[f"pi_{p}"].to_numpy(), 1e-3, None) if f"pi_{p}" in train else np.ones(len(train))
        defl = (1 + ft) ** (-(2025 - yrs))
        e_cnt = st[f"f_{p}"].to_numpy() * pi * defl                       # numero atteso dal prior sui dati nuovi
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


# 23/10: correzione GLM con anche le variabili nuove o cambiate dell'emerging (genere, fasce d'età, nuovo business,
# codici di costruzione, canale) oltre a quelle della Parte 2 preparate prima
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
