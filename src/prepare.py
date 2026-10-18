"""GIUDICE CONGELATO (passo 2): dati, fold, holdout, metrica e guardrail.

Questo file non va modificato da chi sperimenta (umano o agente): definisce cosa vuol dire
"migliore". Si sperimenta solo in train.py.

- Fold temporali: F2023 = stima su 2021-22, valuta su 2023; F2024 = stima su 2021-23, valuta su 2024.
- Holdout chiuso: 10% delle polizze (hash stabile del policy_id), escluso da stima e valutazione
  dei fold; si guarda solo a fine lavoro (holdout_score).
- Target: target_b (ultimate per riga al livello 2025, tagliato + caricamento large loss).
- Metrica: Tweedie D² con power 1,5 sulle previsioni per riga (CU), come Kaggle.
- Guardrail: solo colonne nella lista ammessa possono entrare nel modello.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, LINES, folds_from_years  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(CFG["model_dir"])
RESULTS = Path(CFG["results"])


def _years():
    try:
        return sorted(int(y) for y in pd.read_parquet(MODEL_DIR / f"model_{LINES[0]}.parquet", columns=["year"])["year"].unique())
    except Exception:
        return [2021, 2022, 2023, 2024]


FOLDS = folds_from_years(_years())
HOLDOUT_SHARE = float(CFG["holdout_share"])
TARGET = "target_b"
POWER = 1.5

FORBIDDEN_PREFIX = ("n_", "cost_", "excess_", "pi_", "ult", "target", "flag_")
FORBIDDEN = {"renewal_decision", "written_premium", "rec_adj", "n_total", "cost_total", "year",
             "period_start", "period_end", "exposure", "exposure_raw", "row_id", "policy_id",
             "customer_id", "address_id", "construction_type_raw", "email_domain", "surname",
             "date_of_birth", "customer_since_year", "year_built", "birth_year",
             "pol_prev_n", "pol_prev_cost", "cust_other_prev_n", "cust_other_prev_cost",
             "addr_prev_n", "addr_prev_cost",   # conteggi grezzi: storia più lunga nel test -> usare i tassi
             # LEAKAGE: contatti col servizio clienti NEL periodo della riga, generati dai sinistri stessi
             # (0 contatti -> frequenza 0,03%; corr. 0,75 con n sinistri; nel test 2025 max 6 vs 36)
             "customer_service_contacts"}


def drift_report(line: str) -> pd.DataFrame:
    """Guardrail: confronta ogni variabile ammessa tra training 2024 e test 2025 (PSI su 10 classi).
    PSI > 0,25 = distribuzione molto diversa: possibile informazione non disponibile al momento del prezzo."""
    tr = load(line); tr = tr[tr["year"] == 2024]; te = load(line, "test")
    rows = []
    for c in allowed_features(tr):
        a, b = tr[c], te[c]
        if pd.api.types.is_numeric_dtype(a) and a.nunique() > 10:
            edges = np.unique(np.nanquantile(a.astype(float), np.linspace(0, 1, 11)))
            pa = np.histogram(a.dropna().astype(float), edges)[0] + 1; pb = np.histogram(b.dropna().astype(float), edges)[0] + 1
        else:
            norm = lambda x: x.map(lambda v: str(round(float(v), 6)) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else str(v))
            va, vb = norm(a).value_counts(), norm(b).value_counts()
            k = va.index.union(vb.index); pa = va.reindex(k).fillna(0).to_numpy() + 1; pb = vb.reindex(k).fillna(0).to_numpy() + 1
        pa, pb = pa / pa.sum(), pb / pb.sum()
        rows.append((c, float(np.sum((pa - pb) * np.log(pa / pb)))))
    return pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)


def tweedie_deviance(y, mu, p=POWER):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    return float(np.mean(2 * (np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
                              - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))))


def d2(y, mu):
    y = np.asarray(y, float)
    return 1 - tweedie_deviance(y, mu) / tweedie_deviance(y, np.full_like(y, y.mean()))


def is_holdout(policy_id: pd.Series) -> np.ndarray:
    h = policy_id.map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return (h < HOLDOUT_SHARE).to_numpy()


def allowed_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in FORBIDDEN and not c.startswith(FORBIDDEN_PREFIX)]


def check_features(cols, df) -> None:
    bad = [c for c in cols if c in FORBIDDEN or c.startswith(FORBIDDEN_PREFIX) or c not in df.columns]
    if bad:
        raise ValueError(f"colonne non ammesse nel modello: {bad}")


def load(line: str, kind: str = "train") -> pd.DataFrame:
    return pd.read_parquet(MODEL_DIR / f"{'model' if kind == 'train' else 'test'}_{line}.parquet")


def evaluate(fit_predict, lines=LINES, folds=FOLDS, verbose=True):
    """fit_predict(train_df, valid_df, line) -> previsioni per riga (CU) su valid_df.
    Restituisce un dict con D² per linea × fold e la media ponderata (punteggio da ottimizzare)."""
    res, pooled = {}, {f: ([], []) for f in folds}
    for line in lines:
        df = load(line)
        ho = is_holdout(df["policy_id"])
        df = df[~ho]
        for fname, (tr_years, va_year) in folds.items():
            tr = df[df["year"].isin(tr_years)]
            va = df[df["year"] == va_year]
            pred = np.asarray(fit_predict(tr, va, line), float)
            if (pred < 0).any() or not np.isfinite(pred).all():
                raise ValueError("previsioni negative o non finite")
            res[(line, fname)] = d2(va[TARGET].to_numpy(), pred)
            pooled[fname][0].append(va[TARGET].to_numpy()); pooled[fname][1].append(pred)
            if verbose:
                print(f"  {line:9s} {fname}: D2 = {res[(line, fname)]:.4f}", flush=True)
    # media con pesi = numero righe del test 2025 per linea (come conta Kaggle su tutte le righe insieme)
    w = {l: len(load(l, "test")) for l in lines}   # come conta Kaggle: righe del test per linea
    score = sum(res[(l, f)] * w[l] for l in lines for f in folds) / (sum(w[l] for l in lines) * len(folds))
    # come Kaggle: tutte le linee insieme in un solo file
    pool = {f: round(d2(np.concatenate(a), np.concatenate(b)), 5) for f, (a, b) in pooled.items()} if len(lines) == len(LINES) else {}
    if verbose and pool:
        print("  pooled (tutte le linee):", pool, flush=True)
    return {"by": {f"{l}|{f}": round(v, 5) for (l, f), v in res.items()}, "score": round(score, 5), "pooled": pool,
            "pooled_mean": round(float(np.mean(list(pool.values()))), 5) if pool else None}


def holdout_score(fit_predict, lines=LINES):
    """Da usare solo a fine lavoro: stima su 2021-23 senza holdout, valuta sull'holdout 2024."""
    out = {}
    for line in lines:
        df = load(line)
        ho = is_holdout(df["policy_id"])
        tr = df[(~ho) & df["year"].isin([2021, 2022, 2023])]
        va = df[ho & (df["year"] == 2024)]
        out[line] = round(d2(va[TARGET].to_numpy(), fit_predict(tr, va, line)), 5)
    return out
