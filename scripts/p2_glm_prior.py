"""Parte 2 (27/10, richiesta di Simone: "di base fare modello GLM"): prior del developed con un GLM Tweedie (p 1,5, log, ridge)
invece del GBM freq×sev, sulle stesse variabili comuni ai due mercati (stessa selezione di transfer.export_source).
Numeriche a 10 classi per quantili, categoriche con livelli rari (< 500 righe) raggruppati; risposta = target_b / esposizione,
peso = esposizione; alpha scelto su un 10% di polizze del developed. Uscita: <results nuovo>/source_pred_<linea>_glm.parquet con
mu_src del GLM e le parti per perilo (f_, s_, ll_src, c_src) copiate dal prior GBM (servono solo a livelli per perilo e numeri attesi).
Uso: IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_glm_prior.py <linea> configs/parte2_llc.json [frazione=0,4]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR, models_extra as M, market_config as MC  # noqa: E402
from glum import GeneralizedLinearRegressor, TweedieDistribution  # noqa: E402

line, new_cfg = sys.argv[1], MC.load(sys.argv[2])
t0 = time.time()
src = TR.harmonize(P.load(line))
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6      # campione di polizze del developed (limite di tempo per chiamata)
src = src[pd.util.hash_array(src["policy_id"].to_numpy() + "_c") % 100 < FRAC * 100].reset_index(drop=True)
print(f"{line}: developed campione {len(src)} righe ({time.time() - t0:.0f}s)", flush=True)
new_tr = pd.read_parquet(Path(new_cfg["model_dir"]) / f"model_{line}.parquet")
new_te = pd.read_parquet(Path(new_cfg["model_dir"]) / f"test_{line}.parquet")
common = [c for c in M._features(src, line) if c in new_tr.columns and c in new_te.columns]


def overlap(c):
    if pd.api.types.is_numeric_dtype(src[c]) and pd.api.types.is_numeric_dtype(new_te[c]):
        return 1.0
    seen = set(src[c].dropna().astype(str).unique()); v = new_te[c].dropna().astype(str)
    return float(v.isin(seen).mean()) if len(v) else 1.0


common = [c for c in common if overlap(c) >= 0.5 and src[c].nunique() > 1]
both = pd.concat([new_tr, new_te], ignore_index=True)
Xs, Xn = TR._bins(src, both, common, q=10)
print(f"{line}: classi pronte, {len(common)} variabili ({time.time() - t0:.0f}s)", flush=True)
y = src[P.TARGET].to_numpy(); e = src["exposure"].to_numpy()
h = pd.util.hash_array(src["policy_id"].to_numpy()) % 10 == 3
sub = (pd.util.hash_array(src["policy_id"].to_numpy() + "_s") % 2 == 0) & ~h
best = (-np.inf, None)
for a in [0.3, 1.0, 3.0, 10.0, 30.0]:
    g = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=a, l1_ratio=0.0, max_iter=200)
    g.fit(Xs[sub], y[sub] / e[sub], sample_weight=e[sub])
    sc = P.d2(y[h], g.predict(Xs[h]) * e[h])
    print(f"{line} alpha {a}: D² holdout {sc:.5f} ({time.time() - t0:.0f}s)", flush=True)
    if sc > best[0]:
        best = (sc, a)
g = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=best[1], l1_ratio=0.0, max_iter=200)
g.fit(Xs[~h], y[~h] / e[~h], sample_weight=e[~h])
print(f"{line}: alpha {best[1]}, D² holdout developed {P.d2(y[h], g.predict(Xs[h]) * e[h]):.5f}, {len(common)} variabili", flush=True)
mu = g.predict(Xn) * both["exposure"].to_numpy()
res = Path(new_cfg["results"])
gb = pd.read_parquet(res / f"source_pred_{line}.parquet").set_index("row_id")
out = gb.reindex(both["row_id"]).reset_index()
out["mu_src_gbm"] = out["mu_src"]; out["mu_src"] = mu
out.to_parquet(res / f"source_pred_{line}_glm.parquet", index=False)
print(f"{line}: -> source_pred_{line}_glm.parquet; corr log(GLM, GBM) {np.corrcoef(np.log(mu), np.log(out.mu_src_gbm))[0, 1]:.3f} ({time.time() - t0:.0f}s)")
