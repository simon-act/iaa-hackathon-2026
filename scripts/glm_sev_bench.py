"""Banco GLM contro GBM per la SEVERITÀ di ogni perilo (21/10). Stessi dati, fold e metrica di scripts/sev_bench.py
(D² Gamma per perilo sui sinistri dell'anno dopo, base = media del training).
Modelli: base (severità di sub_40: GBM / GLM log SI / media come oggi; per le lesioni auto conta anche 'media'),
media, glm = GLM Gamma (glum) sui driver di severità (numeriche in 10 classi, auto + encoding make_model), ridge scelto
su un 20% dei sinistri del training; mix50 = media geometrica base × glm.
Output results_s3/glm_sev_bench.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from glum import GeneralizedLinearRegressor
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/sev_bench.py").read_text().split("\nrows = []")[0]
exec(compile(src, "sev_bench", "exec"))   # C, DRIV, CONFIGS, fit_predict, gdev, M
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def design(tr, va, cols):
    A, B = {}, {}
    for c in cols:
        s, t = tr[c], va[c]
        if pd.api.types.is_numeric_dtype(s) and s.dtype != bool and s.nunique() > 10:
            e = np.unique(np.nanquantile(s.astype(float), np.linspace(0, 1, 11))); e[0], e[-1] = -np.inf, np.inf
            a = pd.cut(s.astype(float), e, labels=False).astype(float).fillna(-1).astype(int).astype(str)
            b = pd.cut(t.astype(float), e, labels=False).astype(float).fillna(-1).astype(int).astype(str)
        else:
            a = s.astype(object).where(s.notna(), "NA").astype(str); b = t.astype(object).where(t.notna(), "NA").astype(str)
            vc = a.value_counts(); keep = set(vc.index[vc >= 50])
            a = a.where(a.isin(keep), "altro"); b = b.where(b.isin(keep), "altro")
        lv = sorted(set(a))
        A[c] = pd.Categorical(a, categories=lv); B[c] = pd.Categorical(b.where(b.isin(lv), lv[0]), categories=lv)
    return pd.DataFrame(A), pd.DataFrame(B)


def glm_pred(tr, va, line):
    y = tr["sev"].to_numpy(float)
    tr = tr.copy(); va = va.copy(); cols = [c for c in DRIV[line] if c in tr.columns and c != "make_model"]
    if line == "motor" and "make_model" in tr.columns:
        te_tr, te_va = M._te_make_model(tr, va, y, np.full(len(tr), y.mean()), k=20.0 * y.mean())
        tr["mm_sev_te"] = te_tr; va["mm_sev_te"] = te_va; cols = cols + ["mm_sev_te"]
    A, B = design(tr, va, cols)
    inner = pd.util.hash_array(tr["row_id"].to_numpy()) % 5 == 2
    best = (np.inf, None)
    for a in ALPHAS:
        g = GeneralizedLinearRegressor(family="gamma", alpha=a, l1_ratio=0.0, drop_first=False, max_iter=200).fit(A[~inner], y[~inner])
        d = gdev(y[inner], g.predict(A[inner]))
        if d < best[0]:
            best = (d, a)
    g = GeneralizedLinearRegressor(family="gamma", alpha=best[1], l1_ratio=0.0, drop_first=False, max_iter=200).fit(A, y)
    return g.predict(B), best[1]


rows = []
for line in ("motor", "buildings", "contents"):
    cl = C[C.line == line]
    for last in (2022, 2023):
        for p, g in cl.groupby("peril"):
            tr = g[g.ay <= last]; va = g[g.ay == last + 1]
            base = fit_predict(tr, va, line, p, CONFIGS["base"])
            mean = fit_predict(tr, va, line, p, CONFIGS["media"])
            gl, alpha = glm_pred(tr, va, line) if len(tr) > 150 else (mean, None)
            ref = gdev(va.sev, np.full(len(va), tr.sev.mean()))
            for name, pr in (("base", base), ("media", mean), ("glm", gl), ("mix50", np.sqrt(base * gl))):
                rows.append({"linea": line, "perilo": p, "fold": f"F{last + 1}", "modello": name, "d2": 1 - gdev(va.sev, pr) / ref,
                             "peso": float(va.sev.sum()), "sinistri_stima": len(tr), "alpha": alpha})
            print(line, p, f"F{last + 1}", {r["modello"]: round(r["d2"], 4) for r in rows[-4:]}, "alpha", alpha, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results_s3/glm_sev_bench.csv", index=False)
