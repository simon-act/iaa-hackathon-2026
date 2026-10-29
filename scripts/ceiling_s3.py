"""21/10 (Simone's question): explainable vs unexplainable error, and where we stand. Reruns src/ceiling.py (17/10) on the
current configuration (s3), with more seeds and more values of the frailty φ, and shows the actual fold score alongside.
Simulated truth = per-peril freq×sev fitted on the fold (expected count λ_p, severity s_p) × frailty G ~ Gamma(1, φ).
Simulated target built like target_b: capped claims (shapes resampled from actual closed claims AY 2021–22), reported with
prob. π, + uniform expected IBNR + LL load. Predictions compared:
  noi_senza_storia = true mean without G; cred_fold / cred_test = credibility on the simulated history (fold history /
  test history: +1 year vs the 2024 fold, +2 vs 2023), k = 1/φ; oracolo = true mean with G.
Output: results_s3/ceiling_s3.csv (by line, fold, φ, seed) and a printed summary with the Kaggle-style weighted mean.
Usage: IAA_MARKET=configs/parte1_s3.json python scripts/ceiling_s3.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare as P, models_extra as M, experience as X  # noqa: E402
PHIS, SEEDS = [0.0, 0.15, 0.3, 0.45], [1, 2, 3]
W = {"motor": 254986, "buildings": 184155, "contents": 187498}
cl = pd.read_parquet(Path(P.CFG["interim"]) / "claims_clean.parquet")
cl = cl[(cl.status == "closed") & (cl.ay <= 2022) & (cl.net_capped > 0)]
real = {n: pd.read_parquet(P.RESULTS / "oof" / f"{n}.parquet", columns=["row_id", "line", "fold", "target_b", "pred"]) for n in ("fs2_v2_cred", "blend_opt")}
out = []
for line in P.LINES:
    df = P.load(line); df = df[~P.is_holdout(df.policy_id)]
    H0, _, _ = X.history(line)
    for fname, (tys, vy) in P.FOLDS.items():
        tr, va = df[df.year.isin(tys)], df[df.year == vy]
        M._fs2(tr, va, line, ll_si=True)
        parts, ll, c = M._fs2.last_parts["perils"], M._fs2.last_parts["ll"], M._fs2.last_parts["c"]
        e = va.exposure.to_numpy()
        H = H0.reindex(va.row_id).fillna(0).to_numpy()
        lam_tot = sum(f for f, _ in parts.values()) * c / e
        for nm, o in real.items():
            s = o[(o.line == line) & (o.fold == fname)]
            out.append(dict(line=line, fold=fname, phi=np.nan, seed=0, what=f"vero_{nm}", d2=P.d2(s.target_b, s.pred)))
        for phi in PHIS:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * seed + int(phi * 100))
                G = rng.gamma(1 / phi, phi, len(va)) if phi > 0 else np.ones(len(va))
                kk = 1 / phi if phi > 0 else 1e9
                Ghat = {}
                for tag, Hx in (("cred_fold", H), ("cred_test", H + (1 if vy == 2024 else 2))):
                    Eh = Hx * lam_tot; Nh = rng.poisson(Eh * G); Ghat[tag] = (Nh + kk) / (Eh + kk)
                y = np.zeros(len(va)); mu = np.zeros(len(va)); muG = np.zeros(len(va)); fpart = np.zeros(len(va))
                for p, (f, s) in parts.items():
                    lam = f * c; pi = va[f"pi_{p}"].clip(0, 1).to_numpy()
                    x = cl[(cl.line == line) & (cl.peril == p)]
                    shp = (x.net_capped / x.net_capped.mean()).to_numpy()
                    N = rng.poisson(lam * G); idx = np.repeat(np.arange(len(va)), N)
                    amt = s[idx] * shp[rng.integers(0, len(shp), len(idx))]
                    rep = rng.random(len(idx)) < pi[idx]
                    y += np.bincount(idx, weights=amt * rep, minlength=len(va))
                    ib = (1 - pi) * e * ((lam * s).sum() / e.sum())
                    y += ib; mu += lam * s * pi + ib; muG += lam * G * s * pi + ib
                    fpart += lam * s * pi
                y += ll; mu += ll; muG += ll
                r = {"noi_senza_storia": mu, "oracolo": muG}
                for tag, g in Ghat.items():   # credibility acts only on the frequency-driven part
                    q = mu + fpart * (g - 1); r[tag] = q * mu.sum() / q.sum()
                for k, v in r.items():
                    out.append(dict(line=line, fold=fname, phi=phi, seed=seed, what=k, d2=P.d2(y, v)))
        print(line, fname, "done", flush=True)
        pd.DataFrame(out).to_csv(P.RESULTS / "ceiling_s3.csv", index=False)
R = pd.DataFrame(out)
g = R.groupby(["what", "phi", "line", "fold"], dropna=False)["d2"].mean().reset_index()
g["w"] = g.line.map(W)
k = g.groupby(["what", "phi", "fold"], dropna=False).apply(lambda z: (z.d2 * z.w).sum() / z.w.sum()).unstack("fold")
k["media"] = k.mean(axis=1)
print(k.round(4).to_string())
print(R.groupby(["what", "phi"], dropna=False)["d2"].std().round(4).to_string())
