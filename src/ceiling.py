"""Score ceiling: the D² a PERFECT model (i.e. one with the true mean) would achieve on simulated claims.
Truth = per-peril freq×sev prediction (fs2_all_ll): expected count λ_p and mean severity s_p per row.
Simulation per row and peril: N ~ Poisson(λ_p · G), G = optional Gamma frailty (mean 1, variance φ);
amounts = s_p × shape resampled from the peril's actual closed claims (AY 2021–22), normalised to mean 1.
Two targets:
  T1 "like target_b": capped claims, reported with prob. π_p(row) + uniform line-level expected IBNR + LL loading
  T2 "like Kaggle": all 2025 claims at ultimate, NOT capped (shape from the full amounts)
Output: D² of the perfect model by line and for all lines together, by fold; results/diag/ceiling.json
Usage: python -I ceiling.py [phi]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import prepare as P, models_extra as M
ROOT = Path(__file__).resolve().parents[1]
phi = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
cl = pd.read_parquet(ROOT / "data" / "interim" / "claims_clean.parquet")
cl = cl[(cl.status == "closed") & (cl.ay <= 2022) & (cl.net_capped > 0)]
rng = np.random.default_rng(42)
res, pooled = {}, {f: {"T1": ([], [], []), "T2": ([], [], [])} for f in P.FOLDS}
for line in P.LINES:
    df = P.load(line); df = df[~P.is_holdout(df.policy_id)]
    for fname, (tys, vy) in P.FOLDS.items():
        tr, va = df[df.year.isin(tys)], df[df.year == vy]
        M._fs2(tr, va, line, ll_si=True)
        parts, ll, c = M._fs2.last_parts["perils"], M._fs2.last_parts["ll"], M._fs2.last_parts["c"]
        e = va.exposure.to_numpy()
        mu1 = np.zeros(len(va)); mu2 = np.zeros(len(va)); m1G = np.zeros(len(va)); m2G = np.zeros(len(va)); y1 = np.zeros(len(va)); y2 = np.zeros(len(va))
        G = rng.gamma(1 / phi, phi, len(va)) if phi > 0 else np.ones(len(va))
        tot_rate = sum(f * s for f, s in parts.values()).sum() * c / e.sum()
        # history simulated with the same frailty: E_history = H × row frequency (as in experience.py)
        import experience as X
        H, _, _ = X.history(line); H = H.reindex(va.row_id).fillna(0).to_numpy()
        lam_tot = sum(f for f, _ in parts.values()) * c / e        # expected claims per exposure year
        Ghat = {}
        for tag, Hx in (("fold", H), ("test", H + 1.5)):          # test: on average ~1.5 more years of history
            Eh = Hx * lam_tot
            Nh = rng.poisson(Eh * G)
            kk = 1 / phi if phi > 0 else 1e9
            Ghat[tag] = (Nh + kk) / (Eh + kk)
        for p, (f, s) in parts.items():
            lam = f * c; pi = va[f"pi_{p}"].clip(0, 1).to_numpy()
            x = cl[(cl.line == line) & (cl.peril == p)]
            shp_c = (x.net_capped / x.net_capped.mean()).to_numpy()
            shp_u = (x.net_incurred.clip(lower=0) / x.net_capped.mean()).to_numpy()
            N = rng.poisson(lam * G)
            idx = np.repeat(np.arange(len(va)), N)
            k = rng.integers(0, len(shp_c), len(idx))
            amt_c = s[idx] * shp_c[k]; amt_u = s[idx] * shp_u[k]
            rep = rng.random(len(idx)) < pi[idx]
            y1 += np.bincount(idx, weights=amt_c * rep, minlength=len(va))
            y2 += np.bincount(idx, weights=amt_u, minlength=len(va))
            mu1 += lam * s * pi; m1G += lam * G * s * pi
            mu2 += lam * s * shp_u.mean(); m2G += lam * G * s * shp_u.mean()
            # uniform line-level expected IBNR (as in target_b): unreported share × mean peril rate
            ib = (1 - pi) * e * ((lam * s).sum() / e.sum())
            y1 += ib; mu1 += ib; m1G += ib
        y1 += ll; mu1 += ll; m1G += ll
        def cr(mu, base, g):
            q = base + (mu - base) * g; return q * mu.sum() / q.sum()
        base1 = mu1 - (mu1 - ll)  # part not tied to frequency: LL only (the uniform IBNR stays inside mu1)
        r = {"T1_noi": P.d2(y1, mu1), "T1_cred_fold": P.d2(y1, cr(mu1, ll, Ghat["fold"])), "T1_oracolo": P.d2(y1, m1G),
             "T2_noi": P.d2(y2, mu2), "T2_cred_fold": P.d2(y2, cr(mu2, 0 * mu2, Ghat["fold"])),
             "T2_cred_test": P.d2(y2, cr(mu2, 0 * mu2, Ghat["test"])), "T2_oracolo": P.d2(y2, m2G)}
        res[f"{line}|{fname}"] = {k: round(v, 4) for k, v in r.items()}
        for t, (yy, a, b) in {"T1": (y1, mu1, m1G), "T2": (y2, mu2, m2G)}.items():
            pooled[fname][t][0].append(yy); pooled[fname][t][1].append(a); pooled[fname][t][2].append(b)
        print(line, fname, res[f"{line}|{fname}"], flush=True)
for f in P.FOLDS:
    for t in ("T1", "T2"):
        yy = np.concatenate(pooled[f][t][0])
        res[f"tutte|{f}|{t}"] = {"noi": round(P.d2(yy, np.concatenate(pooled[f][t][1])), 4), "oracolo": round(P.d2(yy, np.concatenate(pooled[f][t][2])), 4)}
print({k: v for k, v in res.items() if k.startswith("tutte")})
(ROOT / "results" / "diag" / f"ceiling_phi{phi}.json").write_text(json.dumps(res, indent=1))
