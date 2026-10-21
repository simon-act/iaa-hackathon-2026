"""21/10 (domanda di Simone): errore spiegabile, non spiegabile e dove siamo. Rifà src/ceiling.py (17/10) sulla
configurazione attuale (s3), con più semi e più valori della frailty φ, e mette accanto il punteggio vero sui fold.
Verità simulata = freq×sev per perilo stimato sul fold (numero atteso λ_p, severità s_p) × frailty G ~ Gamma(1, φ).
Target simulato come target_b: sinistri tagliati (forme ricampionate dai sinistri chiusi veri AY 2021–22), denunciati con
prob. π, + IBNR atteso uniforme + caricamento LL. Previsioni confrontate:
  noi_senza_storia = media vera senza G; cred_fold / cred_test = credibilità sulla storia simulata (storia dei fold /
  storia del test: +1 anno rispetto al fold 2024, +2 rispetto al 2023), k = 1/φ; oracolo = media vera con G.
Uscita: results_s3/ceiling_s3.csv (per linea, fold, φ, seme) e stampa riassuntiva con la media pesata come Kaggle.
Uso: IAA_MARKET=configs/parte1_s3.json python scripts/ceiling_s3.py"""
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
                for tag, g in Ghat.items():   # la credibilità agisce solo sulla parte legata alla frequenza
                    q = mu + fpart * (g - 1); r[tag] = q * mu.sum() / q.sum()
                for k, v in r.items():
                    out.append(dict(line=line, fold=fname, phi=phi, seed=seed, what=k, d2=P.d2(y, v)))
        print(line, fname, "fatto", flush=True)
        pd.DataFrame(out).to_csv(P.RESULTS / "ceiling_s3.csv", index=False)
R = pd.DataFrame(out)
g = R.groupby(["what", "phi", "line", "fold"], dropna=False)["d2"].mean().reset_index()
g["w"] = g.line.map(W)
k = g.groupby(["what", "phi", "fold"], dropna=False).apply(lambda z: (z.d2 * z.w).sum() / z.w.sum()).unstack("fold")
k["media"] = k.mean(axis=1)
print(k.round(4).to_string())
print(R.groupby(["what", "phi"], dropna=False)["d2"].std().round(4).to_string())
