"""GLM vs GBM bench for the FREQUENCY of each peril (21/10, Simone's question: choose the method peril by peril).
Same setup as scripts/freq_bench.py: offset = log(exposure × completeness) − trend, fold 2023 (fit 2021–22) and
2024 (fit 2021–23), Poisson D² on claims REPORTED in the following year, level realigned by peril and fold.
Models:
  gbm   = the sub_40 frequency (FREQ_PARAMS + FREQ_CHOSEN of the peril, 3-leaf subsidence, make_model encoding);
  glm   = Poisson GLM (glum) on the same variables: numerics with > 20 distinct values in 20 quantile bins (NaN as its own bin),
          the others as categories (levels with < 300 rows grouped); for motor also the make_model encoding in 20 bins;
          subsidence with the soil × dry-summer interaction (like the 3-leaf GBM). Ridge penalty chosen on a 10% sample
          of training policies (Poisson deviance), then fitted on the whole training set;
  mix50 = geometric mean of the two (same form as the log-scale blend).
Usage: IAA_MARKET=configs/parte1_s3.json python3 -I scripts/glm_peril_bench.py <line> ; output results_s3/glm_peril_bench.csv"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from glum import GeneralizedLinearRegressor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import prepare as P  # noqa: E402
import models_extra as M  # noqa: E402
import freq_bench as FB  # noqa: E402

ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
OUT = P.RESULTS / "glm_peril_bench.csv"


def pdev(y, mu):
    mu = np.clip(mu, 1e-12, None)
    return 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))


def design(tr, va, cols, peril):
    """categories common to fitting and evaluation"""
    A, B = {}, {}
    for c in cols:
        s, t = tr[c], va[c]
        if pd.api.types.is_numeric_dtype(s) and s.dtype != bool and s.nunique() > 20:
            edges = np.unique(np.nanquantile(s.astype(float), np.linspace(0, 1, 21)))
            edges[0], edges[-1] = -np.inf, np.inf
            a = pd.cut(s.astype(float), edges, labels=False).astype("float").fillna(-1).astype(int).astype(str)
            b = pd.cut(t.astype(float), edges, labels=False).astype("float").fillna(-1).astype(int).astype(str)
        else:
            a = s.astype(object).where(s.notna(), "NA").astype(str); b = t.astype(object).where(t.notna(), "NA").astype(str)
            vc = a.value_counts(); keep = set(vc.index[vc >= 300])
            a = a.where(a.isin(keep), "altro"); b = b.where(b.isin(keep), "altro")
        A[c], B[c] = a.to_numpy(), b.to_numpy()
    if peril == "subsidence" and "soil_type" in tr and "dry_summer_index" in tr:
        e = np.unique(np.nanquantile(tr["dry_summer_index"], [0, .25, .5, .75, 1])); e[0], e[-1] = -np.inf, np.inf
        A["soil_x_dry"] = (tr["soil_type"].fillna("NA").astype(str) + "|" + pd.cut(tr["dry_summer_index"], e, labels=False).astype(str)).to_numpy()
        B["soil_x_dry"] = (va["soil_type"].fillna("NA").astype(str) + "|" + pd.cut(va["dry_summer_index"], e, labels=False).astype(str)).to_numpy()
    A, B = pd.DataFrame(A), pd.DataFrame(B)
    for c in A.columns:
        lv = sorted(set(A[c]))
        A[c] = pd.Categorical(A[c], categories=lv)
        B[c] = pd.Categorical(B[c].where(B[c].isin(lv), lv[0]), categories=lv)
    return A, B


def glm_fit(A, n, off, Bv, off_v, pol):
    inner = (pd.util.hash_array(pol) % 10 == 7)
    best = (np.inf, None)
    for a in ALPHAS:
        g = GeneralizedLinearRegressor(family="poisson", alpha=a, l1_ratio=0.0, drop_first=False, max_iter=200)
        g.fit(A[~inner], n[~inner], offset=off[~inner])
        mu = g.predict(A[inner], offset=off[inner]); mu *= n[inner].sum() / mu.sum()
        d = pdev(n[inner], mu)
        if d < best[0]:
            best = (d, a)
    g = GeneralizedLinearRegressor(family="poisson", alpha=best[1], l1_ratio=0.0, drop_first=False, max_iter=200).fit(A, n, offset=off)
    return g.predict(Bv, offset=off_v), best[1]


def run(line):
    df = FB.load(line)
    dev, _, _ = M._tables(); dl = dev[dev["line"] == line]
    perils = sorted(dl["peril"].unique())
    share = dl[dl["ay"] >= 2023].groupby("peril")["cost_ult"].sum(); share = share / share.sum()
    cols = M._features(df, line)
    if line != "motor" and "prem_rel" not in cols:
        cols = cols + ["prem_rel"]
    has_mm = "make_model" in cols
    base_cols = [c for c in cols if not (has_mm and c == "make_model")]
    done = set(map(tuple, pd.read_csv(OUT)[["linea", "perilo", "fold"]].values)) if OUT.exists() else set()
    for last in (2022, 2023):
        tr = df[df["year"] <= last]; va = df[df["year"] == last + 1]
        Xtr0, cats = M.to_frame(tr, base_cols); Xva0, _ = M.to_frame(va, base_cols, cats)
        for p in perils:
            fold = f"F{last + 1}"
            if (line, p, fold) in done:
                continue
            t0 = time.time()
            ft = float(dl[dl["peril"] == p]["freq_trend"].iloc[0])
            n = tr[f"n_{p}"].to_numpy(float); nv = va[f"n_{p}"].to_numpy(float)
            off = np.log(tr["exposure"].to_numpy() * np.clip(tr[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - tr["year"].to_numpy()) * np.log1p(ft)
            off_v = np.log(va["exposure"].to_numpy() * np.clip(va[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - va["year"].to_numpy()) * np.log1p(ft)
            base = np.log(max(n.sum(), 1) / np.exp(off).sum())
            Xtr, Xva, gl_tr, gl_va = Xtr0, Xva0, tr, va
            if has_mm:
                te_tr, te_va = M._te_make_model(tr, va, n, np.exp(off + base), k=30.0)
                Xtr = Xtr0.assign(mm_freq_te=te_tr); Xva = Xva0.assign(mm_freq_te=te_va)
                gl_tr = tr.assign(mm_freq_te=te_tr); gl_va = va.assign(mm_freq_te=te_va)
            # GBM as in sub_40
            cfg = dict(M.FREQ_CHOSEN.get(line, {}).get(p, M.FREQ_CHOSEN.get(line, {}).get("*", {})))
            mult = cfg.pop("_mult", 4)
            params = dict(M.FREQ_PARAMS, num_threads=4)
            if p == "subsidence":
                params.update(num_leaves=3, max_depth=2)
            params.update(cfg)
            rounds = (300 if n.sum() > 5000 else 150 if n.sum() > 1000 else 80) * mult
            m = lgb.train(params, lgb.Dataset(Xtr, label=n, init_score=off + base), num_boost_round=rounds)
            mu_g = np.exp(m.predict(Xva, raw_score=True) + off_v + base)
            # GLM
            gcols = base_cols + (["mm_freq_te"] if has_mm else [])
            A, B = design(gl_tr, gl_va, gcols, p)
            mu_l, alpha = glm_fit(A, n, off, B, off_v, tr["policy_id"].to_numpy())
            mu_m = np.sqrt(mu_g * mu_l)
            rows = []
            for name, mu in (("gbm", mu_g), ("glm", mu_l), ("mix50", mu_m)):
                mu = mu * nv.sum() / mu.sum()
                rows.append({"linea": line, "perilo": p, "fold": fold, "modello": name, "d2": poisson_d2(nv, mu),
                             "quota_costo": float(share.get(p, 0)), "sinistri_stima": int(n.sum()), "alpha": alpha})
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(line, p, fold, {r["modello"]: round(r["d2"], 5) for r in rows}, "alpha", alpha, f"{time.time() - t0:.0f}s", flush=True)


def poisson_d2(y, mu):
    return 1 - pdev(y, mu) / pdev(y, np.full(len(y), y.mean()))


if __name__ == "__main__":
    run(sys.argv[1])
