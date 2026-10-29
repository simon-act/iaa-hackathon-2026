"""Test bench for the per-peril FREQUENCY of the freq×sev (20/10).
Replicates the frequency part of models_extra._fs2 (offset = log(exposure × completeness) − trend, make_model
encoding, 3-leaf subsidence, iterations = 300/150/80 × mult depending on claim counts) and evaluates it on its own:
Poisson D² of claims REPORTED in the following year (fold 2023: fit 2021–22; fold 2024: fit 2021–23), prediction =
exp(margin + offset of the evaluated year), level realigned by peril and fold (only the shape is judged).
No spread IBNR: counts are compared with the expected completeness. Holdout excluded.
Usage: IAA_MARKET=configs/parte1_s3.json python3 -I scripts/freq_bench.py <line> <config1,config2,...>
Configurations: see CONFIGS. Output: results_s3/freq_bench_<line>.csv (appended)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare as P  # noqa: E402
import models_extra as M  # noqa: E402

CONFIGS = {
    "base": {},
    "giri_x0.5": {"_mult": 2}, "giri_x2": {"_mult": 8},
    "l2_1": {"lambda_l2": 1.0}, "l2_50": {"lambda_l2": 50.0},
    "minleaf_300": {"min_data_in_leaf": 300}, "minleaf_3000": {"min_data_in_leaf": 3000},
    "lr0.05_giri_x2": {"learning_rate": 0.05, "_mult": 8},
    "prof2": {"num_leaves": 3, "max_depth": 2}, "prof3": {"num_leaves": 7, "max_depth": 3, "learning_rate": 0.05, "_mult": 4},
    "ff0.7": {"feature_fraction": 0.7}, "nobag": {"bagging_fraction": 1.0, "bagging_freq": 0},
    "es": {"_es": True}, "no_mm_te": {"_no_mm": True}, "mm_te_k10": {"_mm_k": 10.0}, "mm_te_k100": {"_mm_k": 100.0},
    "path_smooth50": {"path_smooth": 50.0}, "maxbin255": {"max_bin": 255},
    "ff0.7_maxbin255": {"feature_fraction": 0.7, "max_bin": 255}, "ff0.5": {"feature_fraction": 0.5},
    "ff0.7_maxbin255_giri_x2": {"feature_fraction": 0.7, "max_bin": 255, "_mult": 8},
    "tweedie1.1": {"objective": "tweedie", "tweedie_variance_power": 1.1},
    "tweedie1.3": {"objective": "tweedie", "tweedie_variance_power": 1.3},
}


DONE = set()
DONE_PERILS = {}


def poisson_d2(y, mu):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-12, None)
    def dev(m):
        return 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / m), 0) - (y - m))
    return 1 - dev(mu) / dev(np.full(len(y), y.mean()))


def load(line):
    names = pq.ParquetFile(P.MODEL_DIR / f"model_{line}.parquet").schema.names
    feats = M._features(pd.DataFrame({c: [0, 1] for c in names}), line)
    extra = ["row_id", "policy_id", "year", "exposure", "written_premium", "discount_applied", "flag_mileage_zero", "make_model"]
    cols = list(dict.fromkeys([c for c in feats + extra if c in names] + [c for c in names if c.startswith(("n_", "pi_"))]))
    df = pd.read_parquet(P.MODEL_DIR / f"model_{line}.parquet", columns=cols)
    df = df[~P.is_holdout(df["policy_id"])].reset_index(drop=True)
    if line == "motor":
        df = M._mil0(df)
    else:
        df = M._prem(df)
    return df


def run(line, cfgs):
    global DONE
    fpart = P.RESULTS / f"freq_bench_{line}.csv"
    DONE = set(map(tuple, pd.read_csv(fpart)[["perilo", "fold", "config"]].values)) if fpart.exists() else set()
    if all((p, f, c) in DONE for p in DONE_PERILS.get(line, []) for f in ("F2023", "F2024") for c in cfgs) and DONE_PERILS.get(line):
        return None
    df = load(line)
    dev, _, _ = M._tables(); dl = dev[dev["line"] == line]
    perils = sorted(dl["peril"].unique())
    share = dl[dl["ay"] >= 2023].groupby("peril")["cost_ult"].sum(); share = share / share.sum()
    cols = M._features(df, line)
    if line != "motor":
        cols = cols + ["prem_rel"] if "prem_rel" not in cols else cols
    has_mm = "make_model" in cols
    base_cols = [c for c in cols if not (has_mm and c == "make_model")]
    out = []
    for last in (2022, 2023):
        tr = df[df["year"] <= last]; va = df[df["year"] == last + 1]
        Xtr0, cats = M.to_frame(tr, base_cols); Xva0, _ = M.to_frame(va, base_cols, cats)
        for p in perils:
            ft = float(dl[dl["peril"] == p]["freq_trend"].iloc[0])
            n = tr[f"n_{p}"].to_numpy(float); nv = va[f"n_{p}"].to_numpy(float)
            off = np.log(tr["exposure"].to_numpy() * np.clip(tr[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - tr["year"].to_numpy()) * np.log1p(ft)
            off_v = np.log(va["exposure"].to_numpy() * np.clip(va[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - va["year"].to_numpy()) * np.log1p(ft)
            base = np.log(max(n.sum(), 1) / np.exp(off).sum())
            for cname in cfgs:
                if (p, f"F{last + 1}", cname) in DONE:
                    continue
                cfg = dict(CONFIGS[cname]); t0 = time.time()
                mult = cfg.pop("_mult", 4); es = cfg.pop("_es", False); no_mm = cfg.pop("_no_mm", False); mmk = cfg.pop("_mm_k", 30.0)
                Xtr, Xva = Xtr0, Xva0
                if has_mm and not no_mm:
                    te_tr, te_va = M._te_make_model(tr, va, n, np.exp(off + base), k=mmk)
                    Xtr = Xtr0.assign(mm_freq_te=te_tr); Xva = Xva0.assign(mm_freq_te=te_va)
                params = dict(M.FREQ_PARAMS, num_threads=4)
                if p == "subsidence" and cfg.get("num_leaves", 2) <= 3:
                    params.update(num_leaves=3, max_depth=2)
                params.update(cfg)
                rounds = (300 if n.sum() > 5000 else 150 if n.sum() > 1000 else 80) * mult
                if es:
                    rounds = M._rounds_es(Xtr, n, off + base, tr["year"].to_numpy(), params) or rounds
                m = lgb.train(params, lgb.Dataset(Xtr, label=n, init_score=off + base), num_boost_round=rounds)
                mu = np.exp(m.predict(Xva, raw_score=True) + off_v + base); mu *= nv.sum() / mu.sum()
                d2 = poisson_d2(nv, mu)
                row = {"linea": line, "perilo": p, "fold": f"F{last + 1}", "config": cname, "d2": d2, "quota_costo": float(share.get(p, 0)),
                       "giri": rounds, "sec": round(time.time() - t0, 1)}
                out.append(row)
                fpart = P.RESULTS / f"freq_bench_{line}.csv"     # save row by row (the container restarts)
                pd.DataFrame([row]).to_csv(fpart, mode="a", header=not fpart.exists(), index=False)
                print(line, p, f"F{last + 1}", cname, round(d2, 5), rounds, f"{time.time() - t0:.0f}s", flush=True)
    o = pd.read_csv(fpart).drop_duplicates(["perilo", "fold", "config"], keep="last")
    o.to_csv(fpart, index=False)
    return o


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].split(","))
