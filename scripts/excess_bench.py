"""Bench for the RELATIVE DEDUCTIBLE in per-peril frequency (22/10 evening, Simone's question: model gross and apply the deductible afterwards?).
Diagnosis: amounts do not depend on the deductible; frequency does, even in TPL (selection effect), but in own damage and glass
the effect fades with vehicle value (GLM, interaction z 4.6 and 3.1; TPL 0.9). The freq×sev is additive (1 split per tree) and does not capture it.
Same setup as weather_bench.py. Variants: base; xsrel = + (voluntary + compulsory deductible) ÷ insured value
(vehicle value class for motor, sum insured for home).
Usage: IAA_MARKET=configs/parte1_s3.json python3 -I scripts/excess_bench.py <line> ; output results_s3/excess_bench.csv"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import prepare as P  # noqa: E402
import models_extra as M  # noqa: E402
import freq_bench as FB  # noqa: E402

PERILS = {"motor": ["own_damage", "glass_hail", "theft", "third_party_prop"], "buildings": ["accidental_damage", "escape_of_water", "storm"],
          "contents": ["accidental_damage", "escape_of_water", "theft"]}
COMP = {"motor": 250, "buildings": 350, "contents": 350}
OUT = P.RESULTS / "excess_bench.csv"


def poisson_d2(y, mu):
    mu = np.clip(mu, 1e-12, None)
    dev = lambda m: 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / m), 0) - (y - m))
    return 1 - dev(mu) / dev(np.full(len(y), y.mean()))


def run(line):
    df = FB.load(line)
    scale = df["vehicle_value_group"].astype(float) if line == "motor" else df["sum_insured"].astype(float)
    df["xs_rel"] = (df["voluntary_excess"].astype(float) + COMP[line]) / scale.clip(lower=1e-6)
    dev, _, _ = M._tables(); dl = dev[dev["line"] == line]
    cols = M._features(df, line)
    cols = [c for c in cols if c != "xs_rel"]
    if line != "motor" and "prem_rel" not in cols:
        cols = cols + ["prem_rel"]
    has_mm = "make_model" in cols
    base_cols = [c for c in cols if not (has_mm and c == "make_model")]
    df["val_q"] = pd.qcut(scale.rank(method="first"), 5, labels=False)
    # noxs = no deductible; duefasi = "gross then net": model without deductible, then deductible factor × value class
    # estimated on train (A/E per cell, credibility with 50 claims) and applied afterwards
    VAR = {"base": ([], {}), "xsrel": (["xs_rel"], {}), "noxs": ([], {}), "duefasi": ([], {})}
    if len(sys.argv) > 2:
        VAR = {k: v for k, v in VAR.items() if k in sys.argv[2].split(",")}
    rows = []
    for last in (2022, 2023):
        tr = df[df["year"] <= last]; va = df[df["year"] == last + 1]
        for p in PERILS[line]:
            ft = float(dl[dl["peril"] == p]["freq_trend"].iloc[0])
            n = tr[f"n_{p}"].to_numpy(float); nv = va[f"n_{p}"].to_numpy(float)
            off = np.log(tr["exposure"].to_numpy() * np.clip(tr[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - tr["year"].to_numpy()) * np.log1p(ft)
            off_v = np.log(va["exposure"].to_numpy() * np.clip(va[f"pi_{p}"].to_numpy(), 1e-3, None)) - (2025 - va["year"].to_numpy()) * np.log1p(ft)
            base = np.log(max(n.sum(), 1) / np.exp(off).sum())
            cfg = dict(M.FREQ_CHOSEN.get(line, {}).get(p, M.FREQ_CHOSEN.get(line, {}).get("*", {})))
            mult = cfg.pop("_mult", 4)
            res = {}
            for name, (extra, prm) in VAR.items():
                t0 = time.time()
                c2 = [c for c in base_cols if c != "voluntary_excess"] if name in ("noxs", "duefasi") else base_cols + extra
                Xtr, cats = M.to_frame(tr, c2); Xva, _ = M.to_frame(va, c2, cats)
                if has_mm:
                    te_tr, te_va = M._te_make_model(tr, va, n, np.exp(off + base), k=30.0)
                    Xtr = Xtr.assign(mm_freq_te=te_tr); Xva = Xva.assign(mm_freq_te=te_va)
                params = dict(M.FREQ_PARAMS, num_threads=4)
                if p == "subsidence":
                    params.update(num_leaves=3, max_depth=2)
                params.update(cfg); params.update(prm)
                rounds = (300 if n.sum() > 5000 else 150 if n.sum() > 1000 else 80) * mult
                m = lgb.train(params, lgb.Dataset(Xtr, label=n, init_score=off + base), num_boost_round=rounds)
                mu = np.exp(m.predict(Xva, raw_score=True) + off_v + base)
                if name == "duefasi":
                    mu_tr = np.exp(m.predict(Xtr, raw_score=True) + off + base)
                    g = pd.DataFrame({"x": tr["voluntary_excess"].to_numpy(), "q": tr["val_q"].to_numpy(), "n": n, "m": mu_tr}).groupby(["x", "q"]).sum()
                    fac = (g.n + 50) / (g.m + 50)
                    mu = mu * fac.reindex(pd.MultiIndex.from_arrays([va["voluntary_excess"].to_numpy(), va["val_q"].to_numpy()])).fillna(1).to_numpy()
                lvl = nv.sum() / mu.sum(); mu_l = mu * lvl
                res[name] = poisson_d2(nv, mu_l)
                rows.append({"linea": line, "perilo": p, "fold": f"F{last + 1}", "variante": name, "d2": res[name], "AE_livello": lvl,
                             "sec": round(time.time() - t0, 1)})
            print(line, p, f"F{last + 1}", {k: round((v - res['base']) * 1e4, 1) for k, v in res.items()}, "base level AE", round(rows[-len(VAR)]["AE_livello"], 3), flush=True)
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False); rows = []


if __name__ == "__main__":
    run(sys.argv[1])
