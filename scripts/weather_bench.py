"""Banco del METEO nella frequenza per perilo (22/10, idea di Simone: il meteo è probabilmente importante).
Stesso impianto di freq_bench.py / glm_peril_bench.py (offset log(esposizione × completezza) − trend; fold 2023 e 2024;
D² Poisson sui sinistri denunciati dell'anno dopo, livello riallineato per perilo e fold), GBM = frequenza di sub_40.
Varianti (in più delle variabili attuali):
  base   = come oggi;
  anom   = + anomalia dell'anno: meteo dell'anno − media dell'indirizzo 2021–25 (clima vs shock dell'anno);
  lag    = + meteo dell'anno prima allo stesso indirizzo (siccità → cedimento con ritardo, inverni freddi → acqua);
  anomlag= entrambe;
  linear = linear_tree=True (il meteo continua a pesare oltre il range visto).
Diagnosi per la base: A/E nell'anno dopo per decile di ogni variabile meteo (attenuazione?).
Uso: IAA_MARKET=configs/parte1_s3.json python3 -I scripts/weather_bench.py <linea> ; output results_s3/weather_bench.csv, weather_ae.csv"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import prepare as P  # noqa: E402
import models_extra as M  # noqa: E402
import freq_bench as FB  # noqa: E402

WV = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm", "mean_temperature_c"]
PERILS = {"motor": ["glass_hail"], "buildings": ["flood", "storm", "subsidence", "escape_of_water", "fire"],
          "contents": ["flood", "escape_of_water"]}
OUT = P.RESULTS / "weather_bench.csv"; OUT_AE = P.RESULTS / "weather_ae2.csv"


def poisson_d2(y, mu):
    mu = np.clip(mu, 1e-12, None)
    dev = lambda m: 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / m), 0) - (y - m))
    return 1 - dev(mu) / dev(np.full(len(y), y.mean()))


def weather_extra():
    w = pd.read_parquet(Path(P.CFG["raw"]) / "weather.parquet")
    w = w.sort_values(["address_id", "year"])
    mean = w.groupby("address_id")[WV].transform("mean")
    an = (w[WV] - mean).add_prefix("anom_")
    lag = w.groupby("address_id")[WV].shift(1).add_prefix("lag_")
    return pd.concat([w[["address_id", "year"]], an, lag], axis=1)


def run(line):
    df = FB.load(line)
    ids = pd.read_parquet(P.MODEL_DIR / f"model_{line}.parquet", columns=["row_id", "address_id"])
    df = df.merge(ids, on="row_id", how="left").merge(weather_extra(), on=["address_id", "year"], how="left")
    # 22/10: media del meteo per regione e anno (gli eventi sono regionali; l'indice per indirizzo è rumoroso)
    rg = df.groupby(["region", "year"]).apply(lambda z: pd.Series({f"reg_{v}": np.average(z[v], weights=z["exposure"]) for v in WV})).reset_index()
    df = df.merge(rg, on=["region", "year"], how="left")
    dev, _, _ = M._tables(); dl = dev[dev["line"] == line]
    cols = M._features(df, line)
    cols = [c for c in cols if not c.startswith(("anom_", "lag_", "reg_"))]
    if line != "motor" and "prem_rel" not in cols:
        cols = cols + ["prem_rel"]
    has_mm = "make_model" in cols
    base_cols = [c for c in cols if not (has_mm and c == "make_model")]
    A = [f"anom_{v}" for v in WV]; L = [f"lag_{v}" for v in WV]
    G = [f"reg_{v}" for v in WV]
    VAR = {"base": ([], {}), "anom": (A, {}), "lag": (L, {}), "anomlag": (A + L, {}), "linear": ([], {"linear_tree": True, "linear_lambda": 1.0}),
           "reg": (G, {}), "reglin": (G, {"linear_tree": True, "linear_lambda": 1.0})}
    if len(sys.argv) > 2:
        VAR = {k: v for k, v in VAR.items() if k in sys.argv[2].split(",")}
    rows, aes = [], []
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
                c2 = base_cols + extra
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
                lvl = nv.sum() / mu.sum(); mu_l = mu * lvl
                res[name] = poisson_d2(nv, mu_l)
                rows.append({"linea": line, "perilo": p, "fold": f"F{last + 1}", "variante": name, "d2": res[name], "AE_livello": lvl,
                             "sec": round(time.time() - t0, 1)})
                if name == "base":
                    for v in WV:
                        q = pd.qcut(va[v].rank(method="first"), 10, labels=False)
                        g = pd.DataFrame({"q": q.to_numpy(), "n": nv, "e": mu_l, "v": va[v].to_numpy()}).groupby("q").agg(n=("n", "sum"), e=("e", "sum"), v=("v", "mean"))
                        for qq, r in g.iterrows():
                            aes.append({"linea": line, "perilo": p, "fold": f"F{last + 1}", "var": v, "decile": int(qq), "v_medio": r.v, "AE": r.n / r.e, "n": r.n})
            print(line, p, f"F{last + 1}", {k: round((v - res['base']) * 1e4, 1) for k, v in res.items()}, "AE livello base", round(rows[-len(VAR)]["AE_livello"], 3), flush=True)
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False); rows = []
    pd.DataFrame(aes).to_csv(OUT_AE, mode="a", header=not OUT_AE.exists(), index=False)


if __name__ == "__main__":
    run(sys.argv[1])
