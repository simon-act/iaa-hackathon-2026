"""Parte 2 (24/10): estrapolazione del meteo. Nel mercato emerging il clima esce dal range del developed (giorni di grandine
fino a ~40 contro p99 = 19) e il prior (GBM a 1 split) resta piatto oltre il range: sul 2023–24 l'A/E dei cristalli/grandine
sale da 0,86 a 1,14 con i giorni di grandine. Correzione: per un perilo meteo, frequenza del prior × exp(β·(x − media)),
β stimato sui dati emerging con un Poisson a una variabile e offset = numero atteso dal prior; il livello del perilo non cambia.
Fold: β sul 2023, valutazione sul 2024 (forma / senza spalmatura / grandi sinistri veri), con prior × livello e con genere.
Uso: IAA_MARKET=configs/parte2.json python3 scripts/p2_weather.py [scrivi]   ('scrivi' = salva source_pred_<linea>_meteo.parquet con β su 2023+24)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, transfer as TR  # noqa: E402

CAND = {"motor": [("glass_hail", "hail_days")], "buildings": [("storm", "wind_exposure_index"), ("flood", "rainfall_mm")],
        "contents": [("flood", "rainfall_mm")]}
dev = pd.read_parquet(Path(P.CFG["interim"]) / "development_onlevel.parquet")


def fit_beta(df, s, per, var):
    ft = float(dev[(dev.line == df.attrs["line"]) & (dev.peril == per)].freq_trend.iloc[0])
    e = s[f"f_{per}"].to_numpy() * df[f"pi_{per}"].to_numpy() * (1 + ft) ** (-(2025 - df["year"].to_numpy()))
    n = df[f"n_{per}"].to_numpy(float); x = df[var].to_numpy(float); xm = float(np.average(x, weights=e))
    e = e * n.sum() / e.sum()
    nll = lambda b: -(np.sum(n * b * (x - xm)) - np.log(np.sum(e * np.exp(b * (x - xm)))) * n.sum())
    b = minimize_scalar(nll, bounds=(-1, 1), method="bounded").x
    return float(b), xm, n.sum()


def adjust(s, df_rows, per, var, b, xm, e_ref_norm):
    m = np.exp(b * (df_rows[var].to_numpy(float) - xm)) / e_ref_norm
    out = s.copy()
    extra = s[f"f_{per}"].to_numpy() * s[f"s_{per}"].to_numpy() * (m - 1) * s["c_src"].to_numpy()
    out["mu_src"] = s["mu_src"].to_numpy() + extra
    out[f"f_{per}"] = s[f"f_{per}"].to_numpy() * m
    return out


def run_fold():
    rows = []
    for l, cands in CAND.items():
        df = P.load(l); df.attrs["line"] = l
        tr, va = df[df.year == 2023], df[df.year == 2024]
        s_all = pd.read_parquet(P.RESULTS / f"source_pred_{l}.parquet").set_index("row_id")
        y = va[P.TARGET].to_numpy()
        base_s = s_all.copy()
        preds = {"base": TR.src_level(tr, va, l)}
        for per, var in cands:
            st = s_all.reindex(tr.row_id)
            b, xm, nn = fit_beta(tr, st, per, var)
            # normalizzazione: la media pesata del moltiplicatore sul train = 1 (livello del perilo invariato)
            w = st[f"f_{per}"].to_numpy(); mt = np.exp(b * (tr[var].to_numpy(float) - xm)); norm = float(np.sum(w * mt) / np.sum(w))
            both = pd.concat([tr, va])
            adj = adjust(s_all.reindex(both.row_id), both, per, var, b, xm, norm)
            TR._PRIOR_OVERRIDE = adj
            preds[f"{per}~{var} (β={b:+.3f})"] = TR.src_level(tr, va, l)
            TR._PRIOR_OVERRIDE = None
            print(l, per, var, "β", round(b, 4), "sinistri", int(nn), flush=True)
        for k, p in preds.items():
            rows.append(dict(linea=l, variante=k, forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va.row_id, p), unc=P.d2_unc(va.row_id, p)))
    r = pd.DataFrame(rows)
    b = r[r.variante == "base"].set_index("linea")
    for c in ("forma", "own", "unc"):
        r[f"d_{c}_e4"] = (r[c] - r.linea.map(b[c])) * 1e4
    pd.set_option("display.width", 200)
    print(r.round(4).to_string(index=False)); r.to_csv(P.RESULTS / "p2_weather.csv", index=False)


def write_final(keep):
    for l, cands in CAND.items():
        s_all = pd.read_parquet(P.RESULTS / f"source_pred_{l}.parquet").set_index("row_id")
        df = P.load(l); df.attrs["line"] = l; te = P.load(l, "test")
        both = pd.concat([df, te]); adj = s_all.reindex(both.row_id)
        for per, var in cands:
            if (l, per) not in keep:
                continue
            st = s_all.reindex(df.row_id)
            b, xm, nn = fit_beta(df, st, per, var)
            w = st[f"f_{per}"].to_numpy(); mt = np.exp(b * (df[var].to_numpy(float) - xm)); norm = float(np.sum(w * mt) / np.sum(w))
            adj = adjust(adj, both, per, var, b, xm, norm)
            print("finale", l, per, var, "β", round(b, 4), flush=True)
        adj.reset_index().to_parquet(P.RESULTS / f"source_pred_{l}_meteo.parquet", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scrivi":
        write_final({tuple(x.split("|")) for x in sys.argv[2].split(",")})
    else:
        run_fold()
