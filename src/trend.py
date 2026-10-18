"""Passo 1d — trend di frequenza e severità e livello 2025 per linea × perilo.

Regressione log-lineare su AY degli ultimate sviluppati; pesi: frequenza n_ult*pi^2, severità n_ult*quota chiusi.
Un trend non significativo (p > 0,10) viene messo a zero. Perili con severità instabile
(p > 0,10) usano la mediana dei trend significativi della linea.
Livello 2025 = media pesata degli AY riportati al 2025 con il trend.
Uso: python -I trend.py <interim_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, TARGET_YEAR  # noqa: E402
P_MAX = 0.10
# perili guidati dal meteo: la frequenza segue le variabili meteo dell'anno (note anche per il 2025),
# quindi niente trend temporale nel livello base: lo sposta il modello con weather 2025
WEATHER_PERILS = CFG["weather_perils"]


def wls(x, y, w):
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    res = y - X @ beta
    dof = len(x) - 2
    s2 = (w * res ** 2).sum() / dof
    cov = s2 * np.linalg.inv(X.T @ W @ X)
    t = beta[1] / np.sqrt(cov[1, 1])
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    return float(beta[1]), float(p)


def main(interim: str) -> None:
    interim = Path(interim)
    d = pd.read_parquet(interim / "development.parquet")
    # pesi: frequenza ~ completezza^2 (gli AY con molto IBNR contano meno), severità ~ quota chiusi
    d["w_freq"] = d["n_ult"] * d["pi"] ** 2
    d["w_sev"] = d["n_ult"] * (1 - d["open_share"]).clip(lower=0.05)
    rows = []
    for (line, peril), g in d.groupby(["line", "peril"]):
        bf, pf = wls(g["ay"].to_numpy(float), np.log(g["freq_ult"].to_numpy()), g["w_freq"].to_numpy())
        bs, ps = wls(g["ay"].to_numpy(float), np.log(g["sev_ult"].to_numpy()), g["w_sev"].to_numpy())
        rows.append(dict(line=line, peril=peril, freq_slope=bf, freq_p=pf, sev_slope=bs, sev_p=ps))
    t = pd.DataFrame(rows)
    t["freq_trend"] = np.where(t["freq_p"] <= P_MAX, np.expm1(t["freq_slope"]), 0.0)
    t["freq_rule"] = np.where(t["freq_p"] <= P_MAX, "stimato", "zero (non significativo)")
    # trend di linea (perili non meteo insieme) per i perili con stima propria non significativa
    nw = d[[(l, p) not in WEATHER_PERILS for l, p in zip(d["line"], d["peril"])]]
    for line, g in nw.groupby("line"):
        a = g.groupby("ay").agg(n=("n_ult", "sum"), e=("exposure", "first"), pi=("pi", "mean"))
        bl, pl = wls(a.index.to_numpy(float), np.log((a["n"] / a["e"]).to_numpy()), (a["n"] * a["pi"] ** 2).to_numpy())
        use = (t["line"] == line) & (t["freq_p"] > P_MAX) & (pl <= P_MAX)
        t.loc[use, "freq_trend"] = np.expm1(bl)
        t.loc[use, "freq_rule"] = f"trend di linea (p={pl:.3f})"
    wmask = [(l, p) in WEATHER_PERILS for l, p in zip(t["line"], t["peril"])]
    t.loc[wmask, "freq_trend"] = 0.0
    t.loc[wmask, "freq_rule"] = "zero (perilo meteo: lo gestisce il modello con weather)"
    sig = t[t["sev_p"] <= P_MAX]
    line_med = sig.groupby("line")["sev_slope"].median()
    t["sev_trend"] = np.where(t["sev_p"] <= P_MAX, np.expm1(t["sev_slope"]),
                              np.expm1(t["line"].map(line_med).fillna(0.05)))
    t["sev_rule"] = np.where(t["sev_p"] <= P_MAX, "stimato", "mediana della linea")

    d = d.merge(t[["line", "peril", "freq_trend", "sev_trend"]], on=["line", "peril"])
    h = TARGET_YEAR - d["ay"]
    d["freq_onlevel"] = d["freq_ult"] * (1 + d["freq_trend"]) ** h
    d["sev_onlevel"] = d["sev_ult"] * (1 + d["sev_trend"]) ** h
    d["bc_onlevel"] = d["freq_onlevel"] * d["sev_onlevel"]
    d["wt_level"] = d["exposure"] * (1 - 0.5 * d["open_share"])
    lvl = (d.groupby(["line", "peril"])
           .apply(lambda g: pd.Series({
               "freq_2025": np.average(g["freq_onlevel"], weights=g["wt_level"]),
               "sev_2025": np.average(g["sev_onlevel"], weights=g["wt_level"] * g["n_ult"]),
           }), include_groups=False).reset_index())
    lvl["bc_2025"] = lvl["freq_2025"] * lvl["sev_2025"]
    t = t.merge(lvl, on=["line", "peril"])

    # caricamento large loss per linea: eccedenza per esposizione, riportata al 2025 con la
    # mediana del trend di severità della linea
    ll = pd.read_parquet(interim / "large_loss.parquet")
    sev_line = t.groupby("line")["sev_trend"].median()
    ll["excess_onlevel"] = ll["excess_per_expo"] * (1 + ll["line"].map(sev_line)) ** (TARGET_YEAR - ll["ay"])
    ll_load = ll.groupby("line").apply(lambda g: np.average(g["excess_onlevel"], weights=g["exposure"]),
                                       include_groups=False).rename("ll_load_2025")
    line_bc = t.groupby("line")["bc_2025"].sum().rename("bc_2025_capped").to_frame().join(ll_load)
    # Parte 2 — credibilità sul carico large loss: in un mercato piccolo pochi sinistri enormi dominano il carico
    # (visto sul mercato finto: un sinistro da 16M = 32% del premio motor). Z = n_large / (n_large + K), prior = carico
    # del mercato di partenza riscalato al livello (tagliato) del mercato nuovo.
    src_cfg = CFG["transfer"].get("source_market")
    line_bc["ll_load_own"] = line_bc["ll_load_2025"]
    if src_cfg:
        import market_config as MC
        src = pd.read_parquet(Path(MC.load(src_cfg)["interim"]) / "level_2025.parquet")
        cl = pd.read_parquet(interim / "claims_clean.parquet", columns=["line", "is_large"])
        n_large = cl.groupby("line")["is_large"].sum().reindex(line_bc.index).fillna(0)
        K = float(CFG["transfer"].get("ll_credibility_k", 50))
        Z = n_large / (n_large + K)
        prior = src["ll_load_2025"].reindex(line_bc.index) * line_bc["bc_2025_capped"] / src["bc_2025_capped"].reindex(line_bc.index)
        line_bc["ll_load_prior_src"] = prior; line_bc["ll_Z"] = Z; line_bc["n_large"] = n_large
        line_bc["ll_load_2025"] = Z * line_bc["ll_load_own"] + (1 - Z) * prior
    line_bc["bc_2025_total"] = line_bc["bc_2025_capped"] + line_bc["ll_load_2025"]

    d.to_parquet(interim / "development_onlevel.parquet", index=False)
    t.to_parquet(interim / "trend.parquet", index=False)
    line_bc.to_parquet(interim / "level_2025.parquet")
    (interim / "report_trend.json").write_text(json.dumps({
        "trend": t.round(4).to_dict(orient="records"),
        "level_2025_by_line": line_bc.round(2).reset_index().to_dict(orient="records")}, indent=2))
    pd.set_option("display.width", 220)
    print(t.round(4).to_string())
    print(line_bc.round(2))


if __name__ == "__main__":
    main(sys.argv[1])
