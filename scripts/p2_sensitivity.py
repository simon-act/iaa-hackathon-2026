"""Parte 2 (25/10): sensibilità del livello 2025 alle ipotesi. Il livello del modello = Σ target_b / Σ prior sul train, quindi
il tasso previsto si muove in proporzione a Σ target_b (per anno di esposizione, 2023+2024 pesati per esposizione).
Ricalcolo dalla tabella di sviluppo (development_onlevel) con un'ipotesi cambiata alla volta.
Uso: python3 scripts/p2_sensitivity.py → results_p2/sensitivity.csv"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_parquet(ROOT / "data/interim_p2/development_onlevel.parquet")
lv = pd.read_parquet(ROOT / "data/interim_p2/level_2025.parquet")
h = 2025 - d.ay
ex = d.groupby(["line", "ay"]).exposure.first().groupby("line").sum()


def rate(cost, ll="base"):
    r = cost.groupby(d.line).sum() / ex
    l = {"base": lv.ll_load_2025, "own": lv.ll_load_own, "zero": lv.ll_load_2025 * 0, "dev": lv.ll_load_prior_src}[ll]
    return r + l.reindex(r.index)


tf = (1 + d.freq_trend) ** h * (1 + d.sev_trend) ** h
base_cost = d.cost_ult * tf
V = {
    "base (come nel modello)": rate(base_cost),
    "trend = 0": rate(d.cost_ult),
    "trend severità +2 pt/anno": rate(d.cost_ult * (1 + d.freq_trend) ** h * (1 + d.sev_trend + 0.02) ** h),
    "trend frequenza −2 pt/anno": rate(d.cost_ult * (1 + d.freq_trend - 0.02) ** h * (1 + d.sev_trend) ** h),
    "nessun IBNER (severità dei denunciati come osservata)": rate(base_cost / d.ibner_factor),
    "nessun IBNR né IBNER (solo denunciato)": rate(d.cost_rep * tf),
    "IBNR lesioni 2024 +25%": rate(base_cost + np.where((d.peril == "bodily_injury") & (d.ay == 2024), 0.25 * (d.cost_ult - d.cost_rep) * tf, 0)),
    "IBNR lesioni 2024 −25%": rate(base_cost - np.where((d.peril == "bodily_injury") & (d.ay == 2024), 0.25 * (d.cost_ult - d.cost_rep) * tf, 0)),
    "solo 2024 (peso 0 al 2023)": (d[d.ay == 2024].assign(c=base_cost).groupby("line").c.sum() / d[d.ay == 2024].groupby("line").exposure.first()) + lv.ll_load_2025,
    "solo 2023": (d[d.ay == 2023].assign(c=base_cost).groupby("line").c.sum() / d[d.ay == 2023].groupby("line").exposure.first()) + lv.ll_load_2025,
    "grandi sinistri: carico proprio emerging": rate(base_cost, "own"),
    "grandi sinistri: carico del developed (Z=0)": rate(base_cost, "dev"),
    "grandi sinistri: nessun carico": rate(base_cost, "zero"),
}
t = pd.DataFrame(V).T
b = t.iloc[0]
pct = (t / b - 1) * 100
out = t.round(1).astype(str) + " (" + pct.round(1).map(lambda x: f"{x:+.1f}%") + ")"
pd.set_option("display.width", 200)
print(out.to_string())
t.assign(**{f"{c}_pct": pct[c] for c in t.columns}).to_csv(ROOT / "results_p2/sensitivity.csv")
