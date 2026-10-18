"""Passo 1c — sviluppo a ultimate per linea × perilo × anno di accadimento (valutazione 31/12/2024).

Componenti:
  1. ritardo di denuncia F(l) = P(lag <= l) stimato con lo stimatore a tempo inverso per
     dati troncati a destra (un sinistro accaduto il giorno d è visibile solo se lag <= V-d)
  2. completezza attesa pi per riga di polizza e per AY = media di F(V-d) sui giorni coperti
  3. IBNR numero = reported / pi - reported
  4. severità dei tardivi: relatività rel(w) = E[sev | lag > w] / E[sev] su AY maturi
  5. recuperi futuri sugli aperti: tasso osservato sui chiusi dello stesso perilo
  6. IBNER: credibilità tra severità osservata (peso Z = quota chiusi) e severità a priori
     (livello degli AY maturi riportato con l'inflazione del perilo o della linea)
Uso: python -I develop.py <interim_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, LINES  # noqa: E402

V = None      # fissati in main dagli anni di polizza
DAYS = None
YEARS = None
MATURE_OPEN_MAX = 0.10      # un AY è "maturo" per un perilo se gli aperti sono < 10%
MAX_LAG = 1500


def reporting_cdf(lag: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Stimatore a tempo inverso (Lagakos et al. 1988) per lag troncati a destra.
    g(l) = #{lag = l} / #{lag <= l <= window};  F(l) = prod_{k > l} (1 - g(k))."""
    n = np.bincount(lag, minlength=MAX_LAG + 2)[: MAX_LAG + 2].astype(float)
    # rischio in tempo inverso a l: sinistri con lag <= l e finestra >= l
    lag_le = np.cumsum(n)                                    # #{lag <= l}
    w_lt = np.bincount(np.minimum(window, MAX_LAG + 1), minlength=MAX_LAG + 2)[: MAX_LAG + 2]
    # #{lag <= l, window < l}: window >= lag sempre, quindi = #{window < l}
    win_lt = np.concatenate([[0], np.cumsum(w_lt)[:-1]]).astype(float)
    risk = lag_le - win_lt
    g = np.divide(n, risk, out=np.zeros_like(n), where=risk > 0)
    surv = np.cumprod((1 - g)[::-1])[::-1]                   # prod_{k >= l}
    F = np.concatenate([surv[1:], [1.0]])                    # prod_{k > l}
    return np.clip(F, 0, 1)


def completeness_by_day(F: np.ndarray) -> pd.Series:
    w = (V - DAYS).days.to_numpy()
    return pd.Series(F[np.minimum(w, MAX_LAG + 1)], index=DAYS)


def row_completeness(ps: pd.Series, pe: pd.Series, comp_day: pd.Series) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(comp_day.to_numpy())])
    i0 = (ps - DAYS[0]).dt.days.to_numpy()
    i1 = (pe - DAYS[0]).dt.days.to_numpy() + 1
    i1 = np.minimum(i1, len(DAYS))
    return (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)


def main(interim: str) -> None:
    global V, DAYS, YEARS
    interim = Path(interim)
    _p = pd.concat([pd.read_parquet(interim / f"train_{l}.parquet", columns=["year"]) for l in LINES])
    YEARS = sorted(int(y) for y in _p["year"].unique())
    V = pd.Timestamp(year=YEARS[-1], month=12, day=31)
    DAYS = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-30")
    interim = Path(interim)
    c = pd.read_parquet(interim / "claims_clean.parquet")
    out_rows, curves, rel_curves, notes = [], {}, {}, {}

    for line in ("motor", "buildings", "contents"):
        pol = pd.read_parquet(interim / f"train_{line}.parquet",
                              columns=["row_id", "period_start", "period_end", "exposure", "year"])
        cl = c[c["line"] == line]
        comp_cols = {}
        # inflazione di linea dai perili a coda corta (AY maturi, solo chiusi)
        for peril, g in cl.groupby("peril"):
            # 1. ritardo di denuncia
            F = reporting_cdf(g["report_lag_days"].clip(0, MAX_LAG).to_numpy(),
                              g["days_to_valuation"].clip(0, MAX_LAG + 1).to_numpy())
            curves[f"{line}|{peril}"] = F[[0, 7, 14, 30, 60, 90, 180, 365, 730, 1095, 1460]].round(5).tolist()
            comp_day = completeness_by_day(F)
            comp_cols[peril] = row_completeness(pol["period_start"], pol["period_end"], comp_day)

            # 4. severità dei tardivi
            mat = g[g["ay"] <= 2022]
            s = mat.sort_values("report_lag_days")
            lag_s, sev_s = s["report_lag_days"].to_numpy(), s["net_capped"].to_numpy()
            tail_sum = np.cumsum(sev_s[::-1])[::-1]; tail_n = np.arange(len(sev_s), 0, -1)
            grid = np.arange(0, MAX_LAG + 2)
            idx = np.searchsorted(lag_s, grid, side="right")
            mean_all = sev_s.mean()
            rel = np.where(idx < len(sev_s), np.divide(tail_sum[np.minimum(idx, len(sev_s) - 1)],
                                                         tail_n[np.minimum(idx, len(sev_s) - 1)]) / mean_all, 1.0)
            # stabilizza la coda: almeno 30 sinistri oltre w, altrimenti ultimo valore stabile
            ok = (len(sev_s) - idx) >= 30
            last = rel[ok][-1] if ok.any() else 1.0
            rel = np.where(ok, rel, last)
            rel_curves[f"{line}|{peril}"] = rel[[0, 30, 90, 180, 365, 730]].round(3).tolist()

            # 5. recuperi futuri attesi sugli aperti
            closed = g[~g["is_open"]]
            rec_ratio = float(closed["recoveries"].sum() / closed["incurred_to_date"].sum()) if len(closed) else 0.0

            for ay in YEARS:
                gy = g[g["ay"] == ay]
                pa = pol[pol["year"] == ay]
                expo = float(pa["exposure"].sum())
                pi = float(np.average(comp_cols[peril][pol["year"].to_numpy() == ay], weights=pa["exposure"]))
                n_rep = len(gy)
                n_ult = n_rep / pi
                # costo osservato con recuperi futuri attesi sugli aperti
                opn = gy["is_open"].to_numpy()
                cost_rep = float(gy["net_capped"].sum() - rec_ratio * gy.loc[opn, "incurred_to_date"].clip(upper=gy.loc[opn, "net_capped"] / max(1 - rec_ratio, 1e-9)).sum())
                # relatività di severità per gli IBNR: media di rel(w) pesata per (1-F(w)) sui giorni dell'AY
                days_ay = DAYS[DAYS.year == ay]
                w = np.minimum((V - days_ay).days.to_numpy(), MAX_LAG + 1)
                wt = 1 - F[w]
                rel_ibnr = float(np.sum(wt * rel[w]) / wt.sum()) if wt.sum() > 0 else 1.0
                out_rows.append(dict(line=line, peril=peril, ay=ay, exposure=expo, n_rep=n_rep,
                                     open_share=float(opn.mean()) if n_rep else 0.0,
                                     pi=pi, n_ult=n_ult, cost_rep=cost_rep,
                                     sev_rep=cost_rep / n_rep if n_rep else np.nan,
                                     rel_ibnr=rel_ibnr, rec_ratio=rec_ratio))
        comp = pd.DataFrame(comp_cols); comp.insert(0, "row_id", pol["row_id"].to_numpy())
        comp.to_parquet(interim / f"completeness_{line}.parquet", index=False)

    d = pd.DataFrame(out_rows)
    # 6. IBNER: severità a priori dagli AY maturi + inflazione
    d["mature"] = d["open_share"] < MATURE_OPEN_MAX
    infl = {}
    for (line, peril), g in d.groupby(["line", "peril"]):
        m = g[g["mature"] & (g["n_rep"] > 50)]
        if len(m) >= 3:
            b = np.polyfit(m["ay"], np.log(m["sev_rep"]), 1, w=np.sqrt(m["n_rep"]))[0]
            infl[(line, peril)] = (float(np.exp(b) - 1), "perilo (AY maturi)")
    for line in d["line"].unique():
        own = [v[0] for (l, p), v in infl.items() if l == line]
        pooled = float(np.median(own)) if own else 0.05
        for peril in d.loc[d["line"] == line, "peril"].unique():
            infl.setdefault((line, peril), (pooled, "mediana della linea (perilo senza 3 AY maturi)"))
    d["inflation"] = [infl[(l, p)][0] for l, p in zip(d["line"], d["peril"])]
    d["inflation_source"] = [infl[(l, p)][1] for l, p in zip(d["line"], d["peril"])]

    prior = []
    for _, r in d.iterrows():
        g = d[(d["line"] == r["line"]) & (d["peril"] == r["peril"])]
        # riferimento: AY più maturo (minimo open share), riportato all'AY corrente
        ref = g.sort_values(["open_share", "ay"]).iloc[0] if not g["mature"].any() else g[g["mature"]].iloc[0]
        base = g[g["mature"]] if g["mature"].any() else g[g["ay"] == ref["ay"]]
        lvl = np.exp(np.average(np.log(base["sev_rep"]) + np.log1p(r["inflation"]) * (r["ay"] - base["ay"]),
                                weights=base["n_rep"]))
        prior.append(lvl)
    d["sev_prior"] = prior
    d["Z"] = 1 - d["open_share"]
    # severità ultimate dei denunciati (IBNER) e degli IBNR (relatività dei tardivi)
    d["sev_rep_ult"] = d["Z"] * d["sev_rep"] + (1 - d["Z"]) * d["sev_prior"]
    d["ibner_factor"] = d["sev_rep_ult"] / d["sev_rep"]
    d["n_ibnr"] = d["n_ult"] - d["n_rep"]
    d["cost_ult"] = d["n_rep"] * d["sev_rep_ult"] + d["n_ibnr"] * d["sev_rep_ult"] * d["rel_ibnr"]
    d["sev_ult"] = d["cost_ult"] / d["n_ult"]
    d["freq_ult"] = d["n_ult"] / d["exposure"]
    d["bc_ult"] = d["cost_ult"] / d["exposure"]
    d["bc_rep"] = d["cost_rep"] / d["exposure"]
    d["dev_factor"] = d["cost_ult"] / d["cost_rep"]
    d.to_parquet(interim / "development.parquet", index=False)

    # large loss: eccedenza oltre soglia per esposizione (per linea e AY)
    ll = (c.groupby(["line", "ay"])["excess_over_cap"].sum().rename("excess").reset_index())
    ex = d.groupby(["line", "ay"])["exposure"].first().rename("exposure").reset_index()
    ll = ll.merge(ex, on=["line", "ay"], how="right").fillna({"excess": 0})
    ll["excess_per_expo"] = ll["excess"] / ll["exposure"]
    ll.to_parquet(interim / "large_loss.parquet", index=False)

    summary = (d.groupby(["line", "ay"])[["n_rep", "n_ult", "cost_rep", "cost_ult", "exposure"]].sum())
    summary = summary.assign(exposure=d.groupby(["line", "ay"])["exposure"].first())
    summary["dev_factor"] = summary["cost_ult"] / summary["cost_rep"]
    summary["bc_ult"] = summary["cost_ult"] / summary["exposure"]
    notes["reporting_cdf_at_days[0,7,14,30,60,90,180,365,730,1095,1460]"] = curves
    notes["late_report_severity_rel_at_lag[0,30,90,180,365,730]"] = rel_curves
    notes["inflation"] = {f"{l}|{p}": [round(v[0], 4), v[1]] for (l, p), v in infl.items()}
    (interim / "report_development.json").write_text(json.dumps(notes, indent=2))
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string())
    print(d[["line", "peril", "ay", "n_rep", "open_share", "pi", "n_ult", "sev_rep", "sev_prior", "ibner_factor",
             "rel_ibnr", "dev_factor", "inflation"]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
