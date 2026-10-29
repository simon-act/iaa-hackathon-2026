"""Step 1c — development to ultimate by line × peril × accident year (valuation 31/12/2024).

Components:
  1. reporting delay F(l) = P(lag <= l), estimated with the reverse-time estimator for
     right-truncated data (a claim occurring on day d is visible only if lag <= V-d)
  2. expected completeness pi per policy row and per AY = mean of F(V-d) over the covered days
  3. IBNR count = reported / pi - reported
  4. severity of late claims: relativity rel(w) = E[sev | lag > w] / E[sev] on mature AYs
  5. future recoveries on open claims: rate observed on closed claims of the same peril
  6. IBNER: credibility blend of observed severity (weight Z = closed share) and prior severity
     (level of mature AYs trended with the peril or line inflation)
Usage: python -I develop.py <interim_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from market_config import CFG, LINES  # noqa: E402

V = None      # set in main from the policy years
DAYS = None
YEARS = None
MATURE_OPEN_MAX = 0.10      # an AY is "mature" for a peril if open claims are < 10%
MAX_LAG = 1500


def reporting_cdf(lag: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Reverse-time estimator (Lagakos et al. 1988) for right-truncated lags.
    g(l) = #{lag = l} / #{lag <= l <= window};  F(l) = prod_{k > l} (1 - g(k))."""
    n = np.bincount(lag, minlength=MAX_LAG + 2)[: MAX_LAG + 2].astype(float)
    # reverse-time risk set at l: claims with lag <= l and window >= l
    lag_le = np.cumsum(n)                                    # #{lag <= l}
    w_lt = np.bincount(np.minimum(window, MAX_LAG + 1), minlength=MAX_LAG + 2)[: MAX_LAG + 2]
    # #{lag <= l, window < l}: window >= lag always, hence = #{window < l}
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
    # Part 2 (23/10): market with few years (no mature AY) -> late-claim severity, inflation and IBNER by development
    # age taken from the source market (config "development": {"source_interim": ...}); the reporting delay is still
    # estimated on the new market (the reverse-time estimator handles truncation).
    src_dir = CFG.get("development", {}).get("source_interim")
    src = None
    if src_dir:
        sd = Path(src_dir) if Path(src_dir).is_absolute() else Path(__file__).resolve().parents[1] / src_dir
        src = {"claims": pd.read_parquet(sd / "claims_clean.parquet"), "dev": pd.read_parquet(sd / "development.parquet"),
               "report": json.loads((sd / "report_development.json").read_text())}
        src["last"] = int(src["dev"]["ay"].max())
        notes["source_market"] = str(sd)

    for line in LINES:   # used to be a fixed tuple (Qwen review 19/10: Part 2 with different lines)
        pol = pd.read_parquet(interim / f"train_{line}.parquet",
                              columns=["row_id", "period_start", "period_end", "exposure", "year"])
        cl = c[c["line"] == line]
        comp_cols = {}
        # line inflation from short-tail perils (mature AYs, closed claims only)
        for peril, g in cl.groupby("peril"):
            # 1. reporting delay
            F = reporting_cdf(g["report_lag_days"].clip(0, MAX_LAG).to_numpy(),
                              g["days_to_valuation"].clip(0, MAX_LAG + 1).to_numpy())
            if src is not None and CFG.get("development", {}).get("reporting_from_source"):
                # Part 2: reporting curves of the new market equal to the source ones (see docs/WORKLOG.md) -> use the curve
                # estimated on the 160k source claims (tail beyond the 2 observable years, less noise on small perils)
                gs = src["claims"][(src["claims"]["line"] == line) & (src["claims"]["peril"] == peril)]
                curves[f"{line}|{peril}|nuovo"] = F[[0, 7, 14, 30, 60, 90, 180, 365, 730]].round(5).tolist()
                F = reporting_cdf(gs["report_lag_days"].clip(0, MAX_LAG).to_numpy(),
                                  gs["days_to_valuation"].clip(0, MAX_LAG + 1).to_numpy())
            curves[f"{line}|{peril}"] = F[[0, 7, 14, 30, 60, 90, 180, 365, 730, 1095, 1460]].round(5).tolist()
            comp_day = completeness_by_day(F)
            comp_cols[peril] = row_completeness(pol["period_start"], pol["period_end"], comp_day)

            # 4. severity of late claims
            mat = g[g["ay"] <= YEARS[-1] - 2]   # mature AYs: used to be a fixed "<= 2022" (Qwen review 19/10)
            if len(mat) < 30 and src is not None:   # Part 2: late-claim relativities from the source market
                sc = src["claims"]
                mat = sc[(sc["line"] == line) & (sc["peril"] == peril) & (sc["ay"] <= src["last"] - 2)]
            s = mat.sort_values("report_lag_days")
            lag_s, sev_s = s["report_lag_days"].to_numpy(), s["net_capped"].to_numpy()
            tail_sum = np.cumsum(sev_s[::-1])[::-1]; tail_n = np.arange(len(sev_s), 0, -1)
            grid = np.arange(0, MAX_LAG + 2)
            idx = np.searchsorted(lag_s, grid, side="right")
            mean_all = sev_s.mean()
            rel = np.where(idx < len(sev_s), np.divide(tail_sum[np.minimum(idx, len(sev_s) - 1)],
                                                         tail_n[np.minimum(idx, len(sev_s) - 1)]) / mean_all, 1.0)
            # stabilise the tail: at least 30 claims beyond w, otherwise last stable value
            ok = (len(sev_s) - idx) >= 30
            last = rel[ok][-1] if ok.any() else 1.0
            rel = np.where(ok, rel, last)
            rel_curves[f"{line}|{peril}"] = rel[[0, 30, 90, 180, 365, 730]].round(3).tolist()

            # 5. expected future recoveries on open claims
            closed = g[~g["is_open"]]
            rec_ratio = float(closed["recoveries"].sum() / closed["incurred_to_date"].sum()) if len(closed) else 0.0

            for ay in YEARS:
                gy = g[g["ay"] == ay]
                pa = pol[pol["year"] == ay]
                expo = float(pa["exposure"].sum())
                pi = float(np.average(comp_cols[peril][pol["year"].to_numpy() == ay], weights=pa["exposure"]))
                n_rep = len(gy)
                n_ult = n_rep / pi
                # observed cost with expected future recoveries on open claims
                opn = gy["is_open"].to_numpy()
                cost_rep = float(gy["net_capped"].sum() - rec_ratio * gy.loc[opn, "incurred_to_date"].clip(upper=gy.loc[opn, "net_capped"] / max(1 - rec_ratio, 1e-9)).sum())
                # severity relativity for IBNR: mean of rel(w) weighted by (1-F(w)) over the AY's days
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
    # 6. IBNER: prior severity from mature AYs + inflation
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
            if src is not None and f"{line}|{peril}" in src["report"].get("inflation", {}):
                infl.setdefault((line, peril), (float(src["report"]["inflation"][f"{line}|{peril}"][0]), "mercato di partenza (nessun AY maturo)"))
            infl.setdefault((line, peril), (pooled, "mediana della linea (perilo senza 3 AY maturi)"))
    # 18/10 — S3: for perils without 3 mature AYs the line median can be wrong (injury: +6.4% for the
    # line vs +1.9% estimated on the peril); the config allows a per-peril value, documented
    for key, v in CFG.get("development", {}).get("inflation_override", {}).items():
        l, p_ = key.split("|")
        infl[(l, p_)] = (float(v), "valore da config (perilo senza AY maturi)")
    d["inflation"] = [infl[(l, p)][0] for l, p in zip(d["line"], d["peril"])]
    d["inflation_source"] = [infl[(l, p)][1] for l, p in zip(d["line"], d["peril"])]

    prior = []
    for _, r in d.iterrows():
        g = d[(d["line"] == r["line"]) & (d["peril"] == r["peril"])]
        # reference: most mature AY (minimum open share), trended to the current AY
        ref = g.sort_values(["open_share", "ay"]).iloc[0] if not g["mature"].any() else g[g["mature"]].iloc[0]
        base = g[g["mature"]] if g["mature"].any() else g[g["ay"] == ref["ay"]]
        lvl = np.exp(np.average(np.log(base["sev_rep"]) + np.log1p(r["inflation"]) * (r["ay"] - base["ay"]),
                                weights=base["n_rep"]))
        prior.append(lvl)
    d["sev_prior"] = prior
    d["Z"] = 1 - d["open_share"]
    # ultimate severity of reported claims (IBNER) and of IBNR (late-claim relativities)
    d["sev_rep_ult"] = d["Z"] * d["sev_rep"] + (1 - d["Z"]) * d["sev_prior"]
    d["ibner_factor"] = d["sev_rep_ult"] / d["sev_rep"]
    if src is not None:
        # Part 2: no mature AY -> IBNER factor of the source market at the same development age (same peril)
        sdv = src["dev"].set_index(["line", "peril", "ay"])["ibner_factor"]
        fac = [float(sdv.get((l, p_, src["last"] - (YEARS[-1] - ay)), 1.0)) for l, p_, ay in zip(d["line"], d["peril"], d["ay"])]
        d["ibner_factor"] = fac
        d["sev_rep_ult"] = d["sev_rep"] * d["ibner_factor"]
        notes["ibner"] = "fattori del mercato di partenza per età di sviluppo"
    d["n_ibnr"] = d["n_ult"] - d["n_rep"]
    d["cost_ult"] = d["n_rep"] * d["sev_rep_ult"] + d["n_ibnr"] * d["sev_rep_ult"] * d["rel_ibnr"]
    d["sev_ult"] = d["cost_ult"] / d["n_ult"]
    d["freq_ult"] = d["n_ult"] / d["exposure"]
    d["bc_ult"] = d["cost_ult"] / d["exposure"]
    d["bc_rep"] = d["cost_rep"] / d["exposure"]
    d["dev_factor"] = d["cost_ult"] / d["cost_rep"]
    d.to_parquet(interim / "development.parquet", index=False)

    # large loss: excess over threshold per unit of exposure (by line and AY)
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
