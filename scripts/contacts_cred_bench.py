"""22/10 night, T2: complete the last-year history with the late claims expected from contacts (see contacts_late.py).
In the test the 2024 history is incomplete (reported by 31/12/2024). Simulation on the folds: for rows of year Y the history
of year Y−1 is truncated to claims reported by 31/12/(Y−1) and the completeness H reset to its lag-1 value
(same values as the 2024 rows, by period-end month). Three variants of the credibility modifier, k=8:
  piena  = current pipeline (full history: upper bound, not the test situation)
  tronca = what the test actually sees
  fix    = tronca + E[late | contacts, reported] for year Y−1 added to N, H reset to full completeness
Late-claims table estimated on the other year (2022 for F2024, 2023 for F2023). Judges inside the sub_48 blend.
Usage: IAA_MARKET=configs/parte1_s3.json python scripts/contacts_cred_bench.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare as P, experience as X  # noqa: E402
R = P.RESULTS; K = 8
SPEC = {"motor": ("fs2_v2fb_xs_cred", "qwen_gbm73fix_cred", 0.9, False),        # gross not in oof_parts: approximation as in blend_gbmxs
        "buildings": ("fs2_v2_llr_cred", "qwen_gbm73_gross_xs_cred", 0.6, True),
        "contents": ("fs2_v2_sevmix_cred", "qwen_gbm73_cred", 0.8, False)}
QB = pd.read_parquet(R / "oof/qwen_gbm73_cred.parquet", columns=["row_id", "pred"]).set_index("row_id")["pred"]
def part(n, l, f):
    return QB if n == "qwen_gbm73_cred" else pd.read_parquet(R / f"oof_parts/{n}_{l}_{f}.parquet").set_index("row_id")["pred"]
ref = pd.read_parquet(R / "oof/fs2_v2_cred.parquet", columns=["row_id", "line", "fold", "exposure", "target_b"])
cl = pd.read_parquet(ROOT / "data/raw/public/claims.parquet")
cl["loss_date"] = pd.to_datetime(cl.loss_date); cl["report_date"] = pd.to_datetime(cl.report_date)
CS_MAX, NR_MAX = 6, 2
rows = []
for l, (a, b, w, ex) in SPEC.items():
    tr = P.load(l)
    tr["period_start"] = pd.to_datetime(tr.period_start); tr["period_end"] = pd.to_datetime(tr.period_end); tr["mese"] = tr.period_end.dt.month
    perils = [c[3:] for c in tr.columns if c.startswith("pi_")]
    nn = tr[[f"n_{p}" for p in perils]].sum().to_numpy(); wp = nn / nn.sum()
    tr["pi_avg"] = tr[[f"pi_{p}" for p in perils]].to_numpy() @ wp
    excl = X.EXCL_DEFAULT.get(l, []) if ex else []
    keep = [p for p in perils if p not in excl]
    # lag-1 completeness by month (from the 2024 rows)
    pi_lag1 = tr[tr.year == 2024].groupby("mese")["pi_avg"].mean()
    # claims per row: reported by 31/12 of the year, and late (only perils kept in the credibility)
    c = cl[(cl.line == l) & cl.peril.isin(keep)].merge(tr[["row_id", "policy_id", "period_start", "period_end", "year"]], on="policy_id")
    c = c[(c.loss_date >= c.period_start) & (c.loss_date <= c.period_end)]
    c["cut"] = pd.to_datetime(c.year.astype(str) + "-12-31")
    n_rep = c[c.report_date <= c.cut].groupby("row_id").size(); n_late = c[c.report_date > c.cut].groupby("row_id").size()
    tr["n_rep"] = n_rep.reindex(tr.row_id).fillna(0).to_numpy(); tr["n_late"] = n_late.reindex(tr.row_id).fillna(0).to_numpy()
    tr["cs"] = tr.customer_service_contacts.clip(upper=CS_MAX); tr["nr"] = tr.n_rep.clip(upper=NR_MAX)
    # history per policy: current N and H (as in experience.history / history_excl)
    if ex:
        h, n2, sev, frac = X.history_excl(l, excl); N_cur = n2
    else:
        h, N_cur, sev = X.history(l); frac = 1.0
    byrow = tr.set_index("row_id")
    for f in P.FOLDS:
        Y = int(f[1:]); Yp = Y - 1; Yfit = 2022 if Y == 2024 else 2023
        s = ref[(ref.line == l) & (ref.fold == f)]; rid = s.row_id
        tab = tr[tr.year == Yfit].groupby(["cs", "nr"])["n_late"].mean()
        prev = tr[tr.year == Yp].copy()
        prev["late_exp"] = tab.reindex(pd.MultiIndex.from_arrays([prev.cs, prev.nr])).fillna(0).to_numpy()
        prev["pi_sim"] = pi_lag1.reindex(prev.mese).to_numpy()
        # per policy: actual late claims (to subtract from N), exposure and completeness for year Y−1
        g = prev.groupby("policy_id").agg(late=("n_late", "sum"), late_exp=("late_exp", "sum"),
                                          dH=("exposure", lambda e: 0.0)).reset_index()
        prev["dH_tr"] = prev.exposure * (prev.pi_sim - prev.pi_avg)      # truncated H − full H (negative)
        prev["dH_fix"] = prev.exposure * (prev.pi_avg - prev.pi_sim)     # back to full
        g = prev.groupby("policy_id").agg(late=("n_late", "sum"), late_exp=("late_exp", "sum"), dH_tr=("dH_tr", "sum")).reset_index()
        pol = byrow.loc[rid, "policy_id"].to_numpy()
        gg = g.set_index("policy_id").reindex(pol).fillna(0)
        H0 = h.reindex(rid).to_numpy(); N0 = N_cur.reindex(rid).to_numpy(); expo = s.exposure.to_numpy()
        pa = part(a, l, f).reindex(rid).to_numpy(); pb = part(b, l, f).reindex(rid).to_numpy()
        # predictions without credibility (exact inverse of apply with k=8, as in blend_gbmxs)
        ua, ub = X.uncred(pa, s, l, K), X.uncred(pb, s, l, K)
        def cred(p, N, H):
            E = H * p / (expo * sev) * frac
            out = p * (N + K) / (E + K)
            return out * p.sum() / out.sum()
        var = {"piena": (N0, H0),
               "tronca": (N0 - gg.late.to_numpy(), H0 + gg.dH_tr.to_numpy()),
               "fix": (N0 - gg.late.to_numpy() + gg.late_exp.to_numpy(), H0)}
        res = {}
        for name, (N, H) in var.items():
            p = w * cred(ua, N, H) + (1 - w) * cred(ub, N, H)
            res[name] = (P.d2(s.target_b, p), P.d2_own(rid, p), P.d2_unc(rid, p))
        for name in var:
            rows.append(dict(line=l, fold=f, var=name, uff=res[name][0], own=res[name][1], unc=res[name][2],
                             d_uff=(res[name][0] - res["tronca"][0]) * 1e4, d_own=(res[name][1] - res["tronca"][1]) * 1e4, d_unc=(res[name][2] - res["tronca"][2]) * 1e4,
                             late_tot=float(gg.late.sum()), late_exp_tot=float(gg.late_exp.sum())))
        print(l, f, {k: tuple(round(x, 5) for x in v) for k, v in res.items()}, "actual late", int(gg.late.sum()), "expected", round(float(gg.late_exp.sum()), 1), flush=True)
r = pd.DataFrame(rows); r.to_csv(R / "contacts_cred_bench.csv", index=False)
print(r.round(3).to_string(index=False))
