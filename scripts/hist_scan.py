"""21/10 evening: does the history (policy, customer on the other lines, address) leave residuals in the sub_41 blend?
For each history variable: bands, A/E on the own (no spreading) judge (A = target_rep, E = (pred − LL) × π), band factor
estimated on one year (with credibility n/(n+200) claims) and applied to the other, level unchanged; own and
official Δ D² in both directions. Output results_s3/hist_scan.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
R = P.RESULTS
o = pd.read_parquet(R / "oof/base41.parquet")
J = pd.read_parquet(R / "judge_own.parquet", columns=["row_id", "target_rep", "pi_eff", "ll_expo"]).set_index("row_id")
V = ["pol_prev_n", "pol_prev_rate", "pol_prev_cost_rate", "pol_prev_n_last", "cust_other_prev_n", "cust_other_prev_rate",
     "cust_other_prev_n_last", "addr_prev_n", "addr_prev_rate", "addr_prev_cost_rate", "addr_n_customers", "cust_n_lines", "years_at_address"]
rows = []
for line in P.LINES:
    d = pd.read_parquet(P.MODEL_DIR / f"model_{line}.parquet", columns=["row_id", "n_total"] + V)
    x = o[o.line == line].merge(d, on="row_id"); j = J.reindex(x.row_id)
    x["A"] = j.target_rep.to_numpy(); x["E"] = np.clip(x.pred.to_numpy() - j.ll_expo.to_numpy(), 1e-3, None) * j.pi_eff.to_numpy()
    for v in V:
        s = x[v].astype(float)
        if s.nunique() <= 6:
            b = s.clip(upper=3).astype(int).astype(str)
        else:
            pos = s > 0
            b = pd.Series("0", index=x.index)
            if pos.sum() > 1000:
                b[pos] = "q" + pd.qcut(s[pos].rank(method="first"), 4, labels=False).astype(str)
        res = {}
        for fit, sc in (("F2023", "F2024"), ("F2024", "F2023")):
            g = x[x.fold == fit].assign(b=b).groupby("b").agg(A=("A", "sum"), E=("E", "sum"), n=("n_total", "sum"))
            lvl = g.A.sum() / g.E.sum(); ae = g.A / g.E / lvl; z = g.n / (g.n + 200)
            fac = 1 + z * (ae - 1)
            m = (x.fold == sc).to_numpy(); s2 = x[m]
            f_ = b[m].map(fac).fillna(1.0).to_numpy(); q = s2.pred.to_numpy() * f_; q *= s2.pred.sum() / q.sum()
            res[sc] = (P.d2_own(s2.row_id, q) - P.d2_own(s2.row_id, s2.pred.to_numpy()), P.d2(s2.target_b, q) - P.d2(s2.target_b, s2.pred))
            res[sc + "_ae"] = " ".join(f"{k}:{v:.2f}" for k, v in ae.items())
        rows.append(dict(line=line, var=v, own23=res["F2023"][0] * 1e4, own24=res["F2024"][0] * 1e4, uff23=res["F2023"][1] * 1e4,
                         uff24=res["F2024"][1] * 1e4, ae_fit23=res["F2024_ae"], ae_fit24=res["F2023_ae"]))
r = pd.DataFrame(rows); r["min_own"] = r[["own23", "own24"]].min(axis=1)
r.sort_values("min_own", ascending=False).to_csv(R / "hist_scan.csv", index=False)
print(r.sort_values("min_own", ascending=False).round(2).to_string())
