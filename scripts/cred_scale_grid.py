"""21/10 evening: in the credibility M = (N + k)/(E + k) the history expectations E = H × pred/(exposure × line mean severity)
depend on the mean severity used to convert cost into counts. If it is biased, E is biased. Test c × E (c 0.7–1.2)
by line on the sub_41 blend (OOF), own (no spreading) and official judges, two folds. Output results_s3/cred_scale_grid.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
b = pd.read_parquet(Path(__file__).resolve().parents[1] / "autoresearch_storia/base_oof.parquet")
S = pd.read_parquet(Path(__file__).resolve().parents[1] / "autoresearch_storia/storia.parquet").set_index("row_id")
EXCL = {"buildings": ["subsidence"]}
rows = []
for l in P.LINES:
    for f in P.FOLDS:
        s = b[(b.line == l) & (b.fold == f)]; h = S.reindex(s.row_id)
        per = [c[7:] for c in h.columns if c.startswith("hist_n_") and h[c].notna().any()]
        keep = [p for p in per if p not in EXCL.get(l, [])]
        N = h[[f"hist_n_{p}" for p in keep]].sum(axis=1).to_numpy()
        lam = s.base_nocred.to_numpy() / (s.exposure.to_numpy() * h.sev_line_mean.to_numpy())
        E0 = h.hist_expo_pi.to_numpy() * lam
        nn = b  # placeholder
        def pred(c, k=8.0):
            p = s.base_nocred.to_numpy() * (N + k) / (c * E0 + k); return p * s.base_pred.sum() / p.sum()
        p0 = s.base_pred.to_numpy(); rid = s.row_id
        o0, u0 = P.d2_own(rid, p0), P.d2(s.target_b, p0)
        for c in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            p = pred(c)
            rows.append(dict(line=l, fold=f, c=c, own=(P.d2_own(rid, p) - o0) * 1e4, uff=(P.d2(s.target_b, p) - u0) * 1e4,
                             N_over_E=N.sum() / E0.sum()))
r = pd.DataFrame(rows); r.to_csv(P.RESULTS / "cred_scale_grid.csv", index=False)
print(r.pivot_table(index=["line", "c"], columns="fold", values=["own", "uff", "N_over_E"]).round(2).to_string())
