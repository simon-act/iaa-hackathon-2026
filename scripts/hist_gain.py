"""21/10: quanto vale la storia della polizza sui dati veri (per tarare la frailty φ della simulazione del tetto).
fs2 (ll_si) con variabili di storia + credibilità k=8, contro fs2 senza variabili di storia e senza credibilità. Fold 2023/2024."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import prepare as P, models_extra as M, experience as X  # noqa: E402
W = {"motor": 254986, "buildings": 184155, "contents": 187498}
out = []
for line in P.LINES:
    df = P.load(line); df = df[~P.is_holdout(df.policy_id)]
    for fname, (tys, vy) in P.FOLDS.items():
        tr, va = df[df.year.isin(tys)], df[df.year == vy]
        p_no = M._fs2(tr, va, line, ll_si=True, drop_hist=True)
        p_h = M._fs2(tr, va, line, ll_si=True)
        p_hc = X.apply(p_h, va, line, 8)
        y = va[P.TARGET].to_numpy()
        out.append(dict(line=line, fold=fname, senza_storia=P.d2(y, p_no), variabili_storia=P.d2(y, p_h), variabili_e_cred=P.d2(y, p_hc)))
        print(out[-1], flush=True)
R = pd.DataFrame(out); R["w"] = R.line.map(W); R.to_csv(P.RESULTS / "hist_gain.csv", index=False)
for f, z in R.groupby("fold"):
    print(f, {c: round((z[c] * z.w).sum() / z.w.sum(), 5) for c in ["senza_storia", "variabili_storia", "variabili_e_cred"]})
