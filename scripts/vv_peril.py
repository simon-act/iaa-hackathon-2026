"""21/10: residuo sul valore del veicolo (A/E 0,93 → 1,07 nel modello di sub_40): da quale perilo, frequenza o severità?
Stima il freq×sev auto di sub_40 (fs2_v2fb, senza credibilità) sui due fold e scompone per perilo e fascia di valore:
A/E frequenza = n denunciati / (λ × π); A/E severità = costo denunciato tagliato / (n × s). Output results_s3/vv_peril.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare as P, models_extra as M  # noqa: E402
line = "motor"
df = P.load(line); df = df[~P.is_holdout(df.policy_id)]
out = []
for fname, (tys, vy) in P.FOLDS.items():
    tr, va = df[df.year.isin(tys)], df[df.year == vy]
    M._SEV_PERIL = {"motor": {"bodily_injury": {"_mean": True}}}; M._FREQ_OVERRIDE = M.FREQ_CHOSEN
    M._fs2(tr, va, line, ll_si=True)
    M._SEV_PERIL = {}; M._FREQ_OVERRIDE = {}
    parts, c = M._fs2.last_parts["perils"], M._fs2.last_parts["c"]
    band = pd.cut(va.vehicle_value_group, [0, 8, 10, 12, 14, 20]).astype(str).to_numpy()
    for p, (f, s) in parts.items():
        n = va[f"n_{p}"].to_numpy(); pi = va[f"pi_{p}"].clip(0, 1).to_numpy(); cost = va[f"cost_{p}"].to_numpy()
        z = pd.DataFrame({"b": band, "n": n, "En": f * pi, "cost": cost, "Ec": n * s})
        g = z.groupby("b")[["n", "En", "cost", "Ec"]].sum()
        g["AE_freq"] = g.n / g.En / (g.n.sum() / g.En.sum()); g["AE_sev"] = g.cost / g.Ec / (g.cost.sum() / g.Ec.sum())
        g["quota_costo"] = cost.sum()
        for b, r in g.iterrows():
            out.append(dict(fold=fname, peril=p, band=b, n=int(r.n), AE_freq=round(r.AE_freq, 3), AE_sev=round(r.AE_sev, 3)))
    print(fname, "fatto", flush=True)
R = pd.DataFrame(out); R.to_csv(P.RESULTS / "vv_peril.csv", index=False)
print(R.pivot_table(index=["peril", "band"], columns="fold", values=["n", "AE_freq", "AE_sev"]).to_string())
