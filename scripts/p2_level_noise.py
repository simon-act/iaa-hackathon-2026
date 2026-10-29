"""Part 2 (25/10): how reliable is the level estimated on 2 years given the size of the emerging market? (proposed after the review)
Within the developed market: each region (12) acts as a "market"; as many policies as in the emerging train are drawn at random,
level = cost per policy-year (target_b, already at the 2025 level) estimated on 2022–23, compared with 2024.
Estimators: exposure-weighted mean of the 2 years (as in the model); most recent year only; weights 1:2.
Metrics: sd of the error log(true 2024 level / estimated) with and without subsampling (the difference is the noise due to
size); comparison with the 2023→2024 change seen in the emerging market.
Usage: IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_level_noise.py → results_p2/p2_level_noise.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P  # noqa: E402
N_EMERG = {"motor": 23722, "buildings": 14299, "contents": 17689}
EMERG_23_24 = {"motor": (494.4, 532.5), "buildings": (562.4, 340.2), "contents": (102.2, 94.1)}
rows = []
for l in P.LINES:
    df = P.load(l)[["policy_id", "region", "year", "exposure", P.TARGET]]
    for reg, g in df.groupby("region"):
        g = g[g.year.isin([2022, 2023, 2024])]
        pols = g.policy_id.unique()
        for seed in range(-1, 30):           # -1 = whole region, no subsampling
            if seed >= 0:
                pick = np.random.default_rng(seed).choice(pols, size=min(len(pols), N_EMERG[l]), replace=False)
                s = g[g.policy_id.isin(set(pick))]
            else:
                s = g
            a = s.groupby("year").agg(y=(P.TARGET, "sum"), e=("exposure", "sum"))
            bc = a.y / a.e
            est = {"media 2 anni": (a.y[2022] + a.y[2023]) / (a.e[2022] + a.e[2023]), "solo ultimo anno": bc[2023],
                   "pesi 1:2": (bc[2022] + 2 * bc[2023]) / 3}
            for k, v in est.items():
                rows.append(dict(linea=l, regione=reg, seed=seed, stimatore=k, err=np.log(bc[2024] / v),
                                 salto=np.log(bc[2023] / bc[2022])))
    print(l, "done", flush=True)
r = pd.DataFrame(rows)
r.to_csv(Path(P.RESULTS).parent / "results_p2" / "p2_level_noise.csv", index=False)
sub, full = r[r.seed >= 0], r[r.seed < 0]
t = sub.groupby(["linea", "stimatore"]).err.agg(sd_sottocampione="std", rmse=lambda x: np.sqrt((x ** 2).mean()))
t["sd_regione_intera"] = full.groupby(["linea", "stimatore"]).err.std()
j = sub[sub.stimatore == "media 2 anni"].groupby("linea").salto
t2 = pd.DataFrame({"sd_salto_anno_su_anno": j.std(), "p95_salto": j.apply(lambda x: np.percentile(np.abs(x), 95)),
                   "salto_emerging_23_24": {l: np.log(b / a) for l, (a, b) in EMERG_23_24.items()}})
t2["quota_salti_più_grandi_dell_emerging"] = [float((np.abs(j.get_group(l)) >= abs(t2.loc[l, "salto_emerging_23_24"])).mean()) for l in t2.index]
pd.set_option("display.width", 200)
print(t.round(3).to_string()); print(t2.round(3).to_string())
