"""Part 2 — simulation of a "small and different market" within the developed market, to check the transfer choice
with more than one noisy fold (guardrail for the report).
Source market = regions R01–R06 (all years); "new" market = regions R07–R12, 2023–2024 only, subsampled
by policy to the size of the emerging market (and 4 times that). Fit on 2023, evaluation on 2024 (target_b), as in fold F2024.
Models: scratch (freq×sev on the new data only), prior × level, 0.9 prior + 0.1 scratch.
Usage: IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_sim.py <line> ; output results_p2/p2_sim.csv"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare as P, models_extra as M  # noqa: E402
OUT = Path(__file__).resolve().parents[1] / "results_p2" / "p2_sim.csv"
N_EMERG = {"motor": 23722, "buildings": 14299, "contents": 17689}   # policies in the emerging train
line = sys.argv[1]
df = P.load(line)
reg = sorted(df["region"].dropna().unique())
src_reg = set(reg[: len(reg) // 2])
src = df[df["region"].isin(src_reg)]
tgt = df[~df["region"].isin(src_reg) & df["year"].isin([2023, 2024])]
t0 = time.time()
prior = pd.Series(np.asarray(M._fs2(src, tgt, line, ll_si=True), float), index=tgt["row_id"].to_numpy())
print(line, "prior fitted on", len(src), "rows in", round(time.time() - t0), "s", flush=True)
pols = tgt["policy_id"].unique()
rows = []
for mult in (1, 4):
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        pick = set(rng.choice(pols, size=min(len(pols), N_EMERG[line] * mult), replace=False))
        s = tgt[tgt["policy_id"].isin(pick)]
        tr, va = s[s["year"] == 2023], s[s["year"] == 2024]
        y = va[P.TARGET].to_numpy()
        p_tr, p_va = prior.reindex(tr["row_id"]).to_numpy(), prior.reindex(va["row_id"]).to_numpy()
        lvl = p_va * tr[P.TARGET].sum() / p_tr.sum()
        sc = np.asarray(M.fs2_all_ll(tr, va, line), float)
        res = {"scratch": sc, "prior_livello": lvl, "prior_0.9_scratch_0.1": 0.9 * lvl + 0.1 * sc * lvl.sum() / sc.sum()}
        for k, p in res.items():
            rows.append(dict(linea=line, dimensione=f"{mult}x emerging", seed=seed, modello=k, d2=P.d2(y, p),
                             forma=P.d2(y, p * y.sum() / p.sum()), own=P.d2_own(va["row_id"], p), livello=p.sum() / y.sum()))
        print(line, mult, seed, {k: round(P.d2(y, p * y.sum() / p.sum()), 4) for k, p in res.items()}, flush=True)
r = pd.DataFrame(rows)
r.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
print(r.groupby(["dimensione", "modello"])[["d2", "forma", "own"]].mean().round(4).to_string())
