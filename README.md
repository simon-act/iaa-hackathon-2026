# IAA AI Assisted Hackathon 2026 — team "Simone"

Claim cost pricing for a motor + home (buildings, contents) portfolio: predict the expected 2025 claim cost of every
policy row from 2021–2024 policies and claims. Kaggle: *IAA AI Assisted Hackathon 2026 – Part 1*
(metric: Tweedie deviance D², p = 1.5, averaged over lines weighted by test rows).

Final Part 1 candidates (public leaderboard, a third of the test set):

| Submission | What it is | Internal score (OOF 2023/2024) | Public LB |
|---|---|---|---|
| `sub_41` | final pipeline + contents flood severity = mean of GBM and Gamma GLM | best on all three internal judges | 0.12920 |
| `sub_40` | final pipeline (motor bodily-injury severity constant, buildings GBM on gross target) | +0.00006 below sub_41 | 0.12927 |
| `sub_36` | earlier variant, best public score | lower | 0.12930 |

The choice of the final submissions is made on the internal judges, with the leaderboard only as a check
(the noise between near-identical submissions on the public third is about ±0.0001–0.00015).

## How to reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# competition data (not in this repository, rules §2.5) in data/raw/
./run_all.sh configs/parte1_s3.json      # cleaning → development (IBNR/IBNER) → trend → target → features → judges
./run_final.sh                           # six final fits on all 2021–24 policies → submissions/sub_finale.csv (= sub_40)
                                         # + submissions/sub_41_finale.csv
```
`run_final.sh` was checked on 21/10: the rebuilt sub_40 is identical to the uploaded file (max |Δ| = 0).
Out-of-fold predictions: `scripts/oof_one.py <model> <line> <fold>`; scores on the three judges: `scripts/subs_three_judges.py`.

## How it works (short)

1. **Target** (`src/clean_*.py`, `src/develop.py`, `src/trend.py`, `src/build_datasets.py`): cleaned policies and claims
   (duplicates, repudiated, out-of-period claims, sum insured in wrong units), reporting delay curve with a reverse-time
   estimator for truncated data, expected completeness per row, IBNR counts and costs, IBNER with credibility,
   log-linear trends (no trend on weather perils: 2025 weather is known), large losses capped and reloaded.
2. **Model A — frequency × severity per peril** (`src/models_extra.py::_fs2`): LightGBM Poisson with one split per tree
   (an additive log-linear model), offset log(exposure × completeness) − trend; Gamma severity on a few plausible
   drivers (GLM on log sum insured for small home perils, constant for motor bodily injury); large-loss load.
3. **Experience rating** (`src/experience.py`): Bühlmann–Straub on the policy's past claims, M = (N + k)/(E + k), k = 8
   (subsidence excluded in buildings).
4. **Model B — LightGBM Tweedie** written by a local LLM (Qwen 27B) in an autoresearch loop (`qwen/`), trained on a
   "gross" target (reported cost ÷ expected completeness).
5. **Blend per line**: motor 0.9 A + 0.1 B, buildings 0.6/0.4, contents 0.8/0.2 (`scripts/build_final.py`).

## Validation and guardrails

- Frozen judge (`src/prepare.py`): temporal folds (fit 2021–22 → score 2023; fit 2021–23 → score 2024), 10 % policy
  holdout, forbidden columns (leakage: `customer_service_contacts`, `renewal_decision`), drift check.
- Three internal judges: official (target with spread IBNR, like Kaggle), **own** (reported cost vs prediction ×
  completeness, no spreading — decides the shape), **true large losses**. A change is kept only if it improves both
  folds on the official and own judges (`docs/17_regole_metodo_20ott.md`). No post-hoc transforms of predictions.
- Error decomposition by simulation (`scripts/ceiling_s3.py`, `docs/22_domande_21ott.md` §7): variables ≈ 0.127,
  history +0.002–0.004, latent policy risk not recoverable ≈ 0.043, noise ≈ 0.83; our OOF 0.1290 vs 0.1285 ± 0.002
  for a perfect model with the same information.

## AI use

- Claude (agent): data reconstruction, code, diagnostics, documentation; every step logged in `LOG_AI.md` files.
- Qwen 3.8 27B, local (Ollama), in Karpathy-style autoresearch loops (`autoresearch*/`: `program.md`, editable
  `experiment.py`, frozen `eval_experiment.py`, `run_loop.py`, `results.tsv`): about 640 experiments; ideas kept only
  after re-validation by Claude on the three judges. Plus an independent code review by Qwen (`autoresearch_review/`).
- The human (actuary) set the method rules, challenged the results and took every decision; see the report.

## Map of the repository

| Path | Content |
|---|---|
| `src/` | pipeline, models, judges (`prepare.py`), experience rating, Part 2 tools (`audit_new_market.py`, `transfer.py`) |
| `scripts/` | experiments, checks, submission builders |
| `configs/` | one JSON per market / pipeline variant (`parte1_s3.json` = final) |
| `docs/` | working notes in Italian, one per step (02–22) |
| `autoresearch*/`, `qwen/` | local LLM loops and the code they produced |
| `results*/` | aggregated results (row-level files and data are not versioned) |
| `mac_runner/` | small job runner used to run heavy jobs on the local machine |

Data and predictions are not in the repository (competition rules). Credentials are never stored (`.gitignore`).
