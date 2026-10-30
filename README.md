# IAA AI Assisted Hackathon 2026 — team "Simone"

Claim cost pricing for a motor + home (buildings, contents) portfolio, in two markets: predict the expected 2025 claim cost
(risk premium) of every policy row. Kaggle: *IAA AI Assisted Hackathon 2026* Part 1 (developed market, 2021–24) and
Part 2 (emerging market, 2023–24). Metric: Tweedie deviance D², p = 1.5.

## Start here

| To see | Read |
|---|---|
| The approach and results in five pages | the report (submitted PDF) |
| The models on one page | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| What was decided, by whom, on what evidence | [`docs/DECISIONS_actuary.md`](docs/DECISIONS_actuary.md) |
| How AI was used and checked | [`docs/AI_ledger.md`](docs/AI_ledger.md) |
| The work day by day | [`docs/WORKLOG.md`](docs/WORKLOG.md) |
| How to rebuild the two final files | "Final submissions" and the command blocks below |
| Everything else in `docs/` | [`docs/README.md`](docs/README.md) |

## Final submissions

| | Submission | What it is | md5 |
|---|---|---|---|
| Part 1 — developed | `sub_51` | frequency × severity per peril + experience rating (motor k = 16) + Qwen GBM, blended per line | `1f3c985a9506` |
| Part 2 — emerging | `p2_sub_21_auto_prem_costo` | Part 1 model as prior × emerging level × corrections the emerging data support | `c126730083a5` |

Both files are rebuilt from the code with identical md5 (28/10: sub_51 from the saved final fits with `scripts/build_final.py`, p2_sub_21 with the commands below); the full Part 1 rebuild from raw data was checked on sub_40 (21/10).
Part 2 rebuilt from raw data in a clean clone (28/10): equivalent, not bit-identical, because LightGBM with 4 threads is not
bit-deterministic (per-row ratio within ±0.3% on 98% of motor rows, line totals within 0.003%); details in `docs/MODEL_CARD.md`.
Checks: `python -m pytest hackathon_checks/tests` with `SUBMISSION`, `SAMPLE`, `MODEL_FEATURES` (Part 2: `results_p2/features_used_p2.txt`) passes 8/8 on both finals.
Lesson from Part 1: fine choices made late, on small internal gains, did not hold out of sample; the plain structural model
was the most robust. Part 2 therefore keeps the structure and lets the new data set only what they can support.

## Part 2 — the emerging market (`docs/MODEL_CARD.md`, `configs/parte2.json`, `configs/parte2_llc.json`)

```bash
./run_all.sh configs/parte2.json                                                     # same pipeline, one config file
IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_export.py configs/parte2.json motor,buildings,contents   # prior
./run_all.sh configs/parte2_llc.json                  # after copying results_p2/source_pred_*.parquet to results_p2_llc/
IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_export_p1.py buildings v2 configs/parte2_llc.json   # home priors
IAA_MARKET=configs/parte1_s3.json python3 scripts/p2_export_p1.py contents v2 configs/parte2_llc.json
IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_final.py p2_sub_20_prior_p1v2_casa --spec v13       # md5 cfedcffc39a7
IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_sub_varianti_auto.py premcost p2_sub_21_auto_prem_costo   # final
IAA_MARKET=configs/parte2_llc.json python3 scripts/p2_office_premium.py submissions_p2/p2_sub_21_auto_prem_costo.csv
# -> results_p2_llc/office_premium_reference.csv (kept as results_p2/office_premium_reference_sub21.csv)
```
- Automated audit of what changed (`src/audit_new_market.py`).
- Reporting delays identical to the developed market per peril → delays, late-claim severity, IBNER by age and trends
  borrowed from the developed market (`src/develop.py`, `src/trend.py`, options in the config); large-loss load rescaled
  by the emerging frequency of large claims.
- Prior = Part 1 frequency × severity fitted on the developed market with the shared columns (`src/transfer.py`;
  for buildings and contents the Part 1 variant with relative premium, `scripts/p2_export_p1.py`). The emerging data set
  the level per line (and per peril with credibility in buildings and contents) and the corrections that pass the bench:
  gender, new business, fuel and cover type at half weight, relative premium and a Poisson boost on claim counts (motor),
  storm level (buildings), working from home (contents).
- The insurer's premium: excluded in Part 1 (pricing on the current tariff would be circular); in Part 2 its relative
  index (log premium net of discount minus the year median) enters only the motor correction, as the carrier of local
  rating information the emerging data lack (no credit band, mileage, telematics). It sets the slope, not the level.
- Checks: seven-check bench with a rule fixed in advance (`scripts/p2_bench*.py`, ≥ 5/7 on shape and own judge), then a
  stricter bench on the final pieces (fit one year, score the other, disjoint customers: `scripts/p2_bench27.py`); a bench reweighted to the 2025 mix
  (adversarial weights, `scripts/p2_shift_weights.py`, `scripts/p2_bench32.py`);
  simulation of a small, different market inside the developed data (`scripts/p2_sim.py`: prior wins 16/18).
- Office premium for the 20 reference policies: `results_p2/office_premium_reference_sub21.csv`
  (sensitivities in `results_p2/office_premium_sensitivity_sub21.csv`).

## Part 1 — how to reproduce

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
  folds on the official and own judges (`docs/DECISIONS_actuary.md`). No post-hoc transforms of predictions.
- Error decomposition by simulation (`scripts/ceiling_s3.py`): variables ≈ 0.127,
  history +0.002–0.004, latent policy risk not recoverable ≈ 0.043, noise ≈ 0.83; our OOF 0.1290 vs 0.1285 ± 0.002
  for a perfect model with the same information.

## AI use

- Claude (agent): data reconstruction, code, diagnostics, documentation; summary in `docs/WORKLOG.md`.
- Qwen 3.8 27B, local (Ollama), in Karpathy-style autoresearch loops (`autoresearch*/`: `program.md`, editable
  `experiment.py`, frozen `eval_experiment.py`, `run_loop.py`, `results.tsv`): 731 experiments in Part 1 and 124 in Part 2; ideas kept only
  after re-validation by Claude on the three judges. Plus an independent code review by Qwen (`autoresearch_review/`).
- The human (actuary) set the method rules, challenged the results and took every decision; see the report.

## Map of the repository

| Path | Content |
|---|---|
| `src/` | pipeline, models, judges (`prepare.py`), experience rating, Part 2 tools (`audit_new_market.py`, `transfer.py`) |
| `scripts/` | experiments, checks, submission builders |
| `configs/` | one JSON per market / pipeline variant (`parte1_s3.json` = final) |
| `docs/` | `MODEL_CARD.md`, `DECISIONS_actuary.md`, `WORKLOG.md`, `AI_ledger.md`, dashboard |
| `autoresearch*/`, `qwen/` | local LLM loops and the code they produced |
| `results*/` | aggregated results (row-level files and data are not versioned) |
| `docs/dashboard_IAA_2026_offline.html` | the dashboard as one offline HTML file (all charts and tables read `docs/dashboard_data/`, embedded) |

Data and predictions are not in the repository (competition rules). Credentials are never stored (`.gitignore`).
