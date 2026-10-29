# AI ledger: who did what, and how it was checked

Team Simone, IAA AI Assisted Hackathon 2026. The actuary gave the direction: the order of work, the rules, the checks to run, the key challenges to the results and every final decision. AI executed the work inside those rules and proposed alternatives. The work is summarised day by day in `docs/WORKLOG.md`; the actuary's decisions, with their origin and evidence, are in `docs/DECISIONS_actuary.md`. An interactive version of this ledger, with the numbers drawn from the files below, is the dashboard "IAA Hackathon Workflow" (`docs/dashboard_IAA_2026_offline.html`).

## The agents

| Agent | Where it ran | Role |
|---|---|---|
| Claude (coding agent) | cloud workspace linked to the actuary's Mac | rebuilt the target, wrote the pipeline, the judges, the diagnostics, the simulations and the Part 2 adaptation; ran the internal benches; wrote the logs, the report draft and this ledger |
| Qwen 27B (local, Ollama) | the actuary's Mac; data never left the machine | Karpathy-style autonomous research loops: edits one file, a frozen judge scores it, kept only if it improves; also a code review (125 findings) |
| The actuary (Simone) | — | set the order of work, the rules, the challenges; chose every submission |

## Across the actuarial workflow

| Stage | What AI did | How it was verified | Evidence |
|---|---|---|---|
| Exploratory data analysis | profiled every table; automated audit of the new market; found exposure in days, sums insured 1,000× too large, 598 exact duplicate claims, claims before inception | each fix checked on the ratio gap, the same policy in other years and premiums; independent recomputation of Part 2 from the raw files (48 of 52 checks passed, 4 explained) | `docs/WORKLOG.md`, `results_p2/audit_controlli.csv` |
| Target and reserving | reporting-delay curves (reverse-time estimator), IBNR, IBNER, trends, large-loss load; in Part 2, development borrowed from the developed market | target reconciles exactly with the development table; backtest of the borrowed delay curves: actual/predicted 0.97–1.07 on 15 perils | `docs/WORKLOG.md`, `results_p2/audit_backtest_denuncia.csv` |
| Feature engineering | Qwen loops proposed features and constraints; relative deductible; leakage search | a feature stays only if it beats random decoy columns and improves both validation years; two leaking columns excluded by an automated test | `docs/WORKLOG.md`, `hackathon_checks/tests/test_excluded_features.py` |
| Modelling | additive frequency × severity per peril, experience rating, a Qwen-written boosting partner; Part 2: Part 1 model as prior, emerging level and targeted corrections | frozen judges (two time folds, policy holdout), three judges; Part 2 seven-check bench with a rule fixed in advance (wins ≥ 5 of 7 on shape and own judge), final pieces re-checked on a stricter bench (time direction × disjoint customers) | `docs/WORKLOG.md`, `results_p2/p2_bench*.csv` |
| Diagnostics | residual loops; error decomposition by simulation; leaderboard-noise study from our own public vs private scores | simulation calibrated on our data (about 83% of deviance is claim noise); public gaps under 0.0005 were coin flips on the Part 1 private board | `docs/WORKLOG.md` |
| Testing and governance | automated tests on the submission contract and excluded columns; reproducibility runs | final submissions regenerated from code with identical md5; no post-hoc tweaks to predictions; the AI never handled credentials | `hackathon_checks/tests`, `docs/WORKLOG.md` |

## Autonomous experiments (local Qwen)

| Loop | Experiments | Kept | Discarded by the judge | Blocked by a guardrail | Crashed or timed out |
|---|---|---|---|---|---|
| Part 1 main model | 316 | 9 | 267 | 7 | 33 |
| Part 1 frequency | 55 | 0 | 42 | 2 | 11 |
| Part 1 boosting model | 89 | 0 | 88 | 0 | 1 |
| Part 1 severity | 113 | 1 | 87 | 4 | 21 |
| Part 1 residuals | 64 | 1 | 62 | 0 | 1 |
| Part 1 claims history | 94 | 0 | 84 | 0 | 10 |
| Part 2 emerging market | 124 | 4 | 110 | 0 | 10 |

Source: each loop's `results.tsv`. Guardrails refuse code that reads the validation target, imports modules or changes the function contract. A kept change enters a submission only after it is re-run on our three judges (Part 1) or the seven-check bench (Part 2). One Qwen model is in the Part 1 final blend, after its wrong monotone constraint on vehicle age was found and fixed.

## AI errors caught

| Error | Found by | Fix |
|---|---|---|
| Qwen boosting model: wrong monotone constraint on vehicle age | Claude, re-checking the Qwen code review | fixed before it entered the blend |
| Office-premium table used the risk premium of an older submission | independent recomputation | rebuilt from the final submission |
| Report said contents gender effect +12%; the final model has +9% | independent recomputation | corrected |
| A reused staging folder copied a stale model file to the Mac | md5 check after every copy | new staging folder and md5 check for every copy |
| A calibration that looked good on our folds lost 0.00035 on the leaderboard | leaderboard check | post-hoc transforms of predictions banned |

## Rules the actuary set

1. No model before the target reconciles.
2. Change the model or an assumption, never the predictions afterwards.
3. A change is kept only if it improves the internal judges across folds; the leaderboard is only a check (in Part 1 the plain structural model held best on the private board).
4. The agent states its plan and expected effect before every test.
5. Competition data never in the repository; the repository stays private until submission; credentials only through the shell, never seen by the AI.
6. A written log in every folder an agent changes.
