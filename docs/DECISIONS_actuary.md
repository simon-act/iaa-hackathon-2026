# Decisions register

The decisions taken by the actuary (Simone) during the hackathon, in date order. Each row gives the decision, where it came
from and the evidence it rests on. Where an idea came from the agents' own analysis, the "Origin" column says so; the
decision to adopt or reject it was the actuary's. The work behind each row is in `docs/WORKLOG.md`; the final model is
summarised in `docs/MODEL_CARD.md`.

Units: Δ in e·4 of D² (Tweedie deviance, p = 1.5) unless stated.

## Method rules (set before or during Part 1, binding for every agent)

| Date | Rule | Why |
|---|---|---|
| 16/10 | Local validation drives the work; a submission is made only for a clear internal improvement, not to use the daily quota. | Avoid tuning to the public third of the test. |
| 16/10 | Large losses capped and reloaded on the line (the 11.9 M bodily-injury claim included); model on claim counts with a completeness offset, ultimate-cost target only as a check. | Reserving view of the target: development first, then pricing. |
| 20/10 | No transformation of predictions after the model (no power calibration, no multipliers). Every change must be a model or an actuarial assumption with a written reason. Kept only if it improves both time folds on the official and on the own (no-spreading) judge. The leaderboard is a final check only. | A power calibration that looked good on the folds lost on the leaderboard (19/10); the review of the target showed the official-style judge favours flat predictions. |
| 20/10 | Judge changes inside the final blend, not model by model. | A frequency change that passed alone disappeared inside the blend. |
| 24–25/10 | Part 2: fixed rule before testing — a correction is kept if it wins ≥ 5 of 7 checks on both the shape judge and the own judge. | Two years of emerging data: few, noisy checks; the rule must not be chosen after seeing results. |
| 28/10 | Final Part 2 pieces re-checked on a stricter bench (time direction × three disjoint customer groups, 6 checks; mean > 0, ≥ 4/6, positive in both directions). | The K folds of the 7-check bench shared years, so it was less strict than stated. |
| all | Competition data never in the repository; credentials handled only by the actuary's shell. | Competition rules §2.5; security. |

## Part 1 — developed market

| Date | Decision | Origin | Evidence | Outcome |
|---|---|---|---|---|
| 17/10 | Reconstruct the data-generating process before choosing the model. | Actuary | Generator found multiplicative, few interactions, ~30 decoy columns (`docs/WORKLOG.md`, 17/10). | Led to additive frequency × severity per peril, the main model. |
| 17/10 | Study where models disagree. | Actuary | Disagreement concentrated on policies with past claims → latent policy risk (`docs/WORKLOG.md`, 17/10). | Led to experience rating (Bühlmann–Straub). |
| 17/10 | Use a local LLM (Qwen) in autoresearch loops for search; the main work stays with the coding agent. | Actuary | Seven loops, 855 experiments in total (`docs/AI_ledger.md`). | One Qwen model in the final blend. |
| 18/10 | Test the insurer's written premium as a rating variable (relative to the year, net of discount). | Actuary | Home positive in all folds, motor negative. | Kept in home in Part 1; motor excluded. |
| 18/10 | Zero mileage treated as a real value, not missing. | Agent analysis | Frequency of zero-mileage rows ≈ < 5,000 km; +3 in both folds. | Adopted. |
| 19/10 | Reject the power calibration (sub_21, P1). | Agent analysis | −0.00035 on the leaderboard despite +0.0004 on folds. | Origin of the "no post-hoc transforms" rule. |
| 19/10 | Ask for an external, independent review with recomputed numbers. | Actuary | Found that the official-style target spreads IBNR and large losses uniformly (22% of motor 2024 target). | New own judge without spreading (`src/judge_own.py`). |
| 20/10 | Check whether IBNR and large-loss loads are "flat" and emulate an immature year for the boosting model. | Actuary | Gross target better for motor in both years; buildings equal. | Gross target in motor only. |
| 20/10 | Stack the two models' predictions ("double prediction"). | Actuary | +2/+4 on own judge but buildings coefficients sum to 1.10–1.15 (= steeper, already rejected). | Rejected. |
| 20/10 | Look at correlations between risks within and across lines. | Actuary | Subsidence history behaves differently (A/E 0.62/0.48). | Credibility in buildings excludes subsidence. |
| 20/10 | Correct Qwen's monotone constraint on vehicle age (bell-shaped effect). | Agent verification of the Qwen code review | +3…+11 in motor. | Fixed before the blend. |
| 21/10 | Content flood severity as GBM × GLM mix (sub_41). | Agent analysis, run at the actuary's request | Passes the internal rules (+1.05/+0.24); public −0.00007. | Kept as candidate, not final. |
| 22/10 | Model gross of deductible and apply it afterwards. | Actuary | Amounts carry no deductible trace; relative deductible passes in buildings GBM, two-step approach fails. | Relative deductible kept in buildings (sub_48); two-step rejected. |
| 22/10 | Level probes on the leaderboard (all ×1.03, ×0.97, home ×1.03), as diagnosis only. | Actuary | Optimum ≈ ×1.025, all from home; motor already near its optimum; no assumption found wrong. | No change to the model (rule of 20/10). |
| 22/10 | Choose everything on the official judge as a check. | Agent analysis | Only motor credibility changes (k 8 → 16). | Basis of sub_51. |
| 23/10 | Final Part 1 = sub_51 (motor credibility k = 16). | Actuary | Official judge, frozen holdout and public agree; own judge alone disagrees. | Final submission. |

## Part 2 — emerging market

| Date | Decision | Origin | Evidence | Outcome |
|---|---|---|---|---|
| 23/10 | Use the Part 1 model fitted on the shared columns as prior; emerging data set the level and only the corrections they support. | Agent plan, approved by the actuary | Prior beats emerging-only models in every line (2023 → 2024) and in 16 of 18 simulated small markets. | Structure of every Part 2 submission. |
| 25/10 | Model the acquired broker portfolio (76% new business in 2025). | Actuary | 2025 new business indistinguishable from 2023–24 new business on rating variables (AUC 0.52–0.53); no claims of its own. | New-business correction kept; no reweighting. |
| 26/10 | Own level for the storm peril in buildings. | Agent analysis (per-peril A/E) | 5/7 · 5/7 at the minimum. | Kept. |
| 27/10 | Insurer's relative premium in the motor correction. | Agent analysis | Passes 7/7; strict bench +47/+32 vs no premium. | Kept (sub_19 → sub_21). |
| 27/10 | Part 1 v2 priors (with relative premium) for home. | Agent analysis | 5/7; strict bench +14/+11 buildings, +8 contents. | Kept (sub_20). |
| 28/10 | Premium in the motor cost GLM rather than in frequency. | Agent analysis | 5/7 · 5/7; strict bench +41/+25. | Final model. |
| 28/10 | Challenge: does the premium overfit? | Actuary | Same pattern in both years; new-business/renewal premium ratio stable 1.04–1.08; weighted bench on the 2025 mix +67/+48 (6/6). | Premium kept. |
| 28/10 | Challenge: is the bench right? Explore other benches. | Actuary | Importance-weighted bench (classifier train vs 2025, weights p/(1−p)) and new-business-only bench; they agree in direction with the public score on the three adopted steps, and reject the exact-offset variant that the public score favoured. | Bench confirmed; final unchanged. |
| 28/10 | Try the remaining standard practices (exact Tweedie offset, exposure-rating curve on fire, seasonality). | Actuary | None passes either bench; public gains of the offset variant split into level and a shape the 2023–24 data reject. | Rejected; kept as probes. |
| 28/10 | Final Part 2 = p2_sub_21; no leaderboard positions in the delivered material. | Actuary | Strict bench confirms every piece. | Final submission. |
