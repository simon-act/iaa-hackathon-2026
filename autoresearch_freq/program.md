# Research program — FREQUENCY per peril (IAA Hackathon 2026, from 20/10)

You improve ONE file, `experiment.py`: the frequency part of the team's per-peril frequency × severity model. For one
line and one peril it predicts the expected number of reported claims of each policy row in the next year.
One idea per iteration; be concrete and small.

## Contract
`fit_predict(train, valid, line, peril)` → expected count per valid row (> 0).
`train`: rating features + `exposure` + `year` + `offset` + `n` (claims of this peril); `valid`: same without `n`.
`offset` = log(exposure × expected reporting completeness) − years-to-2025 × log(1 + trend): keep it as the offset
(init_score) — it carries exposure, reporting delay and trend.

## Score
Folds: fit 2021–22 → score 2023; fit 2021–23 → score 2024. Poisson D² of the counts per peril (prediction rescaled to the
observed total: SHAPE only), perils weighted by their share of cost, lines weighted like Kaggle. `pooled_mean` = WORSE
fold. Baseline ≈ 0.11. Gains are small: +0.0003 is real, +0.001 is big.

## What the team already measured with this judge (do not repeat)
- Interactions (2- or 3-level trees) are WORSE: the generator is additive per peril (except subsidence: soil × drought).
- More/fewer trees (×0.5, ×2), lambda_l2 1 or 50, min_data_in_leaf 300/3000, early stopping, path_smooth, no bagging,
  make_model encoding off or with k 10/100: no gain on the whole.
- Helps in buildings (both years): feature_fraction 0.7 (+0.0004/+0.0003) and max_bin 255 (+0.0002/+0.0002), mostly
  fire, flood, escape of water, subsidence (continuous drivers: elevation, distance to river, sum insured, building age).
- Noise variables (do not add back): previous_insurer, campaign_code, branch_id, marketing_segment, body_type, seats,
  driver_assist, fuel_type, vehicle_colour, transmission, telematics_opt_in, marital_status, customer_tenure,
  marketing_consent, start_month, cust_other_prev_*.

## Directions worth trying (ONE per iteration)
1. Per-peril settings: different params for big perils (own damage, third party, glass, water) vs small (fire, flood,
   subsidence, theft) — e.g. feature_fraction / max_bin / rounds per peril.
2. Smooth transforms of continuous drivers that the generator likely uses (log sum insured, distance to river,
   elevation bands, building age), or monotone constraints where the direction is certain for that peril
   (flood: flood_zone ↑, elevation ↓, distance to river ↓; theft: security level ↓).
3. A small GLM (sklearn PoissonRegressor on a few transformed drivers, offset handled by dividing n by exp(offset) with
   sample_weight = exp(offset)) blended with the GBM, or used as init_score for the GBM.
4. Weather variables per peril (storm: wind, roof; flood: rainfall; subsidence: drought × clay; glass_hail: hail days).
5. Feature subsets per peril (drop drivers irrelevant to a peril to cut noise).

## Rules (enforced by code)
- Keep `fit_predict(train, valid, line, peril)` with the same signature. Never use a target of valid.
- Allowed imports: numpy, pandas, lightgbm, sklearn, scipy, math, itertools, functools, collections, re, warnings, os (only os.environ).
- No file, network or process access. Do not repeat ideas already discarded in "Recent attempts".
- Do not post-process predictions with powers or multipliers: change the model.

## Output format (strict)
First line: `IDEA: <one sentence describing the single change>`
Then the COMPLETE new `experiment.py` inside one ```python code block. Nothing after the code block.
