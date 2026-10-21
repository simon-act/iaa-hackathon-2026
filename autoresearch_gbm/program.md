# Research program — GBM of the team (IAA Hackathon 2026, from 20/10)

You improve ONE file, `experiment.py`: a bagged LightGBM Tweedie model that predicts the 2025 expected claim cost of a
policy row. A frozen judge scores it. One idea per iteration; be concrete.

## Contract
`fit_predict(train, valid, line)`: `train` = allowed features + `exposure` + `target` (cost at 2025 level);
`valid` = features + `exposure`. Return the expected cost per valid row (finite, >= 0). Lines: motor, buildings, contents.

## Score (read carefully: it changed on 20/10)
Temporal folds: fit 2021–22 → score 2023; fit 2021–23 → score 2024. Per line and fold the judge computes a Tweedie D²
(p = 1.5) on the **reported, developed cost of the scored year**, comparing it with your prediction minus the large-loss
load times the expected reporting completeness of the row, after rescaling to the same total. It judges the SHAPE of
your predictions (who is riskier than whom), not the level. `pooled_mean` = the WORSE fold (an idea must help in both
years). The old judge (cost with IBNR spread evenly by exposure) rewarded flat predictions; it is still printed as
`official_folds` but it does NOT decide.
Baseline ≈ 0.10 on this judge. Real gains are small: +0.0003 on pooled_mean is real, +0.001 is big.

## What we know (verified by the team on the data, 20/10)
- The data come from a generator with a per-peril frequency × severity structure; most rating effects are smooth and
  roughly multiplicative. Interactions are rare.
- This GBM, compared row by row with the team's frequency × severity model, is WRONG where the two disagree: rows it
  prices higher than the other model have actual/expected 0.77–0.82, rows it prices lower have 1.22–1.38 (motor,
  buildings, contents). In other words its relativities are partly noise in some directions and too flat in others.
- It under-uses vehicle value in motor (residual signal in vehicle_value_group and engine_size_cc in both years).
- In buildings, days_unoccupied carries a little signal both models under-use.
- Verified noise (do not add back): previous_insurer, campaign_code, branch_id, marketing_segment, make_model (no
  residual signal), body_type, seats, driver_assist, fuel_type, vehicle_colour, transmission, telematics_opt_in,
  marital_status, customer_tenure, marketing_consent, mileage_k, licence_age, start_month.
- Monotone constraints are allowed only where the data are monotone: sum_insured (home), flood_zone (home),
  building_age (buildings). NOT vehicle_age (hump-shaped).

## Directions worth trying (ONE per iteration)
1. Regularisation that reduces noisy relativities: more min_data_in_leaf, stronger lambda_l2, fewer leaves, path_smooth,
   extra_trees, lower feature_fraction — per line.
2. Better use of vehicle value: interactions value × age, log value, value bands; vehicle_value_group as numeric.
3. A per-peril-like structure: separate models or an offset per strong driver (flood_zone, sum_insured) then GBM on the
   residual (init_score).
4. Number of rounds / learning rate per line (early stopping on a time split INSIDE the training years only).
5. days_unoccupied for buildings; feature engineering on weather columns.

## Rules (enforced by code)
- Keep `fit_predict(train, valid, line)` with the same signature. Never use a target of valid.
- Allowed imports: numpy, pandas, lightgbm, sklearn, scipy, math, itertools, functools, collections, re, warnings, os (only os.environ).
- No file, network or process access. Do not repeat ideas already discarded in "Recent attempts".
- Do not post-process predictions with powers or per-segment multipliers: change the model, not its output.

## Output format (strict)
First line: `IDEA: <one sentence describing the single change>`
Then the COMPLETE new `experiment.py` inside one ```python code block. Nothing after the code block.
