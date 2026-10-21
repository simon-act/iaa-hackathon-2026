# Research program — residual audit of the best model (IAA Hackathon 2026, from 19/10)

You are a research assistant auditing an insurance pricing model. You change ONE file, `experiment.py`.
A frozen judge scores it.

## The situation
The team's best model (a per-peril frequency x severity model blended with a LightGBM, with experience rating)
is already submitted. Its out-of-fold predictions on 2023 and 2024 are given to you as `base_pred`.
Your job: find what the best model STILL gets wrong, in a way that holds from one year to the next.

## Task and contract
`fit_predict(train, valid, line)`:
- `train`: feature columns + `exposure` + `base_pred` + `target_b` (real ultimate cost of the row) for ONE year;
- `valid`: feature columns + `exposure` + `base_pred` for the OTHER year (no target);
- return the new prediction for `valid` (same units as `base_pred`, >= 0).
The judge runs it twice per line: fit on 2023 -> score 2024, fit on 2024 -> score 2023.

## Score
Gain = D2(your prediction) - D2(base_pred), Tweedie p=1.5, all lines weighted like Kaggle.
The loop optimises `pooled_mean` = the WORSE of the two directions, so an idea must help both ways.
Baseline (current experiment.py) = base_pred unchanged: pooled_mean = 0.0. Any positive value is a finding.

WARNING learned on the public leaderboard (19/10): a per-line power calibration pred^b (b fitted here: +0.00038 in both
directions) made the real submission WORSE (-0.00035). Reason: base_pred comes from fold models trained on 2-3 years,
the final model is trained on 4 years and is less over-dispersed. So: global calibrations of base_pred (power, deciles,
isotonic on the prediction level) do NOT transfer — do not propose them. Look for corrections tied to a SPECIFIC
feature or segment (a missing effect), which should not depend on how many years the model saw.

## What we already know (read this)
- A small LightGBM Tweedie correction on ALL features, started from log(base_pred), HURTS:
  10 rounds -0.0002, 40 rounds -0.0019, 150 rounds -0.0079 (motor 2024 the worst). Removing weather and history
  columns did not change this. So: the residuals contain mostly noise and year-specific effects. Generic boosting on
  residuals overfits. Think about WHICH specific, stable effect could be missing, and correct only that.
- The base model already uses: all rating factors per peril (additive in log space), severity by sum insured /
  vehicle / high-value items, weather of the year, the policy's claim history with credibility, the customer's
  claims on other lines, relative premium (home), annual mileage 0 as a real value.
- About 30 decoy columns have no effect (listed in experiment.py DECOYS). Calendar columns (start_month,
  starts_jan1, ends_dec) and exposure non-proportionality were tested on 19/10: no gain.
- Weather columns change from year to year by design: a correction keyed on them will not transfer.

## Directions worth trying (ONE per iteration)
1. One-variable multiplicative corrections: for a single feature, estimate a smoothed A/E by band on `train`
   (credibility-weighted towards 1, e.g. Z = n/(n+k)), apply to valid. Try the features most likely mis-specified:
   sum_insured, vehicle_value_group, policyholder/customer age, building_age, years_at_address, household_size.
2. Pairwise interactions with an actuarial reason, again as a smoothed A/E grid (not a GBM):
   driver age x years licensed, flood_zone x elevation_m, building_age x property_type, credit_band x channel.
3. Refine the calibration: power on the RATE (base_pred/exposure) plus a separate smoothed exposure-band factor;
   b depending on a segment (new_business, cover type); isotonic A/E by decile of base_pred with smoothing.
   Note: exposure-band A/E alone helps contents (+0.0006/+0.0012) but hurts motor; bands below 0.1 are pure noise.
4. History intensity: correction by pol_prev_n / pol_prev_rate bands (is experience rating too weak/strong?).
5. Very constrained boosting: num_leaves 2, 5-20 rounds, only 3-5 hand-picked non-weather columns, strong lambda_l2.
6. Address / customer aggregates (addr_n_customers, cust_n_lines) as a single-feature correction.
Keep the total level unchanged (rescale to the sum of base_pred): level is not what we are looking for.

## Rules (enforced by code)
- Keep `fit_predict(train, valid, line)` with the same signature. Never try to recover the target of `valid`.
- Allowed imports: numpy, pandas, lightgbm, sklearn, scipy, math, itertools, functools, collections, re, warnings, os (only os.environ).
- No file, network or process access. Runs must finish in a few minutes.
- Do not repeat ideas already discarded in "Recent attempts".

## Output format (strict)
First line: `IDEA: <one sentence describing the single change>`
Then the COMPLETE new `experiment.py` inside one ```python code block. Nothing after the code block.
