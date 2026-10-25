# Research program — severity model (IAA Hackathon 2026, from 19/10 afternoon)

You are a research assistant improving the SEVERITY half of an insurance pricing model. You change ONE file,
`experiment.py`. A frozen judge scores it. Runs take seconds: be bold, but one idea per iteration.

## Task and contract
`fit_predict(train, valid, line, peril)`:
- `train`: one row per claim of that line and peril in the fitting years, with the policy-row features + `sev`
  (claim cost > 0, capped, developed to ultimate, at 2025 price level);
- `valid`: the claims of the following year, features only (no target);
- return the predicted mean severity for each valid row (finite, > 0).
Lines: motor (perils bodily_injury, glass_hail, own_damage, theft, third_party_prop), buildings (accidental_damage,
escape_of_water, fire, flood, storm, subsidence), contents (accidental_damage, escape_of_water, flood, theft).
Claims per peril range from ~1,100 (fire) to ~48,000 (motor own damage).

## Score
Gamma D² on the scored year = 1 - deviance(pred) / deviance(mean of the fit years). Two temporal folds:
fit <= 2022 -> score 2023, fit <= 2023 -> score 2024. Perils weighted by cost, lines like Kaggle.
`pooled_mean` = the worse fold. Baseline (current experiment.py, the team's model) = 0.02685 / 0.0342.
Severity is mostly noise: +0.002 on pooled_mean is a real gain, +0.005 would be big.

## What we know about how the data were generated
- Severity is close to constant per peril, with a few real drivers: sum_insured (buildings, contents; roughly a
  power law), vehicle value group and vehicle age (motor own damage, theft), make_model (motor), high_value_items
  (contents theft). Most other columns are noise on severity (and ~30 columns are decoys everywhere:
  engine_size_cc, body_type, seats, driver_assist, fuel_type, vehicle_colour, transmission, telematics_opt_in,
  marital_status, solar_panels, document_preference, customer_tenure, listed_building, marketing_consent, mortgage,
  smoke_alarms, lodgers, bicycle_cover, previous_insurer, campaign_code, branch_id, marketing_segment,
  start_month, starts_jan1, ends_dec, mileage_k, licence_age, pol_years_seen, cust_has_*).
- Open question we want answered: the main model under-predicts expensive cars (A/E rises from 0.95 to 1.10 with
  vehicle_value_group) — is there a severity effect of vehicle value (or make_model, vehicle_age) that the
  1-split Gamma GBM with strong L2 misses, e.g. a multiplicative slope in log(value), or an interaction value x age?
- Heavy tails: bodily_injury, fire, flood, subsidence have few claims and huge variance. Robust choices (fewer
  parameters, shrinkage towards the mean, log-linear GLMs) usually beat flexible ones there.

## Update (after iteration 9)
- The kept change (MIN_CLAIMS 500, L2 100, 100 rounds) only added models for the small perils that the baseline left at
  the mean. The TEAM'S REAL MODEL already fits those perils with a Gamma GLM on log(sum_insured) (power law), whose
  out-of-fold D² is: fire 0.045/0.053, buildings flood 0.045/0.036, subsidence 0.028/0.021, contents flood 0.076/0.073
  (2023/2024). The current experiment.py is about equal to that. So the bar to beat per peril is now: those numbers for
  the small perils, and the baseline numbers for the others. Gains must come from BETTER drivers or functional forms,
  not from switching a peril from "mean" to "a model".
- Target encoding of make_model with k=20 overfits badly (-0.012). Ridge/IRLS GLMs written by hand have produced
  invalid predictions 6 times: if you use a GLM, use sklearn GammaRegressor with standardized log inputs.

## Directions worth trying (ONE per iteration)
1. Gamma GLM with log link (sklearn GammaRegressor or statsmodels-free numpy IRLS) on 1–3 numeric drivers:
   log(sum_insured), vehicle_value_group, vehicle_age — compare with the GBM.
2. Hyper-parameters of the GBM per peril size: lambda_l2, rounds, learning rate, min_data_in_leaf, 2-leaf vs 3-leaf.
3. Target encoding of make_model with credibility (k claims) computed on train only, then as a feature.
4. Drivers not in the list: excess columns (voluntary_excess, compulsory_excess reduce the net claim), bedrooms,
   year_built / building_age, property_type for water and storm, cover_type for motor.
5. Smaller MIN_CLAIMS with a shrunk model for fire / flood / subsidence (today they fall back to the mean).
6. Blend of GLM and GBM; a log-normal fit instead of Gamma; winsorising the training target.

## Rules (enforced by code)
- Keep `fit_predict(train, valid, line, peril)` with the same signature. Never try to recover `sev` of valid.
- Allowed imports: numpy, pandas, lightgbm, sklearn, scipy, math, itertools, functools, collections, re, warnings, os (only os.environ).
- No file, network or process access. Do not repeat ideas already discarded in "Recent attempts".

## Output format (strict)
First line: `IDEA: <one sentence describing the single change>`
Then the COMPLETE new `experiment.py` inside one ```python code block. Nothing after the code block.
