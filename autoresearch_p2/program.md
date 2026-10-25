# Research program — Part 2, emerging market (IAA Hackathon 2026, 25/10)

You change ONE file, `experiment.py`. A frozen judge scores it. Write clear, simple, actuarially sensible code.

## The situation
An insurer enters a new ("emerging") market: 2 policy years (2023–24), ~47k motor, ~27k buildings, ~35k contents rows.
For every row we have a PRIOR from the established market (4 years, 2.4M rows): the Part 1 frequency x severity model
re-fitted on the columns both markets share. Columns: `prior_mu_src` (expected cost of the row), `prior_f_<peril>`
(expected claim count, 2025 level) and `prior_s_<peril>` (expected severity), `prior_c_src` (calibration), `prior_ll_src`.
Target `target_b` = ultimate claim cost of the row at the 2025 level (only in train).
On this market the prior x one level factor beats any model fitted on the emerging data alone, by a wide margin.

## Contract
`fit_predict(train, valid, line)` -> predicted cost for `valid` (>= 0, same units as target_b). Fit on `train`, apply
to `valid`. Seven checks (2023 -> 2024, 2024 -> 2023, five policy groups across both years). Judge = gain over
"prior x level", shape only (each prediction is rescaled to the actual total). pooled_mean = min(mean gain on reported
cost without IBNR spreading, mean gain on the official-like target). Kept if it beats the best by 0.00003.

## What we already know (25/10)
- Gender (exists only in the emerging market) helps in motor (men +9%, women -9%) and contents (women +12%); not buildings.
- Region (E01–E06): no robust effect; in buildings it overfits fire/storm years badly.
- New business in motor: the extra cost in the data comes from 5 large claims (22% of new-business cost); without them
  new business is at par. Do not chase it.
- Broad corrections (GLM on 10 variables, GBM on all features) fitted on emerging data make things worse.
- Weather extrapolation (emerging hail days up to 40, developed max ~19): glass/hail actual/expected rises 0.86 -> 1.14
  with hail days; storm vs wind exposure slightly; gains tiny.
- (25/10 evening, already tested on the full 7-check bench — do not repeat): policy experience rating (credibility on own claim
  history) makes all lines worse; mixing with an emerging-only model (10–50%) fails; per-peril level factors with credibility help
  CONTENTS (k 100 claims frequency / 200 severity) but not motor or buildings; fixed GLM penalties instead of the automatic one: no
  gain; new business by channel: no gain. Motor new-business excess is severity from 5 bodily-injury claims, frequency is at par.
- Age is given in bands (driver_age_band, policyholder_age_band); the prior already uses them.

## Directions worth trying (ONE per iteration)
1. Better credibility for the gender factor (per peril via prior_f_*, or a Bühlmann k chosen on train only).
2. Per-peril level factors (actual counts vs prior_f_* x completeness are not available: use cost ratios per row group)
   with strong shrinkage to the line level.
3. Interactions of gender with age band, or age band corrections, credibility-weighted.
4. Weather slopes per peril (hail_days, wind_exposure_index, rainfall_mm), shrunk.
5. A small, heavily regularised model of the ratio target_b / (prior x level) on a FEW variables, with shrinkage.
Keep the method simple and explain it in the IDEA line.

## Rules (enforced)
- Keep `fit_predict(train, valid, line)` with the same signature. Never try to recover the target of `valid`.
- Allowed imports: numpy, pandas, scipy, sklearn, lightgbm, math, itertools, functools, collections, re, warnings.
- No file, network or process access. Must finish in a few minutes.
- Do not repeat ideas already discarded in "Recent attempts".

## Output format (strict)
First line: `IDEA: <one sentence describing the single change>`
Then the COMPLETE new `experiment.py` inside one ```python code block. Nothing after the code block.
