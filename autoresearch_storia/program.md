# Research program — experience rating (claims history) of the best model (IAA Hackathon 2026, 21/10)

You change ONE file, `experiment.py`. A frozen judge scores it. Write clear, simple, actuarially sensible code.

## The situation
The best model (per-peril frequency x severity blended with a LightGBM) predicts the 2025 claim cost of each policy row.
Out-of-fold predictions for 2023 and 2024 are given as `base_pred`. They already include experience rating:
  M = (N + k) / (E + k), k = 8 expected claims, N = past reported claims of the policy in the same line
  (all perils; in buildings subsidence excluded), E = hist_expo_pi * (base_nocred / (exposure * sev_line_mean)).
`base_nocred` is the same prediction WITHOUT that factor. Simulation says each policy has a latent risk (frailty,
variance ~0.3) that rating factors cannot see; history recovers only part of it. Your job: use the history better.

## Contract
`fit_predict(train, valid, line)` -> new prediction for `valid` (>= 0, same units). Fit anything on `train`
(one year, with targets), apply to `valid` (the other year). Judge: D2 on reported cost (no IBNR spreading),
pooled_mean = the WORSE of the two directions, lines weighted like Kaggle. Level per line and year is reset to
base_pred, so only the SHAPE counts. Baseline pooled_mean = 0.0. A change is kept if it beats the best by 0.00003.

## History columns (storia.parquet; all only from years BEFORE the row's year)
- hist_n_<peril>: past reported claims of the policy for that peril; last_n_<peril>: same, previous year only
- hist_expo_pi: sum over past rows of exposure x expected reporting completeness (the "exposure" of the history)
- hist_years: number of past rows of the policy; sev_line_mean: average claim cost of the line
- cust_other_prev_n, cust_other_prev_n_last: the customer's past claims on OTHER lines (motor <-> home)
- addr_prev_n: past claims at the same address (other policies); pol_prev_cost: past claim cost of the policy
Perils: motor = bodily_injury, glass_hail, own_damage, theft, third_party_prop; buildings = accidental_damage,
escape_of_water, fire, flood, storm, subsidence; contents = accidental_damage, escape_of_water, flood, theft.

## What we already know (21/10)
- k = 8 is the best single k for all lines (grid 1..16); excluding flood/storm/glass_hail/theft from N HURTS;
  excluding subsidence in buildings helps (already in base_pred).
- One-way residuals of base_pred on the own judge, cross-fitted between years: in buildings, rows whose customer had
  claims on OTHER lines have actual/expected 0.73-0.80 (probably the same weather event counted twice), rows with
  own past claims 1.1-1.3 (experience rating slightly too weak in buildings). Small effects (+0.0001 at most).
- Weather variables change every year by design: a correction keyed on them will not transfer.

## Directions worth trying (ONE per iteration)
1. Recompute the factor from base_nocred with per-peril N and E (perils weighted by their own expected counts),
   or with separate k for frequent and rare perils.
2. Recency: weight last year's claims more than older ones (N_w = sum w^lag * n, E_w likewise).
3. Combine sources: M = (N_pol + a*N_cust_other + b*N_addr + k) / (E_pol + a*E_cust + b*E_addr + k) with E for the
   customer/address approximated by their exposure x average frequency; a, b fitted on train (try also negative a).
4. Different k per line, chosen on train only (small grid).
5. Severity history: past cost per past claim vs expected severity, credibility-weighted, as a cost factor.
Keep the method simple and explain it in the IDEA line.

## Rules (enforced)
- Keep `fit_predict(train, valid, line)` with the same signature. Never try to recover the target of `valid`.
- Allowed imports: numpy, pandas, scipy, math, itertools, functools, collections, re, warnings.
- No file, network or process access. Must finish in a few minutes.
- Do not repeat ideas already discarded in "Recent attempts".

## Output format (strict)
First line: `IDEA: <one sentence describing the single change>`
Then the COMPLETE new `experiment.py` inside one ```python code block. Nothing after the code block.
