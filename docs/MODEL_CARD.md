# Model card

Expected 2025 claim cost (risk premium) per policy row, motor and home (buildings, contents), in a developed market
(Part 1) and an emerging market (Part 2). Metric: Tweedie deviance D², p = 1.5.

## Final submissions

| Part | File | md5 | Rebuilt by |
|---|---|---|---|
| 1 — developed | `sub_51` | `1f3c985a9506` | `run_all.sh configs/parte1_s3.json`, `run_final.sh` |
| 2 — emerging | `p2_sub_21_auto_prem_costo` | `c126730083a5` | command block "Part 2" in the README |

Probe not selected (30/10): `p2_sonda_tw05_nbx`, the motor correction GLM with the exact Tweedie offset (weight prior^0.5)
and new-business-specific terms. +0.005 on the public third, but its new-business terms change sign between 2023 and 2024,
it fails both benches (strict bench −26 / −17 e·4), and applied to a random half of the motor rows it keeps the whole gain:
the gain is concentrated on few rows (likely a few large claims in the public third). Kaggle allows one selection: p2_sub_21.
Scripts: `scripts/p2_bench33.py`, `scripts/p2_sub_tw05_nbx.py`.

## 1. Target

Built like a reserve, then priced:
- **Cleaning**: exposure from dates, sums insured in wrong units, duplicate and out-of-period claims, repudiated claims.
- **Development**: reporting-delay curves per peril estimated from the claims data (loss date to report date; reverse-time
  estimator because the data are truncated at 31/12/2024) → expected completeness per row, IBNR counts and costs; IBNER
  by credibility.
- **Trend**: log-linear per peril; none on weather perils (2025 weather is known per row).
- **Large losses**: capped and reloaded on the line.

## 2. Part 1 model (developed market)

| Piece | Specification |
|---|---|
| Frequency per peril | additive Poisson model: a GAM fitted by boosting with one-split trees (LightGBM), so no interactions — like a log-link GLM whose shapes are learned instead of banded by hand; offset log(exposure × completeness) − trend |
| Severity per peril | Gamma on a few plausible drivers (GLM on log sum insured for small home perils; constant for motor bodily injury); large-loss load |
| Experience rating | Bühlmann–Straub on the policy's past claims, M = (N + k)/(E + k); k = 16 motor, 8 home; subsidence excluded in buildings |
| Second model | LightGBM Tweedie written by a local LLM in an autoresearch loop, trained on a gross target (reported ÷ completeness) |
| Blend per line | motor 0.9 / 0.1, buildings 0.6 / 0.4, contents 0.8 / 0.2 (frequency × severity / LightGBM) |

Main code: `src/models_extra.py` (`_fs2`), `src/experience.py`, `scripts/build_final.py`.

## 3. Part 2 model (emerging market)

**Prior × level × corrections.** Two years of new data cannot support a model of their own, so they adjust one.

| Piece | Specification |
|---|---|
| Prior | Part 1 frequency × severity refitted on the developed market with the columns both markets share (`src/transfer.py`); for home, the Part 1 variant with relative premium (`scripts/p2_export_p1.py`) |
| Development and trend | borrowed from the developed market: delay curves estimated on the emerging claims match the developed ones per peril (backtest 2023 → 2024: actual/expected 0.97–1.07 on 15 perils); large-loss load rescaled by the emerging frequency of large claims |
| Level | per line from the emerging data with credibility; own level for storm (buildings); per-peril levels in contents |
| Motor correction | GLM on top of the prior (prior as offset): ½ (new business, gender, relative premium) + ½ (same + fuel type, cover type); a Poisson boost on claim counts |
| Home corrections | gender and working-from-home (contents); 10% of an emerging-only model in buildings |

Relative premium = log(written premium ÷ (1 − discount)) − median of the year. It carries local rating information the
emerging data lack (no credit band, mileage or telematics) and sets the slope, not the level.

## 4. Validation

- **Part 1**: frozen judge (`src/prepare.py`): fit 2021–22 → score 2023, fit 2021–23 → score 2024, 10% policy holdout,
  forbidden columns. Three judges: official-style target, own target without spreading of IBNR and large losses, true
  large losses. A change is kept only if it improves both years on the official and own judges, inside the final blend.
  Out-of-fold D² 0.1290 against 0.1285 ± 0.002 for a perfect model with the same information (simulation,
  `scripts/ceiling_s3.py`).
- **Part 2**: prior against emerging-only models, fit 2023 → score 2024: motor 0.146 vs 0.131, buildings 0.085 vs 0.025,
  contents 0.118 vs 0.061; small-market simulation inside the developed data, prior wins 16 of 18 (`scripts/p2_sim.py`).
  Every correction passed a seven-check bench with a rule fixed in advance (≥ 5/7 on shape and own judge) and was
  re-checked on a stricter bench (time direction × three disjoint customer groups, `scripts/p2_bench27.py`) and on a bench
  reweighted to the 2025 mix (`scripts/p2_bench32.py`). Relative premium in motor: +47 / +32 e·4 against no premium.
- **Contract tests**: `python -m pytest hackathon_checks/tests` (format, outliers, line levels, excluded columns).

## 5. Limits

1. The motor correction GLM uses the response y/prior with weight prior: an exact offset for Poisson; for Tweedie p = 1.5
   the exact form would weight by prior^0.5. The weighted form is the one measured on the bench; the exact form did not
   pass it.
2. The claim-count boost starts from the GLM correction; where that correction is about severity, the boost can absorb
   part of it (limited by regularisation and early stopping; the bench measures the net result).
3. Contents frequency corrections use the raw prior as offset, not the prior already corrected for gender and levels.
4. Two years of emerging data: buildings level uncertain by about ±16%; 2025 new business (76% of the book, from an
   acquired broker portfolio) has no claims of its own and is priced through the new-business correction.
5. The relative premium is a transfer device, not a lasting rating factor: predictions depend on the centring convention
   (median of the year's whole book, the same rule in train and test), and once a new tariff is live the variable becomes
   circular and must be refitted. Bodily-injury development was re-checked at equal age (emerging vs developed accident
   years 2023–24): open shares and reported severity move together (severity ratio 0.62–0.65), so the borrowed development stands.
6. Part 2 rebuilt from raw data is equivalent but not bit-identical (LightGBM multi-threading); from the saved priors the
   chain gives the identical file.

## 6. Office premium (Part 2 reference policies)

Built bottom-up from the risk premium of the final model: `scripts/p2_office_premium.py`; results in the report appendix.
