# Verification of Qwen's findings (19/10, 23:45) — Claude

The review took 79 minutes and produced 125 findings. In the second pass Qwen itself refuted 93 of them (FALSE_POSITIVE). That leaves 3 real and 8 uncertain; 21 were not re-checked, almost all of low severity. Each of the 11 was checked by hand in the container.

| # | Finding | Verification | Action |
|---|---|---|---|
| 1 | develop.py:71 — lines in a fixed tuple instead of `LINES` | **Real** (Part 2 with different lines) | Fixed: `for line in LINES`. Part 1 identical |
| 2 | develop.py:86 — mature AYs fixed at `<= 2022` | **Real**: with a 2023–25 market no AY would be mature | Fixed: `<= YEARS[-1] − 2`. Part 1 (2022) and mock market (2022) identical |
| 3 | experience.py:28 — credibility `sev` estimated on all years, including those scored in the folds | **Real but negligible**: it is one scalar per line; using only the fitting years changes it by 0.5–2% (motor 4,181 vs 4,087/4,125), an effect on D² much smaller than k 6→12 (≤ 0.0002) | Documented, not changed |
| 4–5 | clean_claims.py:92 — `recoveries` column assumed to exist | Uncertain → **Part 2 robustness** | If missing: recoveries = 0, with a warning |
| 6–8 | features.py:135–138 — `birth_year`, `driver_age`, `years_licensed`, `annual_mileage` assumed to exist | Uncertain → **Part 2 robustness** | Derived columns computed only if their sources exist |
| 9 | build_datasets.py:115 — `credit_band` in the report | Only the control report: the config renames are applied before it | None |
| 10 | models_extra.py:44 — fixed `SEV_DRIVERS` | The code already filters with `if c in base_cols` | None |
| 11 | holdout_weights.py:27 — `row_id` assumed to exist | `row_id` is created by the pipeline (clean_policies), it does not come from the market data | None |

Summary: no relevant leakage and no bug that changes the Part 1 numbers. 4 genuine Part 2 fragilities remain, now fixed. The false-positive rate is high (93 of 104 high/medium), but the second pass filters them well: of the 11 that remained, 7 were genuine points, albeit small.
