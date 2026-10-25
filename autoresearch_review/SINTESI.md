# Revisione del codice di Qwen (qwen3.8:27b) — sintesi

125 segnalazioni; seconda passata (Qwen prova a smontarle) sulle high/medium. Da verificare a mano nel container prima di agire.

| # | Verdetto | Gravità | Lente | File:righe | Segnalazione |
|---|---|---|---|---|---|
| 1 | REAL | high | part2 | develop.py:86-86 | Hard-coded year 2022 is used to define 'mature' accident years for severity estimation. |
| 2 | REAL | high | part2 | develop.py:71-71 | The loop iterates over a hard-coded tuple of lines ('motor', 'buildings', 'contents') instead of the configured LINES. |
| 3 | REAL | high | leakage | experience.py:28-28 | The severity parameter `sev` is calculated using `tr['target_b']` and claim counts from the entire training set, including the scored years (2023/2024) in the validation folds. |
| 4 | UNSURE | high | bugs | clean_claims.py:92-93 | The code references a column 'recoveries' that is not present in the defined key list or standard claim fields, likely causing a KeyError or silent NaN propagation if the column is missing or misnamed. |
| 5 | UNSURE | high | part2 | clean_claims.py:92-93 | The code assumes the existence of a 'recoveries' column to calculate net incurred and net paid. |
| 6 | UNSURE | high | part2 | features.py:137-137 | Hard-coded column names `driver_age` and `years_licensed` will cause a KeyError if the new market renames these fields. |
| 7 | UNSURE | high | part2 | features.py:135-135 | Hard-coded column name `birth_year` will cause a KeyError if the new market uses a different name for the customer's birth year. |
| 8 | UNSURE | high | part2 | features.py:138-138 | Hard-coded column name `annual_mileage` will cause a KeyError if the new market uses a different name for mileage. |
| 9 | UNSURE | medium | part2 | build_datasets.py:115-115 | The report generation assumes a specific customer column `credit_band` exists in the test dataset. |
| 10 | UNSURE | medium | part2 | holdout_weights.py:27-27 | The code assumes the dataframe `va` has a column named `row_id`. |
| 11 | UNSURE | medium | part2 | models_extra.py:44-44 | The `SEV_DRIVERS` dictionary hard-codes specific feature names for severity modeling. |
| 12 | – | low | part2 | clean_policies.py:56-56 | The threshold for detecting sum_insured scaling issues is derived from a median calculation that assumes a specific distribution of bedrooms and sum insured. |
| 13 | – | low | part2 | develop.py:114-114 | The calculation of `cost_rep` uses a complex formula that assumes `rec_ratio` is less than 1, but does not explicitly clamp it, which could lead to negative costs if `rec_ratio` > 1. |
| 14 | – | low | part2 | final.py:71-71 | The diagnostic print statement assumes the `exposure` column exists in the test dataframe. |
| 15 | – | low | part2 | models_extra.py:331-331 | The default number of boosting rounds is determined by hard-coded claim count thresholds. |
| 16 | – | low | part2 | models_extra.py:774-774 | The days-per-month array assumes a non-leap year (28 days in February). |
| 17 | – | low | part2 | models_extra.py:784-784 | The days-per-month array in `_season_expo` also assumes a non-leap year. |
| 18 | – | None | bugs | audit_new_market.py:159-159 | Using `c.get('recoveries', 0)` on a pandas DataFrame returns the integer 0 if the column is missing, causing a TypeError when subtracting from a Series. |
| 19 | – | None | bugs | audit_new_market.py:146-146 | The merge of claims and policies on 'policy_id' can produce duplicate rows if a policy has multiple periods, and the subsequent `groupby('claim_id').any()` logic is fragile if 'claim_id' is not unique in the claims table. |
| 20 | – | None | part2 | models_extra.py:444-444 | The GLM frequency model relies on a hard-coded dictionary `GLM_DGP` that maps line names to specific column names, which will fail or produce empty models if the new market renames columns or has different lines. |
| 21 | – | None | part2 | models_extra.py:452-454 | The code hard-codes the peril name "subsidence" and specific column names "soil_type" and "dry_summer_index" for an interaction term, which are likely specific to the Part 1 market's schema. |
| 22 | – | None | part2 | models_extra.py:547-547 | The weighting scheme for `gbm_clean_recent` assumes the presence of a "year" column and uses a hard-coded decay factor of 0.8 relative to the max year in the training set. |
| 23 | – | None | part2 | models_extra.py:595-597 | The `_mil0` function hard-codes the column names "flag_mileage_zero" and "annual_mileage" to impute zero mileage, which are specific to the Part 1 motor line schema. |
| 24 | – | None | part2 | models_extra.py:618-618 | The `_prem` function hard-codes the column names "written_premium" and "discount_applied" to calculate a relative premium feature. |
| 25 | – | None | part2 | models_extra.py:458-458 | The regularization parameter `alpha` for the Poisson GLM is hard-coded based on a threshold of 3000 claims (`n.sum() > 3000`), which is tuned to the Part 1 data volume. |
| 26 | – | None | bugs | trend.py:126-126 | The MLE estimator for the Pareto shape parameter alpha is mathematically incorrect. |
| 27 | – | None | bugs | trend.py:55-55 | Exposure is aggregated using 'first' instead of 'sum' when calculating line-level frequency trends. |
| 28 | – | None | part2 | trend.py:126-126 | The Pareto alpha estimation formula divides by the sum of log-returns, which will raise a ZeroDivisionError or produce NaN if the new market has fewer than 2 claims above the threshold or if all exceedances are identical to the threshold. |
| 29 | – | None | part2 | trend.py:129-129 | The calculation of the expected large loss load divides by `(alpha - 1)`, which will raise a ZeroDivisionError if the estimated Pareto shape parameter is exactly 1.0. |
| 30 | – | None | part2 | trend.py:115-115 | The credibility prior calculation assumes the source market's `level_2025.parquet` contains the exact same `line` values as the current market, which may not hold if Part 2 introduces new lines or renames them. |
| 31 | – | None | part2 | trend.py:124-124 | The Pareto large loss method is applied to a hard-coded default list of lines `["motor"]` if not specified in the config, which may not match the lines present in the new market. |
| 32 | – | None | part2 | trend.py:127-127 | The exposure sum for the large loss calculation assumes the `large_loss.parquet` file contains an `exposure` column, which might be missing or named differently in the new market's data pipeline. |
| 33 | FALSE_POSITIVE | high | part2 | audit_new_market.py:112-112 | The code assumes the 'period_start' column exists in the policies table without checking. |
| 34 | FALSE_POSITIVE | high | part2 | audit_new_market.py:133-133 | The code assumes the 'loss_date' and 'report_date' columns exist in the claims table. |
| 35 | FALSE_POSITIVE | high | part2 | audit_new_market.py:159-159 | The code assumes the 'incurred_to_date' column exists in the claims table. |
| 36 | FALSE_POSITIVE | high | bugs | build_datasets.py:70-70 | The reindex operation on the development table uses the policy's 'year' column, which likely contains period_start years (2020-2023) rather than Accident Year (AY) (2021-2024), causing a misalignment or NaNs. |
| 37 | FALSE_POSITIVE | high | bugs | build_datasets.py:78-78 | The trend projection exponent uses 'year' (policy start) instead of Accident Year (AY), causing incorrect scaling for policies with multi-year exposure or start years prior to the accident year. |
| 38 | FALSE_POSITIVE | high | part2 | build_datasets.py:78-78 | The trend projection exponent uses a hard-coded `TARGET_YEAR` (2025) instead of the current market's target year from config. |
| 39 | FALSE_POSITIVE | high | part2 | build_datasets.py:82-82 | The large loss load column name `ll_load_2025` is hard-coded, assuming the level table uses a year-specific suffix. |
| 40 | FALSE_POSITIVE | high | bugs | clean_claims.py:80-84 | The merge on 'policy_id' creates a Cartesian product for policies with multiple rows, causing claims to be duplicated in the final dataset. |
| 41 | FALSE_POSITIVE | high | part2 | clean_claims.py:20-20 | The LONG_TAIL set hard-codes specific peril names ('bodily_injury', 'subsidence') and line names ('motor', 'buildings') which may not exist or be renamed in the new market. |
| 42 | FALSE_POSITIVE | high | bugs | clean_policies.py:53-59 | The merge with the addresses table can produce duplicate rows if `address_id` is not unique in the `addresses` table, causing `df` to grow in length and misaligning the boolean mask `big` used for assignment. |
| 43 | FALSE_POSITIVE | high | part2 | clean_policies.py:20-20 | SI_FACTOR is hard-coded to 1000.0, assuming a specific currency/unit mismatch (e.g., thousands) that may not apply to the new market. |
| 44 | FALSE_POSITIVE | high | bugs | develop.py:114-114 | The calculation for `cost_rep` subtracts the expected future recoveries from the total incurred amount, which is actuarially incorrect as recoveries should be netted against the ultimate cost or treated as a separate credit, not subtracted from the reported cost base used for severity calculation. |
| 45 | FALSE_POSITIVE | high | leakage | experience.py:34-34 | The historical claim count `N` is retrieved from the pre-computed column `pol_prev_n` which is derived from the full dataset, potentially including claims from the scored year or test year depending on how `build_datasets` defines 'prior'. |
| 46 | FALSE_POSITIVE | high | part2 | experience.py:25-25 | The code assumes the existence of `pi_` prefixed columns to determine perils, which may not exist or follow this naming convention in the new market. |
| 47 | FALSE_POSITIVE | high | part2 | experience.py:28-28 | The severity calculation `sev` divides by a sum that can be zero if `n_` columns are all zero or missing, leading to division by zero or NaNs. |
| 48 | FALSE_POSITIVE | high | part2 | experience.py:34-34 | The code assumes the existence of a `pol_prev_n` column in both train and test datasets, which may not be present in the new market. |
| 49 | FALSE_POSITIVE | high | part2 | features.py:58-58 | Hard-coded mapping of policy_id prefixes to lines will fail if the new market uses different prefixes or formats. |
| 50 | FALSE_POSITIVE | high | part2 | features.py:140-140 | Hard-coded column names `sum_insured` and `bedrooms` will cause a KeyError if the new market uses different names for these attributes. |
| 51 | FALSE_POSITIVE | high | part2 | final.py:42-43 | Hard-coded years 2021-2023 for training and 2024 for validation in the holdout mode. |
| 52 | FALSE_POSITIVE | high | part2 | holdout_weights.py:19-19 | Hard-coded years 2021-2023 for training and 2024 for validation will fail or produce empty sets on a new market with different years. |
| 53 | FALSE_POSITIVE | high | part2 | holdout_weights.py:20-20 | Hard-coded model function names `fs2_v2_cred` and `qwen_gbm73_cred` assume specific model implementations exist in `models_extra`. |
| 54 | FALSE_POSITIVE | high | bugs | models_extra.py:89-92 | The severity model is trained on claims from the validation set because the filter `claims["row_id"].isin(train["row_id"])` is incorrect if `train` and `valid` share the same `row_id` values (which they do, as `row_id` is the unique policy-period identifier). |
| 55 | FALSE_POSITIVE | high | part2 | models_extra.py:82-82 | The frequency offset calculation hard-codes the year 2025 as the projection target. |
| 56 | FALSE_POSITIVE | high | part2 | models_extra.py:90-90 | The severity inflation calculation hard-codes the year 2025 as the projection target. |
| 57 | FALSE_POSITIVE | high | part2 | models_extra.py:107-107 | The large loss load lookup hard-codes the year 2025 in the column name. |
| 58 | FALSE_POSITIVE | high | bugs | models_extra.py:349-350 | The normalization constant for the `zero_keep` subsampling path is calculated using the sum of the full training set's raw scores, but the model was trained on a subset, causing a scale mismatch. |
| 59 | FALSE_POSITIVE | high | part2 | models_extra.py:322-322 | The frequency offset calculation hard-codes the year 2025 as the projection target. |
| 60 | FALSE_POSITIVE | high | part2 | models_extra.py:352-352 | The severity trend adjustment hard-codes the year 2025 as the projection target. |
| 61 | FALSE_POSITIVE | high | part2 | models_extra.py:387-387 | The large loss load lookup hard-codes the year 2025 in the column name. |
| 62 | FALSE_POSITIVE | high | leakage | models_extra.py:447-447 | The `_bin` function is called with the validation set's column values to determine binning boundaries, causing the validation set to influence the feature engineering of the training set. |
| 63 | FALSE_POSITIVE | high | leakage | models_extra.py:450-450 | The `_bin` function is called with the validation set's column values to determine binning boundaries for the `mm_freq_te` feature, causing the validation set to influence the feature engineering of the training set. |
| 64 | FALSE_POSITIVE | high | leakage | models_extra.py:453-453 | The `_bin` function is called with the validation set's column values to determine binning boundaries for `soil_type` and `dry_summer_index`, causing the validation set to influence the feature engineering of the training set. |
| 65 | FALSE_POSITIVE | high | bugs | models_extra.py:459-460 | The Poisson GLM frequency model is fitted with an incorrect target and sample weight combination, resulting in a constant prediction equal to the mean count rate. |
| 66 | FALSE_POSITIVE | high | bugs | models_extra.py:741-744 | The loop over accident years (AY) overwrites the cumulative claim count `N_p` with the count for the *earliest* AY instead of the *latest* AY <= previous year. |
| 67 | FALSE_POSITIVE | high | bugs | models_extra.py:831-834 | Duplicate definition of `_peril_hist` contains the same cumulative sum logic error where the loop overwrites `N_p` with the earliest year's count. |
| 68 | FALSE_POSITIVE | high | part2 | models_extra.py:734-734 | The credibility calculation hard-codes the year 2025 as the reference point for trend projection. |
| 69 | FALSE_POSITIVE | high | part2 | models_extra.py:729-730 | The code assumes the presence of `pi_<peril>` columns in the training data, which may not exist or have different names in Part 2. |
| 70 | FALSE_POSITIVE | high | part2 | oof.py:57-57 | Hard-coded line names and weights for the scoring aggregation. |
| 71 | FALSE_POSITIVE | high | part2 | oof.py:59-59 | The scoring formula assumes a fixed set of lines and folds defined in P.LINES and P.FOLDS. |
| 72 | FALSE_POSITIVE | high | bugs | prepare.py:120-121 | The weighted score calculation uses the number of rows in the 2025 test set to weight the validation scores from 2023 and 2024. |
| 73 | FALSE_POSITIVE | high | part2 | prepare.py:32-32 | The fallback for missing model data hard-codes the Part 1 year range [2021, 2022, 2023, 2024]. |
| 74 | FALSE_POSITIVE | high | part2 | prepare.py:55-55 | The drift report hard-codes the training year 2024 for comparison against the test set. |
| 75 | FALSE_POSITIVE | high | part2 | prepare.py:136-137 | The holdout score function hard-codes the training years [2021, 2022, 2023] and validation year 2024. |
| 76 | FALSE_POSITIVE | high | bugs | transfer.py:135-135 | The final calibration factor uses the raw target sum `y.sum()` instead of the exposure-weighted target sum `y.sum() / train['exposure'].sum()`, causing a scale mismatch when exposure is not 1.0. |
| 77 | FALSE_POSITIVE | high | bugs | transfer.py:152-152 | The final calibration factor uses the raw target sum `y.sum()` instead of the exposure-weighted target sum `y.sum() / train['exposure'].sum()`, causing a scale mismatch when exposure is not 1.0. |
| 78 | FALSE_POSITIVE | high | part2 | transfer.py:106-108 | The rare-category threshold is hard-coded to 500, which is tuned for Part 1's large dataset size. |
| 79 | FALSE_POSITIVE | high | part2 | transfer.py:147-147 | The `min_data_in_leaf` parameter is hard-coded with a floor of 2000, which is too large for the smaller Part 2 dataset. |
| 80 | FALSE_POSITIVE | medium | part2 | audit_new_market.py:116-116 | The exposure check assumes a 365-day year and a specific tolerance, which may not hold for leap years or different market conventions. |
| 81 | FALSE_POSITIVE | medium | part2 | audit_new_market.py:158-158 | The code hard-codes specific status values ('repudiated', 'declined', 'rejected') to filter claims. |
| 82 | FALSE_POSITIVE | medium | part2 | audit_new_market.py:168-168 | The code assumes the 'line' column exists in the claims table to filter by line. |
| 83 | FALSE_POSITIVE | medium | leakage | build_datasets.py:89-96 | The target variable `target_b` is adjusted using a reconciliation factor (`rec_adj`) derived from the aggregate development table (`devsum`) which includes the scored year's ultimate costs. |
| 84 | FALSE_POSITIVE | medium | bugs | build_datasets.py:92-92 | The reconciliation adjustment 'rec_adj' is calculated using the sum of 'ult_capped' which includes IBNR estimates, but the development table 'cost_ult' might be defined differently (e.g., excluding IBNR or including different recovery assumptions), leading to a mismatch. |
| 85 | FALSE_POSITIVE | medium | part2 | build_datasets.py:30-31 | The code assumes `customer_since_year` and `year_built` columns exist in the customer/address tables to calculate tenure and age. |
| 86 | FALSE_POSITIVE | medium | part2 | build_datasets.py:114-114 | The report generation assumes a specific weather column `hail_days` exists in the test dataset. |
| 87 | FALSE_POSITIVE | medium | bugs | clean_claims.py:36-37 | The `fix_loss_date` function reads `train_{line}.parquet` which may not exist or may contain different data than the main `p` dataframe used later, leading to inconsistent coverage checks. |
| 88 | FALSE_POSITIVE | medium | part2 | clean_claims.py:87-87 | The VALUATION date is derived from the maximum year of 'period_end' in the training data, assuming the training data contains the most recent year. |
| 89 | FALSE_POSITIVE | medium | part2 | clean_claims.py:94-94 | The CAP values are mapped from CFG['cap'] using the 'line' column, assuming the line names in the new market match the keys in the config. |
| 90 | FALSE_POSITIVE | medium | part2 | clean_policies.py:74-78 | The logic for flagging inconsistent driver age/NCB fields assumes specific column names ('ncb_years', 'years_licensed', 'driver_age') and a minimum driving age of 16. |
| 91 | FALSE_POSITIVE | medium | part2 | clean_policies.py:83-83 | The list of columns excluded from 'missing' imputation is hard-coded, including 'renewal_decision' which may not exist or may be renamed in the new market. |
| 92 | FALSE_POSITIVE | medium | leakage | develop.py:79-80 | The reporting lag CDF (F) is estimated using claims from all years (2021-2024), including the scored year (2023 or 2024) in validation folds. |
| 93 | FALSE_POSITIVE | medium | leakage | develop.py:86-98 | The late-report severity relative curve (rel) is estimated using claims from all years, not just mature years prior to the scored year. |
| 94 | FALSE_POSITIVE | medium | leakage | develop.py:102-103 | The recovery ratio (rec_ratio) is estimated using closed claims from all years, including the scored year. |
| 95 | FALSE_POSITIVE | medium | leakage | develop.py:135-136 | The inflation rate is estimated using mature AYs from all years, potentially including the scored year if it is considered mature. |
| 96 | FALSE_POSITIVE | medium | bugs | develop.py:154-154 | The selection of the reference year (`ref`) for the prior severity level uses `iloc[0]` after sorting by `open_share` and `ay`, which arbitrarily picks the first row if there are ties in `open_share`, and does not explicitly select the most mature year (lowest `open_share`) if multiple years have the same share. |
| 97 | FALSE_POSITIVE | medium | part2 | develop.py:139-139 | A hard-coded default inflation rate of 0.05 (5%) is used when no mature accident years are found. |
| 98 | FALSE_POSITIVE | medium | part2 | develop.py:103-103 | The recovery ratio calculation assumes `incurred_to_date` is non-zero and positive for closed claims, but does not guard against division by zero if `incurred_to_date` is 0. |
| 99 | FALSE_POSITIVE | medium | leakage | experience.py:52-52 | The final prediction is rescaled using the sum of the raw predictions `pred.sum()` and the modified predictions `out.sum()` from the entire validation fold. |
| 100 | FALSE_POSITIVE | medium | part2 | experience.py:30-30 | The code assumes the existence of a `year` column in both train and test datasets, which may not be present or named differently in the new market. |
| 101 | FALSE_POSITIVE | medium | part2 | experience.py:26-26 | The code assumes that `n_` columns exist for all perils derived from `pi_` columns, which may not be true in the new market. |
| 102 | FALSE_POSITIVE | medium | part2 | features.py:55-55 | The `FIRST_YEAR` is derived from the training data, but the logic for `addr_prev_rate` and `cust_other_prev_rate` assumes a continuous history starting from this year. |
| 103 | FALSE_POSITIVE | medium | part2 | final.py:35-35 | Hard-coded model names 'glm_b' and 'gbm_b' are forced into the model list for holdout mode. |
| 104 | FALSE_POSITIVE | medium | part2 | final.py:47-47 | The reference blend `sub01` is hard-coded to use 'glm_b' and 'gbm_b' with fixed 0.5 weights. |
| 105 | FALSE_POSITIVE | medium | part2 | final.py:80-80 | The code assumes `sample_submission.parquet` exists in the raw data directory. |
| 106 | FALSE_POSITIVE | medium | part2 | holdout_weights.py:23-24 | The code assumes `blend_weights.json` contains a key for the current line and a specific model name `qwen_gbm73_cred`. |
| 107 | FALSE_POSITIVE | medium | leakage | models_extra.py:89-90 | The severity model is trained on claims from the training set, which includes claims from the scored year (e.g., 2023 in F2023), violating the temporal validity constraint that development/trend parameters for the scored year should not be used in validation folds. |
| 108 | FALSE_POSITIVE | medium | part2 | models_extra.py:39-39 | The code assumes the existence of a file named `level_2025.parquet`. |
| 109 | FALSE_POSITIVE | medium | part2 | models_extra.py:44-46 | The `SEV_DRIVERS` dictionary hard-codes specific column names for severity drivers. |
| 110 | FALSE_POSITIVE | medium | part2 | models_extra.py:121-129 | The `GLM2_EXTRA` dictionary hard-codes specific column names for additional GLM features. |
| 111 | FALSE_POSITIVE | medium | leakage | models_extra.py:389-391 | The calibration constant `c` is computed using the sum of the target variable from the training set, which includes the scored year (e.g., 2023 or 2024) in the validation folds. |
| 112 | FALSE_POSITIVE | medium | leakage | models_extra.py:351-354 | The mean severity `mean_sev` is calculated using claims from the entire training set, including the scored year. |
| 113 | FALSE_POSITIVE | medium | leakage | models_extra.py:322-322 | The frequency trend `ft` is applied to the scored year using a fixed 2025 projection formula, implying the trend was estimated using data from the scored year. |
| 114 | FALSE_POSITIVE | medium | leakage | models_extra.py:352-352 | The severity trend `st` is applied to claims from the scored year to project them to 2025 levels. |
| 115 | FALSE_POSITIVE | medium | bugs | models_extra.py:351-351 | The severity calculation uses `ibner_factor` and `sev_trend` estimated on all years, including the validation year, which constitutes leakage in the temporal validation folds. |
| 116 | FALSE_POSITIVE | medium | bugs | models_extra.py:322-322 | The frequency trend `ft` is estimated on all years and used to adjust the offset for the validation year, introducing leakage. |
| 117 | FALSE_POSITIVE | medium | part2 | models_extra.py:275-275 | The `SMALL_SEV_SI` dictionary hard-codes specific perils and lines for the small severity model. |
| 118 | FALSE_POSITIVE | medium | leakage | models_extra.py:717-746 | The `_peril_hist` function computes claim history features (N_p) using the full claims table, which includes claims from the validation year (e.g., 2023 or 2024) for rows in that same year. |
| 119 | FALSE_POSITIVE | medium | leakage | models_extra.py:762-780 | The `_season_weights` function estimates seasonal claim distribution weights using all claims in the training set, which includes claims from the validation year (e.g., 2023) when validating on 2023. |
| 120 | FALSE_POSITIVE | medium | leakage | models_extra.py:717-746 | The `_peril_hist` function uses `freq_trend` estimated on all years (including the validation year) to calculate the expected exposure `H_p`. |
| 121 | FALSE_POSITIVE | medium | part2 | models_extra.py:764-764 | The cache key for season weights uses `len(train)` and `max(year)`, which may not uniquely identify the dataset in Part 2. |
| 122 | FALSE_POSITIVE | medium | part2 | models_extra.py:718-718 | The `_peril_hist` function assumes that `claims_clean.parquet` contains a `policy_id` column, which may not be present or may have a different name in Part 2. |
| 123 | FALSE_POSITIVE | medium | part2 | transfer.py:122-122 | The alpha selection uses a fixed 10% hash-based holdout (`% 10 == 3`) which may be too small or unstable for the smaller Part 2 dataset. |
| 124 | FALSE_POSITIVE | medium | part2 | transfer.py:45-46 | The `CORRECTION_VARS` list contains specific column names (e.g., `building_age`, `vehicle_age`) that may not exist or be named differently in the new market. |
| 125 | FALSE_POSITIVE | medium | part2 | transfer.py:135-135 | The final scaling factor `y.sum() / (g.predict(Xtr) * p_tr).sum()` uses the training target sum, which may not align with the new market's claim levels if the prior is significantly off. |

## Dettagli

### 1. develop.py:86-86 (part2, high, REAL)
- **Cosa:** Hard-coded year 2022 is used to define 'mature' accident years for severity estimation.
- **Perché conta:** In a new market with different years (e.g., 2020-2023), this will either include immature years (if 2022 is recent) or exclude mature years (if 2022 is old), leading to biased severity estimates and incorrect IBNR factors.
- **Codice:** `mat = g[g["ay"] <= 2022]`
- **Test proposto:** Run the code on a dataset where the max year is 2023 and check if `mat` includes 2023 claims (it shouldn't if 2023 is immature, but the logic is static).
- **Seconda passata:** Line 86 hard-codes `2022` to filter mature accident years for severity estimation, which violates the requirement to adapt to new markets with different years (e.g., 2020-2023) via configuration. This will cause the model to incorrectly classify years as mature or immature in the new market, leading to biased severity estimates and IBNR factors.

### 2. develop.py:71-71 (part2, high, REAL)
- **Cosa:** The loop iterates over a hard-coded tuple of lines ('motor', 'buildings', 'contents') instead of the configured LINES.
- **Perché conta:** If the new market has different lines (e.g., 'auto' or 'home'), the code will crash with a KeyError when trying to read `train_{line}.parquet` or will silently skip lines if the files don't exist, failing to process the actual market data.
- **Codice:** `for line in ("motor", "buildings", "contents"):`
- **Test proposto:** Rename a line in the config to 'auto' and ensure the parquet file is named 'train_auto.parquet'; the code will fail to process it.
- **Seconda passata:** The code at line 71 explicitly iterates over the hard-coded tuple `("motor", "buildings", "contents")` instead of the `LINES` variable imported from `market_config` at line 22. This violates the project requirement that the pipeline must adapt to new markets with different lines via JSON config, causing a `FileNotFoundError` or `KeyError` if the new market uses different line names (e.g., 'auto').

### 3. experience.py:28-28 (leakage, high, REAL)
- **Cosa:** The severity parameter `sev` is calculated using `tr['target_b']` and claim counts from the entire training set, including the scored years (2023/2024) in the validation folds.
- **Perché conta:** In the `F2023` fold, the `sev` used to normalize the expected claims `E` contains information from 2023 claims, which are the target being predicted, violating the temporal validity of the out-of-fold evaluation.
- **Codice:** `sev = tr["target_b"].sum() / (tr[[f"n_{p}" for p in perils]].to_numpy() / tr[[f"pi_{p}" for p in perils]].clip(lower=0.05).to_numpy()).sum()`
- **Test proposto:** Run the pipeline for fold F2023 and print the value of `sev` calculated on the full training set versus the value calculated only on rows where year < 2023; they will differ, confirming leakage.
- **Seconda passata:** The `history` function (lines 20-36) calculates `sev` using `tr["target_b"]` and claim counts from the entire training set (`P.load(line)`), which includes the scored years (2023/2024) in the validation folds. This violates the temporal validity requirement for out-of-fold evaluation, as the severity parameter used to normalize expected claims in `modifier` (line 42) contains information from the target year.

### 4. clean_claims.py:92-93 (bugs, high, UNSURE)
- **Cosa:** The code references a column 'recoveries' that is not present in the defined key list or standard claim fields, likely causing a KeyError or silent NaN propagation if the column is missing or misnamed.
- **Perché conta:** If 'recoveries' is not a valid column in the raw claims data, this will crash. If it exists but is not the correct field for subrogation/recoveries, the net incurred/paid calculations will be wrong, skewing severity predictions.
- **Codice:** `c["net_incurred"] = c["incurred_to_date"] - c["recoveries"]
    c["net_paid"] = c["paid_to_date"] - c["recoveries"]`
- **Test proposto:** Run `pd.read_parquet(raw / 'claims.parquet').columns` to verify 'recoveries' exists and contains non-null values.
- **Seconda passata:** The code at lines 92-93 references `c["recoveries"]`, but the provided context and code snippet do not define the schema of the raw `claims.parquet` file, nor do they show any prior renaming or creation of this column. While the pipeline context mentions "paid/incurred" fields, it does not explicitly guarantee the existence of a `recoveries` column, making it impossible to prove from the provided code alone whether this will cause a `KeyError` or if the column is validly present in the data.

### 5. clean_claims.py:92-93 (part2, high, UNSURE)
- **Cosa:** The code assumes the existence of a 'recoveries' column to calculate net incurred and net paid.
- **Perché conta:** If the new market's claims data does not include a 'recoveries' column, the script will crash with a KeyError, preventing the pipeline from running.
- **Codice:** `c["net_incurred"] = c["incurred_to_date"] - c["recoveries"]
c["net_paid"] = c["paid_to_date"] - c["recoveries"]`
- **Test proposto:** Verify that the 'recoveries' column exists in the new market's raw claims parquet file.
- **Seconda passata:** The code at lines 92-93 directly accesses `c["recoveries"]` without checking for its existence or providing a default, which would indeed cause a `KeyError` if the column is missing in the new market's data. However, the provided code snippet does not show the data ingestion or schema validation logic (e.g., in `market_config.py` or prior cleaning steps) that might guarantee the presence of this column or handle its absence, making it impossible to definitively prove the crash from the shown code alone.

### 6. features.py:137-137 (part2, high, UNSURE)
- **Cosa:** Hard-coded column names `driver_age` and `years_licensed` will cause a KeyError if the new market renames these fields.
- **Perché conta:** The code assumes specific column names exist for the motor line; if they are missing or renamed, the script crashes instead of adapting to the new schema.
- **Codice:** `df["licence_age"] = df["driver_age"] - df["years_licensed"]`
- **Test proposto:** Verify that `driver_age` and `years_licensed` exist in the new market's motor policy table.
- **Seconda passata:** The code at line 137 does indeed hard-code `driver_age` and `years_licensed` without checking for their existence, which would cause a `KeyError` if these columns are missing or renamed in the new market data. However, the provided context states that the pipeline is adapted via a JSON config (`market_config.py`) and `features.py` only imports `LINES` from it, not column mappings; without seeing `market_config.py` or the data cleaning steps, it is impossible to prove whether these columns are guaranteed to exist or if the config handles the renaming, making the claim plausible but not definiti

### 7. features.py:135-135 (part2, high, UNSURE)
- **Cosa:** Hard-coded column name `birth_year` will cause a KeyError if the new market uses a different name for the customer's birth year.
- **Perché conta:** The code assumes `birth_year` exists to calculate `customer_age`; if it is missing or renamed, the script crashes.
- **Codice:** `df["customer_age"] = df["year"] - df["birth_year"]`
- **Test proposto:** Verify that `birth_year` exists in the new market's customer or policy tables.
- **Seconda passata:** The code at line 135 directly accesses `df["birth_year"]` without checking for its existence, which would indeed cause a `KeyError` if the column is missing or renamed in a new market configuration. However, the provided code snippet does not contain the logic for column mapping or renaming (likely in `market_config.py` or `clean_policies`), so it is impossible to prove from this file alone whether the pipeline guarantees the presence of `birth_year` or if it is a valid assumption for the specific "Part 2" adaptation mentioned in the context.

### 8. features.py:138-138 (part2, high, UNSURE)
- **Cosa:** Hard-coded column name `annual_mileage` will cause a KeyError if the new market uses a different name for mileage.
- **Perché conta:** The code assumes `annual_mileage` exists for the motor line; if it is missing or renamed, the script crashes.
- **Codice:** `df["mileage_k"] = df["annual_mileage"] / 1000`
- **Test proposto:** Verify that `annual_mileage` exists in the new market's motor policy table.
- **Seconda passata:** The code at line 138 indeed hard-codes `annual_mileage` without a check, which would cause a `KeyError` if the column is missing or renamed in the new market data. However, the provided code snippet does not include the `market_config.py` or `clean_policies` logic that the context states handles column renaming and standardization; therefore, it is impossible to prove from the shown code alone whether this specific column is guaranteed to exist or if the configuration layer handles this mapping.

### 9. build_datasets.py:115-115 (part2, medium, UNSURE)
- **Cosa:** The report generation assumes a specific customer column `credit_band` exists in the test dataset.
- **Perché conta:** If the new market's customer data uses a different column name for credit risk (e.g., `credit_score` or `risk_band`), this line will raise a KeyError, preventing the completion of the dataset build step.
- **Codice:** `rep[line]["test_missing_customer"] = int(te["credit_band"].isna().sum())`
- **Test proposto:** Verify that `credit_band` is a valid column in the new market's `customers.parquet`.
- **Seconda passata:** The code at line 115 explicitly accesses `te["credit_band"]`, which will indeed raise a `KeyError` if that column is missing from the test dataset. However, the provided code chunk does not contain the logic for `add_context` (line 111) or the definition of the `customers` dataframe (line 38), so it is impossible to verify from the shown code whether `credit_band` is guaranteed to exist or if the pipeline handles missing columns dynamically.

### 10. holdout_weights.py:27-27 (part2, medium, UNSURE)
- **Cosa:** The code assumes the dataframe `va` has a column named `row_id`.
- **Perché conta:** If the new market's data schema uses a different identifier column name (e.g., `policy_id` or `id`), this line will raise a KeyError, preventing the output of predictions.
- **Codice:** `pd.DataFrame({"row_id": va.row_id, "A": pa, "B": pb, "y": y}).to_parquet(P.RESULTS / f"ho_pred_AB_{line}.parquet", index=False)`
- **Test proposto:** Check the columns of the dataframe `df` loaded by `P.load(line)` to verify `row_id` exists.
- **Seconda passata:** The code explicitly references `va.row_id` on line 27, which would indeed raise a `KeyError` if the dataframe `df` (loaded via `P.load` on line 18) does not contain a column named `row_id`. However, the provided context does not define the schema of `P.load` or `prepare.py`, so it is impossible to prove from the code shown whether `row_id` is a guaranteed standard column or if the new market's schema (as mentioned in the project context) might use a different identifier like `policy_id` or `id`.

### 11. models_extra.py:44-44 (part2, medium, UNSURE)
- **Cosa:** The `SEV_DRIVERS` dictionary hard-codes specific feature names for severity modeling.
- **Perché conta:** If Part 2 renames these columns (e.g., `vehicle_age` to `car_age`), the severity model will silently fall back to the mean severity because the columns won't be found in `base_cols`.
- **Codice:** `SEV_DRIVERS = {"motor": ["make_model", "vehicle_age", "vehicle_value_group", "urban_density", "region"],`
- **Test proposto:** Check if `SEV_DRIVERS` is populated from the market config or if it relies on static column names that might change in Part 2.
- **Seconda passata:** The code at line 360 (`scols = [c for c in SEV_DRIVERS[line] if c in base_cols]`) does indeed silently filter out missing columns, which would cause the severity model to rely only on the remaining features (or fail if none exist) rather than explicitly falling back to the mean severity as claimed. However, the provided code chunk (lines 200-409) does not contain the definition of `SEV_DRIVERS` (line 44 is not shown), so I cannot verify the specific hard-coded column names or confirm that `base_cols` (derived from `_features` at line 303) would actually exclude them in a Part 2 scenario with r

### 18. audit_new_market.py:159-159 (bugs, None, None)
- **Cosa:** Using `c.get('recoveries', 0)` on a pandas DataFrame returns the integer 0 if the column is missing, causing a TypeError when subtracting from a Series.
- **Perché conta:** The script crashes if the 'recoveries' column is absent in the new market's claims table, preventing the audit from completing.
- **Codice:** `c["net"] = c["incurred_to_date"] - c.get("recoveries", 0)`
- **Test proposto:** Run the script on a dataset where the claims table lacks a 'recoveries' column.
- **Seconda passata:** –

### 19. audit_new_market.py:146-146 (bugs, None, None)
- **Cosa:** The merge of claims and policies on 'policy_id' can produce duplicate rows if a policy has multiple periods, and the subsequent `groupby('claim_id').any()` logic is fragile if 'claim_id' is not unique in the claims table.
- **Perché conta:** If 'claim_id' is not unique, the `groupby` will aggregate multiple claims, and the `any()` check might incorrectly mark a claim as valid if *any* of its associated policy periods cover the loss date, or fail to detect missing coverage if the logic is misaligned.
- **Codice:** `ok = (pd.to_datetime(m["loss_date"]) >= pd.to_datetime(m["period_start"])) & (pd.to_datetime(m["loss_date"]) <= pd.to_datetime(m["period_end"]))
         r["sinistri senza riga di polizza (quota)"] = round(1 - ok.groupby(m["claim_id"]).any().mean(), 4)`
- **Test proposto:** Check if 'claim_id' is unique in the claims table and if 'policy_id' has multiple rows in the policies table.
- **Seconda passata:** –

### 20. models_extra.py:444-444 (part2, None, None)
- **Cosa:** The GLM frequency model relies on a hard-coded dictionary `GLM_DGP` that maps line names to specific column names, which will fail or produce empty models if the new market renames columns or has different lines.
- **Perché conta:** If `GLM_DGP[line]` contains columns not present in the new market's `train` dataframe, the list comprehension `cols` will be empty or missing key drivers, leading to a model with no features or a crash if `line` is not a key in `GLM_DGP`.
- **Codice:** `cols = [c for c in GLM_DGP[line] if c in train.columns and c != "make_model"]`
- **Test proposto:** Check if `GLM_DGP` is defined in the module and if its keys/values match the column names in the new market's `policies_<line>` table.
- **Seconda passata:** –

### 21. models_extra.py:452-454 (part2, None, None)
- **Cosa:** The code hard-codes the peril name "subsidence" and specific column names "soil_type" and "dry_summer_index" for an interaction term, which are likely specific to the Part 1 market's schema.
- **Perché conta:** In a new market, the peril might be named differently (e.g., "ground_damage") or the columns might be missing/renamed, causing the interaction feature to be silently dropped or raising a KeyError if the columns are expected but not present in the `if` check logic (though the `if` check protects against KeyError, it silently disables the feature).
- **Codice:** `if peril == "subsidence" and "soil_type" in train.columns:
    ka, (va_,) = _bin(train["soil_type"], [valid["soil_type"]]); kb, (vb_,) = _bin(train["dry_summer_index"], [valid["dry_summer_index"]], q=4)`
- **Test proposto:** Verify if the new market uses the peril name "subsidence" and if the columns "soil_type" and "dry_summer_index" exist in the new dataset.
- **Seconda passata:** –

### 22. models_extra.py:547-547 (part2, None, None)
- **Cosa:** The weighting scheme for `gbm_clean_recent` assumes the presence of a "year" column and uses a hard-coded decay factor of 0.8 relative to the max year in the training set.
- **Perché conta:** If the new market does not have a column named "year" (e.g., it uses "period_start" or "policy_year"), this will raise a KeyError. If the year range is different, the relative weighting might not be appropriate, but the primary risk is the hard-coded column name.
- **Codice:** `wt = 0.8 ** (train["year"].max() - train["year"]).to_numpy()`
- **Test proposto:** Check if the new market's dataframe contains a column named "year".
- **Seconda passata:** –

### 23. models_extra.py:595-597 (part2, None, None)
- **Cosa:** The `_mil0` function hard-codes the column names "flag_mileage_zero" and "annual_mileage" to impute zero mileage, which are specific to the Part 1 motor line schema.
- **Perché conta:** If the new market uses different column names for mileage flags or values, this function will silently do nothing (due to the `if` check), potentially leaving missing values or incorrect imputations that were intended to be handled, leading to biased predictions for motor lines.
- **Codice:** `if "flag_mileage_zero" in df.columns and "annual_mileage" in df.columns:
    df = df.copy()
    df.loc[df["flag_mileage_zero"].astype(bool), "annual_mileage"] = 0.0`
- **Test proposto:** Verify if the new market's motor policy table contains columns named "flag_mileage_zero" and "annual_mileage".
- **Seconda passata:** –

### 24. models_extra.py:618-618 (part2, None, None)
- **Cosa:** The `_prem` function hard-codes the column names "written_premium" and "discount_applied" to calculate a relative premium feature.
- **Perché conta:** If these columns are missing or renamed in the new market, the function will raise a KeyError when called, breaking the pipeline for any model that uses this feature engineering step.
- **Codice:** `g = np.log(df["written_premium"].clip(lower=1) / (1 - df["discount_applied"].fillna(0)))`
- **Test proposto:** Check if the new market's policy table contains columns named "written_premium" and "discount_applied".
- **Seconda passata:** –

### 25. models_extra.py:458-458 (part2, None, None)
- **Cosa:** The regularization parameter `alpha` for the Poisson GLM is hard-coded based on a threshold of 3000 claims (`n.sum() > 3000`), which is tuned to the Part 1 data volume.
- **Perché conta:** If the new market has significantly fewer or more claims, this threshold may not be appropriate, leading to under-regularization (overfitting) or over-regularization (underfitting) of the frequency model.
- **Codice:** `g = PoissonRegressor(alpha=1e-4 if n.sum() > 3000 else 1e-3, max_iter=500, solver="newton-cholesky")`
- **Test proposto:** Compare the total number of claims in the new market's training set to the threshold of 3000 to assess if the regularization strength is still suitable.
- **Seconda passata:** –

### 26. trend.py:126-126 (bugs, None, None)
- **Cosa:** The MLE estimator for the Pareto shape parameter alpha is mathematically incorrect.
- **Perché conta:** The correct MLE for alpha given threshold u0 is n / sum(log(x_i/u0)). The code divides by the sum of logs, which yields a value orders of magnitude too small (e.g., 100 / 1000 = 0.1 instead of 100 / 10 = 10). This causes the tail probability (u0/cap)^alpha to be ~1, making the large loss load `raw` explode to the sum of all claims, destroying the prediction.
- **Codice:** `alpha = len(x) / np.log(x / u0).sum()`
- **Test proposto:** Run `x = np.array([60000, 70000, 80000, 90000, 100000]); u0=50000; print(len(x)/np.log(x/u0).sum())` and compare to the theoretical MLE `len(x) / np.sum(np.log(x/u0))` (note: the code is missing the sum in the denominator logic or has a conceptual error in the formula implementation vs standard MLE derivation where alpha = n / sum(log(x/u))). Actually, standard MLE is alpha = n / sum(log(x_i/u0)). The code does `len(x) / np.log(x/u0).sum()`. Wait, `np.log(x/u0).sum()` IS the sum. Let me re-evaluate. 

Correction: The MLE for Pareto(alpha, u0) is indeed alpha = n / sum(log(x_i/u0)). The code implements exactly this. 

Let's look closer at line 128: `lam = len(x) / expo * (u0 / cap) ** alpha`. This is the expected number of claims above cap. 
Line 129: `raw = lam * cap / (alpha - 1)`. This is the expected excess over cap for those claims. 

Is there a bug? 
If alpha is close to 1, `alpha - 1` is small, load is huge. 
If the code is correct, I should not report it. 

Let's look at line 131: `onlev = line_bc.loc[line, "ll_load_2025"] / emp_raw if emp_raw > 0 else 1.0`. 
`line_bc.loc[line, "ll_load_2025"]` is the credibility-adjusted load from lines 100-117. 
`emp_raw` is the empirical excess per exposure from the raw large loss table. 
This ratio scales the Pareto estimate to match the level of the empirical estimate. This seems intentional (calibration). 

Let's look at line 115: `prior = src["ll_load_2025"].reindex(line_bc.index) * line_bc["bc_2025_capped"] / src["bc_2025_capped"].reindex(line_bc.index)`. 
This scales the source market's large loss load by the ratio of the current market's base cost to the source market's base cost. This assumes large losses scale with base cost. Reasonable. 

Let's look at line 99: `ll["excess_onlevel"] = ll["excess_per_expo"] * (1 + ll["line"].map(sev_line)) ** (TARGET_YEAR - ll["ay"])`. 
`sev_line` is the median severity trend. 
`excess_per_expo` is likely the excess loss per unit exposure in year `ay`. 
Projecting to 2025 using severity trend is correct. 

Let's look at line 100: `ll_load = ll.groupby("line").apply(lambda g: np.average(g["excess_onlevel"], weights=g["exposure"]), include_groups=False).rename("ll_load_2025")`. 
This calculates the weighted average excess per exposure. Correct. 

Let's look at line 112: `n_large = cl.groupby("line")["is_large"].sum().reindex(line_bc.index).fillna(0)`. 
`cl` is `claims_clean.parquet`. 
`is_large` is a boolean. Sum gives count. Correct. 

Let's look at line 114: `Z = n_large / (n_large + K)`. 
Standard credibility. Correct. 

Let's look at line 117: `line_bc["ll_load_2025"] = Z * line_bc["ll_load_own"] + (1 - Z) * prior`. 
Standard credibility blend. Correct. 

Let's look at line 126 again. 
`alpha = len(x) / np.log(x / u0).sum()` 
This is the standard MLE. 

Is there a bug in `wls`? 
Line 31: `s2 = (w * res ** 2).sum() / dof`. 
Line 32: `cov = s2 * np.linalg.inv(X.T @ W @ X)`. 
This is the standard WLS covariance estimator. 

Let's look at line 56: `bl, pl = wls(a.index.to_numpy(float), np.log((a["n"] / a["e"]).to_numpy()), (a["n"] * a["pi"] ** 2).to_numpy())`. 
`a` is aggregated by `ay`. 
`n` is sum of `n_ult`. `e` is first `exposure`. 
Wait, `a["e"]` is `exposure` aggregated with `first`. 
If `exposure` varies within the group (line, peril, ay), taking `first` is wrong. It should be `sum`. 
Line 55: `a = g.groupby("ay").agg(n=("n_ult", "sum"), e=("exposure", "first"), pi=("pi", "mean"))`. 
`exposure` is a policy-level feature. In `development.parquet`, each row is likely a policy-period or peril-year aggregate? 
The docstring says `development.parquet` comes from `develop` step. 
Usually, development tables are aggregated by (line, peril, ay). 
If `d` is already aggregated by (line, peril, ay), then `g` in line 54 is a subset of `d` for a specific line. 
`g.groupby("ay")` would have one row per `ay` if `d` is unique on (line, peril, ay). 
If `d` has multiple rows per (line, peril, ay) (e.g., different policy segments), then `exposure` should be summed. 
However, `n_ult` is summed. `pi` is averaged. 
If `exposure` is not summed, the frequency `n/e` will be wrong (too high if there are multiple rows). 
Given `n_ult` is summed, it implies `d` might have multiple rows per (line, peril, ay). 
If so, `e` should be `sum`. 
This is a likely bug. 

Let's check the context. `d` is `development.parquet`. 
In `build_datasets`, targets are per-row. 
`develop` likely produces a table with one row per (policy_id, peril, ay) or (line, peril, ay). 
If it's per policy, then `groupby("ay")` aggregates many policies. 
`n_ult` is sum of ultimate claims. 
`exposure` is sum of exposures. 
Using `first` for exposure is definitely a bug if there are multiple rows per (line, peril, ay). 

Severity: High. It affects the line-level trend estimation for non-significant perils. 

Quote: `a = g.groupby("ay").agg(n=("n_ult", "sum"), e=("exposure", "first"), pi=("pi", "mean"))` 

Test: Check if `development.parquet` has multiple rows per (line, peril, ay). If yes, `sum` is required for exposure.
- **Seconda passata:** –

### 27. trend.py:55-55 (bugs, None, None)
- **Cosa:** Exposure is aggregated using 'first' instead of 'sum' when calculating line-level frequency trends.
- **Perché conta:** If the development table contains multiple rows per (line, peril, accident year) (e.g., per policy or segment), taking the first exposure value while summing the claim counts results in an artificially inflated frequency, leading to incorrect trend estimates for perils falling back to the line-level trend.
- **Codice:** `a = g.groupby("ay").agg(n=("n_ult", "sum"), e=("exposure", "first"), pi=("pi", "mean"))`
- **Test proposto:** Run `pd.read_parquet('interim/development.parquet').groupby(['line', 'peril', 'ay']).size()` to check for duplicates. If >1, `exposure` must be summed.
- **Seconda passata:** –

### 28. trend.py:126-126 (part2, None, None)
- **Cosa:** The Pareto alpha estimation formula divides by the sum of log-returns, which will raise a ZeroDivisionError or produce NaN if the new market has fewer than 2 claims above the threshold or if all exceedances are identical to the threshold.
- **Perché conta:** In a small Part 2 market, it is highly likely to have very few large losses; if `len(x)` is 1, `np.log(x/u0).sum()` is 0, crashing the pipeline or silently injecting NaNs into the large loss load.
- **Codice:** `alpha = len(x) / np.log(x / u0).sum()`
- **Test proposto:** Run the code on a synthetic dataset where a line has exactly one claim with `net_incurred` slightly above `u0` and observe the crash or NaN output.
- **Seconda passata:** –

### 29. trend.py:129-129 (part2, None, None)
- **Cosa:** The calculation of the expected large loss load divides by `(alpha - 1)`, which will raise a ZeroDivisionError if the estimated Pareto shape parameter is exactly 1.0.
- **Perché conta:** With very few data points (small market), the MLE for alpha can easily be 1.0 or very close to it, causing a crash or infinite values in the final prediction.
- **Codice:** `raw = lam * cap / (alpha - 1)`
- **Test proposto:** Force `alpha` to 1.0 in a test case and verify the division error.
- **Seconda passata:** –

### 30. trend.py:115-115 (part2, None, None)
- **Cosa:** The credibility prior calculation assumes the source market's `level_2025.parquet` contains the exact same `line` values as the current market, which may not hold if Part 2 introduces new lines or renames them.
- **Perché conta:** If a line exists in the new market but not in the source config's output, `reindex` will produce NaNs, leading to NaN predictions for that line's large loss load.
- **Codice:** `prior = src["ll_load_2025"].reindex(line_bc.index) * line_bc["bc_2025_capped"] / src["bc_2025_capped"].reindex(line_bc.index)`
- **Test proposto:** Create a new market config with a line name not present in the source market's parquet file and check for NaNs in `line_bc`.
- **Seconda passata:** –

### 31. trend.py:124-124 (part2, None, None)
- **Cosa:** The Pareto large loss method is applied to a hard-coded default list of lines `["motor"]` if not specified in the config, which may not match the lines present in the new market.
- **Perché conta:** If the new market has different line names (e.g., 'auto' instead of 'motor') and the config doesn't explicitly override `lines`, the code will silently skip the Pareto adjustment for the relevant line or fail to find data.
- **Codice:** `for line in llc.get("lines", ["motor"]):`
- **Test proposto:** Run with a config where `large_loss.lines` is missing and the market uses a line name other than 'motor'.
- **Seconda passata:** –

### 32. trend.py:127-127 (part2, None, None)
- **Cosa:** The exposure sum for the large loss calculation assumes the `large_loss.parquet` file contains an `exposure` column, which might be missing or named differently in the new market's data pipeline.
- **Perché conta:** If the column is missing, `ll.loc[ll["line"] == line, "exposure"]` will raise a KeyError, breaking the entire trend calculation for that line.
- **Codice:** `expo = float(ll.loc[ll["line"] == line, "exposure"].sum())`
- **Test proposto:** Remove the 'exposure' column from `large_loss.parquet` and run the script to confirm the KeyError.
- **Seconda passata:** –
