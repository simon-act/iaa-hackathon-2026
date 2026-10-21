# Project context for the code reviewer

Competition: Kaggle "IAA AI Assisted Hackathon 2026 – Part 1". Synthetic personal-lines insurer, three lines:
motor, buildings, contents. Task: predict the 2025 ultimate claim cost of every policy row in the test set
(one row = one policy-period with its exposure in years). Metric: Tweedie D² with power 1.5 (higher is better).
Part 2 (after 23 Oct): a NEW market with different data (other years, renamed/missing/new columns, other units,
fewer policies); the same pipeline must be adapted quickly via a JSON config (configs/*.json, market_config.py).

Raw tables: policies_<line> (2021–2024, one row per policy-period: policy_id, customer_id, address_id,
period_start, period_end, exposure, rating features), test_<line> (2025 rows, no claims), claims (one row per
claim: row/policy link, peril, accident date/ay, report date, status, paid/incurred, development snapshots),
customers, addresses, weather (per region and year), sample_submission.

Pipeline (run_all.sh): clean_policies -> clean_claims -> develop (chain-ladder style development of claims to
ultimate, IBNER factors, completeness/IBNR) -> trend (frequency and severity trend per line/peril, projected to
2025; weather perils get no frequency trend) -> build_datasets (per-row target `target_b` = developed, capped
ultimate at 2025 level + large-loss load; per-peril counts n_<peril>, pi_<peril> completeness) -> features
(feature engineering incl. claim-history features of policy/customer/address computed from PRIOR years only).
Modelling: models_extra.py `_fs2` = per-peril Poisson frequency (LightGBM with 1-split trees, offset =
log(exposure × completeness) − trend) × Gamma severity (1-split GBM or GLM on sum insured) + flat large-loss load,
then Bühlmann–Straub experience rating (experience.py, credibility on the policy's own past claims).
Blend with a second GBM (Qwen). oof.py = out-of-fold predictions on temporal folds; blend.py = weights;
final.py / final_line.py = refit on all years and predict the 2025 test rows.

Validation contract (prepare.py, frozen): temporal folds F2023 (fit 2021–22, score 2023) and F2024 (fit
2021–23, score 2024); a 10% holdout of policies by stable hash of policy_id excluded from both; only whitelisted
columns may be model features (prefixes n_, cost_, excess_, pi_, ult, target, flag_ are targets/forbidden).

What counts as LEAKAGE here: any feature, factor or estimate used to predict year Y that uses claims or outcomes
of year Y or later (or of the test year), e.g. claim-history features that include the current period,
development/trend/credibility/large-loss parameters estimated with the scored year inside a fold, target
encodings fit on the scored rows, test rows influencing train statistics, holdout policies entering fits.
Development, trend and large-loss tables are estimated once on all years (by design for the final model). If you
see them being used inside the validation folds, report it as a finding with severity "medium" and say which
quantity of the scored year enters; a human will judge how much it matters.
