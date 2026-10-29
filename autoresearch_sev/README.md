# Severity loop (from the afternoon of 19/10)

Third loop. The first rewrote a GBM from scratch (316 iterations, 9 kept, nothing after iteration 211); the second looked for corrections on the residuals of the best model (62 iterations, 1 kept: vehicle value; it confirmed that the relativities are right).
This one isolates the half of the model with the most room and the least data: per-peril severity. `claims_sev.parquet` = one record per claim with the features of its policy row, peril, year, cost (capped, developed, at 2025 level).
Judge: Gamma D² on the following year (fit ≤ 2022 → 2023, fit ≤ 2023 → 2024), perils weighted by cost, lines weighted like Kaggle; score = worse fold. Baseline = the current model: 0.02685 / 0.0342. One run takes a few seconds.
Launch: `python run_loop.py` (stop the other loops first). Stop: `touch autoresearch_sev/STOP`.
