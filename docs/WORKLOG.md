# Work summary

How the work went, by phase. Decisions and their evidence are in `DECISIONS_actuary.md`; the final models in
`MODEL_CARD.md`; the use of AI in `AI_ledger.md`.

**1. Data and target (16–17 October).** Every table audited before modelling; data defects fixed (exposure from dates,
sums insured in wrong units, duplicate and out-of-period claims). The target was built like a reserve: reporting-delay
curves, IBNR and IBNER, trends, large losses capped and reloaded; it reconciles exactly with the development tables.
A frozen judge (two time folds, policy holdout, forbidden columns) was set up before the first model; one leaking column
(`customer_service_contacts`) was found and excluded.

**2. Model structure (17–18 October).** Reconstructing how the data were generated showed a multiplicative frequency with
few interactions and many decoy columns, so the main model became an additive frequency × severity per peril. Policies
with past claims were under-predicted by every model, which led to Bühlmann–Straub experience rating. A local LLM ran
autoresearch loops on the boosting model, frequency, severity, residuals and claims history; one of its models joined
the final blend after review.

**3. Validation and method rules (19–20 October).** A calibration that looked good on the folds failed out of sample, and
an independent review showed that the official-style target spreads IBNR and large losses evenly and so rewards flat
predictions. From then on: no transformations of predictions after the model; a change is kept only if it improves both
years on the official judge and on a second judge without spreading, inside the final blend; the leaderboard is a check
only. A code review by the local LLM, verified line by line, found no leakage and led to four robustness fixes.

**4. Part 1 refinements and choice (20–23 October).** Frequency settings per peril, gross target for the motor boosting
model, credibility without subsidence in buildings, relative deductible in the buildings model, motor credibility
k = 16. Many alternatives were tested and rejected (power calibrations, stacking, seasonality, weather extrapolation,
risk-related large-loss loads, two-step deductible). Final Part 1: sub_51.

**5. Part 2 transfer (23–25 October).** An automated audit of the new market showed identical reporting delays, so
development and trends were borrowed. The Part 1 model on the shared columns became the prior; the emerging data set the
level and gender effects. The prior beat emerging-only models in every line and in 16 of 18 simulated small markets.
A seven-check bench with a rule fixed in advance was set up for every correction; the new-business portfolio of 2025 was
analysed and priced through a new-business correction.

**6. Part 2 corrections and stress tests (26–28 October).** On the bench: storm level in buildings, fuel and cover type
and a claim-count boost in motor, working from home in contents, Part 1 priors with relative premium in home, and the
insurer's relative premium in the motor cost model — the strongest single effect. Each piece was re-checked on a stricter
bench (time direction × disjoint customers) and on a bench reweighted to the 2025 mix; a single model across both markets,
the exact Tweedie offset, exposure curves on fire and seasonality were rejected. Final Part 2: p2_sub_21.

**7. Delivery checks (28 October).** Contract tests on both finals, no credentials or competition data in the repository,
both finals rebuilt from code, Part 2 rebuilt from raw data in a clean copy, independent review of the new code with its
limits stated in `MODEL_CARD.md`.
