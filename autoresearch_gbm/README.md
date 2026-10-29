# Qwen loop on the GBM (20/10) — judge without IBNR spreading, two folds, gross target
- `data/`: compact dataset per year and line (s3, holdout excluded), with the judge columns (target_b, target_gross, target_rep, pi_eff, ll_expo).
- `eval_experiment.py`: frozen judge. D² without spreading (shape only), score = worse fold; also prints the official judge.
- `experiment.py`: starting point = Qwen's GBM from iteration 73 as fixed by Claude (no monotone constraint on vehicle_age), fitted on the gross target.
- `program.md`: instructions and facts verified by the team.
- Launch: `python run_loop.py`. Stop: `touch autoresearch_gbm/STOP`.
- Rule: every kept idea is re-run by Claude in the container with the three judges (official, without spreading, with actual large losses) before it enters the blend.
