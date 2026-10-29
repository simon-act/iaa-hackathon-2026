# Qwen loop on per-peril FREQUENCY (20/10)
- `data/`: compact dataset per line and year (s3, holdout excluded) with features, n_<peril>, pi_<peril>; `meta.json` with perils, features, trends, cost shares.
- `eval_experiment.py`: frozen judge, Poisson D² of the counts per peril (shape only), worse fold. Reproduces exactly the frequency bench used in the container (`scripts/freq_bench.py`).
- `experiment.py`: starting point = the current frequency part of the freq×sev model.
- `grid.py`: one-off grid of frequency hyperparameters on the same judge and data (`experiment_base.py` = frozen copy of the starting model).
- Launch: `python run_loop.py` (stop the other loops first). Stop: `touch autoresearch_freq/STOP`.
- Rule: every kept idea is re-implemented by Claude in the full model (OOF with the three judges) before being used.
