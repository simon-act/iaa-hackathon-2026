# Residuals loop (from 19/10)

The first loop (`autoresearch/`) rewrote a GBM from scratch: after 285 iterations it kept repeating the same ideas (9 kept, none after iteration 211).
This loop starts from the best model (the sub_18 blend; out-of-sample predictions for 2023 and 2024 in `base_oof.parquet`) and looks for
what it still gets wrong, in a way that holds from one year to the next: fit on one year and score on the other, in both directions;
the score is the D² gain in the worse direction.

- `eval_experiment.py`: frozen judge. `experiment.py`: the file Qwen edits. `program.md`: instructions and what we know.
- Starting point: per-line power calibration (pred^b), +0.00038 in the worse direction.
- Already tried: a correction GBM on all features makes things worse (−0.0079 with 150 rounds), so the residuals are almost all noise.
- Launch: `python run_loop.py` (stop the first loop beforehand). Stop: `touch autoresearch_residui/STOP`.
