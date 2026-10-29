# Policy-history loop (from the evening of 21/10)
The large piece the model does not explain is the policy's own latent risk (D² ≈ 0.043 not recoverable, see docs/WORKLOG.md);
the claims history recovers only 0.002–0.004 of it. The loop looks for a better use of the history (credibility) on top of the sub_41 blend.
- `base_oof.parquet`: 2023/2024 OOF predictions of the sub_41 blend (`scripts/base41_oof.py`), with and without credibility, + judge columns.
- `storia.parquet`: history per row (past claims per peril, previous year, customer on the other lines, address) — `scripts/build_storia.py`.
- `eval_experiment.py`: frozen judge (without IBNR spreading, cross-fitted between years, worse direction; also reports the official one).
- Launch: `python run_loop.py`. Stop: `touch autoresearch_storia/STOP`.
- Rule: every kept idea is re-implemented by Claude as a change to the model (credibility in src/experience.py) and judged on the three judges before it enters a submission.
