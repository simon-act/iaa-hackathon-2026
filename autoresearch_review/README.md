# Code review with Qwen (evening of 19/10)

Fourth use of Qwen, different from the loops: instead of proposing models, it **reads the pipeline and looks for errors**.
- `code_snapshot/` (not kept in the repo): copy of the pipeline files used for the final submission (container `src/` as of 19/10, commit 5b4efe1 + s4).
- `CONTEXT.md`: what the reviewer knows about the project (competition, tables, pipeline, folds, what counts as leakage).
- `review.py`: 25 code chunks × 3 lenses (leakage, bugs, Part 2 robustness) = 75 calls; then a second pass in which Qwen tries
  to **refute** every high/medium finding (REAL / FALSE_POSITIVE / UNSURE).
- Output: `findings.jsonl`, `SINTESI.md` (sorted table), `review.log`, `reviews/unparsed.txt`.
- `SINTESI.md` is the raw output of the local model (Italian, kept as produced); `VERIFICA.md` is the verification of every finding.
- Launch: `python -I autoresearch_review/review.py` (from the project folder). Stop: `touch autoresearch_review/STOP`.
  It resumes where it stopped (`done.json`).
- Rule: no finding is applied without verification in the container (reproducible test). Verifications go in
  `VERIFICA.md`. This is also material for the "trust/validation" and "use of AI" criteria (independent local reviewer).
