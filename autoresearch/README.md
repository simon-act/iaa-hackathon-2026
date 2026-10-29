# Autoresearch loop (Karpathy-style) with a local Qwen

A local LLM (Qwen via Ollama) proposes **one change at a time** to `experiment.py`. A frozen judge scores it on the temporal folds; the change is kept only if it improves the score.

| File | Who edits it | Role |
|---|---|---|
| `program.md` | Simone only | research direction, rules, ideas to try |
| `experiment.py` | the loop only | the only file the LLM may change (starts from the step-2 GBM) |
| `eval_experiment.py` + `../src/prepare.py` | nobody | judge: folds 2023/2024, closed holdout, Tweedie D² p=1.5 |
| `run_loop.py` | nobody | loop: ask, check, evaluate, keep or discard, log |
| `results.tsv` | the loop | one row per attempt (idea, score, outcome, time) |
| `runs/iter_XXXX.py` | the loop | every proposal, including discarded ones |
| `.git` (in this folder) | the loop | one commit per kept change |

## Guardrails

- The experiment receives only allowed columns; the target is given for fitting rows only, never for validation rows; the holdout (10% of policies) is never passed.
- Static check before running: allowed imports only, no file, network or process access, `fit_predict` signature unchanged.
- Time limit per attempt (`--timeout`, default 900 s); predictions must be finite and ≥ 0.
- A change is kept only if it improves `pooled_mean` by at least 0.0003 (noise filter).
- At the end the closed holdout is checked (winner's curse): improving on the folds is not enough.

## Usage (on the Mac)

```bash
cd <repository>
bash autoresearch/setup_mac.sh      # once: .venv, libraries, data, Ollama check
bash autoresearch/run_loop.sh       # starts in the background, Mac kept awake
tail -f autoresearch/loop.log       # follow
touch autoresearch/STOP             # stop
```

Options: `--model qwen3.6` (default: the best Qwen installed), `--iters 200`, `--timeout 900`, `--min-delta 0.0003`.
Test without an LLM: `.venv/bin/python -I autoresearch/run_loop.py --llm mock --iters 3`.
