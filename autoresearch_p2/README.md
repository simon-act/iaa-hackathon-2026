# Karpathy loop — Part 2 (emerging market), 25/10

Qwen (local, Ollama) edits `experiment.py`; `eval_experiment.py` (frozen) scores it on 7 checks (2023→2024, 2024→2023,
5 policy groups) against "developed-market prior × level"; a change is kept only if it improves both the judge without IBNR
spreading and the official-like shape. Data: `data_p2.parquet` (built by `build_data.py`, NOT in the repo). Launch:
`python run_loop.py`; stop: `touch STOP`. Results: `results.tsv`, `loop.log`, the folder's git history. No loop result enters a
submission without re-verification on the full bench (`scripts/p2_bench.py`).
