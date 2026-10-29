#!/bin/bash
# Prepares the Mac for the autoresearch loop (run once).
#   1. Python >= 3.10 in a project .venv
#   2. libomp for LightGBM (Homebrew)
#   3. rebuilds the modelling data (pipeline steps 1-2a, ~5 minutes)
#   4. checks that Ollama responds and which Qwen models are installed
set -e
cd "$(dirname "$0")/.."
echo "== project: $(pwd)"

PYBIN=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PYBIN=$(command -v "$c"); break
  fi
done
if [ -z "$PYBIN" ]; then
  echo "Python >= 3.10 is required. Install with: brew install python@3.12"; exit 1
fi
echo "== python: $PYBIN ($($PYBIN --version))"

if command -v brew >/dev/null 2>&1; then
  brew list libomp >/dev/null 2>&1 || { echo "== installing libomp (needed by LightGBM)"; brew install libomp; }
fi

[ -d .venv ] || "$PYBIN" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r autoresearch/requirements.txt
.venv/bin/python -c "import lightgbm, pandas, sklearn, pyarrow; print('== libraries ok: lightgbm', lightgbm.__version__, '| pandas', pandas.__version__)"

if [ ! -f data/model/model_motor.parquet ] || ! .venv/bin/python -c "import pandas as pd,sys; sys.exit(0 if 'pol_prev_rate' in pd.read_parquet('data/model/model_motor.parquet').columns else 1)"; then
  echo "== rebuilding the data (about 5 minutes)"
  bash run_all.sh
fi

echo "== Ollama"
if curl -s http://localhost:11434/api/tags >/dev/null; then
  curl -s http://localhost:11434/api/tags | .venv/bin/python -c "import json,sys; print('   models:', ', '.join(m['name'] for m in json.load(sys.stdin)['models']) or 'none')"
else
  echo "   Ollama is not responding: open the Ollama app (or run 'ollama serve') and retry"
fi
echo "== setup complete. Start with: bash autoresearch/run_loop.sh"
