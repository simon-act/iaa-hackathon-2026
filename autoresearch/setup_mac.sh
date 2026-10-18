#!/bin/bash
# Prepara il Mac per il loop autoresearch (una volta sola).
#   1. Python >= 3.10 in un ambiente .venv del progetto
#   2. libomp per LightGBM (Homebrew)
#   3. ricostruisce i dati di modellazione (pipeline passi 1-2a, ~5 minuti)
#   4. controlla che Ollama risponda e quali modelli Qwen ci sono
set -e
cd "$(dirname "$0")/.."
echo "== progetto: $(pwd)"

PYBIN=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PYBIN=$(command -v "$c"); break
  fi
done
if [ -z "$PYBIN" ]; then
  echo "Serve Python >= 3.10. Installa con: brew install python@3.12"; exit 1
fi
echo "== python: $PYBIN ($($PYBIN --version))"

if command -v brew >/dev/null 2>&1; then
  brew list libomp >/dev/null 2>&1 || { echo "== installo libomp (serve a LightGBM)"; brew install libomp; }
fi

[ -d .venv ] || "$PYBIN" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r autoresearch/requirements.txt
.venv/bin/python -c "import lightgbm, pandas, sklearn, pyarrow; print('== librerie ok: lightgbm', lightgbm.__version__, '| pandas', pandas.__version__)"

if [ ! -f data/model/model_motor.parquet ] || ! .venv/bin/python -c "import pandas as pd,sys; sys.exit(0 if 'pol_prev_rate' in pd.read_parquet('data/model/model_motor.parquet').columns else 1)"; then
  echo "== ricostruisco i dati (circa 5 minuti)"
  bash run_all.sh
fi

echo "== Ollama"
if curl -s http://localhost:11434/api/tags >/dev/null; then
  curl -s http://localhost:11434/api/tags | .venv/bin/python -c "import json,sys; print('   modelli:', ', '.join(m['name'] for m in json.load(sys.stdin)['models']) or 'nessuno')"
else
  echo "   Ollama non risponde: apri l'app Ollama (o 'ollama serve') e riprova"
fi
echo "== setup completato. Avvia con: bash autoresearch/run_loop.sh"
