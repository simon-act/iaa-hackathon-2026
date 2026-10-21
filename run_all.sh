#!/bin/bash
# pipeline completa: dati grezzi -> dataset di modellazione con feature (passi 1 e 2a)
# uso: ./run_all.sh [configs/<mercato>.json]     (senza argomento = Parte 1; usa .venv se esiste)
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}; [ -x .venv/bin/python ] && PY=.venv/bin/python
if [ -n "$1" ]; then export IAA_MARKET="$1"; fi
read RAW INTERIM MODEL <<< "$($PY -I -c "import sys; sys.path.insert(0,'src'); from market_config import CFG; print(CFG['raw'], CFG['interim'], CFG['model_dir'])")"
[ -d "$RAW/public" ] && RAW="$RAW/public"
mkdir -p "$INTERIM" "$MODEL"
echo "mercato: ${IAA_MARKET:-parte1}  raw=$RAW"
$PY -I src/clean_policies.py "$RAW" "$INTERIM" > /dev/null
$PY -I src/clean_claims.py   "$RAW" "$INTERIM" > /dev/null
$PY -I src/develop.py        "$INTERIM" > /dev/null
$PY -I src/trend.py          "$INTERIM" > /dev/null
$PY -I src/build_datasets.py "$RAW" "$INTERIM" "$MODEL" > /dev/null
$PY -I src/features.py       "$INTERIM" "$MODEL" > /dev/null
$PY -I src/judge_own.py > /dev/null      # giudice secondario senza IBNR/LL spalmati (19/10)
echo "pipeline ok -> $MODEL"
