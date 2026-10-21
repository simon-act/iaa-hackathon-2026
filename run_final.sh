#!/bin/bash
# Submission finali da zero: stime finali su tutte le polizze 2021–24, poi il blend.
# Prima: pipeline dei dati (./run_all.sh configs/parte1_s3.json).
# Verificato il 21/10: sub_finale identica alla sub_40 caricata (max |Δ| = 0); sub_41_finale = sub_41.
set -e
cd "$(dirname "$0")"; export IAA_MARKET=configs/parte1_s3.json
PY=${PY:-python3}; [ -x .venv/bin/python ] && PY=.venv/bin/python
for job in "fs2_v2fb_cred motor" "qwen_gbm73fix_gross_cred motor" "fs2_v2_llr_cred buildings" \
           "qwen_gbm73_gross_cred buildings" "fs2_v2_cred contents" "qwen_gbm73_cred contents" "fs2_v2_sevmix_cred contents"; do
  $PY -I scripts/final_model_line.py $job
done
$PY scripts/build_final.py sub_finale                       # = sub_40
$PY scripts/build_final.py sub_41_finale --variante sub_41  # = sub_41 (alluvione contenuto GBM×GLM)
