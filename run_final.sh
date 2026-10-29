#!/bin/bash
# Final submissions from scratch: final estimates on all 2021–24 policies, then the blend.
# Prerequisite: data pipeline (./run_all.sh configs/parte1_s3.json).
# Verified 21 Oct: sub_finale identical to the uploaded sub_40 (max |Δ| = 0); sub_41_finale = sub_41.
set -e
cd "$(dirname "$0")"; export IAA_MARKET=configs/parte1_s3.json
PY=${PY:-python3}; [ -x .venv/bin/python ] && PY=.venv/bin/python
for job in "fs2_v2fb_cred motor" "qwen_gbm73fix_gross_cred motor" "fs2_v2_llr_cred buildings" \
           "qwen_gbm73_gross_cred buildings" "fs2_v2_cred contents" "qwen_gbm73_cred contents" "fs2_v2_sevmix_cred contents" \
           "qwen_gbm73_gross_xs_cred buildings" "fs2_v2fb_xs_cred motor"; do   # xs = relative deductible (22 Oct)
  $PY -I scripts/final_model_line.py $job
done
$PY scripts/build_final.py sub_finale                       # = sub_40
$PY scripts/build_final.py sub_41_finale --variante sub_41  # = sub_41 (contents flood GBM×GLM)
$PY scripts/build_final.py sub_48_finale --variante sub_48  # = sub_48 (sub_41 + relative deductible: buildings GBM, motor own-damage/theft frequency)
$PY scripts/build_final.py sub_50_finale --variante sub_50  # = sub_50 (sub_48 + motor 2024 history completed with expected late-reported claims)
$PY scripts/build_final.py sub_51_finale --variante sub_51  # = sub_51 FINAL (sub_50 + motor credibility k=16)
