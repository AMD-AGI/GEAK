#!/usr/bin/env bash
set -euo pipefail
# --- SETUP ---
printf '#!/bin/bash\nexport PYTHONPATH=%s:%s:${PYTHONPATH}\nexport HIP_VISIBLE_DEVICES=%s\nexec python3 "$@"\n' "${GEAK_WORK_DIR}" "${GEAK_REPO_ROOT}" "${GEAK_GPU_DEVICE}" > ${GEAK_WORK_DIR}/run.sh && chmod +x ${GEAK_WORK_DIR}/run.sh
# --- FULL_BENCHMARK ---
${GEAK_WORK_DIR}/run.sh ${GEAK_HARNESS} --flydsl-kernel ${GEAK_WORK_DIR}/hingeloss_flydsl.py --full-benchmark ${GEAK_BENCHMARK_EXTRA_ARGS:-}
