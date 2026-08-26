#!/usr/bin/env bash
# =============================================================================
# slurm_job.sh — the body that runs ON a SPUR compute node (via `sbatch --wrap`).
#
# One model, one allocation. SPUR has already granted this job its GPUs (tp of
# them, on one node). Here we:
#   1. resolve the model weights from the catalog ($HF_MODELS_DIR, symlinks into
#      shared NFS; downloads if absent — NFS is mounted on the compute node),
#   2. forward the SLURM GPU allocation into the container, and
#   3. hand off to the existing ci/node/run_local.sh, which does the real work
#      (D-state/GPU preflight -> docker -> Claude -> GEAK e2e -> monitor -> judge)
#      and writes result.json under geak_runtime/<model>/ci_runs/<RUN_TS>/.
#
# Usage (normally invoked by ci/dispatch/slurm_submit.sh, not by hand):
#   RUN_TS=<ts> bash ci/dispatch/slurm_job.sh <model_key> <budget_s>
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"

MODEL_KEY="${1:?usage: slurm_job.sh <model_key> <budget_s>}"
BUDGET="${2:?usage: slurm_job.sh <model_key> <budget_s>}"
RUN_TS="${RUN_TS:-$(new_ts)}"; export RUN_TS

log "SPUR job: model=$MODEL_KEY budget=${BUDGET}s run_ts=$RUN_TS"
log "  node=$(hostname) job_id=${SLURM_JOB_ID:-<none>} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"

# 1. weights: prefer an existing copy, else download from HuggingFace.
MODEL_PATH="$(stage_weights "$MODEL_KEY")"; export MODEL_PATH
log "  weights=$MODEL_PATH"

# 2. Forward the SLURM-granted GPUs into the container. If SPUR exported
#    ROCR_VISIBLE_DEVICES for this allocation, pass it through (run_local maps
#    GEAK_GPUS -> the container's ROCR_VISIBLE_DEVICES). If it's unset, the
#    allocation already isolates devices, so leave GEAK_GPUS empty (use all
#    visible). An explicit GEAK_GPUS from the caller still wins.
export GEAK_GPUS="${GEAK_GPUS:-${ROCR_VISIBLE_DEVICES:-}}"

# 3. Real run. run_local computes OUT_DIR=$HF_LOGS/$MODEL_KEY/ci_runs/$RUN_TS,
#    does all GPU/Docker/Claude work, and exits non-zero on any failure.
#    GEAK_CI_PROBE=1 (set by slurm_submit --probe) instead verifies the infra up
#    to the GEAK e2e doorstep and stops — a fast end-to-end harness check.
if [ "${GEAK_CI_PROBE:-0}" = "1" ]; then
  log "  PROBE mode: verify infra up to GEAK entry, then stop (no e2e workflow)"
  exec bash "$GEAK_ROOT/ci/node/run_local.sh" "$MODEL_KEY" --probe
fi
exec bash "$GEAK_ROOT/ci/node/run_local.sh" "$MODEL_KEY" --budget "$BUDGET"
