#!/usr/bin/env bash
# =============================================================================
# held_job.sh — the body that runs ON a held node (inside an `srun --overlap`
# step of a "$USER-hold" allocation). The held-node analogue of slurm_job.sh.
#
# The holder already owns the whole node and all 8 GPUs, so unlike the sbatch
# path there is nothing to allocate here: we just pick the GPUs this model may
# use, resolve its weights, and hand off to ci/node/run_local.sh.
#
# Usage (invoked by ci/dispatch/held_submit.sh, not by hand):
#   RUN_TS=<ts> bash ci/dispatch/held_job.sh <model_key> <budget_s>
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"

MODEL_KEY="${1:?usage: held_job.sh <model_key> <budget_s>}"
BUDGET="${2:?usage: held_job.sh <model_key> <budget_s>}"
RUN_TS="${RUN_TS:-$(new_ts)}"; export RUN_TS

TP="$(model_tp "$MODEL_KEY")"
log "held job: model=$MODEL_KEY tp=$TP budget=${BUDGET}s run_ts=$RUN_TS"
log "  node=$(hostname) holder_job=${SLURM_JOB_ID:-<none>} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"

# 1. Weights from the catalog (NFS is mounted on the compute node).
MODEL_PATH="$(stage_weights "$MODEL_KEY")"; export MODEL_PATH
log "  weights=$MODEL_PATH"

# 2. GPU selection. The holder exposes all 8 GPUs to every step, so a tp<8 model
#    would otherwise grab the whole box. Pin it to the first tp devices; one model
#    owns one node in this dispatch mode, so a fixed low range is safe and keeps
#    the mapping predictable. An explicit GEAK_GPUS from the caller still wins.
if [ -z "${GEAK_GPUS:-}" ]; then
  GEAK_GPUS="$(seq -s, 0 $(( TP - 1 )))"
fi
export GEAK_GPUS
log "  GEAK_GPUS=$GEAK_GPUS"

# 3. The D-state pre-check cannot work from here and must not be trusted.
#    An overlapping srun step runs in its OWN PID namespace while /proc is still
#    the one mounted for the holder's namespace, so this step sees exactly two
#    PIDs and /proc/self dangles. The scan would find no D-state task and "pass"
#    every time — a false all-clear is worse than no check. (docker --pid=host,
#    which would expose the real host /proc, is refused by the spur-authz plugin.)
#    Two real gates still cover a wedged GPU: discovery probes each node with
#    rocm-smi under a timeout and drops any node that will not answer, and the
#    in-container GPU healthcheck (rocminfo + torch) still runs below.
#    Set GEAK_FORCE_DSTATE_CHECK=1 to run it anyway. (GEAK_SKIP_DSTATE_CHECK is
#    no use as the opt-in here: config.sh always gives it a default of 0, so an
#    explicit request is indistinguishable from the default.)
if [ "${GEAK_FORCE_DSTATE_CHECK:-0}" = "1" ]; then
  log "  D-state pre-check forced on (GEAK_FORCE_DSTATE_CHECK=1) — note its /proc view is not this step's"
else
  export GEAK_SKIP_DSTATE_CHECK=1
  log "  D-state pre-check skipped: a step's /proc is another namespace's (see held_nodes.sh); GPU healthcheck still runs"
fi

# 3. Real run — same entry point the sbatch path uses.
if [ "${GEAK_CI_PROBE:-0}" = "1" ]; then
  log "  PROBE mode: verify infra up to GEAK entry, then stop (no e2e workflow)"
  exec bash "$GEAK_ROOT/ci/node/run_local.sh" "$MODEL_KEY" --probe --budget "$BUDGET"
fi
exec bash "$GEAK_ROOT/ci/node/run_local.sh" "$MODEL_KEY" --budget "$BUDGET"
