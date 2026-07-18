#!/usr/bin/env bash
# =============================================================================
# slurm_submit.sh — submit ONE model as a SPUR batch job and print its job id.
#
# Derives the allocation from the model's handoff.json (tp = GPU count, on one
# node) and the shared SPUR account/partition/qos in lib.sh. The job body is
# ci/slurm_job.sh (weights + run_local on the compute node). Env is propagated
# to the job by SPUR's `--export=ALL` default, so LITELLM_* secrets exported in
# the runner reach the container.
#
# Usage:
#   ci/slurm_submit.sh <model_key> [--budget SECONDS] [--run-ts TS] [--print]
#
# Output (stdout, one line): "<job_id>\t<out_dir>"   (nothing on --print/error)
# All human logging goes to stderr, so callers can capture the id cleanly.
#
# --print / SPUR_DRYRUN=1 : show the sbatch command instead of submitting it
# (used to validate wiring without touching the shared cluster).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"

MODEL_KEY="${1:?usage: slurm_submit.sh <model_key> [--budget N] [--run-ts TS] [--probe] [--print]}"; shift || true
BUDGET="${PERFSKILLS_E2E_TIMEOUT_S:-1800}"
RUN_TS="${RUN_TS:-$(new_ts)}"
PRINT="${SPUR_DRYRUN:-0}"
PROBE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --budget) BUDGET="${2:?}"; shift ;;
    --run-ts) RUN_TS="${2:?}"; shift ;;
    --probe)  PROBE=1 ;;
    --print)  PRINT=1 ;;
    *) die "unknown arg: $1" ;;
  esac; shift
done

is_enrolled "$MODEL_KEY" || die "model '$MODEL_KEY' not enrolled in $MODELS_TSV"
[ -f "$(_handoff_path "$MODEL_KEY")" ] || die "no handoff.json for $MODEL_KEY under $HF_LOGS"

FW="$(model_framework "$MODEL_KEY")"; [ -n "$FW" ] || die "no framework in handoff for $MODEL_KEY"
TP="$(model_tp "$MODEL_KEY")"
case "$TP" in ''|*[!0-9]*) die "bad tp='$TP' for $MODEL_KEY (handoff.tp must be an integer)";; esac
[ "$TP" -ge 1 ] || die "tp must be >=1 for $MODEL_KEY (got $TP)"

GPUS="$TP"
CPUS=$(( GPUS * SPUR_CPUS_PER_GPU ))
# Probe: fixed short wall time (image pull + optional Claude, no e2e); else budget + headroom.
if [ "$PROBE" = "1" ]; then
  TIME="$SPUR_PROBE_TIME"
else
  WALL=$(( BUDGET + SPUR_TIME_HEADROOM_S ))
  TIME="$(fmt_slurm_time "$WALL")"
fi

OUT_DIR="$HF_LOGS/$MODEL_KEY/ci_runs/$RUN_TS"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/slurm.out"
SAFE="${MODEL_KEY//[^A-Za-z0-9_.-]/_}"
JOB="geak_$([ "$PROBE" = 1 ] && echo probe || echo l1)_${SAFE}_${RUN_TS}"

log "submit $MODEL_KEY: fw=$FW tp=$TP -> GPUs=$GPUS cpus=$CPUS time=$TIME budget=${BUDGET}s${PROBE:+ probe=$PROBE}"
log "  account=$SPUR_ACCOUNT partition=$SPUR_PARTITION qos=$SPUR_QOS out=$OUT_DIR"
[ "$PROBE" = "1" ] && log "  PROBE: infra-only (stops at GEAK entry), wall=$TIME"

# Job body env prefix; probe sets GEAK_CI_PROBE=1 so slurm_job.sh runs run_local --probe.
WRAP="RUN_TS='$RUN_TS' "
[ "$PROBE" = "1" ] && WRAP+="GEAK_CI_PROBE=1 "
WRAP+="bash '$GEAK_ROOT/ci/dispatch/slurm_job.sh' '$MODEL_KEY' '$BUDGET'"

# Single node so a tp-way tensor-parallel run stays co-located on one box.
SBATCH=(sbatch
  -A "$SPUR_ACCOUNT" -p "$SPUR_PARTITION" --qos "$SPUR_QOS"
  -J "$JOB" -N 1 -G "$GPUS" -c "$CPUS" -t "$TIME"
  --chdir "$WS" -o "$LOG" -e "$LOG"
  --wrap "$WRAP")

if [ "$PRINT" = "1" ]; then
  { printf 'DRY-RUN sbatch for %s:\n  ' "$MODEL_KEY"; printf '%q ' "${SBATCH[@]}"; printf '\n'; } >&2
  exit 0
fi

OUT="$("${SBATCH[@]}")" || die "sbatch failed for $MODEL_KEY"
log "  $OUT"
JID="$(grep -oE '[0-9]+' <<<"$OUT" | tail -1)"
[ -n "$JID" ] || die "could not parse job id from sbatch output: $OUT"
printf '%s\t%s\n' "$JID" "$OUT_DIR"
