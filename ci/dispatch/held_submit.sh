#!/usr/bin/env bash
# =============================================================================
# held_submit.sh — run ONE model on ONE already-held node.
#
# The held-node analogue of slurm_submit.sh. The crucial difference: this does
# NOT queue anything, so it never pends. It attaches to an allocation we already
# own and runs there immediately.
#
# It BLOCKS for the whole run. That is forced by the platform, not a design
# choice: processes inside an srun step are reaped when the step exits (see the
# header of held_nodes.sh), so the srun client must stay alive for the duration.
# run_matrix.sh backgrounds one of these per model to get parallelism.
#
# Usage:
#   ci/dispatch/held_submit.sh <model_key> --jobid <holder_job> [--budget N]
#                              [--run-ts TS] [--probe] [--print]
#
# Output (stdout, one line): "<holder_job>:<node>\t<out_dir>"
# All human logging goes to stderr. Exit status is the run's exit status.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../lib.sh"
# shellcheck source=/dev/null
source "$HERE/held_nodes.sh"

MODEL_KEY="${1:?usage: held_submit.sh <model_key> --jobid <holder_job> [--budget N] [--run-ts TS] [--probe] [--print]}"; shift || true
BUDGET="$PERFSKILLS_E2E_TIMEOUT_S"   # defaults live in ci/config.sh
RUN_TS="${RUN_TS:-$(new_ts)}"
PRINT="$SPUR_DRYRUN"
PROBE=0
HOLDER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --jobid)  HOLDER="${2:?}"; shift ;;
    --budget) BUDGET="${2:?}"; shift ;;
    --run-ts) RUN_TS="${2:?}"; shift ;;
    --probe)  PROBE=1 ;;
    --print)  PRINT=1 ;;
    *) die "unknown arg: $1" ;;
  esac; shift
done

is_enrolled "$MODEL_KEY" || die "model '$MODEL_KEY' not enrolled in $MODELS_TSV"
[ -f "$(_handoff_path "$MODEL_KEY")" ] || die "no handoff.json for $MODEL_KEY under $HF_LOGS"
[ -n "$HOLDER" ] || die "--jobid <holder_job> is required (see: ci/dispatch/held_nodes.sh idle)"

FW="$(model_framework "$MODEL_KEY")"; [ -n "$FW" ] || die "no framework in handoff for $MODEL_KEY"
TP="$(model_tp "$MODEL_KEY")"
case "$TP" in ''|*[!0-9]*) die "bad tp='$TP' for $MODEL_KEY (handoff.tp must be an integer)";; esac

NODE="$(held_running | awk -v j="$HOLDER" '$1 == j {print $2}')"
[ -n "$NODE" ] || die "holder job $HOLDER is not RUNNING (see: ci/dispatch/held_nodes.sh list)"

OUT_DIR="$HF_LOGS/$MODEL_KEY/ci_runs/$RUN_TS"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/held.out"

# ---- environment forwarding ----
# An srun step does NOT inherit the client's environment (verified: a variable
# exported here is <MISSING> on the far side), unlike the sbatch path where
# --export=ALL carried LITELLM_* into the job. Anything the run needs must be
# named on the command line, so forward an explicit allowlist.
#
# The list is deliberately narrow — secrets and caller intent only. Forwarding
# every GEAK_* would push this box's ci/config.sh defaults onto the node and
# silently override the node's own config; the node sources config.sh itself.
HELD_FORWARD_ENV_RE="${HELD_FORWARD_ENV_RE:-^(LITELLM_|ANTHROPIC_|CLAUDE_|HF_TOKEN$|HUGGINGFACE_|HUGGING_FACE_|GEAK_PROBE_SKIP_CLAUDE$|GEAK_FORCE_DSTATE_CHECK$|IMAGE$)}"
ENV_PREFIX=""
ENV_REDACTED=""
ENV_NAMES=()
while IFS= read -r _n; do
  [[ "$_n" =~ $HELD_FORWARD_ENV_RE ]] || continue
  [ -n "${!_n:-}" ] || continue
  ENV_PREFIX+=" $_n=$(printf '%q' "${!_n}")"
  ENV_REDACTED+=" $_n=<set>"
  ENV_NAMES+=("$_n")
done < <(compgen -v | sort)

# Derive the probe env once so the executed command and the --print preview can
# never disagree: ${PROBE:+...} would fire on PROBE=0 (non-empty), making the
# preview claim probe mode for a real run.
PROBE_ENV=""
[ "$PROBE" = "1" ] && PROBE_ENV=" GEAK_CI_PROBE=1"

REMOTE="RUN_TS=$(printf '%q' "$RUN_TS")"
REMOTE+="$PROBE_ENV"
REMOTE+="$ENV_PREFIX"
REMOTE+=" bash $(printf '%q' "$GEAK_ROOT/ci/dispatch/held_job.sh")"
REMOTE+=" $(printf '%q' "$MODEL_KEY") $(printf '%q' "$BUDGET")"

# Never echo the values — REMOTE carries live API keys.
REMOTE_SAFE="RUN_TS=$RUN_TS$PROBE_ENV$ENV_REDACTED bash $GEAK_ROOT/ci/dispatch/held_job.sh $MODEL_KEY $BUDGET"

log "run $MODEL_KEY: fw=$FW tp=$TP -> holder=$HOLDER node=$NODE budget=${BUDGET}s probe=$PROBE"
log "  out=$OUT_DIR log=$LOG"
log "  forwarding env: ${ENV_NAMES[*]:-<none>}"

if [ "$PRINT" = "1" ]; then
  { printf 'DRY-RUN held step for %s (holder %s on %s):\n  ' "$MODEL_KEY" "$HOLDER" "$NODE"
    printf 'srun --jobid=%s --overlap --nodes=1 --ntasks=1 bash -lc %q\n' "$HOLDER" "$REMOTE_SAFE"; } >&2
  exit 0
fi

printf '%s:%s\t%s\n' "$HOLDER" "$NODE" "$OUT_DIR"

# Stream the run into $LOG. HELD_EXEC_TIMEOUT_S caps the client so a wedged node
# cannot hang the matrix forever; default is the budget plus the same headroom the
# sbatch path adds to its wall clock.
TMO="${HELD_EXEC_TIMEOUT_S:-$(( BUDGET + SPUR_TIME_HEADROOM_S ))}"
{
  echo "=== held run: model=$MODEL_KEY holder=$HOLDER node=$NODE ts=$RUN_TS budget=${BUDGET}s"
  echo "=== started $(date -u +%FT%TZ)"
} >>"$LOG"
held_exec "$HOLDER" "$TMO" "$REMOTE" >>"$LOG" 2>&1
RC=$?
echo "=== finished $(date -u +%FT%TZ) rc=$RC" >>"$LOG"
log "$MODEL_KEY: held run finished rc=$RC (node=$NODE, log=$LOG)"
exit "$RC"
