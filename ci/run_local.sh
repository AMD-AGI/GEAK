#!/usr/bin/env bash
# Top-level LOCAL launcher (no GitHub yet).
#   real run  -> docker (GPU passthrough + same-path mount + Claude)
#   --dry-run -> host only (no docker/GPU/Claude), validates handoff->args wiring
#
# Usage:
#   ci/run_local.sh <model_key> [--dry-run] [--budget SECONDS]
# Examples:
#   ci/run_local.sh Qwen-Qwen3-8B --dry-run
#   ci/run_local.sh Qwen-Qwen3-8B --budget 1800
#   IMAGE=rocm/vllm-dev:some-gfx950-tag ci/run_local.sh Qwen-Qwen3-8B
#
# The container's TMPDIR always points at a per-run bind-mounted dir so Claude Code's
# background-task tree (/tmp/claude-<uid>/.../tasks/*) survives the container for
# post-mortem debugging. (Note: this also redirects other tools' scratch onto the
# host mount.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib.sh"

MODEL_KEY="${1:?usage: run_local.sh <model_key> [--dry-run] [--budget N]}"; shift || true
DRY=""; BUDGET="${PERFSKILLS_E2E_TIMEOUT_S:-1800}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--dry-run" ;;
    --budget)  BUDGET="${2:?}"; shift ;;
    *) die "unknown arg: $1" ;;
  esac; shift
done

FW="$(model_framework "$MODEL_KEY")" || die "unknown model: $MODEL_KEY (add it to $MODELS_TSV)"
[ -n "$FW" ] || die "unknown model: $MODEL_KEY (add it to $MODELS_TSV)"

# RUN_TS may be provided by the caller (e.g. CI) so it can predict OUT_DIR and
# collect artifacts; otherwise mint a fresh one here.
RUN_TS="${RUN_TS:-$(new_ts)}"
OUT_DIR="$HF_LOGS/$MODEL_KEY/ci_runs/$RUN_TS"
mkdir -p "$OUT_DIR"

# ---- dry-run: host only, no container ----
if [ "$DRY" = "--dry-run" ]; then
  log "DRY-RUN on host (no docker/GPU/Claude) -> $OUT_DIR"
  RUN_TS="$RUN_TS" OUT_DIR="$OUT_DIR" bash "$HERE/run_model.sh" "$MODEL_KEY" --dry-run
  exit $?
fi

# ---- real run: container ----
IMAGE="$(resolve_image "$FW")"
WEIGHTS="$(model_weights "$MODEL_KEY")"; MODELS_ROOT="$(dirname "$WEIGHTS")"
[ -d "$WEIGHTS" ] || die "weights dir not found: $WEIGHTS"
log "model=$MODEL_KEY fw=$FW image=$IMAGE weights=$WEIGHTS ts=$RUN_TS budget=${BUDGET}s"

# Persist Claude Code's temp tree (incl. background-task transcripts) by redirecting
# TMPDIR into the per-run bind-mounted output dir. Same-path so it's identical inside
# and outside the container, and survives the (--rm) container for debugging.
DBG_TMP="$OUT_DIR/claude_tmp"
mkdir -p "$DBG_TMP"
log "TMPDIR=$DBG_TMP (Claude task tree persisted here)"

# Same-path bind mounts so paths are identical inside and outside the container:
#   $WS          — workspace (geak_runtime, InferenceX, ...)
#   $MODELS_ROOT — model weights (usually outside the workspace)
#   $GEAK_ROOT   — the code under test. In CI it's a fresh checkout OUTSIDE $WS
#                  (the runner's _work dir), so mount it too. When it already
#                  lives under $WS (local dev) the $WS mount covers it.
GEAK_MOUNT=()
case "$GEAK_ROOT/" in
  "$WS"/*) : ;;
  *) GEAK_MOUNT=(-v "$GEAK_ROOT:$GEAK_ROOT") ;;
esac

# ---- GPU preflight gate: fail FAST if the GPU is unusable BEFORE the long run ----
# A short, timeout-bounded probe in the SAME image + device flags as the real run.
# Catches a dead/wedged GPU (or a docker/device problem) in seconds instead of
# discovering it hours into the workflow (which then limps to a false-green
# no_gain). Set GPU_HEALTHCHECK_TIMEOUT_S=0 to skip (e.g. CPU-only debugging).
HEALTHCHECK_CAP="${GPU_HEALTHCHECK_TIMEOUT_S:-120}"
if [ "$HEALTHCHECK_CAP" != "0" ]; then
  log "GPU preflight: probing $IMAGE (rocminfo + torch matmul, ${HEALTHCHECK_CAP}s cap)"
  if ! timeout "$HEALTHCHECK_CAP" docker run --rm \
      --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined \
      -v "$WS:$WS" "${GEAK_MOUNT[@]}" \
      --entrypoint bash "$IMAGE" "$HERE/gpu_healthcheck.sh"; then
    echo "::error::GPU preflight failed for $MODEL_KEY (image=$IMAGE) — GPU unusable, or docker/probe error/timeout. Refusing to start the run." >&2
    die "GPU preflight failed (model=$MODEL_KEY image=$IMAGE) — GPU unusable or probe error/timeout; not starting the run"
  fi
  log "GPU preflight OK"
fi

# Named so the host-side monitor can `docker kill` it on a confirmed-stuck run.
CONTAINER_NAME="geak_l1_${MODEL_KEY//[^A-Za-z0-9_.-]/_}_${RUN_TS}"

# Pass the resolved paths through explicitly: inside the container lib.sh would
# otherwise re-derive WS/HF_LOGS/INFERENCEX_PATH from $GEAK_ROOT's location,
# which is wrong when the code under test is a checkout outside $WS.
#
# Launched in the BACKGROUND so a host-side liveness monitor (ci/run_monitor.sh)
# can watch $OUT_DIR/run.log and kill this container if the run wedges (dead GPU,
# NFS stall, OOM loop) instead of hanging until the job's wall-clock timeout. We
# then `wait` for the real exit code so the CI step reports pass/fail correctly.
docker run --rm --name "$CONTAINER_NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32g \
  -v "$WS:$WS" -v "$MODELS_ROOT:$MODELS_ROOT" "${GEAK_MOUNT[@]}" \
  -e WS="$WS" -e HF_LOGS="$HF_LOGS" -e INFERENCEX_PATH="$INFERENCEX_PATH" \
  -e GEAK_ROOT="$GEAK_ROOT" -e MODELS_TSV="$MODELS_TSV" \
  -e LITELLM_API_KEY -e LITELLM_BASE_URL -e NODE_TLS_REJECT_UNAUTHORIZED=0 \
  -e RUN_TS="$RUN_TS" -e OUT_DIR="$OUT_DIR" \
  -e CLAUDE_HOME="$OUT_DIR/claude" \
  -e PERFSKILLS_E2E_TIMEOUT_S="$BUDGET" \
  -e TMPDIR="$DBG_TMP" \
  --entrypoint bash \
  "$IMAGE" -lc "
    set -e
    bash '$GEAK_ROOT/ci/setup_claude.sh'
    bash '$GEAK_ROOT/ci/run_model.sh' '$MODEL_KEY'
  " &
DOCKER_PID=$!

# Start the host-side liveness monitor (Claude arbiter, runs on the host, never
# touches the GPU). Set GEAK_MONITOR=0 to disable. It self-exits when the
# container stops; the EXIT trap tears it down on any early exit of this script.
MON_PID=""
if [ "${GEAK_MONITOR:-1}" != "0" ]; then
  bash "$HERE/run_monitor.sh" "$CONTAINER_NAME" "$OUT_DIR/run.log" "$OUT_DIR" "$DOCKER_PID" &
  MON_PID=$!
  log "monitor started (pid=$MON_PID, container=$CONTAINER_NAME)"
fi
cleanup() { [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true; }
trap cleanup EXIT

RC=0
wait "$DOCKER_PID" || RC=$?
trap - EXIT
cleanup

# Surface a monitor kill as the failure reason even if docker's own rc is generic.
if [ -f "$OUT_DIR/monitor_verdict.json" ]; then
  echo "::error::run killed by liveness monitor (see monitor_verdict.json / monitor.log)" >&2
  log "monitor verdict present -> $OUT_DIR/monitor_verdict.json"
  [ "$RC" -eq 0 ] && RC=1
fi
exit "$RC"
