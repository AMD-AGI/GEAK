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
#   $WS          — workspace (huggingface_logs, InferenceX, ...)
#   $MODELS_ROOT — model weights (usually outside the workspace)
#   $GEAK_ROOT   — the code under test. In CI it's a fresh checkout OUTSIDE $WS
#                  (the runner's _work dir), so mount it too. When it already
#                  lives under $WS (local dev) the $WS mount covers it.
GEAK_MOUNT=()
case "$GEAK_ROOT/" in
  "$WS"/*) : ;;
  *) GEAK_MOUNT=(-v "$GEAK_ROOT:$GEAK_ROOT") ;;
esac

# Pass the resolved paths through explicitly: inside the container lib.sh would
# otherwise re-derive WS/HF_LOGS/INFERENCEX_PATH from $GEAK_ROOT's location,
# which is wrong when the code under test is a checkout outside $WS.
docker run --rm \
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
  "
