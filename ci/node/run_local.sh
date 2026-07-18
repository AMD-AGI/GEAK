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
source "$HERE/../lib.sh"

MODEL_KEY="${1:?usage: run_local.sh <model_key> [--dry-run|--probe] [--budget N]}"; shift || true
DRY=""; PROBE=0; BUDGET="${PERFSKILLS_E2E_TIMEOUT_S:-1800}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--dry-run" ;;
    # --probe: exercise the REAL infra (SPUR alloc, docker, GPU preflight, image
    # pull, weights mount, optional Claude install) but STOP at the GEAK e2e
    # doorstep instead of running the (hours-long) workflow. Fast harness check.
    --probe)   PROBE=1 ;;
    --budget)  BUDGET="${2:?}"; shift ;;
    *) die "unknown arg: $1" ;;
  esac; shift
done

FW="$(model_framework "$MODEL_KEY")" || die "unknown model: $MODEL_KEY (add it to $MODELS_TSV)"
[ -n "$FW" ] || die "unknown model: $MODEL_KEY (add it to $MODELS_TSV)"

# ---- optional GPU pinning ----
# GEAK_GPUS is a comma-separated list of ROCm GPU indices (e.g. "4,5,6,7" to use
# the last 4 of an 8-GPU box). Empty => use all visible GPUs. We pin with ONLY
# ROCR_VISIBLE_DEVICES: it masks at the ROCr level so rocminfo, torch and vLLM
# all see just the selected devices (renumbered 0..N-1). Do NOT also set
# HIP_VISIBLE_DEVICES to the same list — HIP applies its mask ON TOP of the
# already-renumbered ROCr set, so e.g. both set to "4,5,6,7" selects indices
# 4-7 of the 4 remaining devices {0,1,2,3} => nothing, and torch.cuda goes away.
# The /dev/dri passthrough still exposes all render nodes; the mask keeps work
# off the others.
GPUS="${GEAK_GPUS:-}"
GPU_ENV=()
if [ -n "$GPUS" ]; then
  GPU_ENV=(-e ROCR_VISIBLE_DEVICES="$GPUS")
  log "GPU pinning: ROCR_VISIBLE_DEVICES=$GPUS"
fi

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

# Weights are picked up from the catalog ($MODELS_ROOT, e.g. /home/ethany/hf_models)
# whose entries are usually SYMLINKS into shared NFS. Bind-mount the byte-holding
# roots too (read-only, same-path) so those links — and HF hub-cache blob links
# (snapshots/<h>/*.safetensors -> ../../blobs/<h>) — resolve INSIDE the container.
# $WEIGHTS_EXTRA_MOUNTS is a colon-separated list of roots (default /shared_nfs).
WEIGHTS_MOUNTS=()
IFS=: read -r -a _wm <<< "${WEIGHTS_EXTRA_MOUNTS:-/shared_nfs}"
for _d in "${_wm[@]}"; do
  [ -n "$_d" ] && [ -d "$_d" ] && WEIGHTS_MOUNTS+=(-v "$_d:$_d:ro")
done

# ---- GPU wedge pre-check: bail BEFORE touching the GPU if the driver is hung ----
# A driver-level wedge parks tasks in uninterruptible (D) state, which NOTHING can
# kill (not SIGKILL, not `timeout`). If we ran rocminfo/torch against that, our own
# probe would hang forever too. So first cheaply scan /proc (touches no GPU) and
# fail fast if the box is already wedged. Set GEAK_SKIP_DSTATE_CHECK=1 to skip.
if [ "${GEAK_SKIP_DSTATE_CHECK:-0}" != "1" ]; then
  log "GPU wedge pre-check (D-state scan, no GPU access) ..."
  if ! bash "$HERE/../preflight/gpu_dstate_check.sh"; then
    echo "::error::GPU appears wedged at the driver level (process(es) stuck in D-state in the amdgpu/kfd path). Refusing to start — the box likely needs a GPU reset or reboot." >&2
    die "GPU wedge pre-check failed (D-state in amdgpu/kfd) — needs GPU reset/reboot; not starting the run"
  fi
fi

# ---- Pre-pull the image OUTSIDE the healthcheck timeout ----
# On a SLURM cluster each job can land on a FRESH compute node with no cached
# image, so the first-time pull of a multi-GB ROCm image can take minutes. The
# GPU preflight cap below must bound only the rocminfo+torch probe, NOT this
# network pull (otherwise a cold node always "fails" preflight on the pull).
# Set GEAK_SKIP_PULL=1 to skip; IMAGE_PULL_CAP overrides the (generous) cap.
if [ "${GEAK_SKIP_PULL:-0}" != "1" ]; then
  log "ensuring image present: docker pull $IMAGE (cap ${IMAGE_PULL_CAP:-1800}s)"
  timeout --kill-after=60 "${IMAGE_PULL_CAP:-1800}" docker pull "$IMAGE" >&2 \
    || log "WARN: docker pull returned non-zero — will try any locally cached image"
fi

# ---- GPU preflight gate: fail FAST if the GPU is unusable BEFORE the long run ----
# A short, timeout-bounded probe in the SAME image + device flags as the real run.
# Catches a dead/wedged GPU (or a docker/device problem) in seconds instead of
# discovering it hours into the workflow (which then limps to a false-green
# no_gain). Set GPU_HEALTHCHECK_TIMEOUT_S=0 to skip (e.g. CPU-only debugging).
HEALTHCHECK_CAP="${GPU_HEALTHCHECK_TIMEOUT_S:-120}"
PF_NAME="geak_pf_${MODEL_KEY//[^A-Za-z0-9_.-]/_}_${RUN_TS}"
if [ "$HEALTHCHECK_CAP" != "0" ]; then
  log "GPU preflight: probing $IMAGE (rocminfo + torch matmul, ${HEALTHCHECK_CAP}s cap)"
  # --kill-after escalates SIGTERM->SIGKILL so a FRESH wedge (one the D-state
  # pre-check couldn't foresee) still can't hang the job past the cap; the probe
  # is NAMED so we can best-effort force-remove the (possibly orphaned) container.
  if ! timeout --kill-after=30 "$HEALTHCHECK_CAP" docker run --rm --name "$PF_NAME" \
      --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined \
      -v "$WS:$WS" "${GEAK_MOUNT[@]}" "${GPU_ENV[@]}" \
      --entrypoint bash "$IMAGE" "$HERE/../preflight/gpu_healthcheck.sh"; then
    # Best-effort reap of a wedged probe container (may itself refuse if D-state).
    ( docker kill "$PF_NAME" >/dev/null 2>&1; docker rm -f "$PF_NAME" >/dev/null 2>&1 ) &
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
# In-container command. Normal: install Claude then run the GEAK e2e workflow.
# Probe: verify weights are readable in-container, (optionally) install Claude,
# validate the GEAK arg mapping via run_model --dry-run, then STOP — never enter
# the real e2e workflow. Set GEAK_PROBE_SKIP_CLAUDE=1 for the fastest infra-only probe.
if [ "$PROBE" = "1" ]; then
  CONTAINER_CMD="
    set -e
    echo \"== PROBE: container up on \$(hostname) ==\"
    echo \"PROBE: MODEL_PATH=\$MODEL_PATH\"
    if [ -f \"\$MODEL_PATH/config.json\" ]; then echo 'PROBE: weights readable in container OK'; else echo 'PROBE FAIL: weights not readable in container'; exit 3; fi
    if [ \"\${GEAK_PROBE_SKIP_CLAUDE:-0}\" != 1 ]; then bash '$GEAK_ROOT/ci/preflight/setup_claude.sh'; else echo 'PROBE: skipping Claude setup (GEAK_PROBE_SKIP_CLAUDE=1)'; fi
    bash '$GEAK_ROOT/ci/node/run_model.sh' '$MODEL_KEY' --dry-run
    echo '== PROBE OK: infra verified up to GEAK phase entry; stopping before the e2e workflow =='
  "
else
  CONTAINER_CMD="
    set -e
    bash '$GEAK_ROOT/ci/preflight/setup_claude.sh'
    bash '$GEAK_ROOT/ci/node/run_model.sh' '$MODEL_KEY'
  "
fi

docker run --rm --name "$CONTAINER_NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32g \
  -v "$WS:$WS" -v "$MODELS_ROOT:$MODELS_ROOT" "${GEAK_MOUNT[@]}" "${WEIGHTS_MOUNTS[@]}" \
  -e WS="$WS" -e HF_LOGS="$HF_LOGS" -e INFERENCEX_PATH="$INFERENCEX_PATH" \
  -e GEAK_ROOT="$GEAK_ROOT" -e MODELS_TSV="$MODELS_TSV" \
  -e MODEL_PATH="$WEIGHTS" -e GEAK_PROBE_SKIP_CLAUDE \
  -e LITELLM_API_KEY -e LITELLM_BASE_URL -e NODE_TLS_REJECT_UNAUTHORIZED=0 \
  -e RUN_TS="$RUN_TS" -e OUT_DIR="$OUT_DIR" \
  -e CLAUDE_HOME="$OUT_DIR/claude" \
  -e PERFSKILLS_E2E_TIMEOUT_S="$BUDGET" \
  -e TMPDIR="$DBG_TMP" \
  "${GPU_ENV[@]}" \
  --entrypoint bash \
  "$IMAGE" -lc "$CONTAINER_CMD" &
DOCKER_PID=$!

# Start the host-side liveness monitor (Claude arbiter, runs on the host, never
# touches the GPU). Set GEAK_MONITOR=0 to disable. It self-exits when the
# container stops; the EXIT trap tears it down on any early exit of this script.
MON_PID=""
if [ "${GEAK_MONITOR:-1}" != "0" ] && [ "$PROBE" != "1" ]; then
  bash "$HERE/../monitor/run_monitor.sh" "$CONTAINER_NAME" "$OUT_DIR/run.log" "$OUT_DIR" "$DOCKER_PID" &
  MON_PID=$!
  log "monitor started (pid=$MON_PID, container=$CONTAINER_NAME)"
fi
cleanup() { [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true; }
trap cleanup EXIT

RC=0
wait "$DOCKER_PID" || RC=$?
trap - EXIT
cleanup

# Probe mode: drop a marker the dispatcher can judge on (no result.json is produced
# because the e2e workflow never runs).
if [ "$PROBE" = "1" ]; then
  if [ "$RC" -eq 0 ]; then
    echo "ok" > "$OUT_DIR/probe_ok"
    log "PROBE OK -> $OUT_DIR/probe_ok"
  else
    log "PROBE FAILED (rc=$RC)"
  fi
  exit "$RC"
fi

# Surface a monitor kill as the failure reason even if docker's own rc is generic.
if [ -f "$OUT_DIR/monitor_verdict.json" ]; then
  echo "::error::run killed by liveness monitor (see monitor_verdict.json / monitor.log)" >&2
  log "monitor verdict present -> $OUT_DIR/monitor_verdict.json"
  [ "$RC" -eq 0 ] && RC=1
fi
exit "$RC"
