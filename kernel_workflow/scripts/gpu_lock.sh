#!/bin/bash
# GPU lock + per-workspace build isolation wrapper.
# Usage:  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>
#
# Run EVERY kernel command (compile / correctness / benchmark / profile) through this wrapper,
# invoked from inside the workspace directory. It does three generic things — none kernel-specific:
#
#  1. flock per GPU id  -> multiple engineers can share GPUs safely (exclusive during the command).
#  2. TORCH_EXTENSIONS_DIR = <workspace>/.torch_ext  -> isolates the torch cpp_extension build cache
#     PER WORKSPACE. Without this, torch.utils.cpp_extension.load(name=...) compiles every engineer's
#     DIFFERENT source into ONE global cache (~/.cache/torch_extensions/...), which both serializes
#     all parallel compiles on a single global lock AND lets one engineer benchmark another's .so.
#     Deriving it from $PWD makes each isolated workspace get its own cache. (Honors a caller-set
#     TORCH_EXTENSIONS_DIR if already exported.)
#  3. PYTORCH_ROCM_ARCH = the local GPU's gfx arch only -> avoids compiling for ~9 architectures
#     (huge compile speedup). Runtime perf and correctness are unaffected (the kernel runs on the
#     local arch either way). Honors a caller-set PYTORCH_ROCM_ARCH if already exported.

set -euo pipefail

GPU_SPEC="${1:?Usage: gpu_lock.sh <gpu_id|pool> <command...>   (pool = comma list, e.g. 0,1,2,3)}"
shift

LOCK_DIR="/tmp/team_gpu_locks"
mkdir -p "$LOCK_DIR"

# ---- GPU selection ------------------------------------------------------------------------------
# <gpu_spec> is either a single id ("2", the historical contract, unchanged) or a POOL ("0,1,2,3").
# With a pool we do not pre-assign a GPU: we take the first one that is BOTH unlocked AND idle,
# retrying until one frees. Static pre-assignment (the old `GPU_LIST[i % n]`) cannot self-balance,
# because the binding is chosen before anyone knows how long a job runs -- in a real 4-GPU run that
# left one GPU with 194 lock calls and another with 0.
#
# The pool is whatever the caller passes, so this scales to 1, 2, 4, 8 ... GPUs with no code change.
#
# IDLENESS (GEAK_GPU_REQUIRE_IDLE=1, the default) is checked against the KERNEL DRIVER via sysfs,
# not against our own locks -- a foreign tenant's job is invisible to flock but will still corrupt
# timings. Measured on an idle MI350X: gpu_busy_percent=0, mem_info_vram_used=284MB; under load:
# 100% / 1837MB. amd-smi is deliberately NOT used here: it returns EMPTY output while another
# process holds the GPU, i.e. it fails exactly when we need it. Set GEAK_GPU_REQUIRE_IDLE=0 to skip
# (e.g. deliberately co-tenanted screening runs).
_gpu_is_idle() {
    local id="$1" dev busy vram
    dev="$(readlink -f "/sys/class/drm/renderD$((128 + 8 * id))/device" 2>/dev/null)" || return 0
    [ -r "$dev/gpu_busy_percent" ] || return 0   # cannot tell -> do not block the run
    busy="$(cat "$dev/gpu_busy_percent" 2>/dev/null || echo 0)"
    vram="$(( $(cat "$dev/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 ))"
    [ "${busy:-0}" -le "${GEAK_GPU_MAX_BUSY_PCT:-5}" ] && [ "$vram" -le "${GEAK_GPU_MAX_VRAM_MB:-1024}" ]
}

case "$GPU_SPEC" in
  *,*)
    # --- pool mode: block until some lane is free AND idle, then hold it for the whole command ---
    POOL="$(echo "$GPU_SPEC" | tr ',' ' ')"
    _deadline=$(( SECONDS + ${GEAK_GPU_POOL_WAIT:-1200} ))
    GPU_ID=""
    while [ -z "$GPU_ID" ]; do
        for _g in $POOL; do
            exec {_fd}>"${LOCK_DIR}/gpu_${_g}.lock"
            if flock -n -x "$_fd"; then
                # We hold the lane. Only now check idleness -- checking before locking would race.
                if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ] && ! _gpu_is_idle "$_g"; then
                    flock -u "$_fd"; exec {_fd}>&-   # foreign job on this GPU: try the next lane
                    continue
                fi
                GPU_ID="$_g"; POOL_FD="$_fd"; break
            fi
            exec {_fd}>&-
        done
        if [ -z "$GPU_ID" ]; then
            [ "$SECONDS" -ge "$_deadline" ] && { echo "ERROR: no free+idle GPU in pool [$GPU_SPEC] after ${GEAK_GPU_POOL_WAIT:-1200}s" >&2; exit 1; }
            sleep 0.2
        fi
    done
    ;;
  *)
    # Single-GPU mode: unchanged from before this change. Falls through to the flock below.
    GPU_ID="$GPU_SPEC"
    ;;
esac

LOCK_FILE="${LOCK_DIR}/gpu_${GPU_ID}.lock"

# (0) Reap ORPHANED hung rocm_agent_enumerator procs before running. aiter's import spawns one such
# subprocess per Python process for gfx detection; under GPU/KFD contention they HANG instead of
# exiting (<1s normally). With many parallel kernel jobs they pile up by the hundreds -> kernel
# task-count explosion -> whole-box hang (observed: 561 enumerators / 37k tasks on a swap=0 box).
# We kill ONLY ppid==1 (parent already dead) AND >60s old -> a live, in-use enumerator is never
# touched. Best-effort; must never fail the wrapper (set -e). Opt out with KERNEL_ENV_SKIP_ENUM_REAP=1.
if [ "${KERNEL_ENV_SKIP_ENUM_REAP:-0}" != "1" ]; then
    for _p in $(pgrep -f rocm_agent_enumerator 2>/dev/null || true); do
        _pp="$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        _et="$(ps -o etimes= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        if [ "${_pp:-0}" = "1" ] && [ -n "${_et:-}" ] && [ "${_et:-0}" -gt 60 ] 2>/dev/null; then
            kill -9 "$_p" 2>/dev/null || true
        fi
    done
fi

# (2) Per-workspace torch extension build cache (default: a hidden dir in the current workspace).
: "${TORCH_EXTENSIONS_DIR:=$PWD/.torch_ext}"
export TORCH_EXTENSIONS_DIR
mkdir -p "$TORCH_EXTENSIONS_DIR" 2>/dev/null || true

# (3) Compile for the local GPU arch only. The environment's default PYTORCH_ROCM_ARCH is often a
# long multi-arch list (~9 targets) → ~9x slower compiles for no benefit on a single-arch box. We
# OVERRIDE it to the detected local arch. Set KERNEL_ENV_KEEP_ARCH=1 to opt out (multi-arch boxes).
if [ "${KERNEL_ENV_KEEP_ARCH:-0}" != "1" ]; then
    _ARCH="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
    [ -n "${_ARCH:-}" ] && export PYTORCH_ROCM_ARCH="$_ARCH"
    # Also pin GPU_ARCHS so aiter's JIT (chip_info.get_gfx_list) takes the env branch instead of
    # _detect_native(), which shells to rocm_agent_enumerator -> rocminfo PER cold-build worker
    # (~77 per cold aiter import). Under the parallel bake-off (isolated per-workspace build caches =>
    # many cold builds) those rocminfo calls hang on the contended KFD driver and pile up by the
    # hundreds -> kernel task-count explosion + ~2x serving-throughput degradation. Setting GPU_ARCHS
    # eliminates the spawn at the source (the reap above is now just a backstop). Honor a caller value.
    [ -n "${_ARCH:-}" ] && export GPU_ARCHS="${GPU_ARCHS:-$_ARCH}"
fi

if [ -n "${POOL_FD:-}" ]; then
    # Pool mode: this process ALREADY holds the lane exclusively (and verified it idle). Re-locking
    # the same file from the same process would be a no-op at best, so just run -- the lane stays
    # held until we exit, which is what guarantees no two evaluations share a GPU.
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    "$@"
else
    (
        flock -x -w 1200 200 || { echo "ERROR: Failed to acquire GPU $GPU_ID lock after 1200s"; exit 1; }
        # Idleness is OPT-IN here (default 0) but on by default in pool mode. Deliberate: a pool can
        # step to another GPU when one is busy, whereas a single GPU has no alternative, so enabling
        # it by default would turn a previously-working run into a hard failure.
        if [ "${GEAK_GPU_REQUIRE_IDLE:-0}" = "1" ] && ! _gpu_is_idle "$GPU_ID"; then
            echo "ERROR: GPU $GPU_ID has foreign work running (busy/VRAM above threshold); refusing to measure on it" >&2
            exit 1
        fi
        export HIP_VISIBLE_DEVICES="$GPU_ID"
        "$@"
    ) 200>"$LOCK_FILE"
fi
