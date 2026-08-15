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

# The usage line deliberately shows NO concrete ids. It used to read "e.g. 0,1,2,3", and that
# example was copied verbatim into real commands by agents improvising one-off checks -- the
# literal string "0,1,2,3" turned up in 15 invocations from runs that had been allocated neither
# GPU 2 nor 3. An example in a usage line gets read as a default; here the value is never
# defaultable, so it names no ids.
GPU_SPEC="${1:?Usage: gpu_lock.sh <gpu_id|pool> <command...>   (pool = comma list of the GPUs THIS run was allocated)}"
shift

LOCK_DIR="/tmp/team_gpu_locks"
mkdir -p "$LOCK_DIR"

# ---- Allocation fence ---------------------------------------------------------------------------
# GEAK_GPU_ALLOWED (comma list) is the set of GPUs the CALLER was actually allocated. When it is
# set, a <gpu_spec> naming anything outside it is refused. Unset = no fence, so every existing
# caller is unaffected.
#
# Why this exists: the spec arrives as an argv string written by whoever composes the command, and
# in an agent-driven workflow that author is a model. Agents reproduce the usage line's example
# verbatim when improvising a check outside the workflow's normal call sites. The idleness test
# below catches the common case -- a busy foreign card gets skipped -- but it is the wrong
# instrument: it asks "is this GPU free?" when the question is "is this GPU MINE?". A neighbour's
# momentarily-idle card passes idleness and fails ownership, and a run labelled "2 GPUs" silently
# becomes a 3-GPU run, which invalidates the measurement rather than merely slowing it.
#
# Enforced here because this wrapper is the single chokepoint: every compile/correctness/benchmark/
# profile command goes through it by contract, so one check also covers call sites that do not exist
# yet. A hard error rather than a silent intersection -- a caller asking for a GPU it does not own
# has a bug, and quietly running elsewhere would hide it while still producing a number.
if [ -n "${GEAK_GPU_ALLOWED:-}" ]; then
    _bad=""
    for _r in $(echo "$GPU_SPEC" | tr ',' ' '); do
        case ",${GEAK_GPU_ALLOWED}," in *",${_r},"*) ;; *) _bad="$_bad $_r" ;; esac
    done
    if [ -n "$_bad" ]; then
        echo "ERROR: gpu_lock.sh asked for GPU(s)${_bad} but this run is allocated only [${GEAK_GPU_ALLOWED}]." >&2
        echo "       Use the allocated set: gpu_lock.sh ${GEAK_GPU_ALLOWED} <command...>" >&2
        exit 1
    fi
fi

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
# (e.g. deliberately co-tenanted screening runs). The default is 1 on BOTH paths: pool mode steps to
# another GPU when one is busy, single-GPU mode has nowhere to step and so fails loudly instead.
#
# The two criteria are reported SEPARATELY, because they are not the same finding.
#   gpu_busy_percent  -- someone is executing on the CUs. This contaminates a timing. Hard stop.
#   mem_info_vram_used -- someone has memory RESIDENT. At busy=0 that neighbour holds no CUs and no
#                        bandwidth, so its effect on a measurement is ~nil.
# Both used to exit through one message ("foreign work running"), and an idle 161-164 GB allocation
# on GPU 0 -- busy 0% the whole time -- therefore read as "another tenant is computing" and refused
# every verifier launch for two full rounds. The one direction that got a measurement did it by
# setting GEAK_GPU_MAX_VRAM_MB=185000 itself, which relaxes ONLY the VRAM half; its two reps agreed
# to 1.0%, i.e. relaxing was safe and the gate had simply not said what it was actually objecting to.
# So: name the criterion that failed, quote the measured values, and name the switch that changes it.
# GEAK_GPU_ALLOW_IDLE_VRAM=1 accepts a card whose VRAM is over threshold while busy% is under it
# (a resident-but-idle neighbour). Default 0: unchanged behavior, but now a legible refusal.
#
# _gpu_is_idle sets _GPU_IDLE_WHY to a caller-printable explanation whenever it returns non-zero.
_GPU_IDLE_WHY=""
_gpu_is_idle() {
    local id="$1" dev busy vram maxbusy maxvram
    _GPU_IDLE_WHY=""
    dev="$(readlink -f "/sys/class/drm/renderD$((128 + 8 * id))/device" 2>/dev/null)" || return 0
    [ -r "$dev/gpu_busy_percent" ] || return 0   # cannot tell -> do not block the run
    busy="$(cat "$dev/gpu_busy_percent" 2>/dev/null || echo 0)"
    vram="$(( $(cat "$dev/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 ))"
    maxbusy="${GEAK_GPU_MAX_BUSY_PCT:-5}"
    maxvram="${GEAK_GPU_MAX_VRAM_MB:-1024}"

    if [ "${busy:-0}" -gt "$maxbusy" ]; then
        _GPU_IDLE_WHY="foreign work is EXECUTING on GPU $id: gpu_busy_percent=${busy}% > GEAK_GPU_MAX_BUSY_PCT=${maxbusy}% (VRAM ${vram}MB). A timing taken here is contaminated; this is not a threshold to raise casually."
        return 1
    fi
    if [ "$vram" -gt "$maxvram" ]; then
        if [ "${GEAK_GPU_ALLOW_IDLE_VRAM:-0}" = "1" ]; then
            echo "WARNING: GPU $id carries an IDLE VRAM allocation: ${vram}MB > GEAK_GPU_MAX_VRAM_MB=${maxvram}MB, but gpu_busy_percent=${busy}% <= ${maxbusy}%. Proceeding because GEAK_GPU_ALLOW_IDLE_VRAM=1 (resident memory at 0% busy holds no CUs and no bandwidth). Watch for OOM, not for skew." >&2
            return 0
        fi
        _GPU_IDLE_WHY="GPU $id is NOT executing anything (gpu_busy_percent=${busy}% <= GEAK_GPU_MAX_BUSY_PCT=${maxbusy}%) but has ${vram}MB VRAM resident > GEAK_GPU_MAX_VRAM_MB=${maxvram}MB. This is a memory-residency refusal, NOT contention. If that allocation is a parked neighbour, set GEAK_GPU_ALLOW_IDLE_VRAM=1 (accept idle residency, keep the busy% check) or raise GEAK_GPU_MAX_VRAM_MB above ${vram}."
        return 1
    fi
    return 0
}

case "$GPU_SPEC" in
  *,*)
    # --- pool mode: block until some lane is free AND idle, then hold it for the whole command ---
    POOL="$(echo "$GPU_SPEC" | tr ',' ' ')"
    # When the wait started. Time spent blocked here is the SCHEDULER'S COST -- the price paid for
    # sharing GPUs instead of pinning one per engineer -- and it used to leave no trace at all: the
    # loop retries on `sleep 0.2` and only writes the use-log AFTER it wins a GPU, so an acquisition
    # that took ten minutes and one that took none were recorded identically.
    _wait_t0=$SECONDS
    _deadline=$(( _wait_t0 + ${GEAK_GPU_POOL_WAIT:-1200} ))
    GPU_ID=""
    while [ -z "$GPU_ID" ]; do
        _POOL_WHY=""   # reasons from THIS sweep only; the loop re-sweeps every 0.2s
        for _g in $POOL; do
            exec {_fd}>"${LOCK_DIR}/gpu_${_g}.lock"
            if flock -n -x "$_fd"; then
                # We hold the lane. Only now check idleness -- checking before locking would race.
                if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ] && ! _gpu_is_idle "$_g"; then
                    # Keep the newest reason per lane. Stepping to the next GPU is the right move,
                    # but if the WHOLE pool is refused the deadline error below must be able to say
                    # why each lane was refused -- "no free+idle GPU" alone is unactionable, and a
                    # pool refused entirely for idle-VRAM looks identical to one that is genuinely
                    # saturated.
                    _POOL_WHY="$(printf '%s\n  - %s' "${_POOL_WHY:-}" "$_GPU_IDLE_WHY")"
                    flock -u "$_fd"; exec {_fd}>&-   # this GPU is unusable right now: try the next lane
                    continue
                fi
                GPU_ID="$_g"; POOL_FD="$_fd"
                # Record which GPU of the pool was actually taken. Without this a pool acquisition
                # is unobservable after the fact: when a foreign tenant holds part of the pool the
                # loop above silently settles for a smaller set, and the run is still filed under
                # its original "N GPUs" label. One append-only line per acquisition makes the real
                # GPU set recoverable. Off unless GEAK_GPU_USE_LOG names a file.
                #
                # wait_s goes on the SAME line rather than into a new file: it is a property of this
                # acquisition, every reader already parses this line, and appending a field stays
                # backward-compatible with logs written before it existed. Nothing is added inside
                # the timed region -- the arithmetic runs after the GPU is already won.
                [ -n "${GEAK_GPU_USE_LOG:-}" ] && \
                    echo "{\"t\":$(date +%s),\"gpu\":$_g,\"pool\":\"$GPU_SPEC\",\"pid\":$$,\"mode\":\"pool\",\"wait_s\":$(( SECONDS - _wait_t0 ))}" \
                        >> "$GEAK_GPU_USE_LOG" 2>/dev/null
                break
            fi
            exec {_fd}>&-
        done
        if [ -z "$GPU_ID" ]; then
            [ "$SECONDS" -ge "$_deadline" ] && {
                echo "ERROR: no free+idle GPU in pool [$GPU_SPEC] after ${GEAK_GPU_POOL_WAIT:-1200}s" >&2
                [ -n "${_POOL_WHY:-}" ] && echo "  last sweep, per GPU:${_POOL_WHY}" >&2
                exit 1
            }
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
    # Single-GPU mode BLOCKS TOO, and its wait must be measured for the same reason the pool's is.
    # Pinning does not mean "no contention": with 4 engineers over 2 pinned GPUs, two engineers share
    # each card and serialize on exactly this flock. That queueing is the pinned policy's own cost,
    # and comparing a measured pool wait against an assumed-zero pin wait would build the scheduler's
    # advantage into the instrument. Both paths measure, so the comparison is real.
    _wait_t0=$SECONDS
    (
        flock -x -w 1200 200 || { echo "ERROR: Failed to acquire GPU $GPU_ID lock after 1200s"; exit 1; }
        # Default 1, matching pool mode above. It was 0 here, so a PINNED engineer skipped the
        # foreign-work check entirely -- not "sampled it once", never ran it. That is how two
        # measurement cells ended up timed on a GPU another tenant was executing on. Both paths ask
        # the same question of the same driver; there is no reason for them to answer differently,
        # and the unsafe default was the one nobody had to opt into. The original rationale for 0
        # was that a single GPU has no alternative to step to, which is true -- but it argues for a
        # loud failure, not for measuring on a contaminated card. Set GEAK_GPU_REQUIRE_IDLE=0 to
        # restore the old behavior for deliberately co-tenanted runs.
        if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ] && ! _gpu_is_idle "$GPU_ID"; then
            echo "ERROR: refusing to measure on GPU $GPU_ID." >&2
            echo "  $_GPU_IDLE_WHY" >&2
            echo "  (GEAK_GPU_REQUIRE_IDLE=0 disables this gate entirely -- only for deliberately co-tenanted runs, never for a number you intend to report.)" >&2
            exit 1
        fi
        [ -n "${GEAK_GPU_USE_LOG:-}" ] && \
            echo "{\"t\":$(date +%s),\"gpu\":$GPU_ID,\"pool\":\"$GPU_SPEC\",\"pid\":$$,\"mode\":\"pin\",\"wait_s\":$(( SECONDS - _wait_t0 ))}" \
                >> "$GEAK_GPU_USE_LOG" 2>/dev/null
        export HIP_VISIBLE_DEVICES="$GPU_ID"
        "$@"
    ) 200>"$LOCK_FILE"
fi
