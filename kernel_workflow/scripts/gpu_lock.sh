#!/bin/bash
# GPU lock + per-workspace build isolation wrapper.
# Usage:  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>
#         bash gpu_lock.sh group:<ids> <command...>                  hold exactly these GPUs
#         bash gpu_lock.sh pool:<n>:<ids> <command...>       hold any n GPUs in this pool
#         bash gpu_lock.sh --help | --selftest
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
#
# Shared command entry point. DSL shims use this wrapper; multi-GPU requests delegate to
# gpu_lease.py. Both acquire the same /tmp/team_gpu_locks/gpu_<id>.lock files.
#
# A single-card command receives HIP_VISIBLE_DEVICES=<id> and GEAK_GPU_LOCK_HELD=<id>.
# Multi-GPU commands receive GEAK_GPU_LEASE_ACTIVE/IDS from gpu_lease.py; the e2e set wrapper
# publishes GEAK_GPU_LOCKED_SET after acquiring all its cards. Consumers reuse matching ownership
# instead of requesting the same lock again. These inherited markers are a caller contract,
# not a replacement for acquiring a lock. Build-cache and compile-arch variables remain shared.

set -euo pipefail

_GL_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_GL_SELF="$_GL_HERE/$(basename "${BASH_SOURCE[0]}")"

# --help, because the alternative is reading 300 lines of shell. Without a case for it, `--help`
# fell through as a GPU SPEC and left a `gpu_--help.lock` file behind.
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'USAGE'
gpu_lock.sh -- run a command holding its requested GPU or GPU group.

  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>      pin: wait for that GPU's lock
  cd <workspace> && bash gpu_lock.sh <id,id,...> <command...>   pool: first free AND idle GPU
  bash gpu_lock.sh group:<ids> <command...>                  hold exactly these GPUs
  bash gpu_lock.sh pool:<n>:<ids> <command...>               hold any n GPUs in this pool
  bash gpu_lock.sh --help | --selftest

What it does for the command (all generic, nothing kernel-specific):
  1. lock      flock /tmp/team_gpu_locks/gpu_<id>.lock for the whole command; exports
               HIP_VISIBLE_DEVICES=<id> and GEAK_GPU_LOCK_HELD=<id>
  2. idleness  refuses (pin) or skips (pool) a GPU with foreign work on it (sysfs busy%/VRAM)
  3. build     TORCH_EXTENSIONS_DIR=<workspace>/.torch_ext (per-workspace torch build cache)
  4. arch pin  PYTORCH_ROCM_ARCH / GPU_ARCHS = the locked GPU's gfx; refuses a mixed-ISA pool
  5. reap      orphaned rocm_agent_enumerator processes (a pile-up once hung a whole box)
  6. sources   workspace source-provenance check before AND after the command (exit 86 = discard
               every number this invocation printed)

Ownership: single-card commands receive GEAK_GPU_LOCK_HELD=<id>; gpu_lease.py publishes
GEAK_GPU_LEASE_ACTIVE=1 and GEAK_GPU_LEASE_IDS=<ids> for a group. The e2e set wrapper publishes
GEAK_GPU_LOCKED_SET=<ids> once all cards are held. A consumer under matching ownership runs
directly; requesting a held GPU again is refused. Nested single-card acquisition is allowed
only in ascending ID order, as used by the e2e set wrapper.

Exit codes: the command's own, 86 (source provenance invalid), 1 (lock/allocation/idleness/arch
refusal), 2 (unsafe nested acquisition). Ids are the HIP ordinals of THIS process: inside an inherited ROCR_VISIBLE_DEVICES mask
they are logical, so every tenant of a box must use the same numbering for the locks to mean the
same card.

ENVIRONMENT
  GEAK_GPU_ALLOWED=<a,b>     refuse any id outside the caller's allocation
  GEAK_GPU_REQUIRE_IDLE=0    do not refuse a card that has foreign work on it (default 1)
  GEAK_GPU_MAX_BUSY_PCT / GEAK_GPU_MAX_VRAM_MB   idleness thresholds (default 5 / 1024)
  GEAK_GPU_POOL_WAIT=<s>     how long pool mode waits for a free lane (default 1200)
  GEAK_GPU_USE_LOG=<file>    append one JSON line per acquisition: {t,gpu,pool,pid,mode,wait_s}
                             mode = pool | pin (multi-GPU leases use their existing group log)
  KERNEL_ENV_KEEP_ARCH=1     skip the arch pin (intentional multi-arch boxes)
  KERNEL_ENV_SKIP_ENUM_REAP=1  skip the enumerator reap
  GEAK_GPU_LOCK_DIR=<dir>    HERMETIC TESTS ONLY. Default /tmp/team_gpu_locks; two callers
                             serialize only if they resolve the same directory.

This wrapper may wait silently for a lock. Allow its acquisition timeout when supervising it.
USAGE
    exit 0
fi

# --selftest: offline (no GPU, no profiler). Exercises the lock itself in a private namespace so it
# never touches a real lane: two holders of one id must not overlap, a read-only lock file must still
# lock, the command must see HIP_VISIBLE_DEVICES/GEAK_GPU_LOCK_HELD, and the use-log must carry mode.
if [ "${1:-}" = "--selftest" ]; then
    _st_fail=0
    _st_dir="$(mktemp -d "${TMPDIR:-/tmp}/geak_gpu_lock_selftest.XXXXXX")"
    trap 'rm -rf -- "$_st_dir"' EXIT
    mkdir -p "$_st_dir/locks" "$_st_dir/ws"
    : > "$_st_dir/order.log"; : > "$_st_dir/use.log"
    _st_env=(env -u GEAK_GPU_ALLOWED -u GEAK_GPU_LOCK_HELD -u GEAK_GPU_LOCKED_SET
             -u GEAK_GPU_LEASE_ACTIVE -u GEAK_GPU_LEASE_IDS
             GEAK_GPU_LOCK_DIR="$_st_dir/locks" GEAK_GPU_REQUIRE_IDLE=0 KERNEL_ENV_KEEP_ARCH=1
             KERNEL_ENV_SKIP_ENUM_REAP=1 GEAK_GPU_USE_LOG="$_st_dir/use.log")
    _st_cmd='echo "start $1 $(date +%s.%N)" >> "$2"; sleep 1; echo "end $1 $(date +%s.%N)" >> "$2"'
    (cd "$_st_dir/ws" && "${_st_env[@]}" bash "$_GL_SELF" 7 bash -c "$_st_cmd" _ A "$_st_dir/order.log") &
    _st_p1=$!
    sleep 0.3
    (cd "$_st_dir/ws" && "${_st_env[@]}" bash "$_GL_SELF" 7 bash -c "$_st_cmd" _ B "$_st_dir/order.log") &
    _st_p2=$!
    wait "$_st_p1" || { echo "FAIL: first holder exited non-zero"; _st_fail=1; }
    wait "$_st_p2" || { echo "FAIL: second holder exited non-zero"; _st_fail=1; }
    _st_seq="$(awk '{printf "%s%s ", $1, $2}' "$_st_dir/order.log")"
    case "$_st_seq" in
        "startA endA startB endB "|"startB endB startA endA ") ;;
        *) echo "FAIL: two holders of gpu 7 overlapped: $_st_seq"; _st_fail=1 ;;
    esac
    grep -q '"mode":"pin"' "$_st_dir/use.log" || { echo "FAIL: use-log line has no mode field"; _st_fail=1; }
    _st_out="$(cd "$_st_dir/ws" && "${_st_env[@]}" bash "$_GL_SELF" 7 bash -c \
        'echo "HIP=$HIP_VISIBLE_DEVICES HELD=$GEAK_GPU_LOCK_HELD"')"
    [ "$_st_out" = "HIP=7 HELD=7" ] || { echo "FAIL: command env wrong: $_st_out"; _st_fail=1; }
    # A lock file another uid created (mode 664) is readable but not writable by us; flock(2) works
    # on a read-only descriptor, so it must still lock rather than fail the redirection.
    : > "$_st_dir/locks/gpu_8.lock"; chmod a-w "$_st_dir/locks/gpu_8.lock"
    if [ "$(id -u)" != 0 ]; then
        _st_out="$(cd "$_st_dir/ws" && "${_st_env[@]}" bash "$_GL_SELF" 8 echo ro-ok 2>&1)" \
            || { echo "FAIL: read-only lock file was not usable: $_st_out"; _st_fail=1; }
    fi
    chmod u+w "$_st_dir/locks/gpu_8.lock"
    [ "$_st_fail" = 0 ] && echo "GPU_LOCK SELFTEST PASS" || echo "GPU_LOCK SELFTEST FAIL"
    exit "$_st_fail"
fi

# The usage line deliberately shows NO concrete ids. It used to read "e.g. 0,1,2,3", and that
# example was copied verbatim into real commands by agents improvising one-off checks -- the
# literal string "0,1,2,3" turned up in 15 invocations from runs that had been allocated neither
# GPU 2 nor 3. An example in a usage line gets read as a default; here the value is never
# defaultable, so it names no ids.
GPU_SPEC="${1:?Usage: gpu_lock.sh <gpu_id|pool|group:<ids>|pool:<n>:<ids>> <command...>   (ids = comma list of the GPUs THIS run was allocated)}"
shift

# Multi-GPU forms (see "Multi-GPU leases" below): group:<ids> holds exactly those GPUs,
# pool:<n>:<ids> holds any n of them. _SPEC_IDS is the plain id list either way.
case "$GPU_SPEC" in
    group:*) _SPEC_IDS="${GPU_SPEC#group:}"; [ "${1:-}" = "--" ] && shift ;;
    pool:*:*) _SPEC_IDS="${GPU_SPEC#pool:*:}"; [ "${1:-}" = "--" ] && shift ;;
    *) _SPEC_IDS="$GPU_SPEC" ;;
esac

# A command already running under a multi-GPU lease must not request another: its GPUs are
# held by the outer lease, so a nested request would wait on itself or escape the lease.
if [ "${GEAK_GPU_LEASE_ACTIVE:-0}" = "1" ]; then
    echo "ERROR: nested gpu_lock.sh call inside a GPU lease (GPUs ${GEAK_GPU_LEASE_IDS:-?}); run the command directly" >&2
    exit 2
fi

# gpu_lock_set.py acquires single cards in ascending order, then publishes LOCKED_SET.
# Keep that existing path working, while rejecting a full-set re-entry or an acquisition that
# could deadlock with an outer single-card lock. LOCK_HELD is the last (largest) acquired ID.
if [ -n "${GEAK_GPU_LEASE_IDS:-}" ] || [ -n "${GEAK_GPU_LOCKED_SET:-}" ]; then
    echo "ERROR: nested gpu_lock.sh call inside a GPU set (GPUs ${GEAK_GPU_LEASE_IDS:-${GEAK_GPU_LOCKED_SET:-?}}); run the command directly" >&2
    exit 2
fi
if [ -n "${GEAK_GPU_LOCK_HELD:-}" ]; then
    if ! [[ "$GPU_SPEC" =~ ^(0|[1-9][0-9]*)$ && "$GEAK_GPU_LOCK_HELD" =~ ^(0|[1-9][0-9]*)$ ]] \
       || [ "$GPU_SPEC" -le "$GEAK_GPU_LOCK_HELD" ]; then
        echo "ERROR: nested gpu_lock.sh call while GPU $GEAK_GPU_LOCK_HELD is held; only a higher single GPU id may be acquired; run commands on held GPUs directly" >&2
        exit 2
    fi
fi

# Source provenance is checked before executing any command and again on exit:
# generated builders can recreate a stale overlay during the command itself.
# Exit 86 invalidates all output from that invocation, even if it printed PASS.
SOURCE_GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/workspace_sources.py"
SOURCE_WORKSPACE="$PWD"
python3 "$SOURCE_GUARD" check --workspace "$SOURCE_WORKSPACE"
_source_audit_size=$(stat -c %s "$SOURCE_WORKSPACE/.geak/invalid_measurements.jsonl" 2>/dev/null || echo 0)
_check_sources_on_exit() {
    local command_status=$?
    trap - EXIT
    python3 "$SOURCE_GUARD" check --workspace "$SOURCE_WORKSPACE" || exit 86
    # A builder may catch a check-input failure and continue, or repair the link
    # later in the same command. Neither makes that invocation's output valid.
    local audit_size
    audit_size=$(stat -c %s "$SOURCE_WORKSPACE/.geak/invalid_measurements.jsonl" 2>/dev/null || echo 0)
    if [ "$audit_size" != "$_source_audit_size" ]; then
        echo "GEAK_SOURCE_INVALID: source validation failed during this command; discard all measurement output." >&2
        exit 86
    fi
    exit "$command_status"
}
trap _check_sources_on_exit EXIT

# ONE team-wide namespace: two runs serialize only if they flock the SAME file. GEAK_GPU_LOCK_DIR
# exists for hermetic tests and must stay unset in real runs.
LOCK_DIR="${GEAK_GPU_LOCK_DIR:-${KERNEL_EXTRACT_LOCK_DIR:-/tmp/team_gpu_locks}}"
mkdir -p "$LOCK_DIR"

# How to open a lock file. The namespace belongs to whoever created it first; on a shared box that
# is often another uid, leaving `gpu_<id>.lock` mode 664 owned by them. A WRITE open of that file
# fails, and under `set -e` the redirection failure exits before the command runs -- a silent
# non-zero with no output that reads like a workload failure. flock(2) does not need write access,
# so open such a file READ-ONLY: it still serializes against every other tenant. Only a lock file
# that does not exist in a directory we cannot write is a real refusal, and it is reported as one
# (falling back to a private namespace would run the command unserialized against that tenant).
_lock_open_mode() {   # <lock file> -> prints w | r | (nothing = unusable)
    local f="$1"
    if [ -e "$f" ]; then
        if [ -w "$f" ]; then echo w; elif [ -r "$f" ]; then echo r; fi
        return 0
    fi
    [ -w "$(dirname "$f")" ] && echo w
    return 0
}
_lock_refuse() {
    echo "ERROR: gpu_lock.sh cannot open '$1' (uid $(id -u)): the lock namespace '$LOCK_DIR' is owned by" >&2
    echo "       another user and this lock file does not exist yet. Ask the owner to 'chmod 1777 $LOCK_DIR'" >&2
    echo "       (or pre-create the file world-readable). Refusing to run unserialized." >&2
}

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
    for _r in $(echo "$_SPEC_IDS" | tr ',' ' '); do
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
# NOTE: the sysfs path assumes id N <-> renderD(128+8N), i.e. physical numbering with no ROCR mask.
_gpu_is_idle() {
    local id="$1" dev busy vram
    dev="$(readlink -f "/sys/class/drm/renderD$((128 + 8 * id))/device" 2>/dev/null)" || return 0
    [ -r "$dev/gpu_busy_percent" ] || return 0   # cannot tell -> do not block the run
    busy="$(cat "$dev/gpu_busy_percent" 2>/dev/null || echo 0)"
    vram="$(( $(cat "$dev/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 ))"
    [ "${busy:-0}" -le "${GEAK_GPU_MAX_BUSY_PCT:-5}" ] && [ "$vram" -le "${GEAK_GPU_MAX_VRAM_MB:-1024}" ]
}

# gpu_busy_percent is a decaying average: our own previous job's tail clears in a few seconds,
# a foreign tenant does not.
_gpu_settle_idle() {
    local id="$1" timeout="$2" t0=$SECONDS
    while :; do
        _gpu_is_idle "$id" && return 0
        [ "$(( SECONDS - t0 ))" -ge "$timeout" ] && return 1
        sleep 0.25
    done
}

case "$GPU_SPEC" in
  group:*|pool:*:*)
    # Multi-GPU lease: gpu_lease.py acquires the whole group or nothing, after compile setup below.
    GROUP_MODE=1
    GPU_ID="$_SPEC_IDS"
    ;;
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
        for _g in $POOL; do
            # (1) flock: exclusive lane. Held until this process exits.
            _lf="${LOCK_DIR}/gpu_${_g}.lock"
            case "$(_lock_open_mode "$_lf")" in
                w) exec {_fd}>>"$_lf" ;;
                r) exec {_fd}<"$_lf" ;;
                *) _lock_refuse "$_lf"; exit 1 ;;
            esac
            if flock -n -x "$_fd"; then
                # We hold the lane. Only now check idleness -- checking before locking would race.
                if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ] \
                   && ! _gpu_settle_idle "$_g" "${GEAK_GPU_SETTLE_S:-5}"; then
                    flock -u "$_fd"; exec {_fd}>&-   # foreign job on this GPU: try the next lane
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

# (3) Compile for the selected GPU's arch only. HIP_VISIBLE_DEVICES does NOT
# filter rocminfo. When the launcher did not already provide a ROCR allocation,
# scope only this rocminfo subprocess to the locked physical GPU. Preserve an
# inherited ROCR mask because GPU_ID is then logical within that allocation.
# Refuse a genuinely mixed-ISA allocation: compiling for one ISA while locked
# to another is a silent wrong-arch result.
# Set KERNEL_ENV_KEEP_ARCH=1 to opt out (intentional multi-arch boxes).
_rocminfo_gpu_gfx_list() {
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || [ -z "${GPU_ID:-}" ]; then
        rocminfo 2>/dev/null
    else
        ROCR_VISIBLE_DEVICES="$GPU_ID" rocminfo 2>/dev/null
    fi | awk '
          /^ *Name: *gfx[0-9a-f]+/ && $2 != "gfx000" { print $2 }
        '
}

_pin_compile_arch() {
    [ "${KERNEL_ENV_KEEP_ARCH:-0}" = "1" ] && return 0
    local gfxs unique n
    gfxs="$(_rocminfo_gpu_gfx_list || true)"
    [ -z "${gfxs:-}" ] && return 0
    unique="$(printf '%s\n' $gfxs | sort -u)"
    n="$(printf '%s\n' $unique | grep -c . || true)"
    if [ "${n:-0}" -gt 1 ]; then
        echo "ERROR: gpu_lock.sh: mixed GPU ISAs in this pool ($unique); refusing to guess PYTORCH_ROCM_ARCH." >&2
        echo "       Run on a homogeneous pool or set KERNEL_ENV_KEEP_ARCH=1 with an explicit PYTORCH_ROCM_ARCH." >&2
        return 1
    fi
    _ARCH="$(printf '%s\n' $unique | head -1)"
    [ -n "${_ARCH:-}" ] && export PYTORCH_ROCM_ARCH="$_ARCH"
    # aiter's native detector spawns rocm_agent_enumerator per process; pinning
    # GPU_ARCHS avoids an enumerator storm as well as redundant multi-ISA builds.
    [ -n "${_ARCH:-}" ] && export GPU_ARCHS="${GPU_ARCHS:-$_ARCH}"
}

if [ -n "${GROUP_MODE:-}" ]; then
    # ---- Multi-GPU leases --------------------------------------------------------------------
    # For jobs that need several GPUs at once (multi-rank ops). gpu_lease.py takes the same
    # per-GPU flock files as the single-GPU paths, all-or-nothing under an allocator lock, so it
    # never holds part of a group while waiting for the rest; older group requests are served
    # first. The command sees HIP_VISIBLE_DEVICES / GEAK_GPU_GROUP = the leased ids. It is
    # stopped after GEAK_GPU_RUN_TIMEOUT seconds (default 900, <0 = no limit), and its process
    # group plus any reparented lease workers under GEAK_GPU_REAP_ROOT (default: this
    # workspace) are killed, so a hung multi-rank job cannot keep the GPUs.
    _pin_compile_arch || exit 1
    _lease_args=(--wait-timeout "${GEAK_GPU_POOL_WAIT:-1200}"
                 --run-timeout "${GEAK_GPU_RUN_TIMEOUT:-900}"
                 --lock-dir "$LOCK_DIR"
                 --reap-root "${GEAK_GPU_REAP_ROOT:-$PWD}")
    case "$GPU_SPEC" in
        group:*) _lease_args+=(--fixed-ids "$_SPEC_IDS") ;;
        *) _n="${GPU_SPEC#pool:}"; _lease_args+=(--pool "$_SPEC_IDS" --count "${_n%%:*}") ;;
    esac
    if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ]; then
        _lease_args+=(--require-idle --max-busy-pct "${GEAK_GPU_MAX_BUSY_PCT:-5}"
                      --max-vram-mb "${GEAK_GPU_MAX_VRAM_MB:-1024}")
    fi
    python3 "$(dirname "$SOURCE_GUARD")/gpu_lease.py" run "${_lease_args[@]}" -- "$@"
elif [ -n "${POOL_FD:-}" ]; then
    # Pool mode: (1) this process ALREADY holds the lane exclusively (and verified it idle).
    # Re-locking the same file from the same process would be a no-op at best, so just run --
    # the lane stays held until we exit, which is what guarantees no two evaluations share a GPU.
    export GEAK_GPU_LOCK_HELD="$GPU_ID"
    export HIP_VISIBLE_DEVICES="$GPU_ID"
    _pin_compile_arch || exit 1   # (3) after the selected GPU is visible
    "$@"
else
    # Single-GPU mode BLOCKS TOO, and its wait must be measured for the same reason the pool's is.
    # Pinning does not mean "no contention": with 4 engineers over 2 pinned GPUs, two engineers share
    # each card and serialize on exactly this flock. That queueing is the pinned policy's own cost,
    # and comparing a measured pool wait against an assumed-zero pin wait would build the scheduler's
    # advantage into the instrument. Both paths measure, so the comparison is real.
    _wait_t0=$SECONDS
    _lock_mode="$(_lock_open_mode "$LOCK_FILE")"
    [ -n "$_lock_mode" ] || { _lock_refuse "$LOCK_FILE"; exit 1; }
    (
        # (1) flock this GPU (exclusive for the duration of the command). fd 200 is opened
        # read-only when the file belongs to another uid (see _lock_open_mode).
        if [ "$_lock_mode" = "w" ]; then exec 200>>"$LOCK_FILE"; else exec 200<"$LOCK_FILE"; fi
        flock -x -w 1200 200 || { echo "ERROR: Failed to acquire GPU $GPU_ID lock after 1200s"; exit 1; }
        # Default 1, matching pool mode above. It was 0 here, so a PINNED engineer skipped the
        # foreign-work check entirely -- not "sampled it once", never ran it. That is how two
        # measurement cells ended up timed on a GPU another tenant was executing on. Both paths ask
        # the same question of the same driver; there is no reason for them to answer differently,
        # and the unsafe default was the one nobody had to opt into. The original rationale for 0
        # was that a single GPU has no alternative to step to, which is true -- but it argues for a
        # loud failure, not for measuring on a contaminated card. Set GEAK_GPU_REQUIRE_IDLE=0 to
        # restore the old behavior for deliberately co-tenanted runs.
        if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ] \
           && ! _gpu_settle_idle "$GPU_ID" "${GEAK_GPU_SETTLE_S:-30}"; then
            echo "ERROR: GPU $GPU_ID still shows foreign work after ${GEAK_GPU_SETTLE_S:-30}s" \
                 "(busy>${GEAK_GPU_MAX_BUSY_PCT:-5}% or VRAM>${GEAK_GPU_MAX_VRAM_MB:-1024}MB); refusing to measure on it." >&2
            echo "       This is a real co-tenant, not our own decaying tail -- that clears in ~3s." >&2
            echo "       Raise GEAK_GPU_SETTLE_S, or set GEAK_GPU_REQUIRE_IDLE=0 to measure anyway." >&2
            exit 1
        fi
        [ -n "${GEAK_GPU_USE_LOG:-}" ] && \
            echo "{\"t\":$(date +%s),\"gpu\":$GPU_ID,\"pool\":\"$GPU_SPEC\",\"pid\":$$,\"mode\":\"pin\",\"wait_s\":$(( SECONDS - _wait_t0 ))}" \
                >> "$GEAK_GPU_USE_LOG" 2>/dev/null
        export GEAK_GPU_LOCK_HELD="$GPU_ID"
        export HIP_VISIBLE_DEVICES="$GPU_ID"
        _pin_compile_arch || exit 1   # (3) after the selected GPU is visible
        "$@"
    )
fi
