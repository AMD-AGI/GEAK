#!/bin/bash
# GPU lock + per-workspace build isolation wrapper.
# Usage:  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>
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
# THIS IS THE ONLY GPU LOCK IN THE REPO. The Gluon skill pack's scripts/gpu_lock.sh is a shim that
# execs this file, so a pack command and a GEAK role command on the same GPU id flock the same file
# in the same namespace (/tmp/team_gpu_locks) and serialize against each other. Absorbed from the
# pack's former copy: --help, --selftest, the read-only lock-file open (below) and the OPTIONAL fleet
# broker seam (GEAK_GPU_BROKER=1; off by default). Not absorbed: the pack's per-user lock namespace
# (~/.cache/tile-runtime/gpu-broker/locks, which does not interlock with GEAK), its silent fallback to
# a private namespace, its first-agent arch pin, and --no-lease (no consumer ships here; it ran a
# command unarbitrated).
#
# Exported to the wrapped command: HIP_VISIBLE_DEVICES=<id>, GEAK_GPU_LOCK_HELD=<id> (tools that must
# run under this lock -- the kernel_tools profiler wrappers -- check it instead of re-locking; a
# nested gpu_lock.sh on the SAME id from inside the command would deadlock on its own flock),
# TORCH_EXTENSIONS_DIR, and PYTORCH_ROCM_ARCH/GPU_ARCHS when the arch pin applies.

set -euo pipefail

_GL_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_GL_SELF="$_GL_HERE/$(basename "${BASH_SOURCE[0]}")"
_GL_REPO="$(cd "$_GL_HERE/../.." && pwd)"
_GL_SCHED_DEFAULT="$_GL_REPO/perf_knowledge/expert_skills/skills/gluon_authoring/scheduler"

# --help, because the alternative is reading 300 lines of shell. Without a case for it, `--help`
# fell through as a GPU SPEC and left a `gpu_--help.lock` file behind.
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'USAGE'
gpu_lock.sh -- run ONE GPU command holding an exclusive lane on this box.

  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>      pin: wait for that GPU's lock
  cd <workspace> && bash gpu_lock.sh <id,id,...> <command...>   pool: first free AND idle GPU
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

Exit codes: the command's own, 86 (source provenance invalid), 1 (lock/allocation/idleness/arch
refusal). Ids are the HIP ordinals of THIS process: inside an inherited ROCR_VISIBLE_DEVICES mask
they are logical, so every tenant of a box must use the same numbering for the locks to mean the
same card.

ENVIRONMENT
  GEAK_GPU_ALLOWED=<a,b>     refuse any id outside the caller's allocation
  GEAK_GPU_REQUIRE_IDLE=0    do not refuse a card that has foreign work on it (default 1)
  GEAK_GPU_MAX_BUSY_PCT / GEAK_GPU_MAX_VRAM_MB   idleness thresholds (default 5 / 1024)
  GEAK_GPU_POOL_WAIT=<s>     how long pool mode waits for a free lane (default 1200)
  GEAK_GPU_USE_LOG=<file>    append one JSON line per acquisition: {t,gpu,pool,pid,mode,wait_s}
                             mode = pool | pin | broker
  KERNEL_ENV_KEEP_ARCH=1     skip the arch pin (intentional multi-arch boxes)
  KERNEL_ENV_SKIP_ENUM_REAP=1  skip the enumerator reap
  GEAK_GPU_LOCK_DIR=<dir>    HERMETIC TESTS ONLY. Default /tmp/team_gpu_locks; two callers
                             serialize only if they resolve the same directory.
  GEAK_GPU_BROKER=1          OPT-IN fleet broker (gluon_authoring/scheduler, default off): when a
                             broker socket exists, queue for the lane through it; the granted lane
                             still takes this flock. GEAK_GPU_BROKER_SOCK / a .geak_gpu_sock marker
                             name the socket; GEAK_GPU_GROUP / .geak_gpu_group the fair-share key.

LONG COMMANDS: this wrapper prints nothing once the command starts and BLOCKS while queueing. A
harness that kills a silent child after N seconds should wrap it, e.g. (standalone pack runs)
  bash <gluon_authoring>/scripts/wait_for.sh --run --log run.log -- bash gpu_lock.sh <id> <cmd...>
USAGE
    exit 0
fi

# --selftest: offline (no GPU, no profiler). Exercises the lock itself in a private namespace so it
# never touches a real lane: two holders of one id must not overlap, a read-only lock file must still
# lock, the command must see HIP_VISIBLE_DEVICES/GEAK_GPU_LOCK_HELD, and the use-log must carry mode.
if [ "${1:-}" = "--selftest" ]; then
    _st_fail=0
    _st_dir="${TMPDIR:-/tmp}/geak_gpu_lock_selftest.$(id -u)"   # reused, never rm'd
    mkdir -p "$_st_dir/locks" "$_st_dir/ws"
    : > "$_st_dir/order.log"; : > "$_st_dir/use.log"
    _st_env=(env GEAK_GPU_LOCK_DIR="$_st_dir/locks" GEAK_GPU_REQUIRE_IDLE=0 KERNEL_ENV_KEEP_ARCH=1
             KERNEL_ENV_SKIP_ENUM_REAP=1 GEAK_GPU_USE_LOG="$_st_dir/use.log" GEAK_GPU_BROKER=0)
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
    # The scheduler's group resolver (only when the pack's scheduler is present).
    _st_resolver="${GEAK_GPU_BROKER_SCHED_DIR:-$_GL_SCHED_DEFAULT}/group_identity.py"
    if [ -f "$_st_resolver" ]; then
        mkdir -p "$_st_dir/kernel_a/dir_17"
        _st_group="$(cd "$_st_dir/kernel_a/dir_17" && python3 "$_st_resolver" --path "$PWD" --ignore-env)"
        [ "$_st_group" = "kernel_a" ] || { echo "FAIL: direction leaf resolved as '$_st_group'"; _st_fail=1; }
    fi
    [ "$_st_fail" = 0 ] && echo "GPU_LOCK SELFTEST PASS" || echo "GPU_LOCK SELFTEST FAIL"
    exit "$_st_fail"
fi

# The usage line deliberately shows NO concrete ids. It used to read "e.g. 0,1,2,3", and that
# example was copied verbatim into real commands by agents improvising one-off checks -- the
# literal string "0,1,2,3" turned up in 15 invocations from runs that had been allocated neither
# GPU 2 nor 3. An example in a usage line gets read as a default; here the value is never
# defaultable, so it names no ids.
GPU_SPEC="${1:?Usage: gpu_lock.sh <gpu_id|pool> <command...>   (pool = comma list of the GPUs THIS run was allocated)}"
shift

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
LOCK_DIR="${GEAK_GPU_LOCK_DIR:-/tmp/team_gpu_locks}"
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
    for _r in $(echo "$GPU_SPEC" | tr ',' ' '); do
        case ",${GEAK_GPU_ALLOWED}," in *",${_r},"*) ;; *) _bad="$_bad $_r" ;; esac
    done
    if [ -n "$_bad" ]; then
        echo "ERROR: gpu_lock.sh asked for GPU(s)${_bad} but this run is allocated only [${GEAK_GPU_ALLOWED}]." >&2
        echo "       Use the allocated set: gpu_lock.sh ${GEAK_GPU_ALLOWED} <command...>" >&2
        exit 1
    fi
fi

# ---- Optional fleet broker (OFF unless GEAK_GPU_BROKER=1) ------------------------------------------
# The Gluon pack ships a GPU broker (gluon_authoring/scheduler: priority queue, per-kernel fair share,
# resident leases). It is opt-in: with GEAK_GPU_BROKER unset or 0, or with no broker socket listening,
# this block is skipped and the flock allocator below is all that runs -- the GEAK default.
#
# When enabled, the broker only CHOOSES THE LANE. We re-enter this script with the granted single id,
# so the source check, the per-workspace build cache, the arch pin, the reap and the flock all still
# run; the flock on the granted lane is what keeps a brokered command mutually exclusive with every
# non-brokered GEAK process on the box (same lock file). GEAK_GPU_BROKER_ACTIVE bounds the recursion.
# The inner call skips the sysfs idleness test because the broker already judged the lane and, unlike
# that test, it can tell its own parked residents from a foreign tenant.
#
# Not exec'd: the client runs as a child so the source-provenance EXIT trap above still fires.
_GL_SCHED="${GEAK_GPU_BROKER_SCHED_DIR:-$_GL_SCHED_DEFAULT}"
_GL_CLIENT="$_GL_SCHED/gpu_client.py"
if [ "${GEAK_GPU_BROKER:-0}" = "1" ] && [ -z "${GEAK_GPU_BROKER_ACTIVE:-}" ] && [ -f "$_GL_CLIENT" ]; then
    # The socket: explicit env, else a `.geak_gpu_sock` marker at a parent of $PWD (the campaign
    # names its broker; the commands are composed by agents, so a marker file is the only channel
    # that reaches them), else the canonical path. gpu_client applies the same order and refuses an
    # unnamed canonical socket, degrading to an unbrokered run of this same script.
    _GL_SOCK="${GEAK_GPU_BROKER_SOCK:-}"
    if [ -z "$_GL_SOCK" ]; then
        _GL_SD="$PWD"
        while [ "$_GL_SD" != "/" ] && [ -n "$_GL_SD" ]; do
            if [ -f "$_GL_SD/.geak_gpu_sock" ]; then
                _GL_SOCK="$(head -1 "$_GL_SD/.geak_gpu_sock" 2>/dev/null | tr -d '[:space:]')"
                [ -n "$_GL_SOCK" ] && export GEAK_GPU_BROKER_SOCK="$_GL_SOCK"
                break
            fi
            _GL_SD="$(dirname "$_GL_SD")"
        done
    fi
    _GL_SOCK="${_GL_SOCK:-${GEAK_GPU_RUNTIME_ROOT:-$HOME/.cache/tile-runtime/gpu-broker}/gpu.sock}"
    if [ -S "$_GL_SOCK" ]; then
        export GEAK_GPU_BROKER_ACTIVE=1
        # Fair-share key and priority band ride marker files for the same reason as the socket.
        # Nothing in GEAK writes them; a campaign that wants per-kernel fair share writes
        # `.geak_gpu_group` (and optionally `.geak_gpu_priority`) at its root, or exports the vars.
        if [ -z "${GEAK_GPU_GROUP:-}" ] || [ -z "${GEAK_GPU_PRIORITY:-}" ]; then
            _GL_D="$PWD"
            while [ "$_GL_D" != "/" ] && [ -n "$_GL_D" ]; do
                if [ -z "${GEAK_GPU_GROUP:-}" ] && [ -f "$_GL_D/.geak_gpu_group" ]; then
                    GEAK_GPU_GROUP="$(head -1 "$_GL_D/.geak_gpu_group" 2>/dev/null | tr -d '[:space:]')"
                    [ -n "$GEAK_GPU_GROUP" ] && export GEAK_GPU_GROUP
                fi
                if [ -z "${GEAK_GPU_PRIORITY:-}" ] && [ -f "$_GL_D/.geak_gpu_priority" ]; then
                    GEAK_GPU_PRIORITY="$(head -1 "$_GL_D/.geak_gpu_priority" 2>/dev/null | tr -d '[:space:]')"
                    [ -n "$GEAK_GPU_PRIORITY" ] && export GEAK_GPU_PRIORITY
                fi
                [ -n "${GEAK_GPU_GROUP:-}" ] && [ -n "${GEAK_GPU_PRIORITY:-}" ] && break
                _GL_D="$(dirname "$_GL_D")"
            done
        fi
        if [ -z "${GEAK_GPU_GROUP:-}" ] && [ -f "$_GL_SCHED/group_identity.py" ]; then
            # Same resolver as the Python client, so a `dir_<n>` leaf is never its own tenant.
            _GL_GROUP="$(python3 "$_GL_SCHED/group_identity.py" --path "$PWD" 2>/dev/null || true)"
            [ -n "$_GL_GROUP" ] && export GEAK_GPU_GROUP="$_GL_GROUP"
        fi
        # Band / kind / runtime estimate derived from the command (an explicit GEAK_GPU_PRIORITY
        # wins): profiler replays are long, verifies gate a kernel's completion.
        _gl_kind() {
            case "$*" in
                *rocprof-compute*|*rocprofv3*|*--att*|*capture.sh*|*profile_kernel.sh*) printf 'profile' ;;
                *plain_autotune*|*sweep_driver*) printf 'sweep' ;;
                *task_runner*correctness*|*ab_bench*) printf 'verify' ;;
                *setup.py*|*build_ext*|*compile*) printf 'compile' ;;
                *) printf 'run' ;;
            esac
        }
        _GL_KIND="$(_gl_kind "$@")"
        case "$_GL_KIND" in
            profile) _GL_BAND=bulk; _GL_EST=600; _GL_YIELD=profile_pass_boundary ;;
            sweep)   _GL_BAND=sweep; _GL_EST=90; _GL_YIELD=sweep_reading ;;
            verify)  _GL_BAND=verify; _GL_EST=90; _GL_YIELD=none ;;
            compile) _GL_BAND=measure; _GL_EST=240; _GL_YIELD=compile_unit_boundary ;;
            *)       _GL_BAND=measure; _GL_EST=180; _GL_YIELD=none ;;
        esac
        _GL_REQUIRE_GROUP=()
        [ "${GEAK_GPU_REQUIRE_GROUP:-0}" = "1" ] && _GL_REQUIRE_GROUP=(--require-group)
        _gl_rc=0
        python3 "$_GL_CLIENT" run \
            --lanes "$(echo "$GPU_SPEC" | tr -d ' ')" \
            --group "${GEAK_GPU_GROUP:-${GEAK_KERNEL_NAME:-}}" \
            --priority "${GEAK_GPU_PRIORITY:-$_GL_BAND}" \
            --command-kind "$_GL_KIND" \
            --estimated-runtime-s "$_GL_EST" \
            --safe-yield-kind "$_GL_YIELD" \
            --timeout-s "${GEAK_GPU_POOL_WAIT:-1200}" \
            --label "${GEAK_GPU_LABEL:-gpu_lock:$$}" \
            "${_GL_REQUIRE_GROUP[@]}" \
            -- bash -c '
                if [ -n "${GEAK_GPU_LANE:-}" ]; then
                    export GEAK_GPU_REQUIRE_IDLE=0     # the broker already judged this lane
                    exec bash "$0" "$GEAK_GPU_LANE" "${@:2}"
                fi
                exec bash "$0" "$1" "${@:2}"           # unbrokered: the original spec, unchanged
            ' "$_GL_SELF" "$GPU_SPEC" "$@" || _gl_rc=$?
        exit "$_gl_rc"
    fi
fi
_GL_MODE_PIN="pin"
[ -n "${GEAK_GPU_BROKER_ACTIVE:-}" ] && [ -n "${GEAK_GPU_LANE:-}" ] && _GL_MODE_PIN="broker"

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
                if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ] && ! _gpu_is_idle "$_g"; then
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

if [ -n "${POOL_FD:-}" ]; then
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
        if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ] && ! _gpu_is_idle "$GPU_ID"; then
            echo "ERROR: GPU $GPU_ID has foreign work running (busy/VRAM above threshold); refusing to measure on it" >&2
            exit 1
        fi
        [ -n "${GEAK_GPU_USE_LOG:-}" ] && \
            echo "{\"t\":$(date +%s),\"gpu\":$GPU_ID,\"pool\":\"$GPU_SPEC\",\"pid\":$$,\"mode\":\"$_GL_MODE_PIN\",\"wait_s\":$(( SECONDS - _wait_t0 ))}" \
                >> "$GEAK_GPU_USE_LOG" 2>/dev/null
        export GEAK_GPU_LOCK_HELD="$GPU_ID"
        export HIP_VISIBLE_DEVICES="$GPU_ID"
        _pin_compile_arch || exit 1   # (3) after the selected GPU is visible
        "$@"
    )
fi
