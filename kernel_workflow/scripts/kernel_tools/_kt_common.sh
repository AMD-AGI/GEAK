#!/usr/bin/env bash
# _kt_common.sh -- shared helpers for the kernel_tools bash wrappers (sourced, never run).
#
# GEAK conventions every profiler wrapper here follows:
#   * GPU work runs under GEAK's gpu_lock.sh (kernel_workflow/scripts/gpu_lock.sh). A wrapper started
#     outside the lock re-execs itself under it when given a GPU id, and refuses otherwise. Inside
#     the lock it never locks again: a nested gpu_lock.sh on the same id would deadlock on its own
#     flock. "Inside" = GEAK_GPU_LOCK_HELD / a lease exported, or (held_lock.py) a parent process
#     holds the team flock for the one GPU in HIP_VISIBLE_DEVICES -- so `gpu_lock.sh <ids> bash
#     <tool>` works without GEAK_KERNEL_TOOLS=1 set first.
#   * No inline HIP_VISIBLE_DEVICES: the device is whatever gpu_lock.sh exported.
#   * No `rm`: stale artifact dirs are moved aside (kt_move_aside), as profile_kernel.sh does.
#   * The profiled command may be argv (`-- cmd args...`) or one shell string (`--cmd "<string>"`,
#     run as `bash -c "<string>"`), matching profile_kernel.sh's <benchmark_cmd>.
#   * Execution locus: shared locus.sh supports an optional external container.
#     Embedded GEAK runs already inside their workload container use host passthrough.

if [ -z "${BASH_VERSION:-}" ]; then
    # Run (or sourced) by a POSIX sh such as dash: BASH_SOURCE below would abort with "Bad
    # substitution", so say what this file needs instead.
    echo "_kt_common.sh: bash-only helpers for the kernel_tools wrappers; source it from bash, do not run it." >&2
    case "${1:-}" in -h|--help) return 0 2>/dev/null || exit 0 ;; *) return 2 2>/dev/null || exit 2 ;; esac
fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    # Run instead of sourced: say what this file is rather than exit silently.
    echo "_kt_common.sh: shared helpers for the kernel_tools bash wrappers; source it, do not run it." >&2
    case "${1:-}" in -h|--help) exit 0 ;; *) exit 2 ;; esac
fi

KT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KT_GPU_LOCK="$(cd "$KT_DIR/.." && pwd)/gpu_lock.sh"
KT_REPO="$(cd "$KT_DIR/../../.." && pwd)"

# kt_move_aside <path>: move an existing file/dir out of the way (never delete).
kt_move_aside() {
  local p="${1:?kt_move_aside <path>}"
  [ -e "$p" ] || [ -L "$p" ] || return 0
  mv "$p" "${p%/}.old_$(date +%s)_$$_${RANDOM}" 2>/dev/null || true
}

# kt_kill_tree <pid>: SIGKILL a process and every descendant (a hung profiler ignores SIGTERM).
kt_kill_tree() {
  local pid="${1:?}" child
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null); do kt_kill_tree "$child"; done
  kill -KILL "$pid" 2>/dev/null || true
}

# kt_under_lock: rc 0 when this process already runs inside gpu_lock.sh.
kt_under_lock() { [ -n "${GEAK_GPU_LOCK_HELD:-}" ] || [ "${GEAK_GPU_LEASE_ACTIVE:-0}" = "1" ]; }

# kt_adopt_outer_lock <self>: gpu_lock.sh exports GEAK_GPU_LOCK_HELD only when GEAK_KERNEL_TOOLS=1
# was set before it. Without it, a tool run as `gpu_lock.sh <ids> bash <tool>` is still inside the
# lock: adopt it when a parent process holds the team flock for the one GPU in HIP_VISIBLE_DEVICES
# (held_lock.py). Nothing is adopted outside a real held lock.
kt_adopt_outer_lock() {
  [ -z "${GEAK_GPU_LOCK_HELD:-}${GEAK_GPU_LOCKED_SET:-}" ] || return 0
  [ "${GEAK_GPU_LEASE_ACTIVE:-0}" != "1" ] || return 0
  local held
  held="$(python3 "$KT_DIR/held_lock.py" 2>/dev/null)" || return 0
  [ -n "$held" ] || return 0
  export GEAK_GPU_LOCK_HELD="$held"
  echo "[$(basename "${1:-kernel_tools}")] under gpu_lock.sh: GPU $held is held by a parent process; using it." >&2
}

# kt_ensure_lock <gpu_id|""> <self> <args...>: return if already under the lock; else re-exec
# `bash <self> <args...>` under gpu_lock.sh <gpu_id>; with no id, refuse (exit 2).
kt_ensure_lock() {
  local gpu="$1" self="$2"; shift 2
  local ownership_rc=0
  kt_adopt_outer_lock "$self"
  python3 "$KT_DIR/../gpu_ownership.py" --gpu-spec "$gpu" 2>/dev/null || ownership_rc=$?
  if [ "$ownership_rc" = 2 ] && [ -n "$gpu" ] && [ -n "${GEAK_GPU_LOCK_HELD:-}" ] \
     && [ -z "${GEAK_GPU_LOCKED_SET:-}" ] && [ "${GEAK_GPU_LEASE_ACTIVE:-0}" != 1 ] \
     && [ "${GEAK_GPU_LOCK_HELD}" = "${HIP_VISIBLE_DEVICES:-}" ] \
     && [[ "$gpu" =~ ^[0-9]+(,[0-9]+)*$ ]] \
     && python3 "$KT_DIR/../gpu_ownership.py" --gpu-spec "" 2>/dev/null; then
    # Already inside a valid single-GPU lock, but --dev/--gpu names another id (e.g. the pool's
    # other member: `gpu_lock.sh 4,5` took 5, the script says --dev 4). The work runs on the
    # held, locked GPU; the id passed is not used for anything else.
    echo "[$(basename "$self")] NOTE: already under gpu_lock.sh holding GPU ${GEAK_GPU_LOCK_HELD};" \
         "--dev/--gpu $gpu is ignored and the held GPU is used (inside a lock, omit --dev/--gpu)." >&2
    ownership_rc=0
  fi
  case "$ownership_rc" in
    0)
      python3 "$KT_DIR/../workspace_sources.py" check --workspace "$PWD" || exit 2
      return 0 ;;
    1) ;;  # No inherited ownership: acquire the outer lock below.
    *)  # Never bypass a conflicting allocation, even in test mode.
      python3 "$KT_DIR/../gpu_ownership.py" --gpu-spec "$gpu" || true   # print the reason
      if [ -n "${GEAK_GPU_LOCK_HELD:-}" ] && [ -z "${GEAK_GPU_LOCKED_SET:-}" ] \
         && [ "${GEAK_GPU_LEASE_ACTIVE:-0}" != 1 ] && [ "${GEAK_GPU_LOCK_HELD}" != "${HIP_VISIBLE_DEVICES:-}" ]; then
        echo "[$(basename "$self")] GEAK_GPU_LOCK_HELD must be the GPU id gpu_lock.sh exported (not a flag):" \
             "it is '${GEAK_GPU_LOCK_HELD}' while HIP_VISIBLE_DEVICES='${HIP_VISIBLE_DEVICES:-}'. Under an" \
             "external lock set GEAK_GPU_LOCK_HELD=<that GPU id>, or pass --gpu <id>." >&2
      fi
      exit 2 ;;
  esac
  if [ "${KT_ALLOW_UNLOCKED:-0}" = "1" ]; then
    echo "[$(basename "$self")] WARNING: KT_ALLOW_UNLOCKED=1 -- running WITHOUT gpu_lock.sh; any number" \
         "this produces is unserialized against other tenants (tests / profiler-less dry runs only)." >&2
    return 0
  fi
  if [ -n "$gpu" ]; then
    export GEAK_KERNEL_TOOLS=1   # the lock exports GEAK_GPU_LOCK_HELD for the re-exec'd tool
    exec bash "$KT_GPU_LOCK" "$gpu" bash "$self" "$@"
  fi
  echo "[$(basename "$self")] REFUSING: no GEAK GPU lock is held by this process or a parent" \
       "(HIP_VISIBLE_DEVICES='${HIP_VISIBLE_DEVICES:-}'; GEAK_KERNEL_TOOLS=1 alone is not a lock). Either" >&2
  echo "  cd <workspace> && bash $KT_GPU_LOCK <gpu_id> bash $self ..." >&2
  echo "  or pass the GPU id this run was allocated (--dev/--gpu <id>) and this script locks itself." >&2
  exit 2
}

# kt_att_decoder_dir: print the directory holding the ATT decoder library (rocprof-trace-decoder),
# rc 1 when absent. GEAK does not vendor it; ROCPROF_ATT_LIBRARY_PATH (colon list) names it, else the
# ROCm lib dirs are searched (KT_ATT_SYSTEM_DIRS overrides that list, for tests). Callers DEGRADE the
# ATT layer when this fails -- they never fail the profile.
kt_att_decoder_dir() {
  python3 "$KT_DIR/profiler_deps.py" --decoder-path
}

# Set KT_ATT_ARGS in the calling shell. The optional report path is host-visible.
# Legacy candidate libraries must load and export V1; newer SDK-managed decoders
# are allowed to attempt capture, whose decoded output remains the real check.
kt_att_prepare() {
  local where="${1:-local}" report="${2:-}" data script="$KT_DIR/profiler_deps.py"
  KT_ATT_ARGS=()
  KT_ATT_STATUS=dependency_probe_failed
  if [ "$where" = locus ]; then
    data="$(locus_run python3 "$(locus_path "$script")" --att-only)" || return 1
  else
    data="$(python3 "$script" --att-only)" || return 1
  fi
  [ -z "$report" ] || printf '%s\n' "$data" > "$report"
  KT_ATT_STATUS="$(printf '%s' "$data" | python3 -c 'import json,sys; print(json.load(sys.stdin)["att"]["status"])')" || return 1
  case "$KT_ATT_STATUS" in
    external_candidate)
      local dir
      dir="$(printf '%s' "$data" | python3 -c 'import json,sys; print(json.load(sys.stdin)["att"]["directory"])')" || return 1
      KT_ATT_ARGS=(--att-library-path "$dir") ;;
    sdk_managed_unverified) ;;
    *) echo "[ATT] $KT_ATT_STATUS; see kernel_tools/profiling-dependencies.md" >&2; return 1 ;;
  esac
}

# Shared helpers never depend on a language pack. KT_NO_PACK_LOCUS remains a
# compatibility switch for explicit host passthrough in existing callers/tests.
if [ "${KT_NO_PACK_LOCUS:-0}" != "1" ]; then
  source "$KT_DIR/locus.sh"
fi
declare -F locus_active >/dev/null || locus_active() { return 1; }
declare -F locus_run >/dev/null || locus_run() { "$@"; }
declare -F locus_timeout >/dev/null || locus_timeout() { local s="${1:?}"; shift; timeout "$s" "$@"; }
declare -F locus_path >/dev/null || locus_path() { printf '%s' "${1:?}"; }
declare -F locus_fetch >/dev/null || locus_fetch() { return 0; }
declare -F locus_push >/dev/null || locus_push() { return 0; }
declare -F locus_workdir_shared >/dev/null || locus_workdir_shared() { [ -d "${1:?}" ]; }
declare -F locus_have >/dev/null || locus_have() { command -v "${1:?}" >/dev/null 2>&1; }
declare -F locus_preflight >/dev/null || locus_preflight() { LOCUS_DEGRADE_CAUSE=""; [ -z "${TILE_KERNEL_CONTAINER:-}" ]; }
