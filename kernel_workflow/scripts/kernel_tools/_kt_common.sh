#!/usr/bin/env bash
# _kt_common.sh -- shared helpers for the kernel_tools bash wrappers (sourced, never run).
#
# GEAK conventions every profiler wrapper here follows:
#   * GPU work runs under GEAK's gpu_lock.sh (kernel_workflow/scripts/gpu_lock.sh). A wrapper started
#     outside the lock re-execs itself under it when given a GPU id, and refuses otherwise. Inside
#     the lock (GEAK_GPU_LOCK_HELD set) it never locks again: a nested gpu_lock.sh on the same id would
#     deadlock on its own flock.
#   * No inline HIP_VISIBLE_DEVICES: the device is whatever gpu_lock.sh exported.
#   * No `rm`: stale artifact dirs are moved aside (kt_move_aside), as profile_kernel.sh does.
#   * The profiled command may be argv (`-- cmd args...`) or one shell string (`--cmd "<string>"`,
#     run as `bash -c "<string>"`), matching profile_kernel.sh's <benchmark_cmd>.
#   * Execution locus: the Gluon pack's scripts/locus.sh (STANDALONE runs that profile inside a
#     container) is sourced when present; otherwise every locus_* helper is a host passthrough, which
#     is exactly what an EMBEDDED GEAK run (already inside its container) needs.

KT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KT_GPU_LOCK="$(cd "$KT_DIR/.." && pwd)/gpu_lock.sh"
KT_REPO="$(cd "$KT_DIR/../../.." && pwd)"
KT_PACK_DIR="${GEAK_GLUON_PACK_DIR:-$KT_REPO/perf_knowledge/expert_skills/skills/gluon_authoring}"

# kt_move_aside <path>: move an existing file/dir out of the way (never delete).
kt_move_aside() {
  local p="${1:?kt_move_aside <path>}"
  [ -e "$p" ] || [ -L "$p" ] || return 0
  mv "$p" "${p%/}.old_$(date +%s)_$$_${RANDOM}" 2>/dev/null || true
}

# kt_under_lock: rc 0 when this process already runs inside gpu_lock.sh.
kt_under_lock() { [ -n "${GEAK_GPU_LOCK_HELD:-}" ]; }

# kt_ensure_lock <gpu_id|""> <self> <args...>: return if already under the lock; else re-exec
# `bash <self> <args...>` under gpu_lock.sh <gpu_id>; with no id, refuse (exit 2).
kt_ensure_lock() {
  local gpu="$1" self="$2"; shift 2
  kt_under_lock && return 0
  if [ "${KT_ALLOW_UNLOCKED:-0}" = "1" ]; then
    echo "[$(basename "$self")] WARNING: KT_ALLOW_UNLOCKED=1 -- running WITHOUT gpu_lock.sh; any number" \
         "this produces is unserialized against other tenants (tests / profiler-less dry runs only)." >&2
    return 0
  fi
  if [ -n "$gpu" ]; then
    exec bash "$KT_GPU_LOCK" "$gpu" bash "$self" "$@"
  fi
  echo "[$(basename "$self")] REFUSING: GPU work must run under GEAK's gpu_lock.sh. Either" >&2
  echo "  cd <workspace> && bash $KT_GPU_LOCK <gpu_id> bash $self ..." >&2
  echo "  or pass the GPU id this run was allocated (--dev/--gpu <id>) and this script locks itself." >&2
  exit 2
}

# kt_att_decoder_dir: print the directory holding the ATT decoder library (rocprof-trace-decoder),
# rc 1 when absent. GEAK does not vendor it; ROCPROF_ATT_LIBRARY_PATH (colon list) names it, else the
# ROCm lib dirs are searched (KT_ATT_SYSTEM_DIRS overrides that list, for tests). Callers DEGRADE the
# ATT layer when this fails -- they never fail the profile.
kt_att_decoder_dir() {
  local d f IFS=': '
  for d in ${ROCPROF_ATT_LIBRARY_PATH:-} ${KT_ATT_SYSTEM_DIRS:-${ROCM_PATH:-/opt/rocm}/lib:/opt/rocm/lib:/opt/rocm/lib64}; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    for f in "$d"/librocprof-trace-decoder*.so* "$d"/libatt_decoder_trace*.so*; do
      [ -e "$f" ] && { printf '%s' "$d"; return 0; }
    done
  done
  return 1
}

# Locus helpers: the pack's locus.sh when present, else host passthroughs. A pack locus.sh that
# predates one of these helpers gets the host passthrough for that helper only, so an older pack
# degrades to host behaviour for what it does not define instead of failing with "command not found".
if [ -f "$KT_PACK_DIR/scripts/locus.sh" ] && [ "${KT_NO_PACK_LOCUS:-0}" != "1" ]; then
  # shellcheck source=/dev/null
  source "$KT_PACK_DIR/scripts/locus.sh"
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
