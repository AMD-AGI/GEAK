#!/usr/bin/env bash
# Server teardown driven by LAUNCH-TIME process identity.  Sourced, never executed.
#
# WHY THIS EXISTS: resolving `ps -o pgid= -p $PID` during cleanup answers a question
# about whoever owns that pid *now*. Once the server exits and its pid is recycled the
# answer is an unrelated process's group, and a group TERM aimed at it can reach the
# caller's orchestrator or container PID 1 (a benchmark teardown was observed TERMing
# the parent coordinator running as PID 1, which failed the whole task ~72min later).
# So WHAT we launched is frozen at launch and never re-derived at kill time. The same
# rule is already enforced elsewhere in the org: a kill path must not consult
# getpgid(pid), because pid reuse makes the answer a stranger.
#
# Contract for callers (scripts/bench_e2e.sh and any agent-authored capture script):
#     source "<scripts>/server_teardown.sh"
#     trap server_teardown EXIT
#     <launch the server, setting SERVER_PID>
#     server_record_identity "$SERVER_PID"
#
# Teardown picks ONE of two modes and says which, plus why, on stdout:
#   GROUP kill  -- only when the launch PROVED pgid == pid, i.e. the group can contain
#                  nothing but our server's descendants.
#   PID kill    -- the server pid plus its verified transitive descendants. Keeps the
#                  anti-orphan intent (setsid'd vLLM/sglang workers live outside
#                  SERVER_PID and leak VRAM) without ever signalling a group.
#
# Optional env:
#   GEAK_PROTECTED_PGIDS  space-separated pgids that must NEVER be group-signalled
#                         (a caller exports its own so our teardown cannot reach it).
#   SERVER_GROUP_UNVERIFIED=1  set by a launcher that cannot vouch for the group
#                         (e.g. the pid came from an external script's pid file).
#   SERVER_STOP_GRACE_S   seconds between TERM and KILL (default 10).

# `setsid` skips its internal fork only when the caller is NOT a process-group leader.
# Non-interactive bash has job control off, so a backgrounded `setsid cmd` keeps
# $! == the server pid and the server stays our child (wait / kill -0 keep working).
# With job control ON, $! would be a wrapper that exits immediately and the real
# server would be unreachable, so pin job control off for every sourcing shell.
set +m

# The launch prefix EVERY adapter must launch through (see the adapter contract in
# bench_e2e.sh): `${SERVER_LAUNCH_PREFIX:-} env ... server ... &`.
# `setsid` makes the server lead its own session, so pgid == pid -- the one condition
# under which teardown can PROVE a process group is ours and reap the whole tree
# (framework worker procs escape SERVER_PID and otherwise leak VRAM). Defined ONCE
# here rather than per adapter so a new backend inherits it by following the contract
# instead of by remembering to copy a line. Empty when the binary is absent, and a
# scalar (not an array) so `${SERVER_LAUNCH_PREFIX:-}` degrades to NO argument under
# `set -u` even in a shell that never sourced this file.
SERVER_LAUNCH_PREFIX=""
command -v setsid >/dev/null 2>&1 && SERVER_LAUNCH_PREFIX="setsid"

SERVER_PID="${SERVER_PID:-}"
SERVER_PGID=""                                    # set ONLY when the group is proven ours
SERVER_START_TICKS=""                             # /proc starttime -> detects pid reuse
SERVER_GROUP_UNVERIFIED="${SERVER_GROUP_UNVERIFIED:-0}"
BENCH_PID=$$
BENCH_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
# 1 is always protected: container init / a PID-1 orchestrator. Our own group is
# protected because signalling it would take down the benchmark shell itself.
SERVER_PROTECTED_PGIDS="1 ${BENCH_PGID} ${GEAK_PROTECTED_PGIDS:-}"

_pgid_of() {  # _pgid_of PID -> pgid, or "" when the pid is gone
  ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '
}

# Root of the proc filesystem. Overridable so the start-time parser can be exercised
# against fixture files — `comm` may contain spaces AND parens, and getting that wrong
# silently disables the pid-reuse gate (an empty start time reads as "cannot verify").
SERVER_PROC_ROOT="${SERVER_PROC_ROOT:-/proc}"

_start_ticks_of() {  # _start_ticks_of PID -> /proc/PID/stat starttime (field 22)
  local _stat
  _stat="$(cat "$SERVER_PROC_ROOT/$1/stat" 2>/dev/null)" || return 1
  # comm (field 2) can contain spaces AND parens -> cut through the LAST ") ".
  _stat="${_stat##*\) }"
  # The remainder begins at `state` (field 3), so starttime (field 22) is $20.
  printf '%s\n' "$_stat" | awk '{print $20}'
}

# server_record_identity PID -- freeze the launched server's identity. Call ONCE,
# immediately after the launch, before anything can exit and recycle the pid.
server_record_identity() {
  SERVER_PID="$1"
  SERVER_PGID=""
  SERVER_START_TICKS="$(_start_ticks_of "$SERVER_PID" || true)"
  local _pgid
  _pgid="$(_pgid_of "$SERVER_PID")"
  # pgid == pid is the ONLY structural proof that the group belongs to this launch:
  # a group led by our own server can contain nothing but its own descendants, so it
  # cannot contain PID 1 or the caller's tree. Anything else (a group inherited from
  # us, or a pid read out of an external pid file) is treated as unproven.
  if [ -n "$_pgid" ] && [ "$_pgid" = "$SERVER_PID" ] && [ "$SERVER_GROUP_UNVERIFIED" != "1" ]; then
    SERVER_PGID="$SERVER_PID"
  fi
  local _mode="pid+descendants"
  [ -n "$SERVER_PGID" ] && _mode="group"
  echo ">>> server identity: pid=$SERVER_PID pgid=${_pgid:-?} start_ticks=${SERVER_START_TICKS:-?}" \
       "teardown_mode=$_mode | bench pid=$BENCH_PID pgid=${BENCH_PGID:-?}" \
       "protected_pgids='$SERVER_PROTECTED_PGIDS'"
}

# Sets $_deny with the refusal reason when it returns non-zero.
_group_kill_allowed() {
  local _p
  [ -n "$SERVER_PGID" ] || { _deny="group was not verified at launch"; return 1; }
  case "$SERVER_PGID" in ''|*[!0-9]*) _deny="pgid '$SERVER_PGID' is not numeric"; return 1 ;; esac
  # `kill -TERM -1` is NOT a group kill: it is a broadcast to every process we are
  # allowed to signal. It must be unreachable, including from an empty/zero lookup.
  [ "$SERVER_PGID" -gt 1 ] || { _deny="pgid<=1 (would broadcast, not group-kill)"; return 1; }
  for _p in $SERVER_PROTECTED_PGIDS; do
    if [ "$SERVER_PGID" = "$_p" ]; then
      _deny="pgid $SERVER_PGID is protected (matches $_p)"
      return 1
    fi
  done
  # Defence in depth: refuse if the LIVE group somehow holds pid 1 or ourselves.
  for _p in $(ps -eo pid=,pgid= 2>/dev/null | awk -v g="$SERVER_PGID" '$2==g{print $1}'); do
    if [ "$_p" = "1" ] || [ "$_p" = "$BENCH_PID" ]; then
      _deny="group $SERVER_PGID contains pid $_p"
      return 1
    fi
  done
  return 0
}

# Does the frozen identity still hold? "" start ticks means we could never verify in
# the first place (unreadable /proc) -- that fails OPEN by design, since the pgid==pid
# proof and the protected-pgid veto still stand between us and a stranger's group.
_identity_still_matches() {
  local _now
  [ -n "$SERVER_START_TICKS" ] || return 0
  _now="$(_start_ticks_of "$SERVER_PID" || true)"
  [ -z "$_now" ] || [ "$_now" = "$SERVER_START_TICKS" ]
}

# Is PID an ancestor of the benchmark shell? Such a pid CANNOT be a server we launched:
# it is our parent chain -- the run_e2e driver, the caller's orchestrator, init. A pid
# file that names one (stale, or recycled into the caller's tree) would otherwise send
# the teardown walking UP the process tree, which is exactly how issue #397 killed the
# coordinator. Checked on the pid path too, where the pgid==pid proof does not apply.
_is_bench_ancestor() {  # _is_bench_ancestor PID
  local _map _p="$BENCH_PID" _parent _hops=0
  _map="$(ps -eo pid=,ppid= 2>/dev/null)"
  while [ "$_hops" -lt 64 ]; do
    _parent="$(printf '%s\n' "$_map" | awk -v k="$_p" '$1==k{print $2; exit}')"
    [ -n "$_parent" ] || return 1
    [ "$_parent" = "$1" ] && return 0
    [ "$_parent" -gt 1 ] 2>/dev/null || return 1
    _p="$_parent"
    _hops=$((_hops + 1))
  done
  return 1
}

# Transitive descendants of SERVER_PID, excluding pid 1 and ourselves.
_server_descendants() {
  ps -eo pid=,ppid= 2>/dev/null | awk -v root="$SERVER_PID" -v self="$BENCH_PID" '
    { parent[$1] = $2 }
    END {
      mark[root] = 1
      changed = 1
      while (changed) {
        changed = 0
        for (p in parent) if (!mark[p] && parent[p] in mark && mark[parent[p]]) { mark[p] = 1; changed = 1 }
      }
      for (p in mark) if (mark[p] && p != root && p != self && p != 1) print p
    }'
}

server_teardown() {
  [ -n "${SERVER_PID:-}" ] || return 0
  # A server we launched can never BE init, and a non-numeric pid means a pid file /
  # launcher handed us garbage. Refuse before any signal, so the pid-only fallback
  # can no more reach PID 1 than the group path can.
  case "$SERVER_PID" in
    *[!0-9]*)
      echo "!!! SERVER_PID='$SERVER_PID' is not a pid; no signal sent." >&2
      return 0 ;;
  esac
  if [ "$SERVER_PID" -le 1 ]; then
    echo "!!! refusing to signal pid $SERVER_PID (container init / invalid pid)." >&2
    return 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">>> server pid $SERVER_PID already gone; no signal sent."
    wait "$SERVER_PID" 2>/dev/null || true
    return 0
  fi
  # PID-reuse gate: a pid whose start time moved is a DIFFERENT process. Send nothing.
  if ! _identity_still_matches; then
    echo "!!! pid $SERVER_PID start_time changed (was $SERVER_START_TICKS): the pid was REUSED;" \
         "refusing to signal an unrelated process." >&2
    return 0
  fi
  # Never signal our own ancestors, whatever the mode.
  if _is_bench_ancestor "$SERVER_PID"; then
    echo "!!! pid $SERVER_PID is an ANCESTOR of this benchmark shell (pid $BENCH_PID), so it cannot" \
         "be a server we launched; no signal sent." >&2
    return 0
  fi
  local _grace="${SERVER_STOP_GRACE_S:-10}" _deny="" _i _k _kids _kids_now
  # An unusable grace makes `seq` fail, which would collapse TERM->KILL to no wait at all.
  case "$_grace" in ''|*[!0-9]*) _grace=10 ;; esac
  if _group_kill_allowed; then
    echo ">>> Shutting down server (pid $SERVER_PID) -- GROUP kill pgid=$SERVER_PGID ..."
    kill -TERM "-$SERVER_PGID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _i in $(seq 1 "$_grace"); do
      kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
    done
    # Re-check identity before escalating: `kill -0` also succeeds against a stranger
    # who inherited the pid during the grace window, and SIGKILL cannot be ignored.
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      if _identity_still_matches; then
        kill -KILL "-$SERVER_PGID" 2>/dev/null || true
      else
        echo "!!! pid $SERVER_PID was REUSED during the grace window; skipping SIGKILL." >&2
      fi
    fi
  else
    echo ">>> Shutting down server (pid $SERVER_PID) -- PID+descendants kill;" \
         "group kill REFUSED: $_deny"
    _kids="$(_server_descendants)"   # collect BEFORE signalling anything
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _k in $_kids; do kill -TERM "$_k" 2>/dev/null || true; done
    for _i in $(seq 1 "$_grace"); do
      kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      if _identity_still_matches; then
        kill -KILL "$SERVER_PID" 2>/dev/null || true
      else
        echo "!!! pid $SERVER_PID was REUSED during the grace window; skipping SIGKILL." >&2
        _kids=""   # the tree we mapped belonged to the old process; do not escalate into it
      fi
    fi
    # Escalate only against pids that are STILL descendants: one that no longer is was
    # either reaped (and possibly recycled) or was never ours.
    _kids_now="$(_server_descendants)"
    for _k in $_kids; do
      case " $_kids_now " in *" $_k "*) ;; *) continue ;; esac
      kill -0 "$_k" 2>/dev/null && kill -KILL "$_k" 2>/dev/null || true
    done
  fi
  wait "$SERVER_PID" 2>/dev/null || true
}
