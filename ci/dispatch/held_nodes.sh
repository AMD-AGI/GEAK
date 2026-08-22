#!/usr/bin/env bash
# =============================================================================
# held_nodes.sh — discover and probe the long-lived "holder" allocations that
# this user is sitting on, so CI can run on them WITHOUT queueing a new job.
#
# Why this exists: the cluster is saturated by long-lived holds, so a freshly
# submitted sbatch job can pend indefinitely and the L1 matrix never runs. We
# already hold nodes (see ~/spur_hold.sh: batch jobs named "$USER-hold" that
# `sleep infinity`). This module finds those holders, decides which are steady
# idle, and gives the dispatcher a way to execute on them.
#
# Execution contract (both facts verified on this cluster, 2026-08-19):
#   1. srun REQUIRES A PTY here. Without one the SPUR shim prints
#      "raw mode unavailable (stdin is not a TTY)", exits 128, and runs NOTHING
#      at all — it is not merely an output-streaming problem. Every remote call
#      therefore goes through `script -qec ... /dev/null`.
#   2. A step's processes CANNOT outlive the step. `nohup setsid ... &` inside
#      an srun step is reaped the moment the step exits (SLURM tears down the
#      step cgroup), so "launch and detach" does not work. The caller must keep
#      the srun client alive for the whole run.
#
# Usage:
#   ci/dispatch/held_nodes.sh list          # every holder (jobid node state)
#   ci/dispatch/held_nodes.sh probe         # per-holder GPU/container activity
#   ci/dispatch/held_nodes.sh idle          # jobid+node of steady-idle holders
#   ci/dispatch/held_nodes.sh exec <jobid> <cmd...>   # run a command on one
# =============================================================================
set -uo pipefail

HELD_NAME="${HELD_NAME:-${USER}-hold}"
# A GPU at or below this busy% counts as idle.
HELD_IDLE_PCT="${HELD_IDLE_PCT:-5}"
# "Steady" = this many consecutive idle samples, this far apart. Guards against
# catching a node in a momentary lull between two phases of somebody's run.
HELD_IDLE_SAMPLES="${HELD_IDLE_SAMPLES:-2}"
HELD_IDLE_GAP_S="${HELD_IDLE_GAP_S:-3}"
# Cap a single remote probe so one wedged node cannot stall discovery.
HELD_PROBE_TIMEOUT_S="${HELD_PROBE_TIMEOUT_S:-60}"
# Directory whose writability decides whether a node can host a run at all.
# A node whose shared mount has dropped to read-only looks PERFECTLY idle (no GPU
# load, no containers) and will keep being handed work, but the run dies seconds
# in, the moment the monitor writes its first line — while the holder stays alive,
# so the "holder vanished" retry never fires and the model is simply lost
# (observed 2026-08-21 on crsuse2-m2m-243). Probe the workspace itself, because
# that is exactly what a run writes to.
HELD_RW_PROBE_DIR="${HELD_RW_PROBE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)}"
# Nodes that pass every probe but cannot actually serve. crsuse2-m2m-254 is idle,
# writable and reports a healthy MI355X, yet the sglang server dies under the first
# real benchmark load: the campaign bundle's own scripts brought it up, verified the
# config, completed warmup, then lost all 192 requests in 8.7s with the server gone
# (2026-08-21). The same node produced a 25s TTFT and 1580 tok/s for Llama-3.1-8B in
# the 10-model smoke, against 2593 on a healthy node. Space-separated; override to
# re-admit a node once it has been repaired.
HELD_EXCLUDE_NODES="${HELD_EXCLUDE_NODES:-crsuse2-m2m-254}"
# Containers that are part of the machine, not of somebody's run. Crusoe keeps a
# log collector, a metrics exporter and a vector agent resident on some nodes for
# weeks at a time; counting them as tenancy made crsuse2-m2m-036 read as "busy"
# with 3 containers and silently cost us a node (observed 2026-08-21). Extended
# regex, matched against container names.
HELD_IGNORE_CONTAINERS="${HELD_IGNORE_CONTAINERS:-^crusoe-}"

# held_excluded <node> -> 0 when the node is on the exclusion list.
held_excluded() {
  local node="$1" ex
  for ex in $HELD_EXCLUDE_NODES; do [ "$node" = "$ex" ] && return 0; done
  return 1
}

# ---- remote execution -------------------------------------------------------
# held_exec <jobid> <timeout_s> <command-string>
# Runs the command inside the holder allocation and echoes its stdout+stderr.
# `script` supplies the pty srun demands; we strip the NUL/CR it injects.
# stdin is redirected from /dev/null: `script` allocates its own pty, and without
# this the remote command would swallow the caller's stdin — which silently ate
# the input of every `while read ... done < <(held_running)` loop below.
#
# SPUR's srun prefixes its first output line with a LITERAL two-character "^@"
# (caret + at-sign, not a NUL byte), so `tr -d '\000'` does not remove it and any
# anchored match on the first line silently fails. Strip it explicitly.
#
# The pty also makes every tool on the far side believe it is interactive, so
# docker pull et al. emit cursor-movement progress bars — thousands of escape
# sequences that made the first captured run log unreadable. TERM=dumb plus an
# ANSI/CR filter turns that back into plain lines.
held_exec() {
  local jid="$1" tmo="$2" cmd="$3" srun_cmd
  srun_cmd="srun --jobid=$jid --overlap --nodes=1 --ntasks=1 env TERM=dumb bash -lc $(printf '%q' "$cmd")"
  timeout "$tmo" script -qec "$srun_cmd" /dev/null </dev/null 2>&1 \
    | tr -d '\000\r' | sed -E 's/\^@//g; s/\x1b\[[0-9;?]*[a-zA-Z]//g' \
    | awk '!/: (Downloading|Extracting|Waiting|Pulling fs layer|Verifying Checksum) /'
}

# ---- discovery --------------------------------------------------------------
# Every holder job, one "jobid state node" per line (node is '-' while PENDING).
held_holders() {
  squeue -h -u "$USER" -o '%i|%j|%t|%N' 2>/dev/null \
    | awk -F'|' -v n="$HELD_NAME" '$2 == n {print $1, $3, ($4 == "" ? "-" : $4)}'
}

# Holders that are RUNNING (i.e. actually own a node right now).
held_running() { held_holders | awk '$2 == "R" {print $1, $3}'; }

# held_activity <jobid> -> "<max_gpu_busy_pct> <container_count> <rw|ro>", or
# "? ? ?" if the node could not be probed. Mirrors ~/probe_idle.sh, plus the
# writability check described at HELD_RW_PROBE_DIR.
held_activity() {
  local jid="$1" out busy ncont fs
  out="$(held_exec "$jid" "$HELD_PROBE_TIMEOUT_S" \
    "HELD_RW_DIR=$(printf '%q' "$HELD_RW_PROBE_DIR") HELD_IGN=$(printf '%q' "$HELD_IGNORE_CONTAINERS")"'
    b=$(rocm-smi --showuse --csv 2>/dev/null | awk -F, "NR>1 && \$2 ~ /^[0-9]+$/ {print \$2}" | sort -rn | head -1)
    c=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -Ev "$HELD_IGN" | wc -l)
    w=ro
    t="$HELD_RW_DIR/.held_rw_probe_$$_${RANDOM}"
    if touch "$t" 2>/dev/null; then w=rw; rm -f "$t" 2>/dev/null; fi
    echo "ACT ${b:-?} ${c:-?} ${w:-?}"
  ')"
  busy="$(awk '/^ACT /{print $2; exit}' <<<"$out")"
  ncont="$(awk '/^ACT /{print $3; exit}' <<<"$out")"
  fs="$(awk '/^ACT /{print $4; exit}' <<<"$out")"
  printf '%s %s %s\n' "${busy:-?}" "${ncont:-?}" "${fs:-?}"
}

# held_is_idle <jobid> -> 0 if steady idle. Samples HELD_IDLE_SAMPLES times.
held_is_idle() {
  local jid="$1" i busy ncont fs
  for (( i = 0; i < HELD_IDLE_SAMPLES; i++ )); do
    [ "$i" -gt 0 ] && sleep "$HELD_IDLE_GAP_S"
    read -r busy ncont fs < <(held_activity "$jid")
    case "$busy" in ''|*[!0-9]*) return 1 ;; esac   # unreadable -> not idle
    case "$ncont" in ''|*[!0-9]*) return 1 ;; esac
    [ "$fs" = "rw" ] || return 1                    # read-only mount -> unusable
    [ "$busy" -le "$HELD_IDLE_PCT" ] || return 1
    [ "$ncont" -eq 0 ] || return 1
  done
  return 0
}

# All steady-idle holders, one "jobid node" per line.
held_idle_nodes() {
  local jid node
  while read -r jid node; do
    [ -n "$jid" ] || continue
    held_excluded "$node" && continue
    held_is_idle "$jid" && echo "$jid $node"
  done < <(held_running)
}

# ---- CLI --------------------------------------------------------------------
# Only when executed directly; when sourced, the functions above are the API.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  case "${1:-list}" in
    list)
      printf '%-10s %-7s %s\n' JOBID STATE NODE
      held_holders | while read -r j s n; do printf '%-10s %-7s %s\n' "$j" "$s" "$n"; done
      ;;
    probe)
      printf '%-10s %-20s %-9s %-11s %-5s %s\n' JOBID NODE GPU_BUSY CONTAINERS FS VERDICT
      while read -r j n; do
        read -r busy ncont fs < <(held_activity "$j")
        v="busy"
        [ "$busy" = "?" ] && v="unreachable"
        [ "$fs" = "ro" ] && v="read-only"
        if [[ "$busy" =~ ^[0-9]+$ && "$ncont" =~ ^[0-9]+$ && "$fs" = "rw" ]]; then
          { [ "$busy" -le "$HELD_IDLE_PCT" ] && [ "$ncont" -eq 0 ]; } && v="idle"
        fi
        held_excluded "$n" && v="excluded"
        printf '%-10s %-20s %-9s %-11s %-5s %s\n' "$j" "$n" "${busy}%" "$ncont" "$fs" "$v"
      done < <(held_running)
      ;;
    idle)  held_idle_nodes ;;
    exec)
      jid="${2:?usage: held_nodes.sh exec <jobid> <cmd...>}"; shift 2
      held_exec "$jid" "${HELD_EXEC_TIMEOUT_S:-3600}" "$*"
      ;;
    *) echo "usage: held_nodes.sh {list|probe|idle|exec <jobid> <cmd...>}" >&2; exit 2 ;;
  esac
fi
