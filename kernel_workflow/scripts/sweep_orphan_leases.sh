#!/bin/bash
# Report (and optionally kill) GPU-lease jobs that have outlived the workflow that spawned them.
#
# A lease job started in the background (`nohup ... gpu_lock.sh ... &`) has no owner once the
# workflow finishes: it can still be queued for the GPU group, acquire it later, and run with
# nobody reading its output while contending with other jobs.
#
# Usage:
#   sweep_orphan_leases.sh [--kill] [--eval-dir <dir>]
#
#   (default)         report only, exit 0 if clean, exit 3 if orphans found
#   --kill            terminate what it finds (TERM to the process group, then KILL after a grace)
#   --eval-dir <dir>  only consider leases whose command line mentions <dir> (this run's jobs)
#
# Reporting is the default on purpose: killing a lease job that a HUMAN started, from an automated
# closeout, would be worse than the leak. Only pass --kill when you know the pool is yours.
set -uo pipefail

DO_KILL=0
EVAL_DIR=""
GRACE="${SWEEP_GRACE_SEC:-10}"

while [ $# -gt 0 ]; do
    case "$1" in
        --kill) DO_KILL=1; shift ;;
        --eval-dir) EVAL_DIR="${2:?--eval-dir needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "sweep_orphan_leases.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

# Match on ownership, not lineage: the leaked process is usually the backgrounded wrapper shell,
# whose child (gpu_lease.py) still has a live parent, so a PPID==1 filter would miss it. A lease
# job that names this run's EVAL_DIR after the run has finished is stale. --kill therefore
# requires --eval-dir; without it the script only lists lease processes.
if [ "$DO_KILL" = "1" ] && [ -z "$EVAL_DIR" ]; then
    echo "sweep_orphan_leases.sh: --kill requires --eval-dir (refusing to kill leases that may not be ours)" >&2
    exit 2
fi

SELF=$$
mapfile -t CANDIDATES < <(
    ps -eo pid=,ppid=,etimes=,args= \
    | grep -E 'gpu_lease\.py|gpu_lock\.sh' \
    | grep -v -e 'grep -E' -e 'sweep_orphan_leases'
)

FOUND=0
for line in "${CANDIDATES[@]}"; do
    [ -n "$line" ] || continue
    read -r pid ppid age cmd <<<"$(tr -s ' ' <<<"${line# }")"
    [ "$pid" = "$SELF" ] && continue
    [ "$ppid" = "$SELF" ] && continue

    if [ -n "$EVAL_DIR" ] && ! grep -qF -- "$EVAL_DIR" <<<"$cmd"; then
        continue
    fi

    FOUND=$((FOUND + 1))
    echo "[ORPHAN-LEASE] pid=$pid ppid=$ppid age=${age}s"
    echo "               $cmd"

    if [ "$DO_KILL" = "1" ]; then
        # Kill the process group: a multi-rank payload forks one child per rank.
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -n "$pgid" ]; then
            kill -TERM -- "-$pgid" 2>/dev/null
            for _ in $(seq 1 "$GRACE"); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null
            echo "               -> killed pgid=$pgid"
        else
            echo "               -> already gone"
        fi
    fi
done

if [ "$FOUND" = "0" ]; then
    echo "[ORPHAN-LEASE] none found${EVAL_DIR:+ for $EVAL_DIR}"
    exit 0
fi

echo "[ORPHAN-LEASE] $FOUND orphan(s)$([ "$DO_KILL" = "1" ] && echo ' killed' || echo '; re-run with --kill to terminate')"
[ "$DO_KILL" = "1" ] && exit 0
exit 3
