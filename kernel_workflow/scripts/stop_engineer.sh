#!/bin/bash
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stop an engineer the lane has stopped waiting for.
#
# kernel_lane.js bounds every agent() call with AGENT_TIMEOUT_MS, but a call that times out is only
# abandoned: a workflow cannot cancel an agent, so the engineer keeps running beside its own verify,
# rebuilding, benchmarking on the shared GPUs and rewriting best_patch.diff. Its verify runs this
# script before it freezes the patch:
#   1. write <out_dir>/STOP. From then on gpu_lock.sh refuses every command run from
#      <out_dir>/workspace (exit 87) and save_patch.sh refuses to rewrite best_patch.diff (exit 3),
#      whatever the engineer agent does next;
#   2. terminate every process whose cwd is inside <out_dir>/workspace, with its descendants
#      (TERM, then KILL after a grace period), and scan again until none is left.
# Matched on cwd, not on argv: engineers run `cd $KERNEL_PATH && bash gpu_lock.sh ...`, so the
# workspace path need not appear on any command line, but every build and benchmark runs inside it.
# Killed one PID tree at a time, never by process group: a tool's shell can share its group with the
# agent host, and killing that group would end the whole run. This script's own ancestors are spared.
#
# Failure is loud: if STOP cannot be written, or processes are still running in the workspace after
# STOP_ENGINEER_ROUNDS kill rounds, the survivors are listed and the exit status is 3. Verify must then
# NOT freeze or measure the patch (it reports status "stop_failed").
#
# Usage: stop_engineer.sh <engineer_out_dir>
#   exit 0 stopped (finding nothing to stop is success), 2 bad usage, 3 could not stop it
# Env: STOP_ENGINEER_GRACE_SEC (TERM->KILL grace, default 10), STOP_ENGINEER_ROUNDS (default 5)
set -uo pipefail

if [ $# -ne 1 ] || [ -z "$1" ] || [ ! -d "$1" ]; then
    echo "usage: stop_engineer.sh <engineer_out_dir>  (an existing directory)" >&2
    exit 2
fi
OUT_DIR="$(cd "$1" && pwd -P)"
WS="$OUT_DIR/workspace"
GRACE="${STOP_ENGINEER_GRACE_SEC:-10}"
ROUNDS="${STOP_ENGINEER_ROUNDS:-5}"
# Run from / so this script's own children never match the workspace.
cd /

if ! touch "$OUT_DIR/STOP" 2>/dev/null || [ ! -e "$OUT_DIR/STOP" ]; then
    echo "STOP_FAILED: cannot write $OUT_DIR/STOP; the engineer is not fenced" >&2
    exit 3
fi

declare -A SPARE=()
p=$$
while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    SPARE[$p]=1
    p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
done

in_ws() {
    case "$1" in
        "$WS"|"$WS"/*) return 0 ;;
    esac
    return 1
}

descendants() {
    local c
    for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do
        echo "$c"
        descendants "$c"
    done
}

alive() {
    local s
    s="$(ps -o stat= -p "$1" 2>/dev/null)"
    [ -n "$s" ] && [ "${s:0:1}" != "Z" ]
}

# Live processes running inside the workspace, plus everything they spawned (wherever it runs).
scan() {
    declare -A seen=()
    local d pid cwd c
    for d in /proc/[0-9]*; do
        pid="${d#/proc/}"
        [ -n "${SPARE[$pid]:-}" ] && continue
        cwd="$(readlink "$d/cwd" 2>/dev/null)" || continue
        in_ws "$cwd" || continue
        alive "$pid" || continue
        seen[$pid]=1
        for c in $(descendants "$pid"); do
            [ -n "${SPARE[$c]:-}" ] || seen[$c]=1
        done
    done
    echo "${!seen[@]}"
}

TOTAL=0
for round in $(seq 1 "$ROUNDS"); do
    read -r -a PIDS <<< "$(scan)"
    [ ${#PIDS[@]} -eq 0 ] && break
    TOTAL=$((TOTAL + ${#PIDS[@]}))
    kill -TERM "${PIDS[@]}" 2>/dev/null
    for _ in $(seq 1 "$GRACE"); do
        left=0
        for pid in "${PIDS[@]}"; do alive "$pid" && left=1 && break; done
        [ "$left" = "0" ] && break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do alive "$pid" && kill -KILL "$pid" 2>/dev/null; done
    sleep 1
done

read -r -a LEFT <<< "$(scan)"
if [ ${#LEFT[@]} -gt 0 ]; then
    echo "STOP_FAILED: ${#LEFT[@]} process(es) still running under $WS after $ROUNDS kill rounds:" >&2
    for pid in "${LEFT[@]}"; do
        echo "  pid=$pid $(ps -o user=,stat=,etimes=,args= -p "$pid" 2>/dev/null | cut -c1-200)" >&2
    done
    exit 3
fi

echo "[STOP-ENGINEER] $OUT_DIR: wrote STOP, stopped $TOTAL process(es) under $WS"
exit 0
