#!/bin/bash
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stop an engineer the lane has stopped waiting for.
#
# kernel_lane.js bounds every agent() call with AGENT_TIMEOUT_MS, but a call that times out is only
# abandoned: a workflow cannot cancel an agent, so the engineer keeps running beside its own verify,
# rebuilding, benchmarking on the shared GPUs and rewriting best_patch.diff. Its verify runs this
# script before it copies the patch:
#   1. write <out_dir>/STOP, which the engineer roles treat as "run nothing more, return now";
#   2. terminate every process whose cwd is inside <out_dir>/workspace, with its descendants
#      (TERM, then KILL after a grace period).
# Matched on cwd, not on argv: engineers run `cd $KERNEL_PATH && bash gpu_lock.sh ...`, so the
# workspace path need not appear on any command line, but every build and benchmark runs inside it.
# Killed one PID tree at a time, never by process group: a tool's shell can share its group with the
# agent host, and killing that group would end the whole run. This script's own ancestors are spared.
#
# Usage: stop_engineer.sh <engineer_out_dir>
#   exit 0 when the engineer is stopped (finding nothing to stop is success), 2 on bad usage.
set -uo pipefail

if [ $# -ne 1 ] || [ -z "$1" ] || [ ! -d "$1" ]; then
    echo "usage: stop_engineer.sh <engineer_out_dir>  (an existing directory)" >&2
    exit 2
fi
OUT_DIR="$(cd "$1" && pwd -P)"
WS="$OUT_DIR/workspace"
GRACE="${STOP_ENGINEER_GRACE_SEC:-10}"
# Run from / so this script's own children never match the workspace.
cd /

touch "$OUT_DIR/STOP"

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

declare -A TARGET=()
for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    [ -n "${SPARE[$pid]:-}" ] && continue
    cwd="$(readlink "$d/cwd" 2>/dev/null)" || continue
    in_ws "$cwd" || continue
    TARGET[$pid]=1
    for c in $(descendants "$pid"); do
        [ -n "${SPARE[$c]:-}" ] || TARGET[$c]=1
    done
done

alive() {
    local s
    s="$(ps -o stat= -p "$1" 2>/dev/null)"
    [ -n "$s" ] && [ "${s:0:1}" != "Z" ]
}

PIDS=("${!TARGET[@]}")
if [ ${#PIDS[@]} -gt 0 ]; then
    kill -TERM "${PIDS[@]}" 2>/dev/null
    for _ in $(seq 1 "$GRACE"); do
        left=0
        for pid in "${PIDS[@]}"; do alive "$pid" && left=1 && break; done
        [ "$left" = "0" ] && break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do alive "$pid" && kill -KILL "$pid" 2>/dev/null; done
fi

echo "[STOP-ENGINEER] $OUT_DIR: wrote STOP, stopped ${#PIDS[@]} process(es) under $WS"
exit 0
