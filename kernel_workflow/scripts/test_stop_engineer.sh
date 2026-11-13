#!/bin/bash
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Contract test for stop_engineer.sh (no GPU, no model).
#
# A verify that has been told to STOP_ENGINEER must end exactly the abandoned engineer's processes:
# everything running inside <out_dir>/workspace (and what it spawned, wherever that runs), and nothing
# else — not the verify's own scratch under <out_dir>/verify, not a sibling directory that merely
# shares the name prefix, and not the caller that runs the script, even from inside the workspace.
#
# Run:  bash kernel_workflow/scripts/test_stop_engineer.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP="$HERE/stop_engineer.sh"
T="$(mktemp -d)"
T="$(cd "$T" && pwd -P)"
ENG="$T/round_1/engineer_0"
mkdir -p "$ENG/workspace/build" "$ENG/verify" "$ENG/workspace2" "$T/elsewhere"

PIDS=()
cleanup() {
    for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done
    pkill -KILL -f "stop_engineer_test_marker_$$" 2>/dev/null
    rm -rf "$T"
}
trap cleanup EXIT

FAIL=0
ok()  { echo "  ok: $1"; }
bad() { echo "  FAIL: $1" >&2; FAIL=$((FAIL + 1)); }
alive() {
    local s
    s="$(ps -o stat= -p "$1" 2>/dev/null)"
    [ -n "$s" ] && [ "${s:0:1}" != "Z" ]
}
# One long sleeper per scenario; the marker lets cleanup find them even if a PID is lost.
spawn() {  # spawn <cwd> <bash snippet>; echoes the pid of the started shell
    (cd "$1" && exec bash -c "$2" "stop_engineer_test_marker_$$") >/dev/null 2>&1 &
    echo $!
}

# --- processes the script must stop -----------------------------------------------------
in_ws=$(spawn "$ENG/workspace" 'exec sleep 300');                 PIDS+=("$in_ws")
in_sub=$(spawn "$ENG/workspace/build" 'exec sleep 300');          PIDS+=("$in_sub")
# A benchmark whose parent runs in the workspace but which itself chdir'd elsewhere.
parent=$(spawn "$ENG/workspace" "(cd '$T/elsewhere' && exec sleep 300) & wait"); PIDS+=("$parent")
# A process that ignores SIGTERM must still go (KILL after the grace period).
stubborn=$(spawn "$ENG/workspace" 'trap "" TERM; while :; do sleep 1; done'); PIDS+=("$stubborn")
# --- processes the script must leave alone ----------------------------------------------
verify=$(spawn "$ENG/verify" 'exec sleep 300');                   PIDS+=("$verify")
prefix=$(spawn "$ENG/workspace2" 'exec sleep 300');               PIDS+=("$prefix")
other=$(spawn "$T/elsewhere" 'exec sleep 300');                   PIDS+=("$other")
sleep 1
child=$(ps -o pid= --ppid "$parent" | tr -d ' ' | head -1)
[ -n "$child" ] && PIDS+=("$child")

# Usage errors are refused, and do not write anything.
bash "$STOP" >/dev/null 2>&1;            [ $? -eq 2 ] && ok "no argument -> exit 2" || bad "no argument should exit 2"
bash "$STOP" "$T/missing" >/dev/null 2>&1; [ $? -eq 2 ] && ok "missing dir -> exit 2" || bad "missing dir should exit 2"

# Run it from INSIDE the workspace: the calling shell must survive.
out="$(cd "$ENG/workspace" && STOP_ENGINEER_GRACE_SEC=2 bash "$STOP" "$ENG"; echo "rc=$?")"
echo "$out" | sed 's/^/    /'
echo "$out" | grep -q "rc=0" && ok "exit 0" || bad "expected exit 0"
echo "$out" | grep -q "STOP-ENGINEER" && ok "reports what it did" || bad "no report line"
[ -e "$ENG/STOP" ] && ok "STOP sentinel written" || bad "STOP sentinel missing"
sleep 1

for v in in_ws in_sub parent stubborn; do
    alive "${!v}" && bad "$v (pid ${!v}) should have been stopped" || ok "$v stopped"
done
if [ -n "$child" ]; then
    alive "$child" && bad "descendant outside the workspace (pid $child) should have been stopped" \
        || ok "descendant outside the workspace stopped"
else
    bad "could not find the descendant to check"
fi
for v in verify prefix other; do
    alive "${!v}" && ok "$v left running" || bad "$v (pid ${!v}) must not be stopped"
done

# Nothing left to stop is still success (idempotent re-run).
out2="$(STOP_ENGINEER_GRACE_SEC=1 bash "$STOP" "$ENG")"; rc=$?
[ $rc -eq 0 ] && echo "$out2" | grep -q "stopped 0 process" && ok "re-run with nothing to stop -> exit 0" \
    || bad "re-run should exit 0 and stop 0 processes (rc=$rc: $out2)"

if [ "$FAIL" -ne 0 ]; then
    echo "test_stop_engineer: $FAIL failure(s)" >&2
    exit 1
fi
echo "test_stop_engineer: all checks passed"
