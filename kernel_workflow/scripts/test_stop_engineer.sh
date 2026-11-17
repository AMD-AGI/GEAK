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
# A stop that does not take fails loudly: processes that keep reappearing in the workspace, or a STOP
# that cannot be written, give exit 3 with STOP_FAILED. And once STOP is written, gpu_lock.sh refuses any
# command from that workspace (exit 87) while verify clones and the canonical tree are not fenced.
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

# --- a process that reappears once after the first kill is caught by a later round -------------
# (an engineer whose command was killed and that immediately issues one more)
ENG1="$T/round_1b/engineer_0"
mkdir -p "$ENG1/workspace"
once=$(spawn "$T/elsewhere" "(cd '$ENG1/workspace' && exec sleep 300) & wait \$!; (cd '$ENG1/workspace' && exec sleep 300) & wait \$!")
PIDS+=("$once")
sleep 1
out4="$(STOP_ENGINEER_GRACE_SEC=1 bash "$STOP" "$ENG1" 2>&1)"; rc=$?
sleep 1
left=$(for d in /proc/[0-9]*; do c=$(readlink "$d/cwd" 2>/dev/null) && [ "$c" = "$ENG1/workspace" ] && echo "${d#/proc/}"; done)
[ $rc -eq 0 ] && [ -z "$left" ] && ok "a process started after the first kill round is stopped too" \
    || bad "re-spawned process not handled (rc=$rc, left: $left): $out4"

# --- a stop that does not take must fail loudly (exit 3), never report success ------------------
# Something outside the workspace keeps starting new processes inside it (as the uncancellable engineer
# agent can): every round kills them, more appear, and the script must give up with STOP_FAILED.
ENG2="$T/round_2/engineer_0"
mkdir -p "$ENG2/workspace"
respawner=$(spawn "$T/elsewhere" "while :; do (cd '$ENG2/workspace' && exec sleep 30) & sleep 0.2; done")
PIDS+=("$respawner")
sleep 1
err="$(STOP_ENGINEER_GRACE_SEC=1 STOP_ENGINEER_ROUNDS=2 bash "$STOP" "$ENG2" 2>&1 >/dev/null)"; rc=$?
[ $rc -eq 3 ] && ok "processes that keep coming back -> exit 3" || bad "expected exit 3, got $rc"
echo "$err" | grep -q "STOP_FAILED: .* still running under $ENG2/workspace" && ok "survivors are reported" \
    || bad "no STOP_FAILED report: $err"
echo "$err" | grep -q "pid=" && ok "survivors are listed by pid" || bad "survivor list missing"
kill -KILL "$respawner" 2>/dev/null
sleep 1
out3="$(STOP_ENGINEER_GRACE_SEC=1 bash "$STOP" "$ENG2" 2>&1)"; rc=$?
[ $rc -eq 0 ] && ok "once the source is gone the stop succeeds" || bad "expected exit 0 after the respawner died: $out3"

# STOP that cannot be written is a failed stop (skipped as root, where permissions do not bite).
if [ "$(id -u)" != "0" ]; then
    ENG3="$T/round_3/engineer_0"
    mkdir -p "$ENG3/workspace"
    chmod a-w "$ENG3"
    err="$(bash "$STOP" "$ENG3" 2>&1 >/dev/null)"; rc=$?
    chmod u+w "$ENG3"
    [ $rc -eq 3 ] && echo "$err" | grep -q "STOP_FAILED: cannot write" && ok "unwritable STOP -> exit 3" \
        || bad "unwritable STOP should exit 3 with STOP_FAILED (rc=$rc: $err)"
else
    echo "  skip: unwritable STOP check (running as root)"
fi

# --- gpu_lock.sh refuses a stopped engineer's workspace, and only that -------------------------
LOCK="$HERE/gpu_lock.sh"
FENCE="$T/fence"
mkdir -p "$FENCE/eng/workspace/sub" "$FENCE/eng/verify/ws_1" "$FENCE/eval/workspace"
touch "$FENCE/eng/STOP"
for where in "$FENCE/eng/workspace" "$FENCE/eng/workspace/sub"; do
    res="$(cd "$where" && KERNEL_EXTRACT_LOCK_DIR="$T/locks" timeout 60 bash "$LOCK" 0 echo RAN_UNDER_LOCK 2>&1)"; rc=$?
    [ $rc -eq 87 ] && ok "gpu_lock refuses from ${where#$FENCE/} (exit 87)" || bad "gpu_lock from ${where#$FENCE/}: expected 87, got $rc"
    echo "$res" | grep -q GEAK_ENGINEER_STOPPED && ok "  ...and says why" || bad "  no GEAK_ENGINEER_STOPPED message"
    echo "$res" | grep -q RAN_UNDER_LOCK && bad "  the command ran anyway" || ok "  ...and the command never runs"
done
# Verify clones under the same engineer dir, and a workspace with no STOP beside it, are not fenced.
# (They may fail later for unrelated reasons on a GPU-less runner; only the fence verdict is checked.)
for where in "$FENCE/eng/verify/ws_1" "$FENCE/eval/workspace"; do
    res="$(cd "$where" && KERNEL_EXTRACT_LOCK_DIR="$T/locks" timeout 60 bash "$LOCK" 0 true 2>&1)"; rc=$?
    [ $rc -ne 87 ] && ! echo "$res" | grep -q GEAK_ENGINEER_STOPPED && ok "gpu_lock does not fence ${where#$FENCE/}" \
        || bad "gpu_lock wrongly fenced ${where#$FENCE/} (rc=$rc)"
done

if [ "$FAIL" -ne 0 ]; then
    echo "test_stop_engineer: $FAIL failure(s)" >&2
    exit 1
fi
echo "test_stop_engineer: all checks passed"
