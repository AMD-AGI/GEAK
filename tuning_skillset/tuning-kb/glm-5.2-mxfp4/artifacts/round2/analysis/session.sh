#!/usr/bin/env bash
# One measurement session: restart the server on a given arm, then run N unmodified benches.
#
#   session.sh <arm> <session-label> [nruns]
#
# One session == one fresh server. Runs within a session give the within-instance spread;
# means across session labels give the restart-to-restart spread. Interleave arms across sessions.
#
# Two failure modes bit once and are guarded here:
#   * launch_server.sh --stop returns before the listening socket is released, so the next bind
#     fails with EADDRINUSE, the server dies after CUDA-graph capture, and the benches then run
#     against nothing. Wait for the port to actually clear.
#   * `launch_server.sh | tail -3 || exit` tests tail's status, not the launcher's, so a dead
#     server was not detected. Check PIPESTATUS.
#   * 43111 is inside the default ephemeral port range, so an outbound socket opened during the
#     ~2 min of model load and graph capture can steal it, and uvicorn then dies with EADDRINUSE
#     *after* a startup that looked entirely healthy. The port cannot be changed (frozen launch
#     config), so retry the launch instead. Seen twice in ~12 starts.
set -uo pipefail
cd /work
ARM="$1"; LABEL="$2"; N="${3:-3}"
PORT="${PORT:-43111}"

start_server() {
    bash scripts/launch_server.sh --stop >/dev/null 2>&1
    for _ in $(seq 1 60); do
        ss -ltn 2>/dev/null | grep -q ":${PORT}\b" || break
        sleep 1
    done
    if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
        echo "[session] port ${PORT} still bound after 60s" >&2; return 1
    fi
    bash analysis/set_arm.sh "$ARM" || return 1
    bash scripts/launch_server.sh 2>&1 | tail -3
    return "${PIPESTATUS[0]}"
}

ok=0
for attempt in 1 2 3; do
    if start_server; then ok=1; break; fi
    echo "[session] launch attempt $attempt failed on arm $ARM, retrying" >&2
    sleep 10
done
if [ "$ok" -ne 1 ]; then
    echo "[session] ABORT: server failed to start on arm $ARM after 3 attempts" >&2; exit 1
fi

for i in $(seq 1 "$N"); do
    TAG="${LABEL}_r${i}" bash scripts/run_bench.sh 2>&1 \
        | grep -E "^Output token throughput|result ->"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[session] ABORT: bench $i failed" >&2; exit 1
    fi
done
echo "[session] $LABEL done"
