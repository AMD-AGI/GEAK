#!/usr/bin/env bash
# Start the container for this experiment. Run on the HOST, on a node with MI355X GPUs.
#
#   ./start_container.sh           # start and exec into it
#   ./start_container.sh --rm      # remove a previous one first
#
# The image is not recorded anywhere in the session state -- the schema does not carry it. This tag
# matches the framework version the reference run reported (sglang 0.5.17); confirm
# it inside the container before you measure, and record what you actually used in FINDINGS.md.
set -uo pipefail

NAME="${NAME:-qwen3_5_397b_a17b_mxfp4_tuning}"
IMAGE="${IMAGE:-harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix}"
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${1:-}" = "--rm" ]; then
    docker rm -f "$NAME" 2>/dev/null; shift
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[container] $NAME exists; exec-ing into it"
    exec docker exec -it "$NAME" bash
fi

# This experiment needs 4 of the node's eight devices, and the render nodes are NOT numbered
# contiguously on every host: these MI355X nodes expose renderD128, renderD136, renderD144 ... in
# steps of eight. Naming renderD129 on such a host does not fail -- docker creates a device node that
# maps to nothing, and the container comes up seeing fewer GPUs than the config needs, which shows up
# much later as an unexplained server start failure. So map the first 4 that actually exist, and
# refuse to start if they are not there. Override with RENDER_NODES to pick specific devices, e.g.
# when a neighbour holds the low-numbered ones.
NGPU=4
RENDER_NODES="${RENDER_NODES:-$(ls -1 /dev/dri/renderD* 2>/dev/null | sort -V | head -"$NGPU" | tr '\n' ' ')}"
DEV=""
FOUND=0
for d in $RENDER_NODES; do
    [ -e "$d" ] || { echo "[container] $d does not exist" >&2; exit 1; }
    DEV="$DEV --device $d"
    FOUND=$((FOUND + 1))
done
if [ "$FOUND" != "$NGPU" ]; then
    echo "[container] need $NGPU render nodes, found $FOUND: $RENDER_NODES" >&2
    exit 1
fi
echo "[container] devices:$DEV"

echo "[container] starting $NAME from $IMAGE"
# --entrypoint bash matters: the vLLM images declare ENTRYPOINT ["vllm serve"], so a bare
# `sleep infinity` is parsed as arguments to the server ("unrecognized arguments: infinity") and the
# container exits about half a minute later -- long enough to look like it started, and to let a
# docker exec succeed, before everything after it fails with "no such container".
docker run -d --name "$NAME" \
    --ipc=host --network=host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    $DEV --device /dev/kfd \
    --shm-size 64g \
    -v /shared_nfs:/shared_nfs:ro \
    -v /home/ethany:/home/ethany \
    -v "$BUNDLE:/work" \
    -w /work \
    -e HF_HUB_OFFLINE=1 \
    --entrypoint bash \
    "$IMAGE" -c 'sleep infinity'

# Being up for a moment is not the same as being up. Wait a beat and confirm, so an entrypoint or
# device problem is reported here rather than as a puzzling failure in the next script.
sleep 5
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[container] $NAME is not running; exit=$(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null)" >&2
    docker logs --tail 20 "$NAME" 2>&1 | sed 's/^/    /' >&2
    exit 1
fi

echo "[container] verifying the stack matches the reference"
if [ -x "$BUNDLE/scripts/preflight.sh" ]; then
    docker exec "$NAME" bash -lc './scripts/preflight.sh' \
        || echo "[container] preflight FAILED -- see above; do not measure until it passes" >&2
else
    # importlib.metadata rather than framework.__version__: sglang builds do not define one, and the
    # packaging metadata is also what carries the vendor build suffix.
    docker exec "$NAME" bash -lc '
      python3 -c "import torch; print(\"torch\", torch.__version__)" 2>/dev/null
      python3 -c "from importlib.metadata import version; print(\"sglang\", version(\"sglang\"))" 2>/dev/null
      echo "reference: sglang 0.5.17, rocm 7.2, 4x MI355X gfx950"
      rocm-smi --showid 2>/dev/null | head -20
    '
fi
exec docker exec -it "$NAME" bash
