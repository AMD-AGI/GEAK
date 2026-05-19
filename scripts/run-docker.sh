#!/bin/bash
# Build and run GEAK-agent Docker container.
#
# Usage:
#   scripts/run-docker.sh                          # Interactive bash shell
#   scripts/run-docker.sh --rebuild                # Rebuild image, then bash
#   scripts/run-docker.sh --editable               # Mount host repo for live code editing
#   scripts/run-docker.sh -- test-discovery /path  # Run a command inside container
#   scripts/run-docker.sh --rebuild -- geak --help # Rebuild, then run command
#
# Modular pipeline commands (chainable via intermediate files):
#   resolve-kernel-url <url> --json -o resolved.json
#   test-discovery --from-resolved resolved.json -o discovery.json
#   kernel-profile --from-discovery discovery.json --json -o profile.json
#   baseline-metrics build --from-profile profile.json --all -o baseline_metrics.json
#   commandment --from-discovery discovery.json -o COMMANDMENT.md
#   task-generator --from-discovery discovery.json --profiling profile.json \
#       --commandment COMMANDMENT.md --baseline-metrics baseline_metrics.json \
#       -o tasks/round_1/
#
# Iterative refinement (round 2+):
#   task-generator ... --from-results results/round_1/ --round 2 -o tasks/round_2/
#
# Run individual tasks:
#   geak --from-task tasks/round_1/10_triton-autotune.md --gpu-ids 2
#
# Other tools:
#   validate-commandment <path>              Validate a COMMANDMENT.md
#   select-patch --patch-dir <dir> ...       Select best patch from runs
#   geak <github_url>                        Full optimization pipeline

set -e

IMAGE_NAME="geak-agent:latest"
CONTAINER_NAME="geak-agent-${USER}"

# Repo root (directory containing scripts/); used with --editable to mount live code
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Set directories to user's home
PARENT_DIR="${HOME}"
HOST_CODE_DIR="${HOME}"

REBUILD=false
EDITABLE=false
EXEC_CMD=()  # Command to run inside the container (empty = bash)

#######################################
# Auto-resolve BASE_IMAGE from host GPU arch (override with BASE_IMAGE env var)
#######################################
SGLANG_VERSION="v0.5.11"
ROCM_TAG="rocm720"

if [ -z "$BASE_IMAGE" ]; then
    _gfx_arch=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)
    case "$_gfx_arch" in
        gfx942) _gpu_suffix="mi30x" ;;
        gfx950) _gpu_suffix="mi35x" ;;
        *)      _gpu_suffix="mi35x" ;;
    esac
    BASE_IMAGE="lmsysorg/sglang:${SGLANG_VERSION}-${ROCM_TAG}-${_gpu_suffix}"
    if [ -n "$_gfx_arch" ]; then
        echo "Detected GPU arch: ${_gfx_arch} → BASE_IMAGE=${BASE_IMAGE}"
    else
        echo "Could not detect GPU arch, using default: BASE_IMAGE=${BASE_IMAGE}"
    fi
fi

#######################################
# Parse options
#######################################
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --editable)
            EDITABLE=true
            shift
            ;;
        --)
            shift
            EXEC_CMD=("$@")
            break
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [-- COMMAND [ARGS...]]"
            echo ""
            echo "Options:"
            echo "  --rebuild     Stop/remove container if present, then rebuild image (no cache)"
            echo "  --editable    Mount host repo over /workspace for live code editing"
            echo "  -h, --help    Show this help"
            echo ""
            echo "If COMMAND is provided after --, it runs inside the container instead of bash."
            echo ""
            echo "Pipeline commands (chainable via --from-* flags, run after --):"
            echo "  resolve-kernel-url <url> --json -o resolved.json"
            echo "  test-discovery --from-resolved resolved.json -o discovery.json"
            echo "  kernel-profile --from-discovery discovery.json --json -o profile.json"
            echo "  baseline-metrics build --from-profile profile.json --all -o baseline_metrics.json"
            echo "  commandment --from-discovery discovery.json -o COMMANDMENT.md"
            echo "  task-generator --from-discovery discovery.json --profiling profile.json \\"
            echo "      --commandment COMMANDMENT.md --baseline-metrics baseline_metrics.json -o tasks/round_1/"
            echo ""
            echo "Iterative refinement (round 2+):"
            echo "  task-generator ... --from-results results/round_1/ --round 2 -o tasks/round_2/"
            echo ""
            echo "Run individual tasks:"
            echo "  geak --from-task tasks/round_1/10_triton-autotune.md --gpu-ids 2"
            echo ""
            echo "Other tools:"
            echo "  validate-commandment <path>            Validate COMMANDMENT.md"
            echo "  select-patch --patch-dir <dir> ...     Select best patch from runs"
            echo "  geak <github_url>                      Full optimization pipeline"
            echo ""
            echo "Requires: AMD_LLM_API_KEY environment variable"
            echo ""
            echo "USER and GEAK_USER are forwarded from the host so the AMD LLM gateway"
            echo "can attribute requests; existing containers must be rebuilt with"
            echo "--rebuild to pick up the change."
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use -- before commands)"
            exit 1
            ;;
    esac
done

#######################################
# Rebuild: stop and remove container, then rebuild image (fresh, no cache)
#######################################
if [ "$REBUILD" = true ]; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping container ${CONTAINER_NAME}..."
        docker stop ${CONTAINER_NAME}
    fi
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Removing container ${CONTAINER_NAME}..."
        docker rm ${CONTAINER_NAME}
    fi
    echo "Rebuilding image ${IMAGE_NAME} (--no-cache, BASE_IMAGE=${BASE_IMAGE})..."
    docker build --network=host --no-cache --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t ${IMAGE_NAME} .
    echo ""
fi

# Helper: exec into the container with the chosen command (or bash)
_exec_into_container() {
    if [[ ${#EXEC_CMD[@]} -gt 0 ]]; then
        echo "Running: ${EXEC_CMD[*]}"
        docker exec ${CONTAINER_NAME} "${EXEC_CMD[@]}"
        exit $?
    else
        exec docker exec -it ${CONTAINER_NAME} bash
    fi
}

# If container already exists (running or stopped), just use it -- no pre-flight needed
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running."
    _exec_into_container
fi
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} exists but is stopped. Restarting..."
    docker start ${CONTAINER_NAME}
    _exec_into_container
fi

#######################################
# Pre-flight checks (only when creating a new container)
#######################################
echo "Checking environment configuration..."

# Check for required AMD_LLM_API_KEY
if [ -z "$AMD_LLM_API_KEY" ]; then
    echo ""
    echo "❌ ERROR: AMD_LLM_API_KEY environment variable is not set!"
    echo ""
    echo "The GEAK-agent requires an API key to function."
    echo ""
    echo "To fix this:"
    echo "  export AMD_LLM_API_KEY=your-api-key-here"
    echo ""
    echo "Optionally, you can also set:"
    echo "  export AMD_LLM_BASE_URL=https://your-llm-gateway-url"
    echo "  (default: https://llm-gateway-dev.apps.amdcloud.com/api/gateway/v1)"
    echo ""
    exit 1
fi

# Show what we're using (first 20 chars only for security)
echo "✅ AMD_LLM_API_KEY: ${AMD_LLM_API_KEY:0:20}..."
if [ -n "$AMD_LLM_BASE_URL" ]; then
    echo "✅ AMD_LLM_BASE_URL: $AMD_LLM_BASE_URL"
else
    echo "ℹ️  AMD_LLM_BASE_URL: (using default)"
fi
if [ -n "$GEAK_MODEL" ]; then
    echo "✅ GEAK_MODEL: $GEAK_MODEL"
else
    GEAK_MODEL="claude-opus-4.6"
    echo "ℹ️  GEAK_MODEL: (using default: $GEAK_MODEL)"
fi
echo ""

# Check if image exists, build if not (unless we already rebuilt)
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "Image ${IMAGE_NAME} not found. Building (BASE_IMAGE=${BASE_IMAGE})..."
    docker build --network=host --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t ${IMAGE_NAME} .
elif [ "$REBUILD" != true ]; then
    echo "Using existing image ${IMAGE_NAME}"
    echo "To rebuild from scratch, run: $0 --rebuild"
fi

# Build volume mount args; --editable overlays host repo for live code editing
VOLUME_ARGS=(
    -v /cephfs:/cephfs
    -v "${PARENT_DIR}:${PARENT_DIR}"
    -v /mnt:/mnt
    -v /shared-nfs:/shared-nfs
    -v /shared-aig:/shared-aig
)
EDITABLE_ENV=()
if [ "$EDITABLE" = true ]; then
    echo "Editable mode: mounting ${REPO_ROOT} -> /workspace"
    VOLUME_ARGS+=(-v "${REPO_ROOT}:/workspace")
    EDITABLE_ENV=(-e GEAK_EDITABLE=1)
fi

# Run new container in detached mode with persistent process
echo "Creating and starting new container ${CONTAINER_NAME}..."
docker run -d \
    --name ${CONTAINER_NAME} \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -e AMD_LLM_API_KEY="${AMD_LLM_API_KEY}" \
    -e AMD_LLM_BASE_URL="${AMD_LLM_BASE_URL}" \
    -e GEAK_MODEL="${GEAK_MODEL}" \
    -e USER="${USER}" \
    -e GEAK_USER="${GEAK_USER:-${USER}}" \
    "${EDITABLE_ENV[@]}" \
    --shm-size 8G \
    "${VOLUME_ARGS[@]}" \
    -w /workspace \
    "${IMAGE_NAME}"

# Now exec into the running container
echo "Entering container ${CONTAINER_NAME}..."
_exec_into_container
