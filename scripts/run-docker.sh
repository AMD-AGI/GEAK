# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
# Build and run GEAK-agent Docker container.
#
# Usage:
#   scripts/run-docker.sh                          # Interactive bash shell
#   scripts/run-docker.sh --rebuild                # Rebuild image, then bash
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
#   openevolve-worker --from-task tasks/round_1/00_openevolve-inner.md --gpu 0
#   geak --from-task tasks/round_1/10_triton-autotune.md --gpu-ids 2
#
# Other tools:
#   validate-commandment <path>              Validate a COMMANDMENT.md
#   openevolve-worker --kernel-path <p> ...  Run OpenEvolve optimizer (manual)
#   select-patch --patch-dir <dir> ...       Select best patch from runs
#   geak <github_url>                        Full optimization pipeline

set -e

IMAGE_NAME="geak-agent:latest"
CONTAINER_NAME="geak-agent-${USER}"

# Repo root (directory containing scripts/); mount this over /workspace for live code without rebuild
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Set directories to user's home
PARENT_DIR="${HOME}"
HOST_CODE_DIR="${HOME}"

REBUILD=false
EXEC_CMD=()  # Command to run inside the container (empty = bash)

#######################################
# Parse options
#######################################
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
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
            echo "  openevolve-worker --from-task tasks/round_1/00_openevolve-inner.md --gpu 0"
            echo "  geak --from-task tasks/round_1/10_triton-autotune.md --gpu-ids 2"
            echo ""
            echo "Other tools:"
            echo "  validate-commandment <path>            Validate COMMANDMENT.md"
            echo "  openevolve-worker --kernel-path <p> ...  Run OpenEvolve optimizer (manual)"
            echo "  select-patch --patch-dir <dir> ...     Select best patch from runs"
            echo "  geak <github_url>                      Full optimization pipeline"
            echo ""
            echo "Requires: AMD_LLM_API_KEY environment variable"
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
    echo "Rebuilding image ${IMAGE_NAME} (--no-cache)..."
    docker build --no-cache -t ${IMAGE_NAME} .
    echo ""
fi

# Helper: exec into the container with the chosen command (or bash)
_exec_into_container() {
    if [[ ${#EXEC_CMD[@]} -gt 0 ]]; then
        echo "Running: ${EXEC_CMD[*]}"
        exec docker exec -it ${CONTAINER_NAME} "${EXEC_CMD[@]}"
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
    GEAK_MODEL="claude-sonnet-4.5"
    echo "ℹ️  GEAK_MODEL: (using default: $GEAK_MODEL)"
fi
echo ""

# Check if image exists, build if not (unless we already rebuilt)
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "Image ${IMAGE_NAME} not found. Building..."
    docker build -t ${IMAGE_NAME} .
elif [ "$REBUILD" != true ]; then
    echo "Using existing image ${IMAGE_NAME}"
    echo "To rebuild from scratch, run: $0 --rebuild"
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
    -v /cephfs:/cephfs \
    --shm-size 8G \
    -v "${REPO_ROOT}:/workspace" \
    -v ${PARENT_DIR}:${PARENT_DIR} \
    -v /mnt:/mnt \
    -v /shared-nfs:/shared-nfs \
    -v /shared-aig:/shared-aig \
    -w /workspace \
    ${IMAGE_NAME}

# Now exec into the running container
echo "Entering container ${CONTAINER_NAME}..."
_exec_into_container
