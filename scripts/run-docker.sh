#!/bin/bash
# Simple script to build and run GEAK-agent Docker container

set -e

IMAGE_NAME="geak-agent:latest"
CONTAINER_NAME="geak-agent-${USER}"

# Set directories to user's home
PARENT_DIR="${HOME}"
HOST_CODE_DIR="${HOME}"

REBUILD=false

#######################################
# Parse options
#######################################
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rebuild     Stop/remove container if present, then rebuild image (no cache)"
            echo "  -h, --help    Show this help"
            echo ""
            echo "Requires: AMD_LLM_API_KEY environment variable"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
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

# If container already exists (running or stopped), just use it — no pre-flight needed
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running. Executing bash..."
    exec docker exec -it ${CONTAINER_NAME} bash
fi
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} exists but is stopped. Restarting..."
    docker start ${CONTAINER_NAME}
    exec docker exec -it ${CONTAINER_NAME} bash
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
else
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
    -v ${PARENT_DIR}:${PARENT_DIR} \
    -v /mnt:/mnt \
    -v /shared-nfs:/shared-nfs \
    -v /shared-aig:/shared-aig \
    -w ${HOST_CODE_DIR} \
    ${IMAGE_NAME}

# Now exec into the running container
echo "Entering container ${CONTAINER_NAME}..."
exec docker exec -it ${CONTAINER_NAME} bash
