#!/bin/bash
# Simple script to build and run GEAK-agent Docker container

set -e

IMAGE_NAME="geak-agent:latest"
CONTAINER_NAME="geak-agent-${USER}"

# Set directories to user's home
PARENT_DIR="${HOME}"
HOST_CODE_DIR="${HOME}"

# Check if image exists, build if not
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "Image ${IMAGE_NAME} not found. Building..."
    docker build -t ${IMAGE_NAME} .
else
    echo "Using existing image ${IMAGE_NAME}"
    echo "To rebuild, run: docker build -t ${IMAGE_NAME} ."
fi

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running. Executing bash..."
    exec docker exec -it ${CONTAINER_NAME} bash
fi

# Check if stopped container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} exists but is stopped. Restarting..."
    docker start ${CONTAINER_NAME}
    exec docker exec -it ${CONTAINER_NAME} bash
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
    -v /cephfs:/cephfs \
    --shm-size 8G \
    -v ${PARENT_DIR}:${PARENT_DIR} \
    -v /mnt:/mnt \
    -v /shared-nfs:/shared-nfs \
    -v /shared-aig:/shared-aig \
    -w ${HOST_CODE_DIR} \
    ${IMAGE_NAME} \
    tail -f /dev/null

# Now exec into the running container
echo "Entering container ${CONTAINER_NAME}..."
exec docker exec -it ${CONTAINER_NAME} bash
