#!/bin/bash

# Minimal parallel optimization script using only auto-detect
# This demonstrates the most concise way to run parallel optimization
# All configuration is auto-detected from the task file!

set -e

# if [ $# -eq 0 ]; then
#     echo "Usage: $0 <kernel_name> [num_agents] [gpu_start_idx]"
#     echo "Example: $0 device_merge_sort 4 0"
#     exit 1
# fi

KERNEL_NAME=$1

echo "======================================"
echo "Minimal Auto-Detect Optimization"
echo "Kernel: ${KERNEL_NAME}"
echo "Agents: ${NUM_AGENTS}"
echo "======================================"

# Paths
TEST_DIR="/mnt/raid0/yueliu14/mini-swe-agent"
ROCPRIM_DIR="${TEST_DIR}/rocPRIM_${KERNEL_NAME}"
TASK_FILE="${TEST_DIR}/rocprim_prompts/rocprim_prompt_${KERNEL_NAME}.md"
CONFIG_FILE="${TEST_DIR}/src/minisweagent/config/mini_system_prompt.yaml"

# Check task file
if [ ! -f "${TASK_FILE}" ]; then
    echo "Error: Task file not found: ${TASK_FILE}"
    exit 1
fi

# Clone repo if needed
if [ ! -d "${ROCPRIM_DIR}" ]; then
    echo "Cloning rocPRIM..."
    git clone https://github.com/ROCm/rocPRIM.git "${ROCPRIM_DIR}" > /dev/null 2>&1
fi

# Generate GPU IDs
GPU_IDS=()
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    GPU_IDS+=($((i % 8 + GPU_START_IDX)))
done
GPU_IDS_STR=$(IFS=','; echo "${GPU_IDS[*]}")

echo ""
echo "Running mini with full auto-detect..."
echo ""

# This is the SIMPLEST possible command!
# Everything else is auto-detected from the task file
mini -t "${TASK_FILE}" \
    # --yolo
# Optional: add --yolo for auto-confirmation mode

echo ""
echo "Done! Check optimization_logs/ for results"
