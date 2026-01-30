#!/bin/bash

# Parallel optimization script for a single kernel with multiple agents
# Usage: ./batch_test_patch_parallel.sh <kernel_name> [num_agents]
# Note: mini-swe-agent handles parallel execution internally, this script just sets up the config

set -e  # Exit on error

# Check if kernel name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <kernel_name> [num_agents]"
    echo "Example: $0 device_merge_sort 4"
    exit 1
fi

# Kernel name
KERNEL_NAME=$1

# Number of parallel optimization agents (default: 4)
NUM_AGENTS=4

# Get the number of available GPUs
NUM_GPUS=8
GPU_START_IDX=${2:-0}

# Generate GPU IDs list for parallel agents
GPU_IDS=()
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    gpu_id=$((i % NUM_GPUS + GPU_START_IDX))
    GPU_IDS+=(${gpu_id})
done

# Convert GPU IDs array to comma-separated string for command line
GPU_IDS_STR=$(IFS=','; echo "${GPU_IDS[*]}")

echo "======================================"
echo "Starting parallel optimization for kernel: ${KERNEL_NAME}"
echo "Number of parallel agents: ${NUM_AGENTS}"
echo "GPU IDs: ${GPU_IDS_STR}"
echo "======================================"

# Setup paths
TEST_DIR="/mnt/raid0/yueliu14/mini-swe-agent/"
ROCPRIM_DIR="${TEST_DIR}/rocPRIM_${KERNEL_NAME}"
OUTPUT_REPO="20260106_${KERNEL_NAME}"
PATCH_OUTPUT_DIR="${TEST_DIR}/${OUTPUT_REPO}"
TRAJ_PATH="${PATCH_OUTPUT_DIR}/last_mini_run.traj.json"
PROMPT_FILE="/mnt/raid0/yueliu14/mini-swe-agent/rocprim_prompts/rocprim_prompt_${KERNEL_NAME}.md"
BASE_CONFIG="/mnt/raid0/yueliu14/mini-swe-agent/src/minisweagent/config/mini_patch_agent.yaml"

# Create directories
mkdir -p "${TEST_DIR}"
if [ -d "${PATCH_OUTPUT_DIR}" ]; then
    echo "Removing existing patch output directory: ${PATCH_OUTPUT_DIR}"
    rm -rf "${PATCH_OUTPUT_DIR}"
fi
mkdir -p "${PATCH_OUTPUT_DIR}"

# Check if prompt file exists
if [ ! -f "${PROMPT_FILE}" ]; then
    echo "Error: Prompt file not found: ${PROMPT_FILE}"
    exit 1
fi

# Clone rocPRIM repository if it doesn't exist
if [ -d "${ROCPRIM_DIR}" ]; then
    echo "Repository already exists: ${ROCPRIM_DIR}"
else
    echo "Cloning rocPRIM repository..."
    git clone https://github.com/ROCm/rocPRIM.git "${ROCPRIM_DIR}" > /dev/null 2>&1
fi

echo ""
echo "Running mini command with ${NUM_AGENTS} parallel agents..."
echo ""

# Run mini command - it will handle parallel execution internally
cd "${ROCPRIM_DIR}"

mini -c "${BASE_CONFIG}" \
    --task "${PROMPT_FILE}" \
    --yolo \
    --save-patch \
    --output "${TRAJ_PATH}" \
    --num-parallel "${NUM_AGENTS}" \
    --repo "${ROCPRIM_DIR}" \
    --patch-output "${PATCH_OUTPUT_DIR}" \
    --test-command "cd /mnt/raid0/yueliu14/mini-swe-agent/test_scripts && python test_correctness_benchmark.py benchmark_${KERNEL_NAME} WORK_REPO" \
    --metric "extract bytes_per_second G/s from test output, note you should change T/s or other units to G/s. To select the best patch, you should calculate the speedup ratio on all datatypes first and get the average speedup ratio." \
    --parallel-gpu-ids "${GPU_IDS_STR}" \
    > "${PATCH_OUTPUT_DIR}/mini_output.log" 2>&1

cd "${TEST_DIR}"

echo ""
echo "======================================"
echo "Parallel optimization completed!"
echo "======================================"
echo ""
echo "Summary for kernel: ${KERNEL_NAME}"
echo "Results are stored in: ${PATCH_OUTPUT_DIR}"
echo "  Each agent's results: ${PATCH_OUTPUT_DIR}/parallel_{0..$((NUM_AGENTS-1))}/"
echo "  GPU IDs used: ${GPU_IDS_STR}"
echo ""
