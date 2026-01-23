#!/bin/bash

# New parallel optimization script using auto-detect and config file features
# Usage: ./batch_patch_auto.sh <kernel_name> [num_agents]
# This script leverages the new three-tier configuration system:
#   1. Command-line arguments (highest priority)
#   2. Config file extra_config section
#   3. Auto-detect from task file (lowest priority)

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
NUM_AGENTS=${2:-4}

# Get the number of available GPUs
NUM_GPUS=8
GPU_START_IDX=0

# Generate GPU IDs list for parallel agents
GPU_IDS=()
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    gpu_id=$((i % NUM_GPUS + GPU_START_IDX))
    GPU_IDS+=(${gpu_id})
done

# Convert GPU IDs array to comma-separated string
GPU_IDS_STR=$(IFS=','; echo "${GPU_IDS[*]}")

echo "======================================"
echo "Starting auto-detect parallel optimization"
echo "Kernel: ${KERNEL_NAME}"
echo "Parallel agents: ${NUM_AGENTS}"
echo "GPU IDs: ${GPU_IDS_STR}"
echo "======================================"

# Setup paths
TEST_DIR="/home/yueliu14/mini-swe-agent"
ROCPRIM_DIR="${TEST_DIR}/rocPRIM_${KERNEL_NAME}"
OUTPUT_REPO="$(date +%Y%m%d_%H%M%S)_${KERNEL_NAME}"
PATCH_OUTPUT_DIR="${TEST_DIR}/optimization_logs/${OUTPUT_REPO}"
PROMPT_FILE="${TEST_DIR}/rocprim_prompts/rocprim_prompt_${KERNEL_NAME}.md"

# Create config file with project defaults
CONFIG_FILE="${TEST_DIR}/config_${KERNEL_NAME}.yaml"

cat > "${CONFIG_FILE}" << EOF
# Auto-generated config for ${KERNEL_NAME} optimization
agent:
  system_template: |
    You are a helpful assistant that can interact with a computer.
    Your response must contain exactly ONE bash code block with ONE command.
    Include a THOUGHT section before your command.
  instance_template: |
    Please solve this issue: {{task}}
    You can execute bash commands and edit files to implement the necessary changes.
  step_limit: 0
  cost_limit: 0
  mode: confirm

env:
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: 'off'
    TQDM_DISABLE: '1'
  timeout: 3600

model:
  model_class: amd_llm
  model_name: claude-opus-4.5
  api_key: ""
  model_kwargs:
    temperature: 0.0
    max_tokens: 16000

# Project-specific defaults (will be used if not provided via command-line)
extra_config:
  repo: ${ROCPRIM_DIR}
  test_command: cd ${TEST_DIR}/test_scripts && python test_correctness_benchmark.py benchmark_${KERNEL_NAME} WORK_REPO
  metric: "Extract bytes_per_second G/s from test output. Convert other units (T/s, M/s) to G/s. Calculate average speedup ratio across all datatypes."
  num_parallel: ${NUM_AGENTS}
  gpu_ids: [$(echo ${GPU_IDS_STR} | sed 's/,/, /g')]
  patch_output_dir: ${PATCH_OUTPUT_DIR}
EOF

echo "Generated config file: ${CONFIG_FILE}"
echo ""

# Create directories
mkdir -p "${TEST_DIR}/optimization_logs"
if [ -d "${PATCH_OUTPUT_DIR}" ]; then
    echo "Removing existing output directory: ${PATCH_OUTPUT_DIR}"
    rm -rf "${PATCH_OUTPUT_DIR}"
fi
mkdir -p "${PATCH_OUTPUT_DIR}"

# Check if prompt file exists
if [ ! -f "${PROMPT_FILE}" ]; then
    echo "Error: Prompt file not found: ${PROMPT_FILE}"
    echo "Please create: ${PROMPT_FILE}"
    exit 1
fi

# Clone rocPRIM repository if it doesn't exist
if [ -d "${ROCPRIM_DIR}" ]; then
    echo "Repository already exists: ${ROCPRIM_DIR}"
else
    echo "Cloning rocPRIM repository..."
    git clone https://github.com/ROCm/rocPRIM.git "${ROCPRIM_DIR}" > /dev/null 2>&1
    echo "Repository cloned to: ${ROCPRIM_DIR}"
fi

echo ""
echo "======================================"
echo "Running mini with auto-detect..."
echo "======================================"
echo ""
echo "Configuration priority:"
echo "  1. Command-line args (highest)"
echo "  2. Config file (${CONFIG_FILE})"
echo "  3. Auto-detect from task (${PROMPT_FILE})"
echo ""

# Change to repository directory
cd "${ROCPRIM_DIR}"

# Run mini command - Much simpler now!
# The config file provides all defaults, auto-detect can fill in any gaps
mini -c "${CONFIG_FILE}" \
    -t "${PROMPT_FILE}" \
    --yolo \
    > "${PATCH_OUTPUT_DIR}/mini_output.log" 2>&1

EXIT_CODE=$?

cd "${TEST_DIR}"

echo ""
echo "======================================"
echo "Optimization completed!"
echo "======================================"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Success!"
else
    echo "✗ Failed with exit code: ${EXIT_CODE}"
    echo "Check log: ${PATCH_OUTPUT_DIR}/mini_output.log"
fi

echo ""
echo "Kernel: ${KERNEL_NAME}"
echo "Results directory: ${PATCH_OUTPUT_DIR}"
echo "  - Each agent's results: ${PATCH_OUTPUT_DIR}/parallel_{0..$((NUM_AGENTS-1))}/"
echo "  - Best patch selection: ${PATCH_OUTPUT_DIR}/best_results.json"
echo "  - GPU IDs used: ${GPU_IDS_STR}"
echo "  - Output log: ${PATCH_OUTPUT_DIR}/mini_output.log"
echo "  - Config used: ${CONFIG_FILE}"
echo ""

# Cleanup config file
if [ -f "${CONFIG_FILE}" ]; then
    echo "Cleaning up temporary config file..."
    rm "${CONFIG_FILE}"
fi

echo "Done!"
