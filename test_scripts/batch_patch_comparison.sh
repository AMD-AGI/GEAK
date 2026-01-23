#!/bin/bash

# Comparison script showing OLD vs NEW way of running parallel optimization
# This demonstrates the improvement from using auto-detect and config files

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <kernel_name> [mode]"
    echo "  mode: 'old' or 'new' (default: new)"
    echo "Example: $0 device_merge_sort new"
    exit 1
fi

KERNEL_NAME=$1
MODE=${2:-new}
NUM_AGENTS=4

# Setup paths
TEST_DIR="/home/yueliu14/mini-swe-agent"
ROCPRIM_DIR="${TEST_DIR}/rocPRIM_${KERNEL_NAME}"
PROMPT_FILE="${TEST_DIR}/rocprim_prompts/rocprim_prompt_${KERNEL_NAME}.md"
BASE_CONFIG="/home/yueliu14/mini-swe-agent/src/minisweagent/config/mini_system_prompt.yaml"

# Generate GPU IDs
GPU_IDS="0,1,2,3"

# Create timestamp-based output directory
OUTPUT_DIR="${TEST_DIR}/optimization_logs/$(date +%Y%m%d_%H%M%S)_${KERNEL_NAME}_${MODE}"
mkdir -p "${OUTPUT_DIR}"

echo "======================================"
echo "Parallel Optimization Comparison"
echo "======================================"
echo "Kernel: ${KERNEL_NAME}"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================"
echo ""

# Clone repo if needed
if [ ! -d "${ROCPRIM_DIR}" ]; then
    echo "Cloning rocPRIM..."
    git clone https://github.com/ROCm/rocPRIM.git "${ROCPRIM_DIR}" > /dev/null 2>&1
fi

cd "${ROCPRIM_DIR}"

if [ "${MODE}" == "old" ]; then
    echo "=== OLD WAY: Manual specification of all parameters ==="
    echo ""
    cat << 'EOF'
mini -c "${BASE_CONFIG}" \
    --task "${PROMPT_FILE}" \
    --yolo \
    --save-patch \
    --output "${OUTPUT_DIR}/traj.json" \
    --num-parallel 4 \
    --repo "${ROCPRIM_DIR}" \
    --patch-output "${OUTPUT_DIR}" \
    --test-command "cd /home/yueliu14/mini-swe-agent/test_scripts && python test_correctness_benchmark.py benchmark_${KERNEL_NAME} WORK_REPO" \
    --metric "extract bytes_per_second G/s from test output, convert units, calculate average speedup" \
    --parallel-gpu-ids "0,1,2,3"
EOF
    echo ""
    echo "Running..."
    
    mini -c "${BASE_CONFIG}" \
        --task "${PROMPT_FILE}" \
        --yolo \
        --save-patch \
        --output "${OUTPUT_DIR}/traj.json" \
        --num-parallel 4 \
        --repo "${ROCPRIM_DIR}" \
        --patch-output "${OUTPUT_DIR}" \
        --test-command "cd ${TEST_DIR}/test_scripts && python test_correctness_benchmark.py benchmark_${KERNEL_NAME} WORK_REPO" \
        --metric "extract bytes_per_second G/s from test output, convert units to G/s, calculate average speedup ratio" \
        --parallel-gpu-ids "${GPU_IDS}" \
        > "${OUTPUT_DIR}/mini_output.log" 2>&1
    
elif [ "${MODE}" == "new" ]; then
    echo "=== NEW WAY: Using config file + auto-detect ==="
    echo ""
    
    # Create config file with defaults
    CONFIG_FILE="${TEST_DIR}/config_${KERNEL_NAME}_temp.yaml"
    cat > "${CONFIG_FILE}" << EOF
agent:
  system_template: |
    You are a helpful assistant that can interact with a computer.
    Your response must contain exactly ONE bash code block with ONE command.
  instance_template: |
    Please solve this issue: {{task}}
  step_limit: 0
  cost_limit: 0
  mode: confirm

env:
  timeout: 3600

model:
  model_class: amd_llm
  model_name: claude-opus-4.5

extra_config:
  repo: ${ROCPRIM_DIR}
  test_command: cd ${TEST_DIR}/test_scripts && python test_correctness_benchmark.py benchmark_${KERNEL_NAME} WORK_REPO
  num_parallel: ${NUM_AGENTS}
  gpu_ids: [0, 1, 2, 3]
EOF
    
    cat << EOF
mini -c "${CONFIG_FILE}" \\
    -t "${PROMPT_FILE}" \\
    --yolo
EOF
    echo ""
    echo "That's it! Everything else comes from:"
    echo "  - Config file: repo, test_command, num_parallel, gpu_ids"
    echo "  - Auto-detect: metric, kernel_name"
    echo ""
    echo "Running..."
    
    mini -c "${CONFIG_FILE}" \
        -t "${PROMPT_FILE}" \
        --yolo \
        > "${OUTPUT_DIR}/mini_output.log" 2>&1
    
    # Cleanup temp config
    rm "${CONFIG_FILE}"
    
else
    echo "Error: Unknown mode '${MODE}'. Use 'old' or 'new'"
    exit 1
fi

cd "${TEST_DIR}"

echo ""
echo "======================================"
echo "Completed!"
echo "======================================"
echo ""
echo "Results: ${OUTPUT_DIR}"
echo ""
echo "Command comparison:"
echo "  OLD: 13 parameters to specify manually"
echo "  NEW: 3 parameters (config, task, yolo)"
echo ""
echo "Improvement: 77% reduction in command complexity!"
