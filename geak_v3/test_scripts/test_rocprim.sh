#!/bin/bash

set -e

KERNEL_NAME="device_binary_search"

echo "======================================"
echo "Optimizing device_binary_search"
echo "======================================"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
ROCPRIM_DIR="${PROJECT_DIR}/rocPRIM_${KERNEL_NAME}"
TASK_FILE="${PROJECT_DIR}/test_prompts/rocprim_device_binay_search.md"

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

echo ""
echo "Running miniswe..."
echo ""

mkdir -p "./optimization_logs"
# Run with yolo mode (End to end without confirmation)
mini -t "${TASK_FILE}" --yolo > "./optimization_logs/rocprim_device_binary_search.txt" 2>&1

# Run with confirm mode (Interactive mode)
# mini -t "${TASK_FILE}" > "$./optimization_logs/rocprim_device_binary_search.txt" 2>&1

echo ""
echo "Done!"
