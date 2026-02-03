#!/bin/bash

set -e

KERNEL_NAME="silu"

echo "======================================"
echo "Optimizing SiLU kernel"
echo "======================================"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
TASK_FILE="${PROJECT_DIR}/test_prompts/prompt_silu.md"
SILU_DIR="${PROJECT_DIR}/test_silu"

# Check task file
if [ ! -f "${TASK_FILE}" ]; then
    echo "Error: Task file not found: ${TASK_FILE}"
    exit 1
fi

# Check silu directory
if [ ! -d "${SILU_DIR}" ]; then
    echo "Error: Silu directory not found: ${SILU_DIR}"
    exit 1
fi

echo ""
echo "Running miniswe on silu kernel..."
echo "Task file: ${TASK_FILE}"
echo "Working directory: ${SILU_DIR}"
echo ""

# Run with yolo mode (End to end without confirmation)
mini -t "${TASK_FILE}" --yolo > ./optimization_logs/mini_silu.log 2>&1

# Run with confirm mode (Interactive mode)
# mini -t "${TASK_FILE}" > ./optimization_logs/mini_silu.log 2>&1

echo ""
echo "Done! Check optimization results."
