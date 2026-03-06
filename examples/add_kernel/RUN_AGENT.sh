# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
# Standard test script for GEAK agent pipeline

set -e

echo "=========================================="
echo "GEAK Agent Pipeline Test"
echo "=========================================="

# Setup
cd "$(dirname "$0")/../.."

# Check for API key
if [ -z "$AMD_LLM_API_KEY" ]; then
    echo "Error: AMD_LLM_API_KEY environment variable not set"
    echo "Please set it with: export AMD_LLM_API_KEY='your-key-here'"
    exit 1
fi

export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Clean previous runs
if [ "$1" == "--clean" ]; then
    echo "Cleaning..."
    rm -rf examples/add_kernel/test_*.py
    rm -rf examples/add_kernel/benchmark*.py
    rm -rf examples/add_kernel/*_optimized.py
    echo "✓ Cleaned"
    echo ""
fi

echo "Starting agent..."
echo ""

python3 -m minisweagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "For examples/add_kernel/kernel.py, CREATE these Python scripts (don't run them):

1. test_add_kernel.py - test cases using kernel.triton_add and kernel.torch_add
2. benchmark_add_kernel.py - that uses StandardBenchmark from minisweagent.benchmark:
   - Import: from minisweagent.benchmark import StandardBenchmark
   - Create: benchmark = StandardBenchmark(kernel_path)
   - Run: benchmark.benchmark_kernel(...)
   - Save: benchmark.save_metrics(...) → benchmark/baseline/metrics.json

Just create the scripts. Don't try to run them or install dependencies." \
  --yolo

echo ""
echo "=========================================="
echo "Done! Check:"
echo "  examples/add_kernel/test_add_kernel.py"
echo "  examples/add_kernel/benchmark_add_kernel.py"
echo "=========================================="
