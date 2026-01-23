#!/bin/bash

# Test script for auto-detect feature (enabled by default)

echo "========================================"
echo "Testing Auto-Detect Feature"
echo "========================================"
echo ""

# Test 1: Basic auto-detection (default behavior)
echo "Test 1: Basic auto-detection with example task (default)"
echo "Command: mini -t examples/optimization_task_example.md --yolo"
echo ""
# Uncomment to run:
# mini -t examples/optimization_task_example.md --yolo

# Test 2: Auto-detection with override
echo "Test 2: Auto-detection with manual override"
echo "Command: mini -t examples/optimization_task_example.md --num-parallel 8 --yolo"
echo ""
# Uncomment to run:
# mini -t examples/optimization_task_example.md --num-parallel 8 --yolo

# Test 3: Auto-detection with inline task
echo "Test 3: Auto-detection with inline task"
echo 'Command: mini -t "Optimize GEMM kernel in /path/to/repo. Test with: python test.py. Measure throughput in GFLOPS. Use 4 GPUs: 0,1,2,3" --yolo'
echo ""
# Uncomment to run:
# mini -t "Optimize GEMM kernel in /path/to/repo. Test with: python test.py. Measure throughput in GFLOPS. Use 4 GPUs: 0,1,2,3" --yolo

# Test 4: Disable auto-detection
echo "Test 4: Disable auto-detection"
echo "Command: mini --no-auto-detect -t examples/optimization_task_example.md --yolo"
echo ""
# Uncomment to run:
# mini --no-auto-detect -t examples/optimization_task_example.md --yolo

echo ""
echo "========================================"
echo "Uncomment the commands in this script to run actual tests"
echo "========================================"
