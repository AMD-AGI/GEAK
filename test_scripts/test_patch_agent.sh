#!/bin/bash

# save path
PATCH_OUTPUT_DIR="./test_patches_merge_sort"
TRAJ_OUTPUT="./test_trajectory_merge_sort.traj.json"
CONFIG="/data/users/yueliu14/ready_mini_swe/mini-swe-agent/src/minisweagent/config/mini_patch_agent.yaml"

rm -rf "$PATCH_OUTPUT_DIR" "$TRAJ_OUTPUT"

echo "Patch output directory: $PATCH_OUTPUT_DIR"
echo "Trajectory output: $TRAJ_OUTPUT"
echo ""

# examples

mini --save-patch \
     --test-command "cd /data/users/yueliu14/mini-sweagent/case_study/rocprim/test_scripts && python test_benchmark.py benchmark_device_merge_sort /data/users/yueliu14/mini-sweagent/case_study/rocprim/rocPRIM_block_histogram/build /data/users/yueliu14/mini-sweagent/case_study/rocprim/test_patch" \
     --patch-output "$PATCH_OUTPUT_DIR" \
     -c "$CONFIG" \
     -o "$TRAJ_OUTPUT" \
     -t "/data/users/yueliu14/ready_mini_swe/mini-swe-agent/rocprim_prompt_patch_test.md" \
     --metric "extract bytes_per_second G/s from test output, note you should change T/s or other units to G/s. To select the best patch, you should calculate the speedup on all datatypes first and get the average speedup. Not the average of bandwidths on all datatypes." > test_patch_with_traj.log 2>&1
