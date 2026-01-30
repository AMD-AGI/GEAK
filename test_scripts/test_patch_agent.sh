#!/bin/bash

# save path
PATCH_OUTPUT_DIR="./test_patches_merge_sort"
TRAJ_OUTPUT="./test_trajectory_merge_sort.traj.json"
CONFIG="/mnt/mnt/raid0/yueliu14/ready_mini_swe/mini-swe-agent/src/minisweagent/config/mini_patch_agent.yaml"

rm -rf "$PATCH_OUTPUT_DIR" "$TRAJ_OUTPUT"

echo "Patch output directory: $PATCH_OUTPUT_DIR"
echo "Trajectory output: $TRAJ_OUTPUT"
# examples

mini --save-patch \
     -c "$CONFIG" \
     -o "$TRAJ_OUTPUT" \
     -t "/mnt/mnt/raid0/yueliu14/ready_mini_swe/rocprim_test.md" > test_patch_agent.log 2>&1

echo "Finished"