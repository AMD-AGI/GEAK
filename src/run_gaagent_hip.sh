#!/bin/bash
set -e  # Exit immediately if any command fails

# Restore the example_hip directory to the last committed version
git restore ../example_hip/mmcv
git restore ../example_hip/point_to_voxel
git restore ../example_hip/rocm-examples
git restore ../example_hip/silu

# Run the Python script
python main_gaagent_hip.py