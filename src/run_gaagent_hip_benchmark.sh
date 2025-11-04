#!/bin/bash

# Define base output path
OUTPUT_BASE="/home/username/github_release"

cd ../example_hip

# Run the extraction and capture the number of files
output=$(python extract_single.py)
num_files=$(echo "$output" | grep -oP '(?<=Split into )\d+(?= files)')

echo "Number of extracted files: $num_files"

cd ../src

# Loop from 1 to num_files
for i in $(seq 1 $num_files); do

  # Restore the example_hip directory to the last committed version
  git restore ../example_hip/mmcv
  git restore ../example_hip/point_to_voxel
  git restore ../example_hip/rocm-examples
  git restore ../example_hip/silu

  config="configs/hipbench_gaagent_config.yaml"

  # Overwrite the config file
  sed \
    -e "s|instruction_path: \".*FromRe_instructions.*\.json\"|instruction_path: \"FromRe_instructions_${i}.json\"|" \
    -e "s|output_path: \".*iteration_logs.*\/iter\"|output_path: \"${OUTPUT_BASE}/iteration_logs_${i}/iter\"|" \
    "$config" > tmp_config.yaml

  mv tmp_config.yaml "$config"

  # Create iteration_logs_i folder
  mkdir -p "${OUTPUT_BASE}/iteration_logs_${i}"

  echo "Running iteration $i with updated config..."

  # Run your Python program and save logs
  python main_gaagent_hip.py 2>&1 | tee "log_ext_${i}.txt"
done
