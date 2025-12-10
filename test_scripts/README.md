# Patch Agent

## Introduction

The **Patch Agent** is an enhanced version of the mini-swe-agent that automatically saves git patches and runs tests during code optimization tasks. It's designed for iterative development workflows where you need to:

- Track code changes as git patches
- Validate each change with automated tests
- Extract performance metrics from test outputs
- Automatically select the best-performing patch

The agent monitors the agent's output for a special `SAVE_PATCH_AND_TEST` command. When detected, it captures the current git diff as a patch file, runs your specified test command, and optionally extracts metrics to help identify the best optimization.


## Usage

### Basic Command

```bash
mini --save-patch \
    --test-command "your test command here, e.g ./kernel_test, it is good to use absolute path or cd to the directory of the test command" \
    --patch-output "the directory to save the patches" \
    --metric "the metric to extract from the test output, how to select the best patch" \
    -c "the config file, you should use mini_patch_agent.yaml for patch saving mode" \
    -o "trajectory.traj.json" \
    -t "task_prompt.md"
```

### Required Arguments

- `--save-patch`: Enables patch agent mode
- `--test-command`: Command to run for testing (use absolute paths or `cd` to test directory)
- `--patch-output`: Directory where patches and test results will be saved
- `--metric`: Description of metric to extract from test output (enables best patch selection)


## How It Works

1. **Agent Execution**: The agent runs normally, making code changes
2. **Patch Trigger**: When the agent outputs `SAVE_PATCH_AND_TEST` as the first line of a command, the patch agent activates
3. **Patch Capture**: The agent runs `git diff` to capture current changes
4. **Test Execution**: Runs your specified test command
5. **Metric Extraction** (if `--metric` provided): Uses LLM to extract metrics from test output
6. **File Saving**: Saves patch file, test output, and updates results.json
7. **Best Patch Selection** (if multiple patches): After all patches are created, LLM selects the best one

## Examples

### Example 1: Performance Benchmarking with Metrics

Extract performance metrics and select the best patch:

```bash
mini --save-patch \
     --test-command "cd /path/to/benchmark && python benchmark.py --output json" \
     --patch-output "./optimization_patches" \
     --metric "extract throughput (operations per second) from the JSON output" \
     -c "config.yaml" \
     -t "optimize_kernel.md"
```

### Example 2: Complex Benchmark with Multiple Datatypes

From the test script, here's a real-world example for GPU kernel optimization:

```bash
PATCH_OUTPUT_DIR="./test_patches_merge_sort"
TRAJ_OUTPUT="./test_trajectory_merge_sort.traj.json"
CONFIG="/path/to/mini.yaml"

mini --save-patch \
     --test-command "cd /path/to/test_scripts && python test_benchmark.py benchmark_device_merge_sort /path/to/build /path/to/test_patch" \
     --patch-output "$PATCH_OUTPUT_DIR" \
     -c "$CONFIG" \
     -o "$TRAJ_OUTPUT" \
     -t "/path/to/prompt.md" \
     --metric "extract bytes_per_second G/s from test output, note you should change T/s or other units to G/s. To select the best patch, you should calculate the speedup on all datatypes first and get the average speedup. Not the average of bandwidths on all datatypes."
```

## Output Structure

The patch output directory will contain:

```
patches/
├── patch_0.patch          # Baseline (no changes)
├── patch_0_test.txt       # Baseline test output
├── patch_1.patch          # First optimization attempt
├── patch_1_test.txt       # First test output
├── patch_2.patch          # Second optimization attempt
├── patch_2_test.txt       # Second test output
└── results.json           # Summary of all patches and metrics
```

### results.json Format

```json
{
  "patch_0": {
    "patch_file": "patch_0.patch",
    "test_output_file": "patch_0_test.txt",
    "test_passed": true,
    "returncode": 0,
    "metric_result": {"values": [10.5, 12.3, 11.8]}
  },
  "patch_1": {
    "patch_file": "patch_1.patch",
    "test_output_file": "patch_1_test.txt",
    "test_passed": true,
    "returncode": 0,
    "metric_result": {"values": [15.2, 16.8, 15.9]}
  },
  "_best_patch": "patch_1"
}
```

## See Also

- `test_patch_agent.sh`: Example script demonstrating patch agent usage
- `patch_agent.py`: Source code for the patch agent implementation

