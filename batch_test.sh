#!/bin/bash

TASK_FILE="GEAK_tasks.txt"
GPUS_PER_TASK=2
TOTAL_GPUS=8
START_GPU=0  # First GPU index to use
SKIP_GPUS=""  # Comma-separated GPU IDs to skip (e.g., "5" or "3,5,7")
LOG_DIR="logs"  # Directory for log files

# Disable cross-session (global) memory: don't read/write the shared KB.
# Within-session working memory remains on. Use GEAK_MEMORY_DISABLE=1 to kill both.
export GEAK_MEMORY_NO_CROSS_SESSION=1

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Build list of available GPUs (excluding skipped ones)
declare -a available_gpu_list=()
IFS=',' read -ra skip_arr <<< "$SKIP_GPUS"
for ((g = START_GPU; g < TOTAL_GPUS; g++)); do
    skip=false
    for s in "${skip_arr[@]}"; do
        if (( g == s )); then skip=true; break; fi
    done
    $skip || available_gpu_list+=("$g")
done

echo "Available GPUs: ${available_gpu_list[*]}"

# Associative array: PID -> comma-separated GPU IDs for that job
declare -A pid_to_gpus=()
# Queue of free GPU slots (each slot is one comma-separated GPU group)
declare -a free_gpu_slots=()

# Pre-fill GPU slots by grouping consecutive available GPUs
idx=0
while (( idx + GPUS_PER_TASK <= ${#available_gpu_list[@]} )); do
    gpu_ids="${available_gpu_list[$idx]}"
    for ((j = 1; j < GPUS_PER_TASK; j++)); do
        gpu_ids+=",${available_gpu_list[$((idx + j))]}"
    done
    free_gpu_slots+=("$gpu_ids")
    idx=$((idx + GPUS_PER_TASK))
done

MAX_PARALLEL=${#free_gpu_slots[@]}
echo "Max parallel tasks: $MAX_PARALLEL"
echo "GPU slots: ${free_gpu_slots[*]}"
echo ""

task_idx=0

while IFS=',' read -r repo_path kernel_url || [[ -n "$repo_path" ]]; do
    # Trim leading/trailing whitespace
    repo_path=$(echo "$repo_path" | xargs)
    kernel_url=$(echo "$kernel_url" | xargs)

    # Skip empty lines
    [[ -z "$repo_path" ]] && continue

    # If no free GPU slot, wait for any background job to finish and reclaim GPUs
    while (( ${#free_gpu_slots[@]} == 0 )); do
        echo "No free GPU slots, waiting for a task to finish..."
        wait -n  # Wait until any background job completes
        # Find finished PIDs and return their GPU groups to the free pool
        for pid in "${!pid_to_gpus[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Task with PID $pid finished, freeing GPUs: ${pid_to_gpus[$pid]}"
                free_gpu_slots+=("${pid_to_gpus[$pid]}")
                unset pid_to_gpus["$pid"]
            fi
        done
    done

    # Take the next free GPU group from the queue
    gpu_ids="${free_gpu_slots[0]}"
    free_gpu_slots=("${free_gpu_slots[@]:1}")

    # repo_name = last path component of repo_path
    repo_name=$(basename "$repo_path")

    echo "=========================================="
    echo "Task $((task_idx + 1)): $repo_path"
    echo "Kernel name: $kernel_url"
    echo "GPU IDs: $gpu_ids"
    echo "=========================================="

    # Timestamp for log filename
    timestamp=$(date +"%Y%m%d_%H%M%S")
    # Per-task log file
    log_file="${LOG_DIR}/${repo_name}_${timestamp}.log"

    # Run geak in the background
    geak --repo "$repo_path" \
         --task "Optimize the repository ${repo_path}, the kernel is ${kernel_url}, test command is python3 scripts/task_runner.py compile && python3 scripts/task_runner.py correctness && python3 scripts/task_runner.py performance" \
         --num-parallel "$GPUS_PER_TASK" \
         --gpu-ids "$gpu_ids" \
         --pipeline-mode planned \
         > "$log_file" 2>&1 &

    pid=$!
    pid_to_gpus[$pid]="$gpu_ids"
    echo "Started task $((task_idx + 1)) with PID $pid on GPUs $gpu_ids"
    echo ""

    task_idx=$((task_idx + 1))
done < "$TASK_FILE"

# Wait for any jobs still running after the task file is exhausted
echo "=========================================="
echo "Waiting for all remaining tasks to complete..."
echo "=========================================="
wait

echo ""
echo "All tasks completed!"