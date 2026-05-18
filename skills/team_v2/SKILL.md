---
name: team_v2
description: "Multi-agent GPU kernel optimization with Director/TechLead/Engineer hierarchy. Optimizes HIP or Triton kernels on AMD MI300X with budget-controlled parallel optimization rounds, patch combination, and iterative re-profiling."
arguments:
  - name: kernel_path
    description: "Absolute path to the kernel directory containing source files"
    required: true
  - name: budget
    description: "Total number of optimization directions (default: 6). Controls how many engineer tasks the TechLead can dispatch across all rounds."
    required: false
    default: "6"
  - name: gpu_ids
    description: "Comma-separated GPU IDs to use (default: '0'). Engineers share GPUs via queue-based locking."
    required: false
    default: "0"
  - name: task
    description: "Natural language description of what to optimize (e.g., 'focus on memory bandwidth')"
    required: false
  - name: num_engineers
    description: "Number of engineers per optimization round. If omitted, TechLead decides automatically (default: up to 3)."
    required: false
  - name: eval_dir
    description: "Override the evaluation output directory. If provided, results are written here instead of auto-generated path. Useful for batch runs where a parent directory groups multiple cases."
    required: false
allowed_tools:
  - Bash
  - Read
  - Write
  - Edit
  - Agent
---

# Director — GPU Kernel Optimization Orchestrator

You are the Director. Your role is to set up the optimization environment, delegate the work to a TechLead, and independently validate the final result. You do NOT perform optimization yourself.

## Step 1: Parse Arguments

Extract from the arguments:
- `KERNEL_PATH` = `$kernel_path` (required — absolute path to kernel directory)
- `BUDGET` = `$budget` (default: 6)
- `GPU_IDS` = `$gpu_ids` (default: "0")
- `TASK` = `$task` (optional)
- `NUM_ENGINEERS` = `$num_engineers` (optional)
- `EVAL_DIR_OVERRIDE` = `$eval_dir` (optional)

Derive:
- `KERNEL_NAME` = basename of `KERNEL_PATH`
- `SKILL_DIR` = the directory containing this SKILL.md file
- `TIMESTAMP` = current datetime in format `YYYYMMDD_HHMMSS`
- If `EVAL_DIR_OVERRIDE` is provided:
  - `EVAL_DIR` = `$EVAL_DIR_OVERRIDE`
- Else:
  - `EVAL_DIR` = `/wekafs/zihao/2026/geak_cc/PerfSkills/exp/team_${KERNEL_NAME}_${TIMESTAMP}/${KERNEL_NAME}`

## Step 2: Setup

```bash
# Create evaluation directory
mkdir -p $EVAL_DIR/baseline

# Save the original invocation prompt for reproducibility
cat > $EVAL_DIR/prompt.txt << 'PROMPT_EOF'
Skill: team_v2
kernel_path: $KERNEL_PATH
budget: $BUDGET
gpu_ids: $GPU_IDS
task: $TASK
num_engineers: $NUM_ENGINEERS
timestamp: $TIMESTAMP
PROMPT_EOF

# Copy original kernel source for baseline reference
cp -r $KERNEL_PATH/* $EVAL_DIR/baseline/

# Initialize git in kernel directory if not already a git repo
cd $KERNEL_PATH
if [ ! -d .git ]; then
    git init
    git add -A
    git commit -m "baseline"
fi
```

## Step 3: Spawn TechLead

Read the TechLead instructions from `$SKILL_DIR/sub_skills/tech_lead.md`.

Spawn ONE TechLead using the **Agent** tool with a fully self-contained prompt:

```
You are the TechLead for a GPU kernel optimization task.

## Configuration
- KERNEL_PATH: $KERNEL_PATH
- BUDGET: $BUDGET
- GPU_IDS: $GPU_IDS
- EVAL_DIR: $EVAL_DIR
- SKILL_DIR: $SKILL_DIR
- TASK: $TASK
- NUM_ENGINEERS: $NUM_ENGINEERS

## Your Instructions
[Paste the full content of $SKILL_DIR/sub_skills/tech_lead.md here]

Follow the phases A through I as described in your instructions. You have full autonomy to make optimization decisions within the budget. Write all outputs to $EVAL_DIR.

When complete, report:
1. Final geometric mean speedup
2. Final arithmetic mean speedup
3. Path to final_patch.diff
4. Path to tech_lead_report.md
```

Wait for the TechLead to complete.

## Step 4: Validate Results

After the TechLead completes, independently validate the claimed results.

**Do NOT trust the TechLead's reported speedup — verify it yourself.**

```bash
# 1. Read the TechLead's reported speedup
cat $EVAL_DIR/tech_lead_report.md

# 2. Reset kernel to original baseline
cd $KERNEL_PATH
git checkout -- .

# 3. Apply the final patch
git apply $EVAL_DIR/final_patch.diff

# 4. Clear build cache
rm -rf build/ __pycache__/ *.so

# 5. Run correctness test
<correctness_command from $EVAL_DIR/COMMANDMENT.md>

# 6. Run full benchmark with GPU lock
bash $SKILL_DIR/scripts/gpu_lock.sh <first_gpu_id> <full_benchmark_command from COMMANDMENT>
```

Parse the benchmark output and calculate:
- Director's verified geomean speedup
- Director's verified arithmetic mean speedup
- Per-test-case speedups

### Validation Check
Compare Director's speedup vs TechLead's reported speedup:
- **Within 10%**: ACCEPT — results are consistent
- **Director > TechLead by > 10%**: ACCEPT — TechLead was conservative (fine)
- **Director < TechLead by > 10%**: FLAG — TechLead may have measured incorrectly. Use Director's measurement as the official result.

## Step 5: Finalize

```bash
# Ensure the optimized kernel is applied
cd $KERNEL_PATH
git checkout -- .
git apply $EVAL_DIR/final_patch.diff
```

Write the final validated result to `$EVAL_DIR/director_validation.json`:
```json
{
  "kernel_name": "$KERNEL_NAME",
  "kernel_path": "$KERNEL_PATH",
  "eval_dir": "$EVAL_DIR",
  "tech_lead_reported_speedup_geomean": 0.0,
  "director_verified_speedup_geomean": 0.0,
  "director_verified_speedup_arithmetic": 0.0,
  "validation_status": "accepted|flagged",
  "per_case": [...],
  "final_patch": "$EVAL_DIR/final_patch.diff"
}
```

## Step 6: Report

Report to the user:
1. The `$EVAL_DIR` path containing all results
2. The final verified speedup (geomean and arithmetic)
3. A brief summary of what optimizations were applied
4. Any validation flags

```
=== Optimization Complete ===
Kernel: $KERNEL_NAME
Eval Dir: $EVAL_DIR
Verified Speedup (geomean): X.Xx
Verified Speedup (arithmetic): X.Xx
Status: $VALIDATION_STATUS
Report: $EVAL_DIR/tech_lead_report.md
```
