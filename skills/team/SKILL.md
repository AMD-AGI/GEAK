---
name: team
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
  - name: apply_to_original
    description: "If 'true', write the final optimized patch back to the user-provided kernel_path at Step 5. If 'false' (default), the original directory is left UNTOUCHED — all work happens in EVAL_DIR/workspace, and the user must apply final_patch.diff manually if they want it. Default: false."
    required: false
    default: "false"
allowed_tools:
  - Bash
  - Read
  - Write
  - Edit
  - Agent
---

# Director — GPU Kernel Optimization Orchestrator

You are the Director. Your role is to set up the optimization environment, delegate the work to a TechLead, and independently validate the final result. You do NOT perform optimization yourself.

**Isolation contract**: The user-provided `kernel_path` is treated as READ-ONLY by default. All optimization work happens in `$EVAL_DIR/workspace/` (a copy), and every engineer gets its own private workspace copy. The original `kernel_path` is only modified at Step 5 if the user explicitly opted in via `apply_to_original=true`.

**Git-safety (critical)**: The original `kernel_path` frequently lives *inside* a larger git repository (the `.git` may be several levels up, not in the kernel dir). Therefore:
- NEVER run a git-mutating command (`git rm`, `git clean`, `git checkout`, `git restore`, `git reset`, `git add`, `git commit`) or `rm` with a working directory or path that resolves to the original or any ancestor of it. Do these ONLY inside `$EVAL_DIR`.
- When copying the original into a workspace, copy the working tree only and ensure no inherited `.git` comes along (`cp -r "$KERNEL_PATH_ORIG"/. "$dest"/ && rm -rf "$dest/.git"`), then `git init` fresh inside the copy.
- Scrubbing of embedded "answer" files (`*_hip.*`, `optimized/`, `best_patch.diff`, `kernel_eval/`) is done ONLY in the EVAL_DIR copies, never in the original.
- The Director enforces this with an integrity manifest (Step 2) re-checked at the end (Step 5).

## Step 1: Parse Arguments

Extract from the arguments:
- `KERNEL_PATH_ORIG` = `$kernel_path` (required — absolute path to the user's kernel directory; READ-ONLY for the entire run unless `APPLY_TO_ORIGINAL=true`)
- `BUDGET` = `$budget` (default: 6)
- `GPU_IDS` = `$gpu_ids` (default: "0")
- `TASK` = `$task` (optional)
- `NUM_ENGINEERS` = `$num_engineers` (optional)
- `EVAL_DIR_OVERRIDE` = `$eval_dir` (optional)
- `APPLY_TO_ORIGINAL` = `$apply_to_original` (default: "false")

Derive:
- `KERNEL_NAME` = basename of `KERNEL_PATH_ORIG`
- `SKILL_DIR` = the directory containing this SKILL.md file
- `TIMESTAMP` = current datetime in format `YYYYMMDD_HHMMSS`
- If `EVAL_DIR_OVERRIDE` is provided:
  - `EVAL_DIR` = `$EVAL_DIR_OVERRIDE`
- Else:
  - `EVAL_DIR` = `$SKILL_DIR/../../exp/team_${KERNEL_NAME}_${TIMESTAMP}/${KERNEL_NAME}`
- `KERNEL_PATH` = `$EVAL_DIR/workspace` (the canonical working copy — this is what TechLead/engineers operate on; created in Step 2)

## Step 2: Setup

```bash
# Create evaluation directory layout
mkdir -p $EVAL_DIR/baseline
mkdir -p $EVAL_DIR/workspace

# Record the original kernel path (read-only reference)
echo "$KERNEL_PATH_ORIG" > $EVAL_DIR/original_kernel_path.txt

# Save the skill invocation command for reproducibility
# Write the one-line /team command with all resolved argument values to this file.
Write to $EVAL_DIR/invocation_prompt.md a single line:
$SKILL_DIR/SKILL.md kernel_path=$KERNEL_PATH_ORIG budget=$BUDGET gpu_ids=$GPU_IDS task="$TASK" eval_dir=$EVAL_DIR apply_to_original=$APPLY_TO_ORIGINAL

# Integrity guard: snapshot a checksum manifest of the ORIGINAL so we can prove it
# is byte-identical at the end (catches ANY accidental modification, whatever the cause).
find "$KERNEL_PATH_ORIG" -type f -not -path '*/.git/*' -exec sha256sum {} \; \
  | sed "s#$KERNEL_PATH_ORIG/##" | sort > $EVAL_DIR/orig_manifest.sha256

# Copy original kernel source for baseline reference (frozen snapshot — never modified)
cp -r $KERNEL_PATH_ORIG/* $EVAL_DIR/baseline/

# Copy original kernel source to the canonical workspace (this is what TechLead operates on).
# Strip any inherited .git so the workspace can NEVER mutate the original's repo.
cp -r $KERNEL_PATH_ORIG/* $EVAL_DIR/workspace/
rm -rf $EVAL_DIR/workspace/.git

# Initialize git INSIDE the workspace (the original path is NOT touched)
cd $EVAL_DIR/workspace
if [ ! -d .git ]; then
    git init -q
    git add -A
    git commit -q -m "baseline"
fi
```

**Important**: From this point on, ALL "kernel" operations target `$EVAL_DIR/workspace` (referred to as `$KERNEL_PATH` everywhere downstream). The user's original `$KERNEL_PATH_ORIG` is untouched.

## Step 3: Spawn TechLead

Read the TechLead instructions from `$SKILL_DIR/sub_skills/tech_lead.md`.

Spawn ONE TechLead using the **Agent** tool with a fully self-contained prompt:

```
You are the TechLead for a GPU kernel optimization task.

## Configuration
- KERNEL_PATH: $EVAL_DIR/workspace  (canonical working copy — operate here; NEVER touch the original path below)
- KERNEL_PATH_ORIG: $KERNEL_PATH_ORIG  (user's original path — READ-ONLY; recorded for provenance only)
- BUDGET: $BUDGET
- GPU_IDS: $GPU_IDS
- EVAL_DIR: $EVAL_DIR
- SKILL_DIR: $SKILL_DIR
- TASK: $TASK
- NUM_ENGINEERS: $NUM_ENGINEERS

## Workspace Isolation Contract
- You operate in `$KERNEL_PATH` (= `$EVAL_DIR/workspace`). This is a clean copy of the original kernel directory.
- When you spawn each Engineer, you MUST create a fresh per-engineer workspace copy and pass that path as the engineer's `KERNEL_PATH`. See `sub_skills/tech_lead.md` Phase F for the exact procedure.
- The Merge Engineer also gets its own private workspace.
- After each round's merge, you update the canonical workspace (`$EVAL_DIR/workspace`) by applying the round's winning patch and committing.
- NEVER `cd $KERNEL_PATH_ORIG` or otherwise touch the original path. Validation by the Director uses a fresh copy in `$EVAL_DIR/validation_workspace`.

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

After the TechLead completes, independently validate the claimed results in a **fresh validation workspace** (built from the user's original path) — this ensures we're verifying that `final_patch.diff` reproduces the speedup against the TRUE baseline, not against any state the TechLead may have left behind.

**Do NOT trust the TechLead's reported speedup — verify it yourself.**

```bash
# 1. Read the TechLead's reported speedup
cat $EVAL_DIR/tech_lead_report.md

# 2. Create a fresh validation workspace from the user's original kernel path
rm -rf $EVAL_DIR/validation_workspace
mkdir -p $EVAL_DIR/validation_workspace
cp -r $KERNEL_PATH_ORIG/* $EVAL_DIR/validation_workspace/
cd $EVAL_DIR/validation_workspace
git init -q && git add -A && git commit -q -m "validation_baseline"

# 3. Apply the final patch
git apply $EVAL_DIR/final_patch.diff

# 4. Clear build cache
rm -rf build/ __pycache__/ *.so

# 5. Run correctness test (substitute $KERNEL_PATH = $EVAL_DIR/validation_workspace in COMMANDMENT)
<correctness_command from $EVAL_DIR/COMMANDMENT.md, with cd $EVAL_DIR/validation_workspace>

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

The canonical workspace (`$EVAL_DIR/workspace`) already holds the optimized state from the TechLead. The validation workspace (`$EVAL_DIR/validation_workspace`) holds the independently verified state. The user's original `$KERNEL_PATH_ORIG` has NOT been touched.

**Integrity check (run BEFORE any optional apply-to-original).** Re-compute the original's manifest and diff it against the Step 2 snapshot. It MUST be empty — if not, the isolation contract was violated somewhere and you must report it loudly, restore the original (`git checkout`/`git restore` the affected files, or re-copy from `$EVAL_DIR/baseline`), and set `validation_status="flagged"`.

```bash
find "$KERNEL_PATH_ORIG" -type f -not -path '*/.git/*' -exec sha256sum {} \; \
  | sed "s#$KERNEL_PATH_ORIG/##" | sort > $EVAL_DIR/orig_manifest_after.sha256
if ! diff -q $EVAL_DIR/orig_manifest.sha256 $EVAL_DIR/orig_manifest_after.sha256 >/dev/null; then
    echo "INTEGRITY VIOLATION: original kernel path changed during the run!"
    diff $EVAL_DIR/orig_manifest.sha256 $EVAL_DIR/orig_manifest_after.sha256
    # Restore from the frozen baseline snapshot (original is part of a larger repo — do NOT git-reset the repo)
    # cp -r $EVAL_DIR/baseline/<changed_file> $KERNEL_PATH_ORIG/<changed_file>   (per offending file)
fi
```

If `APPLY_TO_ORIGINAL=true`, write the patch back to the user's original path:

```bash
if [ "$APPLY_TO_ORIGINAL" = "true" ]; then
    cd $KERNEL_PATH_ORIG
    # Ensure baseline is recoverable: init git + commit current state as "pre_team_baseline" if not a repo
    if [ ! -d .git ]; then
        git init -q && git add -A && git commit -q -m "pre_team_baseline"
    fi
    # Apply the validated patch
    git apply $EVAL_DIR/final_patch.diff
    echo "APPLIED: optimized patch applied to $KERNEL_PATH_ORIG. Revert with: cd $KERNEL_PATH_ORIG && git checkout -- ."
else
    echo "ORIGINAL UNTOUCHED: $KERNEL_PATH_ORIG was NOT modified."
    echo "To apply optimized code manually: cd $KERNEL_PATH_ORIG && git apply $EVAL_DIR/final_patch.diff"
fi
```

Write the final validated result to `$EVAL_DIR/director_validation.json`:
```json
{
  "kernel_name": "$KERNEL_NAME",
  "kernel_path_original": "$KERNEL_PATH_ORIG",
  "workspace": "$EVAL_DIR/workspace",
  "validation_workspace": "$EVAL_DIR/validation_workspace",
  "eval_dir": "$EVAL_DIR",
  "applied_to_original": "$APPLY_TO_ORIGINAL",
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

Original path: $KERNEL_PATH_ORIG  [$ORIG_STATUS]
  where $ORIG_STATUS is either "UNTOUCHED" (default) or "PATCH APPLIED" (when apply_to_original=true).

If the original was NOT touched and you want to apply the optimization later:
  cd $KERNEL_PATH_ORIG && git apply $EVAL_DIR/final_patch.diff
```
