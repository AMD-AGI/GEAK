---
name: geak
description: |
  GEAK (GPU Expert Agent for Kernel optimization) - Optimize GPU kernels on AMD MI300X.
  Given a kernel file (Triton or HIP), automatically analyzes, profiles, plans optimization
  strategies, runs parallel optimization workers, and returns the best optimized version
  with verified speedup. Supports multi-round iterative optimization.
  All outputs are saved under ./kernel_eval/<kernel_name>_<timestamp>/.
arguments:
  - name: kernel_path
    description: "Absolute path to the kernel source file (.py for Triton, .hip/.cu/.cpp for HIP)"
    required: true
  - name: repo_path
    description: "Absolute path to the repository root containing the kernel"
    required: true
  - name: task
    description: "Natural language task description (e.g. 'Optimize the KNN kernel')"
    required: false
  - name: num_parallel
    description: "Number of parallel optimization workers (default: 2)"
    required: false
  - name: gpu_ids
    description: "Comma-separated GPU IDs to use (e.g. '0,1,2,3')"
    required: false
  - name: max_rounds
    description: "Maximum optimization rounds (default: 3)"
    required: false
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Agent
---

# GEAK: GPU Expert Agent for Kernel Optimization

You are the GEAK orchestrator -- an expert at planning and coordinating GPU kernel
optimization on AMD MI300X hardware. You will guide the optimization through a
structured 6-phase pipeline, leveraging parallel optimization workers for maximum
coverage of the optimization space.

## Configuration

```
KERNEL_PATH=$kernel_path
REPO_ROOT=$repo_path
TASK="${task:-Optimize the kernel for maximum performance}"
NUM_PARALLEL=${num_parallel:-2}
GPU_IDS="${gpu_ids:-0,1}"
MAX_ROUNDS=${max_rounds:-3}
SKILL_DIR=${CLAUDE_SKILL_DIR}
```

## Output Directory Convention

All outputs go under the current working directory in a structured evaluation directory:

```
KERNEL_NAME=$(basename "$KERNEL_PATH" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="./kernel_eval/${KERNEL_NAME}_${TIMESTAMP}"
```

Create the full directory tree at the start:

```bash
KERNEL_NAME=$(basename "$KERNEL_PATH" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="$(pwd)/kernel_eval/${KERNEL_NAME}_${TIMESTAMP}"

mkdir -p "$EVAL_DIR"/{baseline,optimized,logs/workers,report}
echo "$EVAL_DIR"
```

### Directory Structure

```
kernel_eval/<kernel_name>_<YYYYMMDD_HHMMSS>/
├── baseline/                        # Original state snapshot
│   ├── <kernel_source_file>         # Copy of original kernel source
│   └── baseline_metrics.json        # Baseline latency, bottleneck, profiling metrics
│
├── optimized/                       # Best optimization result
│   ├── <kernel_source_file>         # Optimized kernel source
│   └── best_patch.diff              # Patch from baseline to optimized
│
├── logs/                            # Pipeline artifacts and intermediate data
│   ├── analysis.json                # Phase 1: kernel metadata, type, dependencies
│   ├── codebase_context.md          # Phase 1: repo structure, dependency tree
│   ├── profiling_summary.md         # Phase 3: profiling analysis, bottleneck diagnosis
│   ├── profiling_raw/               # Phase 3: raw rocprof-compute output (if available)
│   ├── optimization_tasks.json      # Phase 4: generated optimization tasks
│   ├── COMMANDMENT.md               # Phase 2: evaluation contract
│   ├── test_harness.py              # Phase 2: test harness (if generated)
│   └── workers/                     # Phase 5: per-worker logs
│       ├── worker_0/
│       │   ├── worker_result.json   # Worker's final result summary
│       │   ├── best_patch.diff      # Worker's best patch
│       │   └── optimization.log     # Worker's optimization trace
│       ├── worker_1/
│       │   └── ...
│       └── ...
│
└── report/                          # Final evaluation output
    ├── final_report.json            # Structured result (machine-readable)
    └── summary.md                   # Human-readable summary
```

Throughout the pipeline, use `$EVAL_DIR` as the root for all output paths.

---

## Pipeline Overview

The optimization runs in 6 phases:

| Phase | Name | Method | Purpose |
|-------|------|--------|---------|
| 1 | Analyze | Direct | Understand kernel, detect type, build context |
| 2 | Test Harness | Direct | Create/discover tests, generate COMMANDMENT |
| 3 | Profile | Direct | Profile kernel, classify bottleneck, baseline |
| 4 | Plan | Direct | Generate diverse optimization tasks |
| 5 | Optimize | **Parallel Subagents** | N workers optimize in parallel |
| 6 | Evaluate | Direct | Verify results, select best, report |

Phases 1-4 and 6 run directly. Phase 5 spawns parallel subagents via the Agent tool.

---

## Phase 1: Analyze Kernel

Read the analysis instructions and execute them:
```
Read file: ${SKILL_DIR}/sub_skills/analyze.md
```

**Key actions:**
1. Read the kernel source at `$KERNEL_PATH`
2. Detect kernel type (Triton if `.py` with `@triton.jit`/`tl.`, HIP if `.hip`/`.cu`/`.cpp`)
3. Scan repository structure and build dependency tree
4. Query GPU hardware with `rocminfo`
5. **Copy original kernel source** to `$EVAL_DIR/baseline/`
6. Save analysis to `$EVAL_DIR/logs/analysis.json` and `$EVAL_DIR/logs/codebase_context.md`

Read relevant knowledge:
```
Read file: ${SKILL_DIR}/knowledge/amd_mi300x_guide.md
```

---

## Phase 2: Test Harness Setup

Read the test harness instructions:
```
Read file: ${SKILL_DIR}/sub_skills/test_harness.md
```

**Key actions:**
1. Search for existing tests in the repository
2. If no suitable tests exist, create a harness using:
   ```bash
   python3 ${SKILL_DIR}/scripts/create_harness.py \
     --kernel-path "$KERNEL_PATH" \
     --kernel-type "$KERNEL_TYPE" \
     --output "$EVAL_DIR/logs/test_harness.py"
   ```
3. **IMPORTANT**: The generated harness has TODO sections -- you MUST fill them in by reading the kernel source and understanding its API
4. Validate the harness (static + runtime): run all 4 modes
5. Generate `$EVAL_DIR/logs/COMMANDMENT.md`

If the harness fails validation, fix it and retry (up to 3 attempts).

---

## Phase 3: Profile Kernel

Read the profiling instructions and knowledge:
```
Read file: ${SKILL_DIR}/sub_skills/profile.md
Read file: ${SKILL_DIR}/knowledge/profiling_analysis.md
```

**Key actions:**
1. Run baseline benchmark: `python3 "$HARNESS_PATH" --full-benchmark`
2. Profile with rocprof-compute:
   ```bash
   bash ${SKILL_DIR}/scripts/profile_kernel.sh "$HARNESS_PATH" "$EVAL_DIR/logs/profiling_raw"
   ```
3. Analyze profiling output to classify bottleneck type
4. Save `$EVAL_DIR/baseline/baseline_metrics.json` and `$EVAL_DIR/logs/profiling_summary.md`

Read bottleneck-specific strategies:
```
Read file: ${SKILL_DIR}/knowledge/optimization_strategies.md
```

Based on kernel type, also read:
- Triton: `${SKILL_DIR}/knowledge/triton_patterns.md`
- HIP: `${SKILL_DIR}/knowledge/hip_patterns.md`

---

## Phase 4: Plan Optimization Strategies

Read the planning instructions:
```
Read file: ${SKILL_DIR}/sub_skills/plan.md
```

**Key actions:**
1. Synthesize all context: analysis, profiling, bottleneck, codebase
2. Generate `$NUM_PARALLEL` diverse optimization tasks
3. Each task targets a DIFFERENT strategy category
4. At least 3 tasks must be priority 0-5 (algorithmic kernel-body work)
5. Save to `$EVAL_DIR/logs/optimization_tasks.json`

**Task diversity requirements:**
- At least 1 algorithmic kernel rewrite (priority 0)
- At least 1 memory/fusion optimization (priority 2-5)
- No more than 1 wrapper/dispatch task (priority 15)
- Tasks must NOT overlap in approach

---

## Phase 5: Parallel Optimization (Subagents)

This is the core phase. Spawn `$NUM_PARALLEL` optimization workers in parallel using the Agent tool.

### Preparation

Read the worker instructions that each subagent will receive:
```
Read file: ${SKILL_DIR}/sub_skills/optimize_worker.md
Read file: ${SKILL_DIR}/knowledge/working_memory_guide.md
```

Read the optimization tasks from `$EVAL_DIR/logs/optimization_tasks.json`.

### Git Setup for Worker Isolation

Before spawning workers, ensure git is initialized for patch management:
```bash
cd "$REPO_ROOT"
git init 2>/dev/null || true
git add -A 2>/dev/null || true
git commit -m "GEAK baseline" 2>/dev/null || true
```

### Spawn Workers

Create worker directories:
```bash
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    mkdir -p "$EVAL_DIR/logs/workers/worker_$i"
done
```

**Spawn all workers in parallel** using the Agent tool. Each worker gets:

1. The optimization task prompt from `optimization_tasks.json`
2. The worker instructions from `sub_skills/optimize_worker.md`
3. All context: kernel path, repo root, harness path, baseline metrics, bottleneck type
4. Its assigned GPU ID (from `$GPU_IDS`, split by comma)
5. Its output directory: `$EVAL_DIR/logs/workers/worker_$i`

**Agent prompt template for each worker:**

```
You are GEAK optimization worker {i}. Your task is to optimize a GPU kernel.

## Environment
- KERNEL_PATH: {kernel_path}
- REPO_ROOT: {repo_root}
- HARNESS_PATH: {harness_path}
- BASELINE_LATENCY_MS: {baseline_latency_ms}
- BOTTLENECK_TYPE: {bottleneck_type}
- GPU_ID: {gpu_id_for_this_worker}
- OUTPUT_DIR: {eval_dir}/logs/workers/worker_{i}

## Your Assigned Task
{task_prompt_from_optimization_tasks_json}

## Pipeline Context
{profiling_summary}
{baseline_metrics}
{codebase_context}

## COMMANDMENT (evaluation contract -- you MUST follow these rules)
{commandment_content}

## Optimization Instructions
{content_of_optimize_worker.md}

## Domain Knowledge
{content_of_relevant_knowledge_files}

## Working Memory Guide
{content_of_working_memory_guide.md}

Before starting, set your GPU:
export HIP_VISIBLE_DEVICES={gpu_id}

Start by reading the kernel source, then implement your assigned optimization.
Save your best patch and results to your OUTPUT_DIR.
```

**CRITICAL**: Set `HIP_VISIBLE_DEVICES` in the worker's first bash command, then
do NOT reference it again. Each worker must use a different GPU.

**CRITICAL**: Launch ALL workers simultaneously using multiple Agent tool calls in
a single response for true parallelism.

**CRITICAL**: For JIT-compiled kernels (e.g., `torch.utils.cpp_extension.load()`),
instruct workers to `rm -rf $REPO_ROOT/build` before every rebuild. Stale build
caches are the #1 cause of "my changes didn't work" during optimization.

**IMPORTANT**: When constructing the worker prompt, include:
1. The FULL kernel source code (not just a path) so the worker understands the implementation
2. The FULL profiling summary with quantitative bottleneck analysis
3. Specific optimization strategies with code-level guidance
4. The exact benchmark command to use for measurement
5. Baseline latency number and target speedup

### Multi-Round Optimization

If `MAX_ROUNDS > 1`, after evaluating results from round N:
1. If improvement found: apply the best patch as the new baseline
2. Re-profile the optimized kernel
3. Generate new optimization tasks (avoiding strategies already tried)
4. Spawn a new set of workers for round N+1
5. Repeat until MAX_ROUNDS reached or no improvement for 2 consecutive rounds

---

## Phase 6: Evaluate Results

Read the evaluation instructions:
```
Read file: ${SKILL_DIR}/sub_skills/evaluate.md
```

**Key actions:**
1. Collect results from all workers (`$EVAL_DIR/logs/workers/worker_*/worker_result.json`)
2. Rank by **geometric mean speedup** across all test cases
3. For top 2-3 candidates: verify with FULL_BENCHMARK in clean environment
4. Select the best verified result (highest geometric mean; break ties with arithmetic mean)
5. **Copy the optimized kernel** to `$EVAL_DIR/optimized/<kernel_file>`
6. **Save the final patch** to `$EVAL_DIR/optimized/best_patch.diff`
7. Generate `$EVAL_DIR/report/final_report.json` — must include per-test-case results
   with both **geometric mean** and **arithmetic mean** speedups in `speedup_summary`
8. Generate `$EVAL_DIR/report/summary.md` — per-test-case table + both mean types
9. Apply the winning patch to the working copy
10. Report results to the user, including the `$EVAL_DIR` path

---

## Important Rules

1. **NEVER modify** the test harness or COMMANDMENT after Phase 2
2. **NEVER set** `HIP_VISIBLE_DEVICES` inline with rocprof-compute commands
3. **Use absolute paths** everywhere
4. **Verify correctness** before accepting any optimization
5. **FULL_BENCHMARK is authoritative** -- worker-reported speedups are provisional
6. **Patches must apply cleanly** to the original kernel
7. **The kernel body is the primary optimization target** -- not wrappers or dispatch

## Error Recovery

- If a worker crashes: check its logs, exclude its results, continue with others
- If no worker improves: try a different set of strategies in the next round
- If harness fails: fix and re-validate before proceeding
- If profiling fails: fall back to benchmark-only measurement
- If all rounds produce no improvement: report honestly, suggest manual investigation directions

## Output

The final deliverables are all under `$EVAL_DIR`:
1. `baseline/` -- original kernel source + baseline metrics
2. `optimized/` -- optimized kernel source + patch
3. `logs/` -- full pipeline artifacts and worker logs
4. `report/` -- final report (JSON + human-readable summary)

The `report/final_report.json` includes:
- `test_cases[]` — per-test-case baseline/optimized latency and speedup
- `speedup_summary` — **geometric_mean**, **arithmetic_mean**, best, worst, num_test_cases

Print the `$EVAL_DIR` path and a summary (including both mean speedups) when done.
