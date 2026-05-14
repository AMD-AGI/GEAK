# Phase 4: Optimization Strategy Planning

## Objective
Generate a set of diverse, prioritized optimization tasks based on profiling data,
bottleneck classification, and domain knowledge. Each task will be assigned to a
parallel optimization worker.

## Steps

### 4.1 Gather Context

Read and synthesize all preprocessing artifacts:
1. `$EVAL_DIR/logs/analysis.json` -- kernel metadata, type, dependencies
2. `$EVAL_DIR/baseline/baseline_metrics.json` -- baseline performance, bottleneck type
3. `$EVAL_DIR/logs/profiling_summary.md` -- profiling analysis
4. `$EVAL_DIR/logs/codebase_context.md` -- repo structure, dependency tree
5. `$EVAL_DIR/logs/COMMANDMENT.md` -- evaluation contract
6. The kernel source file itself

### 4.2 Apply Workload-Specific Guidance

Based on kernel type and bottleneck, apply the appropriate guidance:

**For Triton kernels** (see `knowledge/triton_patterns.md`):
- Prefer: Algorithmic rewrites, fusion, memory restructuring
- Deprioritize: autotune-only, dispatch changes

**For HIP kernels** (see `knowledge/hip_patterns.md`):
- Prefer: Algorithmic rewrites, branchless patterns, vectorization
- Deprioritize: Launch-config-only, wrapper changes

**For search/pointer-chasing workloads** (KNN, binary search, etc.):
- Prefer: Branchless logic, size-specialized variants, cooperative search
- Deprioritize: Bandwidth maximization, generic vectorization

### 4.3 Generate Optimization Tasks

Create `$NUM_PARALLEL` or more diverse optimization tasks. Each task should:
1. Target a DIFFERENT optimization strategy
2. Focus primarily on the kernel body (priorities 0-5)
3. Include at least 3 genuinely different *algorithmic* approaches
4. NOT repeat strategies from prior rounds (if multi-round)

**Task format:**
```json
{
  "label": "short-kebab-case-id",
  "priority": 0-15,
  "strategy_category": "algorithmic|fusion|memory|tuning|wrapper",
  "description": "Detailed description of the optimization approach",
  "task_prompt": "Full instructions for the optimization worker..."
}
```

### 4.4 Task Prompt Requirements

Each task_prompt MUST include:
1. **Specific optimization focus** -- what algorithm/technique to try
2. **COMMANDMENT adherence** -- read and follow the evaluation contract
3. **Verification instructions**:
   - Run tests after changes to verify correctness
   - Measure performance and compare against baseline
   - If correctness fails, revert changes
4. **Baseline comparison** -- compare results against baseline metrics
5. **Save enforcement** -- save patches when improvement is achieved

Each task_prompt must NOT:
- Repeat kernel_path, COMMANDMENT path, baseline_metrics path (these are provided by the framework)
- Include instructions to modify the test harness or benchmark
- Include instructions to set HIP_VISIBLE_DEVICES

### 4.5 Priority Rules

- At least 3 tasks MUST be priority 0-5 (kernel-body algorithmic work)
- Wrapper/dispatch-only tasks: max 1, always priority 15
- Leave GPUs idle rather than filling with low-priority wrapper work
- Never generate tasks that modify the test harness or benchmark

### 4.6 Output

Save tasks to `$EVAL_DIR/logs/optimization_tasks.json`:
```json
[
  {
    "label": "algorithmic-rewrite-reduction",
    "priority": 0,
    "strategy_category": "algorithmic",
    "description": "Rewrite the reduction tree using a different algorithm...",
    "task_prompt": "You are optimizing a GPU kernel. Your goal is to..."
  },
  ...
]
```

Sort tasks by priority (lowest number first = highest priority).
