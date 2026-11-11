# mega_fusion: mode=mega as an expert workflow

`mode="mega"` is for operators that run across several GPUs at once and are worth rewriting as one
fused, persistent kernel — e.g. an EP MoE layer whose dispatch, GEMM1, GEMM2 and combine are
separate launches today.

**Scope:** validated on EP MoE layers (MegaMoE) only. Other multi-rank operators (attention, a GEMM
fused with its all-reduce, ...) are not validated yet; use it for EP MoE until they are.

This directory is self-contained: the workflow, its roles, knowledge, tools, scripts and tests. The
kernel-workflow dispatcher runs `workflow.js` as a sub-workflow when `mode=mega` is passed (or
`expert_workflow: "mega_fusion"`); nothing in `kernel_workflow/` depends on it. Shared pieces it uses
from `kernel_workflow/` are only the GPU lease scripts (`scripts/gpu_lock.sh`, `gpu_lease.py`), the
profiling scripts and the shared knowledge files (`amd_instinct.md`, `hip_optimization.md`,
`profiling_guide.md`, ...); `knowledge/*_mega.md` hold what mega adds to the shared ones.

## Lifecycle

`workflow.js` keeps a portfolio of independent whole-kernel **candidates** instead of one canonical tree:

- **Setup / Analyze / Benchmark / Profile** freeze the baseline, write an operator-neutral plan
  (MegaPlanIR: work domains, buffers, counters, queues, events, schedule) and build the paired A/B
  harness (rank-max timing, CUDA-graph replay, positive/null controls).
- **Optimize** runs candidate turns: a search lead picks the next candidate, an engineer authors it
  in its own tree, and every candidate is verified on-card — direct accuracy on the run's required
  cases, graph-replay liveness, launch count, and paired A/B against the frozen baseline. Failures are
  recorded as structured blockers that the next turn consumes.
- **Validate** re-measures the finalists and delivers the fastest fully verified candidate (or none).

## Usage

Every job needs the operator's full rank count, so pass `gpus_per_job` (or `job_gpu_ids`) with
`gpu_ids`. `use_expert_skills=true` adds a matched Expert Skill (e.g.
`perf_knowledge/expert_skills/skills/megamoe_ep_mega_fusion`) as knowledge for the same roles; with
`false` the generic lifecycle runs unchanged. Optional `analysis_skill=moe_bottleneck` adds a
route-attribution pass after Profile.

```
Workflow({ scriptPath: "<GEAK>/kernel_workflow/kernel_workflow.js", args: {
  mode: "mega", kernel_path: "<op task dir>", workflow_dir: "<GEAK>/kernel_workflow",
  gpu_ids: "0,1,2,3,4,5,6,7", gpus_per_job: 8,
  budget: 12, task: "...",
  target_guards: ["8192_uniform"],          // cases that create speedup credit
  regression_guards: ["512_uniform"],       // measured, veto-only
  required_accuracy_cases: ["128", "8192"], // direct accuracy + liveness, independent of skills
  launch_target: 2,                         // terminal launch-count ceiling
  state_dir: "<dir>",                       // candidate registry; reuse it to continue a search
}})
```

All other args (time budgets, final top-k, Expert Skill pinning, gates) are read at the top of
`workflow.js` with their defaults. Launching `workflow.js` directly (with `mode: "mega"`,
`workflow_dir` = the kernel_workflow directory) shows its phases individually; a caller that is
itself a workflow (e.g. e2e) calls it directly too, since workflows nest one level only.

## Layout

```
workflow.js   the mode=mega workflow (Setup -> Analyze -> Benchmark -> Profile -> Optimize -> Validate)
roles/        the role set it reads (director, tech_lead, mega_search_lead, engineer, verify_engineer, ...)
knowledge/    distributed_fusion, fusion_preconditions, tile_task_graph, overlap_instrument,
              pipe_occupancy, resource_partition, gfx950_lowering, crash_bisection, jit_arm_isolation,
              mega_* (measurement, regimes, generic lessons, self-monitoring), *_mega.md (additions to the
              shared amd_instinct.md / hip_optimization.md), collectors/, analysis_skills/
tools/        mega_graph_ab, mega_regime, generic_graph_contract, ab_retry, fast_test_key,
              rebase_commandment, seed_fast_test_cache, expert_skill_contract / expert_skill_runtime
scripts/      multi_rank_analysis/ (per-rank trace/counter aggregation), replay_runs.js (re-decide
              finished runs with today's gates, no GPU), reference_leak_sweep.sh, skill_address_scan.sh
tests/        regression tests: `for t in tests/test_*.js; do node "$t"; done`, `python -m pytest tests/`
```

`tools/expert_skill_contract.py` (static checker for a skill's `contract.yaml`) and
`tools/expert_skill_runtime.py` (runs a skill's `runtime_validation.py`) back the pinned-skill path;
`_contribute/validate_skill.py` uses the same files for a skill's constraint evidence and run args.
