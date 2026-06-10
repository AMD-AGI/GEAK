# PerfSkills

GPU kernel optimization skills for LLM-based coding agents. Target: AMD MI300X (gfx942).

## Skills & Workflows


| Method | Type | Architecture | Iteration | Key Idea |
| --- | --- | --- | --- | --- |
| [GEAK skill](skills/geak/SKILL.md) | Skill | Orchestrator + parallel workers | Single round | One agent plans, N workers execute in parallel |
| [Team skill](skills/team/SKILL.md) | Skill | Director → TechLead → Engineers | Multi-round with re-profiling | Hierarchical delegation, budget-controlled iteration, wrapper overhead detection |
| [Team Workflow](workflows/) | Workflow | Director → TechLead → Specialist Engineers | Multi-round with re-profiling | JS-orchestrated deterministic pipeline, independent verification, specialist engineers, cross-round memory |


### Design Comparison


|                        | GEAK v3                                   | GEAK Skill                                 | Team Skill                                                | Team Workflow                                                     |
| ---------------------- | ----------------------------------------- | ------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------- |
| **Agent backend**      | miniswe                                   | Claude/Cursor                              | Claude/Cursor                                             | Claude                                                     |
| **Origin**             | GEAK                                      | Refactored from GEAK_v3, reuses its logic  | Ground-up redesign                                        | Successor to Team Skill — JS control flow                         |
| **Architecture**       | Orchestrator + parallel workers           | Orchestrator + parallel workers      | Hierarchical: Director → TechLead → Engineers → Merge     | Same hierarchy, but orchestration in JS, not LLM-interpreted prose |
| **Iteration**          | Multi-round                               | Single round                               | Multi-round, budget-controlled                            | Multi-round, budget-controlled                                    |
| **Orchestration**      | Python                                    | LLM-driven                                 | LLM-driven                                                | **Deterministic JS** — loop/parallelism/verification in code       |
| **Verification**       | Orchestrator verifies                     | Orchestrator verifies                      | Director independently re-benchmarks                      | **Pipelined** — each patch verified immediately by a separate agent |
| **Engineer types**     | Generic                                   | Generic                                    | Generic                                                   | **Specialist**: algorithm, memory, compute, host_runtime           |
| **Cross-round memory** | miniswe-memory control                    | None                                       | Implicit (TechLead context)                               | **Explicit**: insight blackboard + hypothesis ledger               |
| **Best for**           | Programmatic kernel optimization workflow | Single optimization direction suffices     | Complex kernels, multi-round gains                        | Same as Team Skill, with higher reliability and reproducibility    |


### Skill vs Workflow

**Skills** (GEAK, Team) are markdown-driven: the LLM reads `SKILL.md` and interprets the steps. This is flexible but non-deterministic — the LLM decides when to loop, how many engineers to spawn, and when to stop.

**Workflows** (Team Workflow) use JS orchestration: the budget loop, fan-out, verification, and stop conditions are deterministic code in `team_workflow.js`. LLM agents are called for judgement (analysis, strategy, optimization), but control flow is reliable. Workflows run via the `Workflow` tool.

## Quick Start

### Using Team Skill (via Skill tool or direct invocation)

```
skills/team/SKILL.md \
  kernel_path=/path/to/kernel/ \
  budget=6 \
  gpu_ids=0 \
  task="Optimize this kernel for maximum throughput on AMD MI300X"
```

### Using Team Workflow

**Natural language (recommended)** — just tell Claude Code what to optimize:

```
用 workflow 优化下 /path/to/knn，gpu 4
```

```
用 workflow 优化 /path/to/silu，budget 8，重点优化 wrapper 开销
```

Claude Code will automatically resolve all paths and invoke the Workflow tool. You only need to specify the kernel path and optionally the GPU / budget / focus area.

**Programmatic invocation** (via Workflow tool directly):

```js
Workflow({
  scriptPath: "<PerfSkills>/workflows/team_workflow.js",
  args: {
    kernel_path: "/abs/path/to/kernel/",     // required
    workflow_dir: "<PerfSkills>/workflows/",  // required: directory containing team_workflow.js
    budget: 6,                                // optional, default 6
    gpu_ids: "0",                             // optional, comma-separated, default "0"
    task: "focus on memory bandwidth",        // optional, natural-language steer
    eval_dir: "",                             // optional, override output directory
    apply_to_original: "false"                // optional, write patch back to kernel_path
  }
})
```

### Using Team Workflow E2E (end-to-end serving throughput)

`workflow_e2e_team/` raises the **sglang/vllm serving throughput** of a whole LLM on MI300X. It is a
system layer that wraps — and recursively calls — the unchanged single-kernel `workflows/team_workflow.js`:
it preflights the env, profiles a running server, triages hot kernels by **Amdahl** (`pct_gpu_time ×
achievable_speedup`), then pulls levers cheapest-first — config/backend sweep → head GEMM/attention
bake-off (aiter per-shape DB tune + a Triton kernel **authored** via the recursive kernel layer) →
editable-kernel milestone loop (parallel optimize, serial integrate, cumulative stacking) — overlaying
each accepted change back **reversibly** and gating it on a measured warm-server throughput delta
(interleaved A/B, 0.5% band + engagement proof + output parity). Every run writes a complete Chinese
**`final_report.md`** (with a 阶段树 Phases tree + 产物 tree).

**Natural language (recommended):**

```
用 workflow_e2e_team 优化 /path/to/model 的推理，sglang，ISL/OSL=1024，conc=64
```

**Programmatic invocation:**

```js
Workflow({
  scriptPath: "<PerfSkills>/workflow_e2e_team/team_workflow_e2e.js",
  args: {
    model_path: "/abs/path/to/model",            // required (e2e mode)
    workflow_dir: "<PerfSkills>/workflow_e2e_team", // required
    backend: "sglang",                            // sglang | vllm (scripts/adapters/<backend>.sh)
    isl: 1024, osl: 1024, conc: 64,               // workload (profile + bench use the SAME)
    gpu_ids: "0,1,2,3",                           // optimization-parallelism pool (serving stays TP=1)
    budget: 6, min_kernel_tasks: 4, kernel_budget: 6,
    head_budget: 3, head_author_max: 2,           // head GEMM/attn bake-off + author
    e2e_repeats: 7, config_tune: "true",          // tight A/B; Tier-0 sweep on
    apply_to_original: "false"
  }
})
// Single-kernel pass-through (backward compatible): pass kernel_path instead of model_path.
```

Output lands under `workflow_e2e_team/exp/e2e_<model>_<timestamp>/` — `final_report.md`,
`architect_report.md`, `final/` (overlay + patch + `final_launch.sh`), and per-stage artifacts.
See a real run in [`examples/team_workflow_e2e/`](examples/team_workflow_e2e/).

### Batch optimization (multiple kernels)

For optimizing many kernels in parallel, spawn one agent per kernel with isolated GPU assignments:

```python
# Example: 13 kernels across 4 GPUs
kernels = ["knn", "ball_query", "roiaware_pool3d", ...]
for i, kernel in enumerate(kernels):
    gpu_id = 4 + (i % 4)  # round-robin across GPUs 4-7
    # Spawn Agent with Team Skill or Workflow, each gets its own eval_dir
```

GPU access is serialized via `scripts/gpu_lock.sh` (flock-based), so multiple kernels can safely share GPUs.

## Results

12 HIP kernels on AMD MI300X (excluding mla_decode; FAIL counted as 1.0x):

| Method | LLM | Geo Mean |
| ------ | --- | -------- |
| GEAK_v3 | n/a | 1.90x |
| GEAK Skill | Sonnet 4.6 | 2.33x |
| Team Skill | Opus 4.6 | 3.56x |
| Team Skill | Opus 4.8 | 3.32x |
| Team Workflow | **Opus 4.8** | **3.68x** |

> Team Skill 4.6, Team Skill 4.8, and Team Workflow 4.8 are measured with unified baselines (3 runs, median). GEAK_v3 and GEAK Skill use each run's own baseline.

Per-kernel breakdowns:
- [Original comparison](examples/result/hip2hip_comparison.md)
- [Reproducibility comparison](examples/result/hip2hip_repro_comparison.md)

## Repository Structure

```
PerfSkills/
├── skills/
│   ├── geak/              # Single-round parallel optimization
│   │   ├── SKILL.md
│   │   ├── knowledge/
│   │   ├── scripts/
│   │   └── sub_skills/
│   └── team/              # Multi-round hierarchical optimization (Skill)
│       ├── SKILL.md
│       ├── knowledge/     # 7 knowledge files (MI300X, HIP, Triton, strategies, profiling, wrapper, self-monitoring)
│       ├── scripts/       # gpu_lock.sh, profile_kernel.sh
│       └── sub_skills/    # analyze, benchmark_setup, profile, engineer, evaluate, merge_engineer, tech_lead
├── workflows/             # Single-kernel optimizer (Workflow)
│   ├── team_workflow.js   # Deterministic JS orchestration
│   ├── roles/             # director, tech_lead, engineer, author/benchmark/profile/verify engineers, integrator
│   ├── knowledge/         # optimization strategies, HIP/Triton/wrapper guides, profiling, MI300X, self-monitoring
│   ├── scripts/           # gpu_lock.sh, profile_kernel.sh
│   └── README.md
├── workflow_e2e_team/     # End-to-end LLM serving-throughput optimizer (wraps workflows/)
│   ├── team_workflow_e2e.js  # system-layer orchestration (config/head-GEMM/kernel tracks + e2e gate)
│   ├── roles/             # director, system_architect, profiler, config_tuner, kernel_extractor, op_benchmarker, e2e_integrator
│   ├── knowledge/         # e2e_optimization, aiter_gemm_tuning, gemm_attention_backends, backend_playbook, preflight, sglang_internals, …
│   ├── scripts/           # bench_e2e.sh + adapters/{sglang,vllm}.sh, op_bench.py, capture_shapes.py, overlay_setup.py, parse_profile.py
│   └── README.md / PLAN.md
├── examples/
│   ├── tasks/             # Example kernel tasks (e.g. knn)
│   ├── result/            # Benchmark comparisons
│   └── team_workflow_e2e/ # Example e2e run (final_report.md with 阶段树, final_launch.sh)
└── exp/                   # Experiment outputs (timestamped per run)
```

## Prerequisites

- AMD MI300X GPU (gfx942), ROCm 6.x, `rocprof-compute`, Python 3.8+

## License

Apache License 2.0
