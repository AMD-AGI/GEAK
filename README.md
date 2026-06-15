# PerfSkills

Multi-agent GPU performance optimization for **AMD Instinct MI GPUs** (CDNA, e.g. gfx942 / gfx950 — the
on-box card is auto-detected). Driven by Claude Code, orchestrated by deterministic JS **Workflows**.

Two workflows ship here:


| Workflow                                      | Scope               | What it optimizes                                     |
| --------------------------------------------- | ------------------- | ----------------------------------------------------- |
| **[Team Workflow E2E](workflow_e2e_team/)** ⭐ | Whole-model serving | End-to-end **sglang / vLLM throughput** of a full LLM |
| [Team Workflow](workflows/)                   | Single kernel       | Latency / speedup of a single AMD GPU kernel (Triton, HIP, CK, FlyDSL, …) |


> **Team Workflow E2E is the headline.** It raises the serving throughput of a real model by triaging hot
> kernels and pulling levers cheapest-first, then *recursively* calls the single-kernel Team Workflow to
> author/optimize the kernels worth fixing. If you only want to speed up one kernel, use Team Workflow directly.

---

## Getting started

### 1. Prerequisites

- An **AMD Instinct MI GPU** (CDNA, e.g. gfx942 / gfx950), **ROCm 6+**, a profiler (`rocprof-compute` /
`rocprofv3` / `rocprof`), Python 3.8+.
- **Claude Code ≥ 2.1.177** — the workflows use the **dynamic Workflow** (JS orchestration) feature, which
is only available from this version onward. Check with `claude --version`.
- For E2E: a running-capable serving backend (`sglang` or `vllm`) and the model weights on disk.

### 2. Launch Claude Code in auto mode

The workflows spawn many sub-agents and run profiling / benchmark / build commands on the box, so run
Claude Code with permissions auto-approved. Update Claude Code first to make sure you have the dynamic
Workflow feature (≥ 2.1.177):

```bash
claude update                                    # update Claude Code to the latest version
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

### 3. Point it at this repo and ask

```bash
git clone <this-repo> PerfSkills && cd PerfSkills
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Then just describe what you want in natural language (examples below). Claude Code resolves the paths and
invokes the `Workflow` tool for you.

---

## Team Workflow E2E — whole-model serving throughput ⭐

`workflow_e2e_team/` raises the **sglang / vLLM serving throughput** of a whole LLM. It is a *system layer*
that wraps — and recursively calls — the single-kernel Team Workflow:

1. **Preflight** the env (GPU arch, backend, model).
2. **Profile** a running server on your exact workload.
3. **Triage** hot kernels by **Amdahl** (`pct_gpu_time × achievable_speedup`).
4. **Pull levers cheapest-first** — config/backend sweep → head GEMM/attention bake-off (aiter per-shape
  tune + a Triton kernel *authored* via the recursive kernel layer) → editable-kernel milestone loop.
5. **Overlay** each accepted change back **reversibly**, gated on a measured warm-server throughput delta
  (interleaved A/B, 0.5% band + engagement proof + output parity).

Every run writes a complete `**final_report.md`** (with a Phases tree + artifacts tree).

### Example — natural language (recommended)

```
use path/to/workflow_e2e_team to optimize inference for /models/Qwen3.5-27B-FP8, sglang, ISL/OSL=1024, conc=64, gpus 0,1,2,3
```

### Example — programmatic (`Workflow` tool)

```js
Workflow({
  scriptPath: "<PerfSkills>/workflow_e2e_team/team_workflow_e2e.js",
  args: {
    model_path: "/models/Qwen3.5-27B-FP8",            // required (e2e mode)
    workflow_dir: "<PerfSkills>/workflow_e2e_team",   // required
    backend: "sglang",                                // sglang | vllm
    isl: 1024, osl: 1024, conc: 64,                   // workload (profile + bench use the SAME)
    gpu_ids: "0,1,2,3",                               // optimization-parallelism pool (serving stays TP=1)
    budget: 6, min_kernel_tasks: 4, kernel_budget: 6,
    head_budget: 3, head_author_max: 2,               // head GEMM/attn bake-off + author
    e2e_repeats: 7, config_tune: "true",              // tight A/B; Tier-0 sweep on
    apply_to_original: "false"
  }
})
// Single-kernel pass-through (backward compatible): pass kernel_path instead of model_path.
```

**Output** lands under `workflow_e2e_team/exp/e2e_<model>_<timestamp>/` — `final_report.md`,
`architect_report.md`, `final/` (overlay + patch + `final_launch.sh`), and per-stage artifacts.
See a real run in `[examples/team_workflow_e2e/](examples/team_workflow_e2e/)`.

---

## Team Workflow — single kernel

`workflows/` optimizes a single AMD GPU kernel — Triton, HIP, CK, FlyDSL, or any AMD GPU source: Director → TechLead → specialist engineers
(algorithm / memory / compute / host_runtime), multi-round and budget-controlled, with each patch
independently verified before it's accepted.

### Example — natural language (recommended)

```
use path/to/workflows to optimize /path/to/knn, gpu 4
```

```
use path/to/workflows to optimize /path/to/silu, budget 8, focus on wrapper overhead
```

### Example — programmatic (`Workflow` tool)

```js
Workflow({
  scriptPath: "<PerfSkills>/workflows/team_workflow.js",
  args: {
    kernel_path: "/abs/path/to/kernel/",        // required
    workflow_dir: "<PerfSkills>/workflows/",    // required: dir containing team_workflow.js
    budget: 6,                                  // optional, default 6
    gpu_ids: "0",                               // optional, comma-separated, default "0"
    task: "focus on memory bandwidth",          // optional, natural-language steer
    apply_to_original: "false"                  // optional, write the patch back to kernel_path
  }
})
```

### Batch (many kernels at once)

Spawn one agent per kernel with isolated GPU assignments; GPU access is serialized via
`scripts/gpu_lock.sh` (flock-based), so kernels can safely share GPUs.

---

## Why Workflows (not prose skills)

Control flow — the budget loop, fan-out, verification, and stop conditions — is **deterministic JS** in
`team_workflow.js` / `team_workflow_e2e.js`. LLM agents are called only for judgement (analysis, strategy,
optimization). This makes runs reliable and reproducible. (Markdown-driven `skills/` also exist for
reference, but the workflows are the recommended path.)

## Results — single-kernel

12 HIP kernels, measured on AMD MI300X (gfx942) (excluding mla_decode; FAIL counted as 1.0x):


| Method            | LLM          | Geo Mean  |
| ----------------- | ------------ | --------- |
| GEAK_v3           | n/a          | 1.90x     |
| GEAK Skill        | Sonnet 4.6   | 2.33x     |
| Team Skill        | Opus 4.6     | 3.56x     |
| Team Skill        | Opus 4.8     | 3.32x     |
| **Team Workflow** | **Opus 4.8** | **3.68x** |


> Team Skill and Team Workflow are measured with unified baselines (3 runs, median); GEAK_v3 / GEAK Skill
> use each run's own baseline. Per-kernel breakdowns:
> [original](examples/result/hip2hip_comparison.md) ·
> [reproducibility](examples/result/hip2hip_repro_comparison.md).

## Repository layout

```
PerfSkills/
├── workflow_e2e_team/   # ⭐ End-to-end LLM serving-throughput optimizer (wraps workflows/)
│   ├── team_workflow_e2e.js   # system-layer orchestration (config / head-GEMM / kernel tracks + e2e gate)
│   ├── roles/  knowledge/  scripts/   # adapters/{sglang,vllm}.sh, op_bench.py, parse_profile.py, …
│   └── README.md / PLAN.md
├── workflows/           # Single-kernel optimizer
│   ├── team_workflow.js       # deterministic JS orchestration
│   ├── roles/  knowledge/  scripts/   # gpu_lock.sh, profile_kernel.sh
│   └── README.md
├── skills/              # Markdown-driven skills (GEAK, Team) — reference
├── examples/            # Example tasks, benchmark comparisons, a real e2e run
└── exp/                 # Experiment outputs (timestamped per run)
```

## License

Apache License 2.0