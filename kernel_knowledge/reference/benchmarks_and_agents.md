---
title: Benchmarks & agents — pointer doc (workflow_e2e_team, GEAK, examples, KernelBench/TritonBench)
kind: reference
updated: 2026-06-08
sources:
  - /wekafs/zihao/2026/geak_cc/PerfSkills/workflow_e2e_team/README.md
  - /wekafs/zihao/2026/geak_cc/GEAK (https://github.com/AMD-AGI/GEAK)
  - /wekafs/zihao/2026/geak_cc/PerfSkills/examples/team_workflow_e2e/
  - /wekafs/zihao/2026/geak_cc/PerfSkills/kernel_knowledge/index/sota_registry.yaml
---

# Benchmarks & agents — pointer doc

Where the *agents* and *benchmark harnesses* around kernel_knowledge live, and how they consume this knowledge
base. kernel_knowledge is the **reference layer**; the items below are the **actors** that read it. This file is a
map, not a tutorial — each target owns its own docs.

## workflow_e2e_team (the e2e optimizer skill)
Path: [`/wekafs/zihao/2026/geak_cc/PerfSkills/workflow_e2e_team/`](/wekafs/zihao/2026/geak_cc/PerfSkills/workflow_e2e_team/)
(entry `team_workflow_e2e.js`, `README.md`, `roles/`, `knowledge/`, `scripts/`).

A deterministic JS-orchestrated multi-agent **Workflow** that raises sglang/vLLM serving throughput on
MI300X. It is a *system layer* on top of the unchanged single-kernel `team_workflow`; it profiles a
running server, does Amdahl triage (`pct_gpu_time × achievable_speedup`), tunes config/backend knobs
first, extracts hot kernels into standalone unittests, optimizes them via the kernel layer, overlays
them back, and re-validates e2e throughput.

Roles (in `roles/`) — these are the kernel_knowledge consumers:
- **system_architect** — strategy/routing; reads the SOTA landscape to decide config-vs-kernel tracks.
- **op_benchmarker** — *named in this task* — standardizes per-op benchmark shapes/metrics. (On-box the
  comparable role files are `op_benchmarker.md` plus `profiler.md`; **(verify exact role name)** against
  the installed `roles/` if scripts depend on it.)
- **config_tuner** — Tier-0 env/flag/backend sweep (uses [`env_vars.md`](env_vars.md)).
- **profiler**, **kernel_extractor**, **e2e_integrator**, **director** — trace, extract, reintegrate, gate.

## How the architect / op_benchmarker query kernel_knowledge
The machine-queryable surface is [`../index/sota_registry.yaml`](../index/sota_registry.yaml) —
an auto-generated mirror of `sota_matrix.md` (one entry per operator×backend with
`status / gens / dtypes / regimes / card / sources`). Intended query pattern:

> *"best backend for operator X on gen Y, dtype Z, regime R"* → filter `entries[]` by those keys,
> take `status: sota`, follow `card:` to the SOTA card.

- **system_architect** filters the registry by the served model's hot ops (from the profile Top-N) to
  decide whether a known-SOTA path already exists (config track) or a kernel needs authoring (kernel
  track), and reads the env knobs in [`env_vars.md`](env_vars.md) for the Tier-0 sweep.
- **op_benchmarker** resolves an op id to its SOTA card's *how-to-bench* section for canonical shapes,
  dtypes, and the perf-number format ([`../index/conventions.md`](../index/conventions.md)).

Always edit card frontmatter and regenerate the registry (`index/_gen_registry.py`) — never hand-edit
`sota_registry.yaml`.

## GEAK (AMD agentic kernel generator)
On-box: [`/wekafs/zihao/2026/geak_cc/GEAK/`](/wekafs/zihao/2026/geak_cc/GEAK/) (`examples/`,
`mcp_tools/`, `docs/`). Upstream: https://github.com/AMD-AGI/GEAK.

LLM agent that generates + iteratively refines Triton kernels for AMD GPUs, with a debugging/reflection
loop and an evaluator. kernel_knowledge feeds GEAK target SOTA characteristics and pitfalls; GEAK is invoked by
the kernel layer when a hot op has no SOTA path. Authoring workflow:
[`../workflows/authoring_a_kernel_with_geak.md`](../workflows/authoring_a_kernel_with_geak.md).

## examples/ (recorded runs)
[`/wekafs/zihao/2026/geak_cc/PerfSkills/examples/team_workflow_e2e/`](/wekafs/zihao/2026/geak_cc/PerfSkills/examples/team_workflow_e2e/):
- `qwen3.5-27b_sglang_gemm-tuning-win/` — a GEMM-tuning e2e win (config track).
- `qwen3.5-27b_sglang_isl1024_osl1024_conc64/` — fixed-ISL/OSL/concurrency throughput run.

These are the ground-truth shape of a full run (baseline → triage → tune → validate) and back the
case studies in [`../case_studies/`](../case_studies/).

## KernelBench / TritonBench (external eval harnesses)
- **KernelBench** — https://github.com/ScalingIntelligence/KernelBench — task suite for LLM-generated
  GPU kernels (correctness + speedup vs. reference). Used as an external yardstick for GEAK-style
  generation, not part of the serving loop.
- **TritonBench** — https://github.com/pytorch-labs/tritonbench — Triton operator microbenchmark
  suite; the per-op shapes/metrics op_benchmarker mirrors should stay consistent with TritonBench
  conventions where an op overlaps.

Both are *external* benchmarks; kernel_knowledge's authoritative numbers come from on-box runs in `examples/`
recorded in the [`../index/conventions.md`](../index/conventions.md) perf format.

## Sources
- workflow_e2e_team README + roles — [`/wekafs/zihao/2026/geak_cc/PerfSkills/workflow_e2e_team/README.md`](/wekafs/zihao/2026/geak_cc/PerfSkills/workflow_e2e_team/README.md), `roles/`.
- GEAK — on-box [`/wekafs/zihao/2026/geak_cc/GEAK/`](/wekafs/zihao/2026/geak_cc/GEAK/) ; https://github.com/AMD-AGI/GEAK
- examples — [`/wekafs/zihao/2026/geak_cc/PerfSkills/examples/team_workflow_e2e/`](/wekafs/zihao/2026/geak_cc/PerfSkills/examples/team_workflow_e2e/)
- registry/query contract — [`../index/sota_registry.yaml`](../index/sota_registry.yaml), [`../index/conventions.md`](../index/conventions.md)
- KernelBench — https://github.com/ScalingIntelligence/KernelBench ; TritonBench — https://github.com/pytorch-labs/tritonbench
- Items marked **(verify exact name)** were not confirmed verbatim at write time.
