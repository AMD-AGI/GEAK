---
myst:
  html_meta:
    "description": "GEAK is a multi-agent GPU performance optimizer for AMD Instinct MI GPUs. It raises sglang/vLLM serving throughput and optimizes single kernels (Triton, HIP, CK, FlyDSL), driven by deterministic JS Workflows."
    "keywords": "GEAK, GPU kernel optimization, serving throughput, ROCm, sglang, vLLM, Triton, HIP, CK, FlyDSL, AMD Instinct, multi-agent, Workflow"
---

# What is GEAK?

GEAK (Generating Efficient AI-Centric Kernels) is a multi-agent GPU performance optimizer for **AMD
Instinct MI GPUs** (CDNA; the on-box card is auto-detected). It ships two deterministic **Workflows**,
driven by Claude Code:

| Workflow | Scope | What it optimizes |
| --- | --- | --- |
| **`e2e_workflow`** ⭐ | Whole-model serving | End-to-end **sglang / vLLM throughput** of a full LLM |
| `kernel_workflow` | Single kernel | Latency / speedup of one AMD GPU kernel (Triton, HIP, CK, FlyDSL) |

`e2e_workflow` is the headline: it raises serving throughput by triaging hot kernels and pulling levers
cheapest-first, then *recursively* calls `kernel_workflow` to author or optimize the kernels worth
fixing. To speed up a single kernel, use `kernel_workflow` directly.

## Core design: deterministic control plane, LLM judgement

Control flow — the budget loop, parallel fan-out, verification, and stop conditions — is **deterministic
JS** in `e2e_workflow.js` / `kernel_workflow.js`. LLM agents are invoked only for judgement (analysis,
strategy, kernel authoring). This makes runs reliable and reproducible.

### e2e_workflow (whole-model serving)

A system layer that wraps — and recursively calls — the single-kernel layer. Specialized agents own each
stage: an **e2e Director** (isolated environment + true baseline), a **System Architect** (Amdahl
strategy), a **Profiler**, a **Config Tuner**, a **Kernel Extractor**, an **Op Benchmarker**, and an
**e2e Integrator** (reversible overlay + throughput gate).

### kernel_workflow (single kernel)

A hierarchical single-kernel optimizer: **Director → TechLead → specialist engineers** (algorithm,
memory, compute, host-runtime), multi-round and budget-controlled. Each patch is independently verified
against an immutable correctness oracle before it is accepted.

## Two things keep it honest

- **Profiling** grounds every decision in measured data — nothing is optimized unless it is first shown
  to matter (Amdahl: `pct_gpu_time × achievable_speedup`).
- **A curated knowledge base** (`perf_knowledge/`) supplies operator × backend priors as *reference
  only* — the on-box benchmark and end-to-end gate are always the judge.

## How a run works

1. **Preflight** the environment (GPU arch, serving backend, model).
2. **Profile** a running server on your exact workload and rank hot kernels by Amdahl.
3. **Pull levers cheapest-first** — config/backend sweep → head GEMM/attention bake-off (aiter per-shape
   tune + a kernel *authored* via the recursive kernel layer) → editable-kernel milestone loop.
4. **Overlay** each accepted change back **reversibly**, gated on a measured warm-server throughput delta
   (interleaved A/B, output parity).
5. **Report and validate** — every run writes a complete `final_report.md`, and the Director independently
   re-measures the result.

## Related topics

- [Install GEAK](install/install.md) — set up the environment and Claude Code.
- [Run a workflow](how-to/run-agent.md) — invoke a workflow from Claude Code or the `run_e2e.py` interface.
- [GEAK pipeline](conceptual/geak-pipeline.md) — the phases of an end-to-end run.
- [Reference](reference/api-reference.md) — Workflow arguments, the integration contract, and run artifacts.
