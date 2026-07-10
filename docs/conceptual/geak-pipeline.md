---
myst:
  html_meta:
    "description": "How the GEAK end-to-end pipeline works: setup, profile, Amdahl strategy, config sweep, head-kernel bake-off, editable-kernel milestones, and independent validation."
    "keywords": "GEAK, pipeline, serving throughput, Amdahl, config sweep, head kernel, profiling, e2e gate, sglang, vLLM, ROCm"
---

# GEAK pipeline

The `e2e_workflow` takes a running LLM server and a workload (ISL/OSL/concurrency) and raises its serving
throughput. Control flow is deterministic; each phase is owned by a specialized agent, and every accepted
change is gated on a measured end-to-end delta before it is kept. The single-kernel `kernel_workflow` runs
the same closed loop on one kernel and is called recursively when a kernel needs to be authored or tuned.

```{mermaid}
flowchart LR
  A[Setup<br/>Director] --> B[Profile<br/>Profiler]
  B --> C[Strategize<br/>Architect]
  C --> D[ConfigSweep<br/>Config Tuner]
  D --> E[HeadKernel<br/>Extractor / Op Benchmarker / ⟲ kernel layer / Integrator]
  E --> F[Milestone<br/>⟲ kernel layer / Integrator]
  F --> G[Finalize / Report / Validate<br/>Integrator / Architect / Director]
```

## Inputs

- **Model + serving backend**: a model on disk served by `sglang` or `vllm` (selected per run).
- **Workload**: input/output sequence length and concurrency (ISL/OSL/conc). Profiling and benchmarking
  use the *same* workload so numbers are comparable.

## Phases

### Setup

The Director builds an isolated evaluation directory, launches a warm server, and records the **true
baseline** throughput and its noise band. Model weights and installed packages stay read-only.

### Profile

The Profiler traces the warm server under the real workload and produces one canonical, Amdahl-ranked
hot-kernel list, separating the prefill and decode regimes.

### Strategize

The System Architect routes each hot kernel to a track by Amdahl leverage
(`pct_gpu_time × achievable_speedup`): config, head kernel, editable kernel, or host overhead.

### ConfigSweep

The Config Tuner tries service-level switches one at a time — attention backend, cuda/HIP graph,
scheduling and memory knobs, backend toggles, and (when enabled) lower precision — keeping only changes
that measurably help. This is the cheapest, landscape-reshaping lever, so it runs first; the server is
re-profiled afterward.

### HeadKernel

For the heaviest GEMM/attention kernels: the Kernel Extractor builds a standalone, immutable unit test
from the real workload; the Op Benchmarker bakes off backends and tunes them; and the recursive
`kernel_workflow` authors/optimizes a kernel against that test. The e2e Integrator overlays the winner
back and gates it end-to-end.

### Milestone

The remaining editable kernels above a threshold are swept the same way — extract, optimize via the
kernel layer, integrate, re-profile — until the budget or the Amdahl stop rule ends the loop.

### Finalize, Report, Validate

The Integrator assembles the deliverable (reversible overlay + patch + `final_launch.sh`); the Architect
writes `final_report.md`; and the Director **independently re-measures** the combined result against the
true baseline and arbitrates the official number.

## Integration and gating

Each accepted kernel is folded back into the live server through a **reversible overlay** (never editing
installed packages). A change is accepted only when it **actually runs live** (engagement proof), clears
the noise band under a tight back-to-back A/B, and preserves output quality (greedy parity, or a task
accuracy gate for reduced-precision kernels).

## Outputs

Everything lands under `exp/e2e_<model>_<timestamp>/`: `final_report.md`, `architect_report.md`, the
`final/` bundle (overlay, `final_patch.diff`, `final_launch.sh`), and per-stage artifacts.

## Related topics

- [What is GEAK?](../what-is-geak.md) — overview and design.
- [Run a workflow](../how-to/run-agent.md) — how to start a run.
