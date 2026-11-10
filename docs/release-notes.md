---
myst:
  html_meta:
    "description": "GEAK release notes for 4.2.0 and 4.0.0: knowledge base, Codex CLI backend, AgentX workloads, ATOM serving backend, end-to-end LLM serving optimization, and single-kernel optimizer for AMD Instinct GPUs."
    "keywords": "GEAK, release notes, GEAK 4.2, GEAK 4.0, AMD Instinct, knowledge base, Hyperloom, Codex CLI, AgentX, ATOM, e2e_workflow, kernel_workflow, sglang, vLLM, ROCm 10.0"
---

# GEAK release notes

This topic summarizes the features available in each GEAK release. For the hardware and software versions validated for a release, see the [Compatibility matrix](compatibility.md).

## GEAK 4.2.0

GEAK 4.2.0 extends GEAK with a persistent knowledge base, an alternative agent backend, support for AgentX trace-replay workloads, a new serving backend, and expanded ROCm compatibility.

### Release highlights

GEAK 4.2.0 introduces five major additions.

#### Kernel-level knowledge base and Hyperloom integration

GEAK 4.2.0 introduces a kernel-level knowledge base that captures and reuses kernel optimization strategies, implementations, and validated results across runs. Hyperloom integration enables knowledge retrieval, warm-start optimization, and continuous learning from previous runs.

#### Codex CLI agent backend

Codex CLI is now a supported agent backend alongside Claude Code. It runs the existing `kernel_workflow` and `e2e_workflow` JavaScript Workflows without modification.

#### AgentX workload support

GEAK 4.2.0 adds optimization support for AgentX workloads using real trace replay instead of synthetic fixed-shape benchmarks, along with interactivity-based performance metrics and workload-aligned validation.

#### ATOM serving backend

ATOM is now a first-class serving backend alongside sglang and vLLM. The integration supports automated server launch, profiling, benchmarking, and end-to-end optimization.

#### ROCm 10.0 compatibility

The validated environment expands to include ROCm 10.0.

## What's new in GEAK 4.2.0

The following items were added or improved in GEAK 4.2.0.

### Added

- Kernel-level knowledge base for capturing and reusing optimization strategies, implementations, and validated results across runs.
- Hyperloom integration for knowledge retrieval, warm-start optimization, and continuous learning.
- Codex CLI as an alternative agent backend (runs existing `kernel_workflow` and `e2e_workflow` Workflows without modification).
- AgentX trace-replay workload support with interactivity-based performance metrics and workload-aligned validation.
- ATOM serving backend with automated server launch, profiling, benchmarking, and end-to-end optimization.
- ROCm 10.0 compatibility.

### Improved

- Cross-run knowledge reuse and warm starts through the kernel-level knowledge base.
- AgentX performance measurement and validation.
- Flexibility across agent runtimes and serving backends.

## GEAK 4.0.0

GEAK 4.0.0 is a major redesign of GEAK, upgrading it from a single-kernel optimization agent into an end-to-end GPU performance optimization system for AMD Instinct™ GPUs.

The headline change is e2e_workflow: GEAK can now optimize full LLM serving workloads on sglang or vLLM, not just isolated kernels. It profiles the real workload, ranks bottlenecks by Amdahl impact, pulls the cheapest high-leverage levers first, and recursively calls the single-kernel workflow only for kernels that can truly move model throughput.

### Release highlights

GEAK 4.0.0 introduces three major capabilities.

#### End-to-end LLM serving optimization

GEAK 4.0.0 introduces e2e_workflow, a system-level optimizer for whole-model serving throughput. The workflow:

- Preflights the GPU, backend, model, and profiler environment.
- Profiles a warm sglang or vLLM server on the target workload.
- Ranks hot kernels by `pct_gpu_time × achievable_speedup`.
- Runs config and backend sweeps before expensive source rewrites.
- Extracts high-impact kernels into standalone tasks.
- Recursively invokes kernel_workflow to author or optimize kernels.
- Overlays accepted changes back into the live server reversibly.
- Validates gains with warm-server A/B, engagement proof, output parity, and throughput gating.

#### Workflow-based orchestration

GEAK 4.0.0 moves the optimization control plane into deterministic JavaScript Workflows. Budget loops, parallel fan-out, verification, recursion, and stop conditions are handled in code, while LLM agents focus on judgment-heavy work such as analysis, strategy, kernel authoring, debugging, and integration.

This makes GEAK 4.0.0 easier to run, easier to reproduce, and easier to debug than a fully prompt-driven optimization loop.

#### Stronger single-kernel workflow

The original kernel optimization capability remains first-class through kernel_workflow. It supports Triton, HIP, CK, FlyDSL, and other AMD GPU source paths through a hierarchical multi-agent design:

- Director for workspace setup and final validation.
- TechLead for strategy and round planning.
- Specialist engineers for algorithm, memory, compute, and host/runtime optimization.
- Independent verification for every candidate patch.
- Integration and re-profiling across rounds.

Each patch is measured independently before it can become the new best candidate.

#### Performance knowledge and expert skills

GEAK 4.0.0 adds a structured perf_knowledge layer for AMD GPU optimization. It organizes operator, backend, GPU-generation, dtype, and regime knowledge into a machine-queryable kernel matrix. Expert skills can seed high-value optimization directions, but they never replace measurement: every candidate must still pass on-box validation.

## What’s new

The following items were added or improved in GEAK 4.0.0.

### Added

- `e2e_workflow/` for whole-model sglang or vLLM serving-throughput optimization.
- `kernel_workflow/` as a workflow-native single-kernel optimizer.
- Deterministic JS Workflow orchestration for budget control, parallelism, verification, and recursive workflow calls.
- Amdahl-driven profiling and triage for system-level bottleneck selection.
- Config and backend sweep before source-level kernel authoring.
- Warm-server end-to-end validation with interleaved A/B comparison, engagement proof, output parity, and noise-band gating.
- Reversible overlay-based integration for serving backends.
- `perf_knowledge/` for AMD operator × backend optimization knowledge.
- Expert-skill support for reusable, measurement-gated kernel optimization recipes.
- Example e2e reports and benchmark comparisons under `examples/`.

### Improved

- Clearer separation between deterministic control flow and LLM judgment.
- Better usability through natural-language workflow invocation.
- More reliable benchmarking and verification through independent patch validation.
- Better alignment between kernel optimization and real serving workloads.
- Improved long-run optimization support for multi-kernel and multi-backend exploration.
- Stronger artifact discipline: each run writes structured reports, patches, overlays, and per-stage outputs.

