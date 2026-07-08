---
myst:
  html_meta:
    "description": "Release notes for GEAK. Find what changed in each version, including new features, bug fixes, and breaking changes."
    "keywords": "GEAK, release notes, changelog, ROCm, GPU kernel optimization, version history"
---

# GEAK release notes

This topic lists what changed in each GEAK release, including new features, bug fixes, and breaking changes. For verified hardware and software combinations per release, see the [GEAK compatibility matrix](compatibility.md).

## GEAK v3.2.2

Patch release cut from the HEAD commit (`d44e1cb`) of PR [#290](https://github.com/AMD-AGI/GEAK/pull/290) on branch `chore/subagent-docs-upstream-resource`.

Includes the harness shape-faithfulness improvements, backend-rewrite subagents, and upstreamed subagent documentation and resources, building on the v3.2.1 GEMM tuning entry point.

Tagged independently of PR #290's merge status so downstream consumers have a durable, reproducible reference, regardless of when or how the PR is merged into `main`.

---

## GEAK v3.2.1

Release built from `main` (`c0a1f937`). This release focuses on GPU scheduler reliability, preprocess v3 robustness, and improved model resolution for multi-agent runs.

### GPU scheduler and concurrency overhaul

- Centralized `GPUManager` decouples agent threads from physical GPU count — each agent gets its own worktree slot.
- GPU lease system, split timeouts, CPU-pressure gate, and a reaper for stuck jobs.
- LLM concurrency cap to avoid TPM/RPM blowout under oversubscription; natural-language knobs (`gpu_oversubscribe`, `max_concurrent_llm`).
- JSONL event logging and per-outcome counters. Auto worker count `max(4, 3×gpus)`.

### Preprocess v3 robustness

- Deterministic Path-A; synthesizes harnesses for flag-less and composite tasks; routes shape-bearing tasks to the harness generator.
- Handles truncated tool calls instead of crash-looping; soft cap no longer hard-fails the run. Five legacy capabilities ported into v3.

### Model resolution

- Default model bumped to `claude-opus-4.8`.
- Provider-qualified model names so sub-agent fallback can't 400 mid-run; NTID normalization and core42 gateway header.

### FlyDSL knowledge base and skills

- Rewrote PyTorch→FlyDSL translation knowledge base (zero-compute enforcement, CuTe primitives, epilogue fix); guardrails, prefetch patterns, LDS diagnostics; fp16 deterministic harness.

### Patch capture and packaging

- `save_and_test` falls through to `git diff --no-index` and `diff -ruN`; JIT/compiled cache artifacts excluded from captured patches.
- Package bundles `subagents/`, `skills/`, and reverse-knowledge scripts; install resolves repo root for non-editable installs.

### RAG and miscellaneous

- RAG postprocessor reuses the agent's live model instead of hardcoding; assorted CI, lint, and test-flake fixes.

---

## GEAK v3.2.0

### Unified multi-agent optimization pipeline

Replaced the Triton/HIP language router with a unified orchestrator that dispatches subagents based on task type, following a Claude Code/Codex-style multi-agent architecture. The framework is more flexible and easier to extend: new workflows require new subagent definitions and system prompts. Multi-round optimization is now unified across all kernel optimization workflows, with future support for resuming from intermediate checkpoints.

### FlyDSL support

GEAK adds first-class FlyDSL support for AMD GPU kernels. The new FlyDSL path covers both PyTorch-to-FlyDSL translation and direct FlyDSL optimization, so the system can translate, tune, and debug kernels across patterns such as GEMM, attention, softmax, layernorm, reductions, conv2d, and pooling.

### FP8 GEMM tuning with SGLang and AITER

Added GEAK FP8 GEMM tuning support (via `geak-gemm-tuning`) for the SGLang + AITER workflow, including automated GEMM shape extraction, source localization, tuning orchestration, and performance validation.

### Changes since v3.1.0

Pipeline and orchestration:

- Unified the optimization pipeline behind `run_pipeline`, removing the legacy homogeneous vs. heterogeneous split.
- Added `TaskPlanner` and `Dispatcher` for plan-then-select execution and parallel worker assignment.
- Replaced hard-coded language-specific routing with dynamic subagent dispatch based on task characteristics.
- Shared preprocessing, benchmarking, evaluation, and reporting across kernel types through one common execution path.

Subagent framework:

- Added `SubAgentRegistry` for discovery, registration, and lookup of `SUBAGENT.yaml` definitions.
- Added inherited `model`, `env`, and `tools` blocks so subagents can reuse `geak.yaml` defaults with local overrides.
- Added `register_from_dict` and `create_subagent` for programmatic subagent creation.

FlyDSL support:

- Added the `pytorch-to-flydsl` subagent for translating GPU kernels into FlyDSL.
- Added FlyDSL optimization skills for tile programming, performance tuning, and correctness debugging.
- Added shipped skills including `flydsl` and `pytorch2flydsl-translation`.

FP8 GEMM tuning:

- Added GEMM tuning for shape dumping, tuner execution, kernel table generation, and validation.
- Added the standalone `geak-gemm-tuning` CLI with workspace management and model or environment loading from `geak.yaml`.
- Added the `fp8-gemm-tuning-sglang-aiter` skill.

Run modes and budget system:

- Added `--mode quick|full` to select standard runtime profiles: `quick` (1-hour total budget, up to 2 rounds) and `full` (2-hour total budget, up to 5 rounds).
- Added `--total-budget-s` for explicit wall-clock overrides.

Preprocessing improvements:

- Sandboxed bash and editor tools to the worktree and added path rewriting for `GEAK_REPO_ROOT`.
- Improved support for the AITER Path B scenario.
- Added `--preprocess-only` to stop after preprocessing for lower-cost validation sweeps.
- Added `--target wall|kernel` with dual-signal harness support.
- Simplified harness generator and verifier prompts and reshaped the `COMMANDMENT` contract.

Stability and cleanup:

- Hard-kill watchdog now fires at exactly `started_at + total_s`, independent of preprocess duration.
- Added `--debug` to disable post-run patch apply and artifact cleanup, preserving the full run directory for inspection.
- Unified tool settings and removed deprecated `metrix-mcp` in favor of the `profiler-mcp` metrix backend.

### Requirements

- Python 3.10 or later
- Access to an LLM endpoint (for example, LiteLLM or AMD LLM Gateway)
- Optional: Docker environment via `scripts/run-docker.sh`

---

## GEAK v3.1.0

### RAG-powered knowledge integration

GEAK introduces an MCP-based RAG system that provides dynamic access to a curated GPU optimization knowledge base covering the ROCm stack, architecture insights, and optimization patterns. Retrieved context is injected directly into the optimization loop, improving reasoning quality and convergence. A reverse pipeline supports offline knowledge ingestion and continuous updates.

### Cross-run memory and retrieval

GEAK now carries optimization memory across runs. During optimization, working memory helps track promising strategies, failures, and intermediate insights. Historical kernels and strategies can be stored and retrieved through similarity search, allowing new runs to start from prior experience instead of optimizing from scratch.

### Clearer optimization logs

GEAK improves logging throughout the optimization workflow. You can more easily follow what the agent is doing, inspect optimization progress step by step, and understand the current state of a run while kernel optimization is in progress.

### Changes since v3.0.0

- Integrated MCP-based retrieval into the optimization loop, with offline knowledge ingestion support.
- Introduced cross-run memory with similarity-based retrieval over past optimizations, and within-run working memory.
- Reduced MCP-related race conditions and CLI blocking issues.
- Improved environment compatibility (Docker, paths, shell).
- Improved model and backend compatibility.
- Improved latency/result handling and evaluation patch application.
- Improved harness validation and fallback handling.
- More reliable baseline and evaluation execution under varied conditions.
- Added initial support for reusable agent skills.

### Requirements

- Python 3.10 or later
- Access to an LLM endpoint (for example, LiteLLM or AMD LLM Gateway)
- Optional: Docker environment via `scripts/run-docker.sh`

---

## GEAK v3.0.0

### Repository-level optimization

GEAK now operates on full code repositories instead of isolated kernels, providing an end-to-end workflow from preprocessing (test discovery, harness setup, profiling) to optimization and evaluation.

### Automated optimization loop

Introduces an LLM-driven optimization loop with built-in patching, testing, and benchmarking, supporting multi-round iteration and automatic best-patch selection across parallel runs.

### Reproducible evaluation pipeline

A standardized preprocessing stage establishes fixed test harnesses and measured baselines upfront, ensuring consistent and comparable performance evaluation throughout optimization.

### Extensible tooling via MCP integration

Provides a unified interface to integrate external tools such as profilers and analysis utilities, enabling flexible and extensible optimization workflows.

### Changes since v2.0.0

- Optimization now runs directly on real-world repositories with tracked patches, logs, and execution history, replacing the previous benchmark-centric approach.
- Reorganized into a modular mini-swe-agent-style structure, with clear separation of preprocessing, agents, environments, and tools.
- Expanded test coverage, standardized linting and formatting, and added CI workflows for validation and smoke testing.

### Requirements

- Python 3.10 or later
- Access to an LLM endpoint (for example, LiteLLM or AMD LLM Gateway)
- Optional: Docker environment via `scripts/run-docker.sh`

---

## GEAK v2.0.0

### GEAK-OptimAgentv2 (instruction to Triton)

Builds on OptimAgentv1 with:

- **Multi-offspring evolution:** each iteration produces several candidate kernels from the same instruction, improving diversity and success rate versus a single candidate per generation.
- **LLM-based evaluator:** offspring are scored and ranked on multiple axes (for example, fusion, numerical stability, wavefront utilization) before selecting the next parent.
- **Profiler–analyzer loop:** AMD GPU profiling (rocprof-compute-class telemetry) is summarized by an LLM into actionable text feedback for the next optimization round.

### GEAK-OpenEvolve (Triton to Triton)

Quality-Diversity / MAP-Elites–style search over a large population of kernel variants, with:

- Hybrid parent selection and cascade evaluation.
- Concurrent evaluation on multiple GPUs.
- LLM scoring on kernel feature dimensions.

Aimed at optimizing existing kernels beyond single-trajectory LLM refinement.

### Benchmark outcomes

| Metric | GEAK v1 | GEAK v2 |
|--------|---------|---------|
| TritonBench-modified execution accuracy | 53.80% | 63.04% (+9.76%) |
| ROCm benchmark execution accuracy | 54.84% | 61.29% (+6.45%) |
| Average speedup with profiler + LLM analyzer | 1.38× | 3.32× |

OpenEvolve results: ~56% success rate (speedup > 1); 3.42× average speedup on TritonBench-modified and 7.02× on ROCm-bench over successfully optimized kernels.

### Requirements

- Python 3.8 or later
- Dependencies per `pyproject.toml` at v2.0.0 (Triton, PyTorch, `openai`, `anthropic`)
- Stack used in benchmarks: ROCm 6.4.3, Triton 3.3.0, PyTorch 2.4+ (ROCm)

---

## GEAK v1.0.0

### Agent architecture

Four cooperating modules:

- **Generator:** produces code from a query and retrieved context.
- **Reflector:** applies error-trace–driven fixes when correctness fails.
- **Evaluator:** cascade evaluation — functionality first, then performance.
- **Optimizer:** applies strategies drawn from historically sorted runs to improve latency and efficiency.

### Composable techniques

- 1-shot prompting: retrieve similar Triton kernels by code similarity.
- Knowledge injection: hardware specs and optimization principles.
- Reflexion-style self-correction on failed tests.
- Configurable LLM selection for the agent.
- LLM as optimizer over sorted historical performance records.
- Debugging trap (`max_perf_debug_num`) to abandon stuck strategies.
- Parallel scaling: multiple independent runs with temperature-based diversity and pass@k-style analysis.

### Benchmark outcomes

| Scenario | TritonBench-modified | ROCm 30-kernel set |
|----------|---------------------|-------------------|
| Direct LLM prompting (baseline) | ~8–16% execution accuracy | — |
| GEAK without optimizer (ablation) | ~51% execution accuracy, 1.81× speedup | ~40% execution accuracy, 0.85× speedup |
| Full GEAK | up to 54.89% execution accuracy, 2.59× speedup | 63.33% execution accuracy, 0.92× speedup |

11 of 30 kernels beat expert baselines on latency in the ROCm benchmark set.

## Related topics

- [What is GEAK?](what-is-geak.md)—overview of GEAK's capabilities and design goals.
- [GEAK compatibility matrix](compatibility.md)—verified hardware and software configurations per release.
- [Install GEAK](install/install.md)—install the latest version.
