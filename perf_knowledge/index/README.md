# perf_knowledge — AMD Instinct (MI-series) Kernel-Optimization Knowledge Base

A knowledge base for **writing, selecting, and tuning the fastest kernel** for any operator on any
AMD Instinct MI GPU (CDNA1 MI100 · CDNA2 MI200 · CDNA3 MI300A/X/325X · CDNA4 MI350X/355X).

It has two jobs:
1. **Teach optimization** — hardware, programming models, backends, techniques, profiling.
2. **Be a SOTA registry** — for **every `operator × backend`** cell, point at the *best known
   implementation(s)* with source, measured performance, applicability, knobs, and pitfalls.

## How to navigate
- **Pick an implementation fast** → [`sota_matrix.md`](sota_matrix.md) (human matrix) or
  [`sota_registry.yaml`](sota_registry.yaml) (machine-queryable) → jump to the operator's backend card
  under [`../operators/<op>/backends/<backend>.md`](../operators/).
- **Decide what to even try** → [`decision_trees.md`](decision_trees.md).
- **Learn the hardware / a language / a backend / a technique** → `hardware/ · languages/ · backends/ · optimization/`.
- **Profile / triage a kernel** → `profiling/`.
- **Run the end-to-end optimization workflow** → `kernel_workflow/`.

## Layout
```
index/        navigation + SOTA registry + taxonomy + sourcing rules + templates
hardware/     per-generation arch deep dives (CDNA1–4) + shared (matrix core, memory, numerics)
languages/  top-10 languages, deep: triton, flydsl, hip, ck, asm, tilelang, rocwmma, hipkittens, mojo, cutlass_port
backends/     top-10 libraries, deep: aiter, hipblaslt, ck_lib, rocblas/tunableop, fa-rocm, mori/rccl, miopen, inductor, sglang, vllm
operators/    ★ the Cartesian core — each <op>/ has overview+tuning+numerics+fusion + backends/<backend>.md SOTA cards
optimization/ techniques (tiling, split-K/stream-K, pipelining, LDS, occupancy, fusion, …)
quantization/ fp8/fp6/fp4/int8, block-scaling/microscaling (MXFP), kv-cache quant
profiling/ rocprofv3, rocprof-compute (omniperf), roofline, ISA dump, counter dictionary
kernel_workflow/    kernel dev lifecycle, capture→tune→deploy, agentic authoring, e2e serving tuning
case_studies/ by_model + by_kernel war stories with measured numbers
reference/    MFMA matrix, ISA tables, env-var dictionary, perf-peak tables, repo index, benchmarks & agents
```

## Non-negotiables (see [`sourcing_rules.md`](sourcing_rules.md))
- **Every file ends with `## Sources`** citing primary sources (ROCm docs/blogs, CDNA ISA §, GitHub
  `repo@commit/path`, arXiv, vendor benchmarks).
- **Every performance number is measured** and tagged `value @ hardware + ROCm/lib version + date`.
  No theoretical-peak claims presented as achievable (MI300X commonly sustains ~45% of peak).
- **Cartesian completeness**: a missing `operator×backend` cell is written as `N/A` *with a reason*,
  not left blank.

## Status
Built P0 → P4 (2026-06-08). Coverage: **~580 sourced docs** —
54 operators × {overview,tuning,numerics,fusion} + **204 backend SOTA cards** (the Cartesian core);
~27 hardware, ~42 language, ~35 backend deep-dives; 41 cross-cutting docs (optimization/quantization/profiling/workflows);
case studies; auto-generated [`sota_matrix.md`](sota_matrix.md) + [`sota_registry.yaml`](sota_registry.yaml)
(regenerate via `index/_gen_registry.py`). See [`changelog.md`](changelog.md) for the per-phase log.

## Sources
- ROCm "AMD Instinct MI300X workload optimization": https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Performance reality (≈45% of peak on MI300X): https://arxiv.org/pdf/2510.27583
