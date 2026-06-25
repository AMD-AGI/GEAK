---
okf_version: "0.1"
---

# AMD GPU Kernel Optimization Knowledge Base

Distilled success experience from the agent-kernel-arena optimization campaigns on
AMD Instinct MI300X (gfx942/CDNA3). Built **2026-06-22** from the *documented* corpus
only (KernelForge `optimization_report.md`, spare_kernels `OPT_NOTES.md`, campaign20
`RESULTS.md` + `FINAL_REPORT.md`, task-specific perf notes). High-speedup automated
geak runs (>10×, no authored notes) are deliberately excluded pending source-diff review.

Each entry is OKF-conformant (YAML frontmatter + structural markdown). Primary
navigation axes: **operator domain** and **bottleneck class** (see tags).

## Start here
- [Methodology: bottleneck-first optimization](/methodology/bottleneck-first-classification.md) — classify *before* you tune.

## Patterns (reusable success experience)
The single biggest meta-lesson: **host-side levers (no kernel rebuild) deliver most of
the ROI** — graph replay, backend dispatch swaps, and block_m schedules.

- [Host-side HIP-graph replay](/patterns/host-graph-replay.md) — kill launch overhead on tiny ops.
- [Backend dispatch swap](/patterns/backend-dispatch-swap.md) — route to a faster prebuilt kernel.
- [Per-shape kernel dispatch](/patterns/per-shape-kernel-dispatch.md) — CK-vs-ASM by M.
- [block_m to routing sparsity](/patterns/block-m-routing-sparsity.md) — bm64→bm16 for sparse MoE decode.
- [CK pipeline V3→V1 for occupancy](/patterns/ck-pipeline-v1-occupancy.md) — single LDS buffer → 2 blocks/CU.
- [Triton do_not_specialize launcher](/patterns/triton-launcher-do-not-specialize.md) — cut host dispatch.
- [Hoist K-loop-invariant math](/patterns/hoist-kloop-invariant-math.md) — divides, masks, pointers.
- [int4 load-once-unpack](/patterns/int4-load-once-unpack.md) — cut L2 read BW.
- [L2-locality pid remap](/patterns/l2-locality-pid-remap.md) — super-grouping / XCD-aware.
- [Launch-config autotune](/patterns/launch-config-autotune.md) — warps/nonkdim/kpack/stages/BLOCK_M.
- [Single-pass attention](/patterns/single-pass-attention.md) — online softmax, drop the reduce launch.
- [Empty (not zeros) output buffer](/patterns/output-empty-not-zeros.md) — skip a needless memset.

## Anti-patterns (verified negative results)
- [Non-temporal load regression](/anti-patterns/non-temporal-load-regression.md)
- [Benchmark over-fit](/anti-patterns/benchmark-overfit.md)
- [Body opts invisible when launch-bound](/anti-patterns/launch-bound-body-opts-invisible.md)
- [Numerics-gate violation](/anti-patterns/numerics-gate-violation.md)

## Catalog (all referenced kernels)
The full registry of LLM-inference kernels referenced here, grouped by domain with best
measured speedup and case-study links: [catalog/kernel-registry.md](/catalog/kernel-registry.md).

## Cases (per-kernel evidence)
See [cases/index.md](/cases/index.md).

## Data
Raw aggregated speedups (vendored in [catalog/](/catalog/kernel-registry.md)):
[kernel_speedups.md](/catalog/kernel_speedups.md),
[kernel_speedups_task_results.csv](/catalog/kernel_speedups_task_results.csv) (107 kernels),
[kernel_speedups_llm_inference.csv](/catalog/kernel_speedups_llm_inference.csv) (91 LLM kernels).
