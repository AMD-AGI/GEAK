---
title: FlyDSL GEMM changes to test
kind: language
gens: [gfx942, gfx950]
updated: 2026-10-16
source_commit: AMD-AGI/GEAK@c0a1f937
---

# FlyDSL GEMM changes to test

For an authored tiled matmul, identify logical `M,N,K`, operand layout/dtype,
wave ownership, LDS reuse and the output mapping. For wrapper configuration use
[knobs](knobs.md); for shared rewrite constraints use
[authoring optimization](authoring_optimization.md).

| Part | What to inspect or change |
|---|---|
| Tiles | MFMA divisibility, vector alignment, output-grid parallelism and edge masks; larger tiles trade reuse against register/LDS cost |
| LDS staging | Reuse, layout and load-to-use distance; distinguish allocation pressure, bank conflicts and exposed dependency latency |
| Prefetch | Issue independent loads earlier or carry the next tile; check tail handling and added live state |
| Scheduling | Match `sched_mfma`, `sched_vmem`, `sched_dsrd`, `sched_dswr` to the actual loop; copied constants need verification |
| Epilogue | Compare direct stores with fragment reordering; account for extra LDS/barriers and conversion semantics |

`vmcnt`, `lgkmcnt`, `ds_*` and barrier stalls can locate a cost but do not uniquely
identify its cause. Test a changed layout or schedule and inspect whether the
expected movement changed. Static MFMA density alone cannot establish utilization.
Swizzled stores and loads must use matching address transforms. Padding also changes
LDS allocation, and wider loads need valid alignment/masks.

Measure correctness and unprofiled full-call latency over the required shapes.
A good schedule for one shape may regress another; report the task's actual
aggregate and per-case effects. Neither structural changes nor parameter tuning
has a universally preferred order.

## Sources

- Imported source: `AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_gemm_optimization.md`
- [AMD matrix-core programming](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [CDNA4 GEMM example](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
- [Dense GEMM integration](../../operators/dense_gemm/backends/flydsl.md)
