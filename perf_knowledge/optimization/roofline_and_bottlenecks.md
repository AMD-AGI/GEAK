---
title: roofline and bottleneck classification
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-10-16
sources:
  - https://rocm.github.io/rocprofiler-compute/performance_model.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# Roofline and bottlenecks

Use a roofline as a hypothesis about resource cost. For work `W`, traffic `Q`,
compute capacity `P` and bandwidth `B`, the optimistic service time is
`max(W/P, Q/B)`. Use compatible units, dtype, memory level and measurement
boundary. `W/Q` versus `P/B` identifies the tighter modeled resource; it does
not establish the observed bottleneck.

Start with accepted full-operation timing and the actual source/shape/device.
For GEMM, `2*M*N*K` useful FLOPs and input/output bytes read or written once are
a logical model, not measured instruction work or HBM traffic. Padding, repeated
loads, cache reuse, metadata and workspace may change implementation cost.
See [roofline collection and interpretation](../profiling/roofline_on_mi.md).

| Evidence suggesting a cost | Experiment to consider |
|---|---|
| Arithmetic resources near an applicable comparator | Tile/instruction scheduling, or less executed work; see [MFMA scheduling](mfma_scheduling.md) |
| High traffic at the relevant memory interface | Coalescing, reuse or fewer materialized passes; see [fusion](kernel_fusion_strategy.md) |
| Low compute and memory rates with stall/parallelism evidence | Test dependencies, pipelining or workgroup geometry; see [memory pipelining](memory_pipelining.md) |
| LDS movement or allocation pressure | Test layout and ownership; see [LDS and bank conflicts](lds_and_bank_conflicts.md) |

These are candidate explanations. Occupancy alone does not explain latency, and
memory efficiency alone does not prove traffic is necessary. Check that the
proposed change occurred in the measured path and that competing costs did not
move elsewhere. Preserve the numerical/state contract, then measure unprofiled
full-operation and relevant workload results.

Label specification roofs and empirical/library comparators separately. There
is no universal 45–55% software ceiling or fixed efficiency target across shapes,
dtypes and products. A matched library baseline is useful evidence of an attained
rate under its conditions. Report conditional model percentages alongside latency
and speedup; do not turn the gap into guaranteed headroom or prune an idea because
its current implementation appears saturated.
