---
type: Optimization Pattern
title: block_m sized to MoE routing sparsity
description: Shrink the GEMM row-tile (block_m 64→16) for sparse decode routing so the tile is not mostly padding.
tags: [domain-moe, bottleneck-occupancy, bottleneck-memory, lever-host-side, no-rebuild, gfx942]
bottleneck: occupancy / wasted-bytes
lever_class: host-side
median_speedup: 1.08x-1.31x (kernel); decisive e2e lever (+13%-37%)
timestamp: 2026-06-22T00:00:00Z
---

# When to use
MoE grouped GEMM at **decode** (small tokens, sparse routing). With ~8 tokens/expert a
64-row routing block is ~87–97% padding → wasted weight reads and idle MFMA lanes.

# Mechanism
Host-side `block_m` schedule keyed on token count, e.g.
`token≤256 → bm16 / 257–768 → bm32 / 769–8192 → bm64-V1 / …` (boundaries env-overridable).
No C++ rebuild — just select the right CK instance. Often stacked with
[CK V3→V1 pipeline](/patterns/ck-pipeline-v1-occupancy.md) for the large-token regime.

# Evidence
- [moe_stage2](/cases/moe-stage2.md) — **1.31×** geomean (bm32-V1 dense / bm16 sparse)
- [moe_stage1](/cases/moe-stage1.md) — **~1.08×** (bm16 ≤256 gave 1.196× at token=256)
- **e2e MiniMax-M2.5**: decode `bm64→bm16` is the single decisive lever behind +13% to +37% output tok/s (decode-bound serving).

# Caveats
- A prefill-only tweak (bm64 large-token path) is **e2e-neutral** on a decode-bound
  workload — confirm where the serving time actually goes before claiming an e2e gain.
- The CK tile itself is constraint-locked; this pattern selects instances, it does not
  retile the kernel.

# Citations
1. AgentKernelArena/tasks/campaign20/baseline/moe_stage2/RESULTS.md
2. AgentKernelArena/tasks/campaign20/baseline/moe_stage1/RESULTS.md
3. exp/e2e_minimax_20260615_141446/RESULTS_SUMMARY.md
