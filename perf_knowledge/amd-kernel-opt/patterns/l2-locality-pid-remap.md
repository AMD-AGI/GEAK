---
type: Optimization Pattern
title: L2-locality via pid super-grouping / XCD-aware remap
description: Reorder workgroup ids so adjacent tiles share a warm L2 / stay on one XCD chiplet, cutting redundant HBM reads.
tags: [domain-gemm, domain-moe, bottleneck-memory, lever-kernel-body, gfx942]
bottleneck: memory / L2 reuse
lever_class: kernel-body
median_speedup: 1.05x-1.52x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
A GEMM where consecutive workgroups re-read overlapping A/B tiles from HBM that could
have been served from L2. gfx942 has 8 XCD chiplets; default linear pid ordering scatters
reuse across chiplets.

# Mechanism
- **Super-grouping** (`GROUP_M` widening): order pids so a group of row-tiles reuses the
  same column tiles while they are hot in L2.
- **XCD-aware remap**: chiplet-chunk the workgroup id so consecutive ids land on one XCD,
  sharing that XCD's L2. Sweep the group-N (WGM) width.
- Combine with a K-peel for the ragged tail.

# Evidence
- [_gemm_a16_w16](/cases/gemm-a16-w16.md) — grouped pid (GROUP_M=8 + remap_xcd) → **~1.5×**, plus dropping a forced `.cg` on the B load (use L1) +4%
- [fused_moe (fp8 w8a8)](/cases/fused-moe-fp8-blockscale.md) — adaptive GROUP_M=4 L2 swizzle + XCD-aware pid remap contribute to **1.36×**

# Caveats
- GROUP_M is shape-sensitive — values within ~1% of each other are noise; pick the
  balanced one and re-measure across shapes.
- XCD remap helps only when there is cross-tile reuse; on streaming kernels it is neutral.

# Citations
1. KernelForge/results/_gemm_a16_w16_kernel/tasks/cli/*/workspace/optimization_report.md
2. KernelForge/results/fused_moe_kernel/tasks/cli/*/workspace/optimization_report.md
