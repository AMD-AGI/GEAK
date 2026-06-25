---
type: Optimization Pattern
title: CK pipeline V3→V1 for occupancy
description: Force the single-LDS-buffer CK pipeline (V1) on occupancy-bound short-K GEMMs to fit 2 blocks/CU instead of 1.
tags: [domain-moe, domain-gemm, bottleneck-occupancy, lever-host-side, no-rebuild, gfx942]
bottleneck: occupancy (LDS-limited)
lever_class: host-side / codegen-instance
median_speedup: 1.08x-1.31x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
A CK block-scale GEMM is **occupancy/latency-bound, not HBM-saturated** (rocprofv3 shows
HBM well below achievable, e.g. ~3.0 of ~4.8 TB/s at large M). The V3 2-LDS double-buffer
needs ~64 KB LDS → only 1 block/CU on gfx942.

# Mechanism
Switch the pipeline to **V1** (single LDS buffer, ~24–32 KB) so 2 blocks/CU co-reside,
lifting achieved HBM (e.g. ~3.0→3.9 TB/s) and hiding latency. Done by selecting the V1
DeviceOp instance / `BlockGemmPipelineVersion::v1` in the codegen list — bit-exact, no
algorithm change. Keep V3 as default for the mid regime where it ties or wins.

# Evidence
- [moe_stage2](/cases/moe-stage2.md) — V1 (single ~24 KB buffer) → **1.31×** on short-K down-proj
- [moe_stage1](/cases/moe-stage1.md) — V3→V1 32 KB → 2 blocks/CU, **~1.16× at token≥16384**

# Caveats
- Only pays off when the kernel is genuinely occupancy-bound; re-profile to confirm HBM
  slack first. If HBM is already saturated this does nothing.
- `Nswizzle` / interwave variants in the same instance space caused GPU memory faults /
  compile failures here — validate each instance swap for correctness.

# Citations
1. KernelForge/results/moe_stage2/tasks/cli/*/workspace/optimization_report.md
2. KernelForge/results/moe_stage1/tasks/cli/*/workspace/optimization_report.md
3. AgentKernelArena/tasks/campaign20/baseline/moe_stage1/RESULTS.md
