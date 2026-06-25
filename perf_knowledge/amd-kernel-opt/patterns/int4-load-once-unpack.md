---
type: Optimization Pattern
title: int4 load-once-unpack-both-nibbles + scale/zp dedup
description: For int4 W4A16 GEMM, read each packed byte once and unpack both nibbles; broadcast group-shared scale/zp instead of re-reading.
tags: [domain-moe, domain-gemm, bottleneck-memory, lever-kernel-body, gfx942]
bottleneck: L2-read-BW
lever_class: kernel-body
median_speedup: up to 5.19x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
int4 (W4A16) MoE/GEMM that is **L2-read-bandwidth bound**: the packed weights and their
group scales/zero-points are read redundantly.

# Mechanism
- **Load-once, unpack-both-nibbles**: read each packed int4 byte a single time and unpack
  the high and low nibble in registers, instead of two strided reads (~3.8× alone).
- **Scale/zp group-dedup broadcast**: a group of weights shares one scale/zp — read it
  once per group and broadcast, rather than per-element (stacks to ~5.2×).
- Drop redundant high-nibble masking (`& 0xF`) where the shift already isolates it.

# Evidence
- [fused_moe_int4_w4a16](/cases/fused-moe-int4-w4a16.md) — L10 load-once-unpack (~3.8×) + L11 scale/zp dedup → **5.19×** geomean; near the body-only L2-BW ceiling.

# Caveats
- On AMD gfx942 FP8 is **FNUZ** and int dequant scales are arch-specific — re-derive when
  porting (a correctness gate, not a perf knob).
- This is body-only; the surrounding host/grader path was locked in the source case, so
  the ceiling is the L2 read bandwidth.

# Citations
1. spare_kernels/arena_tasks/triton2triton/fused_moe_int4_w4a16/RESULTS.md
2. AgentKernelArena/tasks/campaign20/baseline/fused_moe_int4_w4a16/RESULTS.md
