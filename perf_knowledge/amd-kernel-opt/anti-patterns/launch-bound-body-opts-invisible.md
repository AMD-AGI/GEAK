---
type: Anti-Pattern
title: GPU-body optimizations are invisible when launch-bound
description: Tuning the kernel body wastes effort when wall-clock is dominated by host dispatch.
tags: [domain-any, bottleneck-launch, methodology, gfx942]
bottleneck: launch
lever_class: kernel-body
timestamp: 2026-06-22T00:00:00Z
---

# The trap
Investing attempts in BLOCK_SIZE, cumsum vectorization, LDS layout, masking, etc. on an op
whose runtime is ~all host dispatch. Every body change measures as noise, leading to wasted
iterations and false "at ceiling" conclusions about the algorithm.

# Why
On this stack a launch costs ~20–25 µs. If the GPU body is ~9 µs it is **fully hidden**
behind the launch; nothing inside the kernel can move wall-clock. Examples:
[_topk_forward](/cases/topk-forward.md) (GPU ~9 µs under ~24 µs dispatch),
[write_req_to_token_pool](/cases/write-req-to-token-pool.md),
[kernel_unified_attention_2d](/cases/unified-attention-2d.md) (body changes <1% noise).

# The fix
Measure GPU utilization **first** (see
[bottleneck-first](/methodology/bottleneck-first-classification.md)). If launch-bound,
move to host-side levers: [graph replay](/patterns/host-graph-replay.md),
[do_not_specialize launcher](/patterns/triton-launcher-do-not-specialize.md). A genuine
resource win (e.g. cutting LDS 34→16 KB) can still be worth keeping for occupancy even with
no wall-clock change — record it as such, don't claim a speedup.

# Citations
1. KernelForge/results/_topk_forward/tasks/cli/*/workspace/optimization_report.md
2. KernelForge/results/kernel_unified_attention_2d/tasks/cli/*/workspace/optimization_report.md
