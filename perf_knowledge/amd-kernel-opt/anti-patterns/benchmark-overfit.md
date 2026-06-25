---
type: Anti-Pattern
title: Benchmark over-fit (false speedups that don't transfer)
description: Gains that exist only because of how the benchmark or harness is built — identity-keyed caches, reused tensors, harness bugs.
tags: [domain-any, methodology, transferability, gfx942]
bottleneck: n/a
lever_class: n/a
timestamp: 2026-06-22T00:00:00Z
---

# The trap
A measured speedup that is real on the benchmark but vanishes (or is meaningless) in real
serving/training. The most dangerous failure mode because the number looks great.

# Forms seen
- **Harness bug, not a kernel win.** The retracted **17.39×** on
  [chunk_scaled_dot_kkt](/cases/chunk-scaled-dot-kkt.md) came from a varlen harness
  launching `grid=(NT, B*H)` instead of `(NT, H)` — over-launching by B. The "grid
  collapse" merely removed redundant work; against a correct baseline the honest gain is
  ~1.5×. **Rule of thumb: a speedup larger than the change can structurally produce is a
  bug until proven otherwise.**
- **Identity-keyed caches on non-weight tensors.** `dict`/`lru_cache`/`weakref` keyed on
  `id(activation)` or `id(grad_out)` — real-step hit rate ≈ 0; only the benchmark's reused
  tensor objects make it look free. Adding `_version`/shape/dtype to the key does not fix it.
- **Graph-replay that games the timing loop.** Accepted only if it survives fresh-tensor
  validation; rejected on [_topk_forward](/cases/topk-forward.md) for this reason.

# How to defend
- Re-measure with **fresh input tensors each iteration**; only weights keep stable identity.
- A/B the baseline against a *correctly-launched* reference before trusting a ratio.
- Verify the win still appears **e2e** (decode-bound vs prefill-only) before claiming it.
- Weight quant/preshuffle caches are allowed but bounded — key on `(id(weight), _version)`,
  invalidate after `optim.step()`; report the bounded gain, not the headline.

# Citations
1. head_kernels/campaign20/FINAL_REPORT.md (Integrity / pos8 retraction)
2. KernelForge system_prompt — "optimizations must transfer to real training"
