---
type: Anti-Pattern
title: Throughput levers that break the numerics gate
description: A faster kernel that fails the correctness/precision gate is a failed attempt, not a win.
tags: [domain-gemm, domain-attention, methodology, transferability, gfx942]
bottleneck: n/a
lever_class: any
timestamp: 2026-06-22T00:00:00Z
---

# The trap
Accepting a speedup whose numerics drifted outside tolerance — a fast-but-wrong kernel.
Most acute in low-precision (FP8/FP6/FP4/MX) where a packing/scale/encoding mistake looks
like a perf win.

# Examples
- [_w8a8_triton_block_scaled_mm](/cases/index.md) — the available throughput levers
  broke the numerics gate, so the kernel stays at **1.00× (at ceiling)** rather than
  shipping a wrong result.
- A numerically-wrong splitKV attention graph (campaign20 pos13) reported 2.50/1.79/1.39×
  but was **FLAGGED**; only the correctness-exact single-pass path counted (honest 1.58×).

# Rules
- **Correctness before performance**, validated *after every change* on representative shapes.
- For low precision, validate in separable steps: (1) format conversion/packing alone,
  (2) scale handling alone, (3) kernel math vs a higher-precision reference. gfx942 FP8 is
  **FNUZ** (E4M3 bias 8, max 240, no Inf) vs gfx950 **OCP** — re-derive scales when porting;
  the encoding is a correctness gate, not a knob.
- Prefer bit-exact / SNR-improving changes; a backend swap that also *raises* SNR (e.g.
  paged_attention_decode 48.7→49.9 dB) is the gold standard.

# Citations
1. KernelForge/results/_w8a8_triton_block_scaled_mm/tasks/cli/*/workspace/optimization_report.md
2. head_kernels/campaign20/FINAL_REPORT.md (pos13 FLAGGED)
