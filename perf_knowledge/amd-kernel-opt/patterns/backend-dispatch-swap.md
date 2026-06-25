---
type: Optimization Pattern
title: Backend dispatch swap to a faster prebuilt kernel
description: Route the op to an already-faster prebuilt kernel (e.g. CK→ASM, ASM→CK) on the host side, gated with a correctness fallback.
tags: [domain-attention, domain-moe, domain-gemm, bottleneck-memory, lever-backend-swap, no-rebuild, gfx942]
bottleneck: memory/compute (kernel-dependent)
lever_class: host-side / backend-swap
median_speedup: 1.19x-4.39x
timestamp: 2026-06-22T00:00:00Z
---

# When to use
The fastest implementation for this shape regime **already exists** in the stack
(aiter ships prebuilt ASM and CK kernels), but the default dispatch path picks a slower
one. This is often the single highest-ROI lever and needs no kernel authoring.

# Mechanism
Swap the host-side dispatch to the faster backend, **gated** with `try/except` fallback
to the original op and a numerics check (cos/SNR) so a missing kernel or layout mismatch
degrades gracefully. Keep any layout massaging (e.g. V-shuffle) **outside the timed /
steady-state path**.

# Evidence
- [paged_attention_decode](/cases/paged-attention-decode.md) — **4.39×** (CK/HIP `paged_attention_v1` → ASM `pa_fwd_asm` / `pa_bf16_noquant_gqa8`); also higher SNR (49.9 vs 48.7 dB)
- [moe_gemm_fp8_blockscale](/cases/moe-gemm-fp8-blockscale.md) — **1.19×** (1-stage ASM → 2-stage CK; +SNR 23→32.7 dB)
- [gemm_a8w8_blockscale](/cases/gemm-a8w8-blockscale.md) — Triton → CK bpreshuffle (also the decisive e2e GEMM lever)

# Caveats
- The swap may be **un-integrable downstream** even when the micro-win is real: in SGLang
  the `pa_fwd_asm` decode winner needs a block-structured 5D K + shuffled-V KV layout;
  the engine uses flat page_size=1, and an on-the-fly shuffle erases the win. Verify the
  consuming framework's data layout before claiming an e2e gain.
- A winning CK/ASM path may be **arch-gated** (gfx950-only); check it is reachable on the
  target (see [per-shape dispatch](/patterns/per-shape-kernel-dispatch.md)).

# Citations
1. head_kernels/campaign20/FINAL_REPORT.md (pos14)
2. spare_kernels/arena_tasks/hip2hip/paged_attention_decode/RESULTS.md
3. e2e_runs/minimax_m25/REPORT.md
