---
key: occupancy and wave-count tuning on an MFMA-bound fp16 dense GEMM whose compiler pins waves-per-eu, gfx950 CDNA4
type: anti-pattern
confidence: ★★
effect: ~1.00x on four occupancy-directed attempts (COARSEN=3, 32x32 MFMA, tile shrink, intra-workgroup ping-pong) and clearly negative on two more: the 8-wave workgroup costs about 80% and the waves-per-eu=1 register double-buffer is ~50% slower, on every case (small-M and large-M alike)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound
description: gfx950 dense GEMM: with waves-per-eu pinned at 2 by the backend, four occupancy-raising directions returned ~1.00x or worse — spend the round elsewhere
keywords: ['dense-gemm', 'occupancy', 'waves-per-eu', 'num-warps', 'ping-pong', 'mfma', 'gfx950', 'compute-bound', 'anti-pattern']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
layer: learned
lifecycle: active
---
# Occupancy axis closes when the backend pins waves-per-eu
- lever: Before budgeting rounds on occupancy for an MFMA-bound GEMM, check whether the AMD backend emits an unconditional amdgpu-waves-per-eu attribute at the end of LLIR generation; if it does, no in-body primitive raises occupancy and the whole axis can be probed with one cheap test instead of four rounds.
- apply: Two knobs that look alike behave differently: waves-per-eu is frozen from inside the kernel body, while num_warps is only a harness default and a launcher-side override can force 8 — so test them separately rather than concluding both are frozen.
- verify: Compare achieved workgroups-per-CU, not just the launch config: a variant that changes the wave count without changing workgroups-per-CU has not engaged, and re-time per case against the frozen baseline.
- pitfall: the 8-wave ping-pong schedule from a published recipe looked like free throughput -> it collapses 2 workgroups/CU to 1 and loses the inter-tile overlap that hides load latency on this low-intensity shape -> the 4-wave, occupancy-2 configuration wins; the intra-workgroup variant additionally corrupted the accumulator.
- caution: also verify the shape regime the published ping-pong result was measured on — that number comes from a large square high-K GEMM, and its own write-up scopes ping-pong out of skinny / low-K shapes like this one.
- source: run _gemm_a16_w16_kernel-ch16h (16h single-kernel budget, 44 passes), 2026-08-12
