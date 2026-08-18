---
key: the late rounds of a bf16 fused norm+GEMM campaign on gfx950/MI355 whose prefill arm is at the HBM read roof and whose decode arm is at the graph-dispatch floor
type: anti-pattern
confidence: ★★
effect: disconfirming, all isolated against the frozen baseline: MFMA on the prefill inner phase is bounded by the measured L2 read roof at its 2.15 MB working set to a whole-direction ceiling below the incumbent; cutting 21% of prefill phase-1 instructions moved runtime 0.0%; halving the chunk costs 3.7-19% before any mechanism runs; a double buffer +38%; wave specialisation +5.5%; non-temporal loads +5.3% on the largest prefill case; guard-free block_m>1 on decode hits a register cliff and occupancy 6 measures ~2x slower than the prediction; a CU-aligned decode grid -2.04% +/- 0.41% SEM with sign count 0/10 on one decode case and null on the other
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both
description: Once a fused norm+GEMM path sits at its HBM/L2/dispatch roofs, MFMA, phase overlap, occupancy, NT loads and CU-aligned grids all return <=1.00x
keywords: ['anti-pattern', 'closed-axis', 'roofline', 'occupancy', 'double-buffering', 'non-temporal-loads', 'mfma', 'software-pipelining', 'wave-quantization', 'dispatch-floor', 'gfx950', 'tilelang']
kernels: ['mhc_pre_big_fuse', 'mhc_fused_decode_tilelang']
platforms: ['gfx950']
kernel_class: fused_norm_gemm
regime: both
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-14
roofline: prefill phase 1 at ~94% of the measured HBM read roof with zero removable bytes; prefill phase 2 at ~91% of the packed-dot issue roof; decode at the graph-dispatch floor
levers: ['compute.mfma', 'compute.software-pipelining', 'compute.occupancy', 'mem.non-temporal']
---
# Axes that stayed closed on a roof-bound fused norm-GEMM path
- lever: Before funding a structural round on an arm already near a roof, price the ceiling: measure bandwidth elasticity (perturb traffic and read the runtime response) and delete the candidate work entirely to get a free upper bound; here elasticity measured ~0.1, so the gap above the byte floor was never serialisation and every overlap/skew/double-buffer variant was paying for a cure to a bound the op does not have.
- apply: Get the bound from a deletion control rather than from a percent-of-roof figure: emptying the whole decode workgroup body left the residual chain intact and flat across a 7x change in workgroup count, which identifies it as a fixed dispatch/drain cost and caps any intra-kernel win at a few percent weighted.
- verify: Gate each of these on a same-process interleaved paired A/B with a per-rep sign count and a stated error bar; the wave-quantization result was a clean unanimous negative rather than the expected null, and only the paired form could show that.
- pitfall: A grid-shape hypothesis looked like it needed a new mechanism -> an existing flag already expressed exactly the shape under test -> flipping the flag off against a control pinned to production settled it in one probe, and showed the fractional tail wave is a purchase, not a cost, because the extra workgroups take the iterative normalisation off the critical path.
- caution: Also verify the non-temporal axis by form before closing it: the loads regressed, while a one-line traversal reversal delivered the cache residency the hint was meant to buy, and stores on write-once outputs were the only surviving form.
- source: run mi355x_vllm_tilelang_mhc_fused_post_pre-bmk7-12h, rounds 5, 8, 10, 13 and 15 (five disproof rounds), director validation accepted 2026-08-14
