---
key: bf16 vLLM Triton fused MoE on gfx950 whose graded mix is dominated by a small-M decode case, after per-bucket tile tuning has already landed
type: anti-pattern
confidence: ★★
effect: over 15 rounds and 38 direction slots: MFMA layout / kpack / waves-per-eu / num_stages all within +/-0.3% at decode and at the large prefill; large BLOCK_K 2-8% slower at every tile; XCD swizzle ~1.00x in both directions; fp32 and bf16 atomic epilogues 2.2-3.8x slower than the round trip they remove, at every size; two edits that strictly REMOVE inner-loop work each lost 6-8%; decode ends at ~0.9 of the measured achievable roof
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both
description: bf16 Triton fused MoE, decode-weighted mix: MFMA-layout knobs, XCD swizzle, atomic epilogues, decode fusion and work-removing edits all priced ~1.00x or worse
keywords: ['anti-pattern', 'closed-axis', 'moe-grouped-gemm', 'bf16', 'triton', 'gfx950', 'mfma-nonkdim', 'xcd-swizzle', 'atomics', 'kernel-fusion', 'roofline', 'launch-overhead']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-17
origin_kernels: ['mi355x_vllm_triton_fused_moe_gemma4']
---
# Axes that stayed closed on a bf16 fused MoE with a decode-weighted mix
- lever: Treat these as low-prior once per-bucket tile configs are tuned: MFMA layout knobs, larger BLOCK_K, XCD swizzle, atomic epilogues, and any fusion at decode; the live axes here were cache policy, dispatch count in the periphery, and per-shape launch configs.
- apply: Price a fusion candidate by grid arithmetic first (M-blocks vs CUs: less than one wave means the N-loop it eats is already free) and a gate/up pairing by hidden-dim arithmetic (an intermediate dim that is not a multiple of BLOCK_N bought +9.1% MACs to delete one dispatch).
- verify: Measure the achievable roof with your own sweep before sizing headroom, and measure the launch floor with an empty kernel at the real grid: ~70% of this periphery was pure launch floor, so a free body in every periphery kernel was worth only ~1.2% weighted.
- pitfall: Deleting an even-K mask and replacing a modulo N-wraparound with a masked tail both removed real inner-loop work and both lost 6-8% -> the removed work was hidden under memory latency while the new control flow was not -> price a work-removal edit against the same A/B as any other candidate.
A closure taken with the large-prefill launch config held for ten rounds at decode and was worth -22% on that dispatch when re-swept at the decode shape -> record the shape a closure was measured at.
- caution: Also verify the roofline you are quoting is measured rather than nameplate: the nameplate figure over-stated the achievable roof by ~13% here and made a dead-end direction look funded.
- source: run mi355x_vllm_triton_fused_moe_gemma4-bmk7-12h, 2026-08-17, rounds 1-15 closed-lane ledger, director-validated
