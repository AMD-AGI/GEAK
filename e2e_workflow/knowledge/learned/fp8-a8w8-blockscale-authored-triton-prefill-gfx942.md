---
name: fp8-a8w8-blockscale-authored-triton-prefill-gfx942
description: After M-routing decode to CK, AUTHOR a Triton gemm_a8w8_blockscale for the RESIDUAL prefill-large-M head — +6.55% e2e on top of the CK decode swap (sglang fp8, gfx942).
keywords: [fp8, a8w8, blockscale, dense-gemm, authored-triton, prefill, large-m, sglang, gfx942, m-routing]
kernels: [gemm_a8w8_blockscale, _gemm_a8w8_blockscale_seed_kernel, triton_gemm_a8w8_blockscale]
platforms: [gfx942]
kernel_class: dense_gemm
regime: prefill
key: fp8_a8w8_blockscale dense GEMM · gfx942 · sglang · PREFILL large-M residual head (authored Triton, on top of the CK decode swap)
type: routing
confidence: ★★★
effect: iso 1.527× (large-M prefill op); e2e +6.55% BANKED (2 search reps/leg non-overlapping, cand_min 1234.9 > ref_max 1219.0), greedy output byte-identical to the accepted stack. In the Director-validated final stack (run 770.3 → 1256.7 tok/s, 1.644× validated_win, parity PASS).
lifecycle: active
last_seen: 2026-10-01
---
# Author a Triton gemm_a8w8_blockscale for the prefill-large-M residual head (after CK takes decode)
- lever: the fp8 a8w8 block-scale dense GEMM is the dominant head (~82% prefill GPU time). Once decode
  (M≤256) is M-routed to tuned CK (see fp8-a8w8-blockscale-overlay-gfx942.md), the RESIDUAL large-M prefill
  is where CK is a measured DEAD-END (CK ~1.4–1.9× SLOWER than stock Triton on every weight family). That
  residual is NOT exhausted by routing: an AUTHORED Triton gemm_a8w8_blockscale beats stock Triton +6.55% e2e.
- apply: hook sglang `fp8_utils` AFTER the CK decode overlay (h0) so dispatch stays M-routed — M>256 →
  authored Triton launcher, M≤256 → the accepted CK path, unchanged. The aiter alias
  `aiter.ops.triton.gemm_a8w8_blockscale` is a lazy __getattr__/back-compat finder, so bind the sglang
  fp8_utils SEAM itself (not the alias); reversible PYTHONPATH overlay, no site-packages edit.
- verify: live marker `[c0-authored-gemm] ENGAGED ... M=2973 N=7168 K=5120` fires on M>256 prefill calls
  ONLY; decode M≤256 still shows the CK path. Greedy parity byte-identical to the accepted stack (tok_agree 1.0).
- caution: single-seam RE-authoring of this head is EXHAUSTED at modest budget — a deepest-seam re-capture +
  re-author returned a BYTE-COPY (weighted 0.9926, iso ~1.0×, max_rel_err 0.0) = zero extra e2e. It stays the
  top target by Amdahl mass (84% head), but the only remaining lever is a GENUINELY NEW kernel (flydsl SOTA
  fp8 GEMM / a real split-K rewrite), not another re-capture — a flydsl author variant here measured iso
  1.014× → +0.15% e2e (rejected). Also verify deepest-seam engagement: a dispatcher that binds its launcher
  by direct reference at import time bypasses a post-hoc marker (installed_but_never_live → false
  deepest_verified); use a post-import hook, certify by 32/32 marked-vs-device match (method-verify-engagement.md).
- source: exp/e2e_*Qwen3-14B-FP8*/ 2026-09-30..10-01 (sglang TP1 gfx942, head h0/c0 authored-Triton leg);
  kernel technique → kernel_workflow/knowledge/learned/ (dense/quantized fp8 GEMM cards).
