# Learned — index of distilled experience cards

Open the cards matching your run's `(kernel_class, gfx, regime)` as **additional, advisory priors** —
they only ADD candidates to try, never remove any or replace measurement. The on-box bake-off + e2e gate
is always the judge (see `README.md` philosophy). One line per card, grouped by reuse key. **Cap: ≤40 lines.**
Confidence (a hint strength, not authority): ★ noise/unverified · ★★ single non-overlap or ≥2 consistent · ★★★ ≥2 non-overlap or verified e2e.

## dense GEMM
- [gfx942 · sglang bf16] aiter per-shape DB tune ★★★ **+2.23% e2e (verified)** — (aiter-bf16-tuned-gemm-gfx942.md)
- [gfx942 · sglang fp8 a8w8 blockscale] per-(N,K) M-bucketed Triton config-JSON overlay ★★★ ~1.10× prefill — (fp8-a8w8-blockscale-overlay-gfx942.md)
- [gfx950 · vLLM MXFP8 E8M0] dense `tl.dot_scaled` STATIC tiles (decode BK256/prefill BM128) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)

## MoE grouped GEMM
- [gfx950 · vLLM MXFP8 E8M0] grouped `dot_scaled` STATIC tiles (GEMM1-decode BN64+BK256) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)

## attention
- [gfx942 · sglang hybrid prefill] `--attention-backend triton` cheap flag win ★★★ +~5% e2e — (attention-backend-triton-gfx942.md)
- [gfx942 · vLLM decode, non-pow2 KV block] live = editable in-tree Triton paged kernel ★★ ~+1% — (paged-attn-nonpow2-gfx942.md)

## linear-attention / FLA / mamba (editable Triton)
- [gfx942 · prefill-dominated hybrid] stack-and-compound cluster; Amdahl pre-dispatch screen ★★★ — (editable-triton-cluster-amdahl.md)

## method (cross-model, applies to any run)
- engagement verification: one-shot stderr banner + log grep ★★★ — (method-verify-engagement.md)
- e2e A/B: pinned port, interleaved, non-overlap gate ★★★ — (method-e2e-ab-harness.md)
- cuda/HIP-graph-safe integration (the #1 e2e killer) ★★★ — (method-cudagraph-safe-integration.md)
