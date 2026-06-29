# Learned — index of distilled experience cards

Open the cards matching your run's `(kernel_class, gfx, regime)` as **additional, advisory priors** —
they only ADD candidates to try, never remove any or replace measurement. The on-box bake-off + e2e gate
is always the judge (see `README.md` philosophy). One line per card, grouped by reuse key. **Cap: ≤40 lines.**
Confidence (a hint strength, not authority): ★ noise/unverified · ★★ single non-overlap or ≥2 consistent · ★★★ ≥2 non-overlap or verified e2e.

## dense GEMM
- [gfx950 · vLLM MXFP8 E8M0 decode-bound] dense-linear split-K/fused decode-tile Triton rewrite ★★★ **+21.8% e2e (verified, gsm8k-clean); decode-driven (converts only at high conc); grouped-MoE GEMM resists (~1.1× ceiling)** — (mxfp8-linear-decode-rewrite-gfx950.md)
- [gfx942 · sglang bf16] aiter per-shape DB tune ★★★ **+2.23% e2e (verified)** — (aiter-bf16-tuned-gemm-gfx942.md)
- [gfx942 · sglang fp8 a8w8 blockscale] per-(N,K) M-bucketed Triton config-JSON overlay ★★★ ~1.10× prefill — (fp8-a8w8-blockscale-overlay-gfx942.md)
- [gfx950 · vLLM MXFP8 E8M0] dense `tl.dot_scaled` STATIC tiles (decode BK256/prefill BM128) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)

## MoE grouped GEMM
- [gfx950 · vLLM MXFP8 E8M0] grouped `dot_scaled` STATIC tiles (GEMM1-decode BN64+BK256) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)
- [gfx942 · vLLM int4 W4A16] per-shape fused-MoE Triton config tune via `VLLM_TUNED_CONFIG_FOLDER` (env, ZERO HBM; N=moe_int//TP) ★★★ +11-18% e2e (13 confirms, TP8 & TP4) — (moe-int4-w4a16-tune-gfx942.md)
- [gfx942 · vLLM int4 W4A16] **apply-back** a FlyDSL MoE rewrite into live vLLM (load-time in-place same-byte convert — re-home BOTH weight AND scale params, key by weight.data_ptr, graph mode; REVERSIBLE-OVERLAY variant = 0 site-packages mutation) ★★★ **+77% e2e at FULL fair config (mem 0.9, 262144, 257→455); re-confirmed via overlay +63.57% (329.6→539.1) & +66.02% (319.6→530.6), 4 Director-verified confirms, all non-overlap; fix scale-dup leak (THAT was KV OOM, not repeat_interleave); eager is a trap; pin FLYDSL_ROOT; sub-op GEMM1-only cand can't rebind at fused seam** — (flydsl-moe-applyback-gfx942.md)
- [gfx942 · vLLM int4 W4A16] **GEMM2-leg levers** on the landed FlyDSL shim (after apply-back GEMM2 is new #1 head ~43-47% eff) ★★ — TWO distinct levers: (1) TILE-TUNE stage-2 prefill tile_m 32→64 via LIVE _pick_tiles (stage-aware, ZERO HBM) **+10.35% e2e (531.7→586.7, non-overlap, 12/12 byte-exact); iso entry does NOT auto-engage (patch _pick_tiles+live counter); same-tile re-tune EXHAUSTED (2 nulls)**; (2) FUSED-XROWS GEMM2 fused-SwiGLU+un-replicated-X (FLYDSL_MOE_FUSED_XROWS=1, the structurally-new angle) **+6.49% e2e STACK provisional (engaged x4, parity-safe, OVERLAPPING at 2 reps→verify more reps+gsm8k; splits MoE-GEMM into gemm1#1 31.5%+gemm2#2 14.9% eff, target gemm1 next)** — (flydsl-moe-gemm2-tiletune-gfx942.md)

## attention
- [gfx942 · sglang hybrid prefill] `--attention-backend triton` cheap flag win ★★★ +~5% e2e — (attention-backend-triton-gfx942.md)
- [gfx942 · vLLM decode, pow2+non-pow2 KV block, +MLA TRITON_MLA] live=editable in-tree Triton → Tier-C rewrite (pow2 ROCm/CK→author); op bake-off N/A ★★★ ~+1-4% — (paged-attn-nonpow2-gfx942.md)
- [gfx950 · vLLM block-sparse NSA GQA prefill] custom kernel, no lib swap; live = editable in-tree Triton → Tier-C rewrite ★★ ~5.6% head — (sparse-attn-nsa-triton-gfx950.md)

## linear-attention / FLA / mamba (editable Triton)
- [gfx942 · prefill-dominated hybrid] stack-and-compound cluster; Amdahl pre-dispatch screen ★★★ — (editable-triton-cluster-amdahl.md)

## method (cross-model, applies to any run)
- engagement verification: one-shot stderr banner + log grep ★★★ — (method-verify-engagement.md)
- e2e A/B: pinned port, interleaved, non-overlap gate ★★★ — (method-e2e-ab-harness.md)
- cuda/HIP-graph-safe integration + load-time weight conversion (the #1 e2e killer: capture hang OR HIP-OOM-at-init → REJECTED; key cache by weight.data_ptr() not A's, convert once, drop --enforce-eager) ★★★ — (method-cudagraph-safe-integration.md)
