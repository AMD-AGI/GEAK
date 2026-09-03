# Learned — index of distilled experience cards

Open the cards matching your run's `(kernel_class, gfx, regime)` as **additional, advisory priors** —
they only ADD candidates to try, never remove any or replace measurement. The on-box bake-off + e2e gate
is always the judge (see `README.md` philosophy). One line per card, grouped by reuse key. **Cap: ≤40 lines.**
Confidence (a hint strength, not authority): ★ noise/unverified · ★★ single non-overlap or ≥2 consistent · ★★★ ≥2 non-overlap or verified e2e.

> ⚠ **This index is hand-maintained and has drifted before** — cards have sat in this folder with no line
> here, which makes them invisible to every reader. `ls` the folder before concluding a card does not
> exist, and add the missing line when you find one. It becomes a **generated** file (from each card's
> discovery header, via `python3 kernel_workflow/scripts/kb.py --kb-dir e2e_workflow/knowledge/learned index`)
> once the cards carry those headers — see `README.md`.

## dense GEMM
- [gfx950 · vLLM MXFP8 E8M0 decode-bound] dense-linear split-K/fused decode-tile Triton rewrite ★★★ **+21.8% e2e (verified, gsm8k-clean); decode-driven (converts only at high conc); grouped-MoE GEMM resists (~1.1× ceiling)** — (mxfp8-linear-decode-rewrite-gfx950.md)
- [gfx942 · sglang bf16] aiter per-shape DB tune ★★★ **+2.23% e2e (verified)** — (aiter-bf16-tuned-gemm-gfx942.md)
- [gfx950 · sglang bf16 dense] backend swap = NO win (hipBLASLt already fastest), but the live seam IS `aiter.tuned_gemm:gemm_a16w16` (unlike vLLM) → aiter per-shape DB tune, `AITER_CONFIG_GEMM_BF16` colon-MERGE, ZERO HBM ★★ **engages; coverage-gated (gate_up/down 0 shipped = headroom, qkv/o already shipped = noise); e2e pending, gfx942 analog +2.23%** — (dense-gemm-bf16-gfx950-sglang.md)
- [gfx950 · vLLM fp8 a8w8 blockscale] per-shape CK tune DB `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`, plus the `VLLM_ROCM_USE_AITER[_LINEAR]=1` Triton→CK swap whenever AITER is off ★★★ **+16.1% / +65.69% / +18.86% e2e (Director-validated); iso 1.86–3.17× serving-wtd — but gain is SHIPPED-coverage-gated (~1.03× when already covered): probe `use default` vs `is tuned` first; swap-only UNTUNED CK confirmed 1.513× serving-wtd on Qwen3-14B-FP8 but REGRESSES prefill 0.66× → needs the CK tune (offline tuner/ckProfiler may be ABSENT from the aiter wheel — provision to recover prefill)** — (fp8-a8w8-blockscale-ck-tune-gfx950-vllm.md)
- [gfx950 · sglang fp8 a8w8 blockscale, ROCm>=7.2] live kernel is CK **bpreshuffle**, not plain-CK/Triton → tune THAT DB (`--preshuffle`, env `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE`), no overlay/swap needed ★★★ **+20.91% e2e (Qwen3-14B-FP8 TP1, Director-validated, byte-identical) at a 53.51% head; iso 1.46× geomean, ZERO shipped coverage, ZERO HBM** — (fp8-a8w8-blockscale-bpreshuffle-ck-tune-gfx950-sglang.md)
- [gfx942 · sglang fp8 a8w8 blockscale] **MANDATED LEVER = the CK skill** `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md` (capture live (M,N,K) → aiter CK tuner → fp8_utils Triton→CK switch overlay + `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`); baseline = the UNTUNED Triton default, so CK-tuned is the real win. The old per-(N,K) Triton config-JSON overlay is **DEPRECATED for this op (do NOT use it — it keeps the slow Triton seam live and bypasses the skill)** — (fp8-a8w8-blockscale-overlay-gfx942.md)
- [gfx950 · vLLM MXFP8 E8M0] dense `tl.dot_scaled` STATIC tiles (decode BK256/prefill BM128) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)
- [gfx950 · vLLM MXFP8 E8M0 dense linear] author/rewrite the in-tree `_mxfp8_linear_kernel`, tune BLOCK_K 128→256 ★★ iso ~1.12x geomean (all 12 cases); e2e unverified — (mxfp8-dense-linear-gemm-gfx950.md)
- [gfx950 · vLLM MXFP8 E8M0 dense linear] ROUTING: live is editable in-tree Triton `dot_scaled`; no env/flag win exists — the author/Tier-C route IS the lever ★★ iso baseline only — (mxfp8-e8m0-dense-linear-gfx950.md)

## MoE grouped GEMM
- [gfx950 · vLLM+sglang MXFP8 1×32 E8M0 grouped fused-MoE] live = editable Triton `tl.dot_scaled` (hardcoded tiles, no autotune, no split-K) → NO env/flag win; author is the lever (flydsl FIRST, then triton rewrite) ★★★ **iso 1.0× baseline, head 14.8–32.6%; two dead-ends pre-recorded (aiter bf16 DB, op_bench blockscale misroute)** — (mxfp8-e8m0-grouped-moe-gfx950.md)
- [gfx950 · vLLM mxfp4 grouped fused-MoE, decode-dominated] seam is the FUSED `triton_kernel_moe_forward`, not a standalone GEMM → author a whole-file Triton replacement ★★★ **+92.5% e2e @ 8.32% head, +26.9% @ 32.43% head, +1.83% @ ~15.6% (all byte-exact); exceeds naive Amdahl (launch-overhead-bound); isolated GEMM only 1.06–1.57× — don't size from it. CAUTION: a split_k/tile rewrite can break byte-parity → corrective re-author preserving accum order recovers the win; routing-metadata chain is not standalone-extractable (fold into rewrite); OCP-MX emulation sub-variant needs a NATIVE-fp4 swap instead** — (moe-mxfp4-grouped-authored-gfx950-vllm.md)
- [gfx950 · vLLM MXFP8 E8M0] grouped `dot_scaled` STATIC tiles (GEMM1-decode BN64+BK256) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)
- [gfx950 · vLLM MXFP8 E8M0 grouped MoE] author/rewrite `_mxfp8_grouped_gemm_kernel`, tune BLOCK_K 128→256 ★★ iso ~1.08x geomean (all 6 buckets); e2e unverified — (mxfp8-grouped-moe-gemm-gfx950.md)
- [gfx950 · vLLM MXFP8 E8M0 grouped MoE] ROUTING: editable in-tree Triton `dot_scaled`; aiter bf16 GEMM DB is the WRONG lever ★★ iso baseline only — (mxfp8-e8m0-grouped-moe-gfx950.md)
- [gfx942 · vLLM int4 W4A16] per-shape fused-MoE Triton config tune via `VLLM_TUNED_CONFIG_FOLDER` (env, ZERO HBM; N=moe_int//TP) ★★★ +11-18% e2e (10 confirms, TP8 & TP4) — (moe-int4-w4a16-tune-gfx942.md)
- [gfx942+gfx950 · vLLM bf16 MoE] same per-shape config-tune lever as int4, for DENSE bf16 (dtype=None filename); nothing ships for unseen `(E,N,device)` → default fallback. Sweep per M-bucket, deploy `VLLM_TUNED_CONFIG_FOLDER` + `--max-num-batched-tokens ≈2·ISL`, ZERO HBM ★★★ **iso 1.01–1.66×/bucket, serving-wtd 1.10–1.40×, +7.01% e2e (Mixtral-8x7B gfx950 TP8, Director byte-exact); 3 confirms** — (moe-bf16-tune-gfx942.md)
- [gfx950 · vLLM fp8_w8a8 block-scale MoE] same `VLLM_TUNED_CONFIG_FOLDER` lever, dtype=fp8_w8a8+block_shape filename; nothing ships for unseen `(E,N,MI355)` → default fallback. fp8-specific: pin BLOCK_SIZE_K=128, BLOCK_SIZE_N∈{128,256}; build qc via `fp8_w8a8_moe_quant_config(...,block_shape=[128,128])`. Authored Triton rewrite of `fused_experts_impl` beat env-tune (iso 1.034×). ZERO HBM ★★ **iso 1.02–1.16×/bucket, serving-wtd ~1.03× (Qwen3.5-122B-A10B-FP8 TP2, 21.4% head); e2e = validated_no_win: milestone +1.36% non-overlapping byte-exact COLLAPSED to Director same-session +0.16% (1.0016×, overlapping); trust the Director A/B, ~20% head × iso 1.03× is inside noise; aiter fmoe_fp8_blockscale_g1u1 swap is a separate Tier-A candidate** — (moe-fp8-blockscale-tune-gfx950.md)

## attention
- [gfx942 · sglang hybrid prefill] `--attention-backend triton` cheap flag win ★★★ +~5% e2e — (attention-backend-triton-gfx942.md)
- [gfx942 · vLLM decode/prefill, pow2+non-pow2 KV block, +MLA TRITON_MLA, +0.21 UNIFIED_ATTENTION / ROCM_AITER_UNIFIED_ATTN] live=editable Triton (in-tree OR aiter copy — read the log) → Tier-C rewrite (pow2 ROCm/CK→author); op bake-off N/A ★★★ ~+1-4% — (paged-attn-nonpow2-gfx942.md)
- [gfx950 · vLLM block-sparse NSA GQA prefill] custom kernel, no lib swap; live = editable in-tree Triton → Tier-C rewrite ★★ ~5.6% head — (sparse-attn-nsa-triton-gfx950.md)
- [gfx942 & gfx950 · vLLM block-sparse GQA] the concrete Triton→Triton rewrite that landed upstream (arch-specialized, constexpr-gated) ★★★ gfx950 TP8 +42% out tok/s, −50% TTFT, −30% TPOT; gsm8k unchanged — (sparse-attn-triton2triton-opt-gfx942-gfx950.md)
- [gfx950 · vLLM dense GQA chunked-prefill (prefix_prefill)] live = editable in-tree Triton `context_attention_fwd`; the aiter/CK swap is a SERVER flag, so op_bench's `winner=none`/rel=0 is expected, not a fault → Tier-C Triton rewrite is the only lever; served `context_len=0` so the paged KV loop is never entered ★★★ **head 8.6–25% over 5 confirms → Amdahl-only: 1.2× buys +1.5% (8.58% head) to +4.4% (25.11%); screen before spending a round** — (prefill-attn-prefix-triton-gfx950-vllm.md)

## editable-Triton op clusters (FLA/mamba, quant-prologue)
- [gfx942 · prefill-hybrid AND decode-bound dense fp8] stack-and-compound cluster; Amdahl pre-dispatch screen run on the DEPLOYABLE iso ★★★ **even a 10.2%-GPU quant-prologue cluster @1.196x → ceiling +1.7%, measured +0.59% overlapping** — (editable-triton-cluster-amdahl.md)

## method (cross-model, applies to any run)
- engagement verification: one-shot stderr banner + log grep + monotonic CALLS counter; for vLLM `torch.ops.vllm.*` ops the seam is `direct_register_custom_op` op_func substitution (post-import rebinds are too late) ★★★ — (method-verify-engagement.md)
- e2e A/B: pinned port, interleaved, non-overlap gate; **box drift ~4% BETWEEN sessions and ~6% ref spread WITHIN one → only same-session pooled ref/cand pairs are comparable** ★★★ — (method-e2e-ab-harness.md)
- iso speedup does NOT rank e2e: same head, iso 1.53–1.70× siblings spanned +19.35% … −32.5%; A/B every sibling, split TTFT vs TPOT ★★ — (method-iso-e2e-gap-decode-tpot.md)
- cuda/HIP-graph-safe integration (the #1 e2e killer); **also verify a cross-op "cluster fusion" that DEFERS a launch — the live decoder consumes each output immediately, so deferred buffers are garbage even though the per-op unittest passes** ★★★ — (method-cudagraph-safe-integration.md)
