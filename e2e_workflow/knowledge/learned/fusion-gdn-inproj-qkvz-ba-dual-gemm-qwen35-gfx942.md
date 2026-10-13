<!-- discovery: key=kernel fusion (GDN linear_attn in_proj qkvz+ba sibling dual GEMM) · gfx942 · sglang Qwen3.5-397B GDN-hybrid fp8 decode TP8, cudagraph; confidence=★★; last_seen=2026-10-11 -->
# fusion · GDN linear_attn in_proj qkvz+ba sibling dual-GEMM · gfx942 · Qwen3.5-397B GDN-hybrid fp8 decode TP8

lever: the two GDN input projections that share the layer activation hidden_states [M,4096] —
in_proj_qkvz (fp8 a8w8 block-scale GEMM, W[2560,4096]) and in_proj_ba (bf16 a16w16 GEMM, W[16,4096]) —
are TWO sibling kernel launches (`_gemm_a8w8_blockscale_kernel` + `aiter::wv_splitk_small_fp16_bf16`).
Fuse into ONE self-authored Triton launch whose grid spans the concatenated N-blocks of both weights;
quantize the activation ONCE (shared) and read the bf16 ba leg from hidden_states directly.

apply: seam = `sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet._forward_input_proj` (lazy class-method
rebind; plain per-call method, not direct_register, so a rebind reaches it). Quant with the SAME op the
stock fp8 linear uses — `aiter_per1x128_quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)`
(gfx942 use_triton path) — then `fused_qkvz_ba_dual_gemm(q, W_qkvz_fp8, x_scale, W_qkvz.weight_scale_inv,
hidden_states_bf16, W_ba_bf16)`. in_proj_a/in_proj_b are in the checkpoint `modules_to_not_convert` → bf16.

verify: `[overlay-e01-qkvz-ba-dual-gemm] ENGAGED` on 8/8 TP ranks + `DIAG use_fused=True
qm=Fp8LinearMethod wbs=[128,128] fails=[]`, 0 fallbacks; cuda-graph safe (single static launch).
Unit: self-authored 1.99x, bit-exact (worst_rel_err 0) vs the LIVE split reference at M=4 decode;
existing aiter `fused_gemm_a8w8_blockscale_a16w16` fits + bit-exact but only 1.23x → keep self-authored.

caution — **TARGET THE RIGHT GDN CLASS.** arch `Qwen3_5MoeForConditionalGeneration` runs
`sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet`, NOT `qwen3_next.Qwen3GatedDeltaNet` (which imports —
so its install banner fires — but is never instantiated). Patching qwen3_next here is SILENTLY inert
(measured 0 ENGAGED, A/B read ~baseline noise). Confirm the arch in config.json, grep the model file
that owns `self.linear_attn`, and log a one-time `use_fused`/guard-fails DIAG so an inert overlay is
visible. Also: the stock method has an alt_stream dual-stream variant under capture — replacing the whole
method drops it, which is intended (one launch replaces the two overlapped ones).

effect: **+10.72% e2e output throughput** (interleaved warm_server A/B, ref med 265.07 → cand med 293.48,
non-overlapping cand_min 292.58 > ref_max 266.84, clear win 10.72% ≥ 10× spread 0.74%), **TPOT -9.9%**
(14.23→12.81 ms), TTFT +3.5% (prefill noise, decode fusion doesn't move it). Accuracy: calibrated gate
`unit_equivalence_only` PASS via unit-equivalence (rel_err 0); informational gsm8k n=200 no-thinking
base 0.365 → cand 0.46 (no drop). Reproduces positive prior `xb-qwen35-qkvz-ba-dual-gemm` (+17.72%,
other branch). ISL=8192 OSL=1024 CONC=4 TP=8.

source: exp/Qwen35/20261010_test_e2e_qwen35/geak/e2e_Qwen3.5-397B-A17B-FP8_...4c70519dZ (apply_one e01,
2026-10-11, single non-overlap run).
