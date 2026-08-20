---
key: fp8_w8a8 block-scale fused-MoE grouped GEMM · gfx950 · vLLM
type: lever
confidence: ★★
confirms: 2
effect: per-shape Triton config tune (winner_kind=env, ZERO HBM) → iso 1.02–1.16× per M-bucket, serving-weighted ~1.03× (decode M64 1.026×, prefill M8192 1.041×). Same VLLM_TUNED_CONFIG_FOLDER mechanism as the int4/bf16 MoE cards, dtype segment = fp8_w8a8 + block_shape. An authored Triton rewrite of `fused_experts_impl` (Tier-C) beat the env-tune bake-off (iso 1.034×). e2e transfer did NOT clear the noise band at the FINAL gate: Director same-session A/B = +0.16% (1.0016×), ranges OVERLAP → validated_no_win, byte-exact parity 12/12. The lever ENGAGES (decode-bucket rebind fired on both TP workers) but at a 21.4% head with iso ~1.03× the Amdahl ceiling (~0.6%) is inside serving noise.
last_seen: 2026-08-20
---
# fp8_w8a8 block-scale fused-MoE → the memory-free vLLM config-tune lever (fp8 analog of int4/bf16 cards)

- path: same as `moe-bf16-tune` / `moe-int4-w4a16-tune` but for fp8 block-quant. (1) check whether a
  tuned config ships for `(E,N,device,fp8_w8a8,block_shape)` — vLLM ships NONE for unseen fp8-blockscale
  MoE shapes on gfx950/MI355 (verified 0 configs match `*MI355*fp8_w8a8*block_shape*`), so the expert
  grouped-GEMM falls back to the slow default tile (`Using default MoE config`). (2) Sweep per M-bucket
  against `fused_experts`+`override_config` with fp8 weights + block scales, parity rel<1e-2. (3) Deploy
  `VLLM_TUNED_CONFIG_FOLDER`, pair with `--max-num-batched-tokens ≈2·ISL` (clamp 8192..32768).
- lookup filename: `get_config_file_name(E, N, "fp8_w8a8", [128,128])` →
  `E=<E>,N=<N>,device_name=<dev>,dtype=fp8_w8a8,block_shape=[128,128].json`. N = moe_intermediate//TP.
- fp8-SPECIFIC constraint (differs from bf16/int4 sweep): the block-scale kernel requires
  `BLOCK_SIZE_K % block_k == 0` and `BLOCK_SIZE_N % block_n == 0` → pin BLOCK_SIZE_K=128, BLOCK_SIZE_N∈{128,256}.
  Build the quant_config via `fp8_w8a8_moe_quant_config(w1_scale,w2_scale,block_shape=[128,128])`; weights
  float8_e4m3fn, scales float32 shaped [E,ceil(2N/128),ceil(K/128)] / [E,ceil(K/128),ceil(N/128)].
- expected gain: iso 1.02–1.16× per bucket (mid buckets M128/256 biggest at ~1.15×; decode M64 1.026×,
  prefill M8192 1.041×), serving-weighted ~1.03×. ZERO extra HBM → sails the mem_footprint gate.
  Naive Amdahl ceiling ~0.6% at a 21.4% head, BUT decode is graph-hidden/under-counted (profiled decode
  share 0.00 → floored 0.30) so measured e2e may exceed it (cf. bf16 card +7.01% > +3.37% ceiling).
- also: editable in-tree Triton MoE present (`kernel_src/fused_moe/fused_moe.py`, hot `fused_moe_kernel`
  @triton.jit + `fused_experts_impl`) → Tier-C `route=rewrite`; flydsl grouped-MoE primitives import on
  this gfx950 image (`aiter.ops.flydsl.flydsl_moe_stage1/stage2`, is_flydsl_available=True) → `route=author`.
  aiter ALSO ships a native fp8 block-scale fused MoE (`aiter.fmoe_fp8_blockscale_g1u1`) reachable via the
  vLLM `VLLM_ROCM_USE_AITER[_MOE]` backend swap — a separate Tier-A candidate the Integrator can A/B
  (mutually exclusive with the Triton config env, since aiter MoE bypasses the Triton seam).
- caution (also verify flydsl viability before routing it): for fp8-[128,128]-block MoE the aiter FlyDSL
  path did NOT compile on this gfx950 image — the high-level `flydsl_moe_stage1/2` wrapper accepts only
  b_dtype∈{fp4,fp8 MXFP8 per-32 e8m0, bf16xint4} and raises ValueError on bf16xbf16, and the
  precision-preserving fallback (dequant fp8-block→bf16/fp16 then `compile_moe_gemm1/2`) hits an internal
  DSL compiler bug (`UnboundLocalError 'a0'` in the non-int4 prefetch pipeline, `moe_gemm_2stage.py`,
  reproduced across tiles). So `is_flydsl_available=True` (imports OK) is NOT sufficient — verify the
  actual dtype path compiles; for [128,128]-block fp8 MoE prefer the Triton rewrite. flydsl would need a
  fixed non-int4 `moe_gemm_2stage` build or an MXFP8 requant path validated to the tol.
- caution (also verify): a milestone interleaved A/B here showed +1.36% NON-overlapping byte-exact for
  the authored Triton rewrite, but the Director SAME-SESSION A/B collapsed it to +0.16% (overlapping
  ranges) = validated_no_win. On a decode-bound serving run always trust the Director same-session A/B
  over the milestone A/B: a milestone win at a ~20% head with iso ~1.03× can be entirely serving noise
  once re-measured against a fresh same-session baseline (base median rose from 2405.9 warm-start to
  2451.5 same-session — most of the apparent gain was baseline drift). Byte-exact parity is NOT evidence
  of a throughput win.
- source: exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-19..08-20 (E=256, N=512, K=3072, topk=8, silu, fp8
  block[128,128], vLLM 0.26.0, TP=2, gfx950/MI355 OAM, MoE 21.4% GPU; no shipped config → default
  fallback; iso per-bucket 1.015–1.158×, authored iso 1.034×; Director same-session +0.16% (1.0016×),
  validated_no_win, byte-exact 12/12).
  driver: `config/tune_moe_fp8_blockscale.py`; tuned artifact under
  `config/moe_tuned_fp8/E=...,N=...,dtype=fp8_w8a8,block_shape=[128,128].json`.
