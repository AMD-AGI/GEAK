---
name: moe-mxfp4-gate-up-separated-gfx950-sglang
description: On sglang+aiter mxfp4 MoE, SGLANG_USE_AITER_MOE_GU_ITLV=0 swaps the PREFILL fused-MoE to separated afp4_wfp4 rows (TTFT -10.8%, ~+1.3-1.9% e2e, accuracy-gated) but ALSO un-fuses the decode swiglu — only HIP graphs keep that free.
keywords: [fused-moe, mxfp4, gate-up-interleave, gu-itlv, prefill, ttft, env-flag, aiter, sglang, gpt-oss, config-fast-path, swiglu-fusion, hip-graph, dispatch-overhead, pad-fusion]
kernels: [flydsl_moe1_afp4_wfp4_bf16, flydsl_moe1_afp8_wfp4_bf16_gui_fp8, swiglu_mxfp4_bf16_cktile, ck_tile::MoeFlatmmKernel, swiglu_act_and_mul_bias_kernel, at::native::FillFunctor, direct_copy_kernel_cuda]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: both (prefill wins; decode pays an un-fusion cost that HIP graphs hide)
key: native-mxfp4 fused-MoE gate/up projection · gfx950 · sglang + aiter (SGLANG_USE_AITER=1), ISL-heavy serving
type: lever
confidence: ★★★
effect: config-sweep +1.25% e2e (5960.9 -> 6035.2 tok/s; both candidate replicas above both baseline replicas, reproducibility control 0.16%); inside the Director-validated bundle TTFT 530.1 -> 472.8 ms (-10.81%) with TPOT flat, i.e. this flag is the PREFILL half of a +1.809% validated result. Zero HBM, zero code. Second confirm on a re-profile of the accepted stack: prefill MoE per 64-request wave 236.5 -> 163.8 ms (-31%), TTFT 403 -> 357 ms (-11.5%), whole-prefill share 4.5% -> 4.0%; net e2e +1.92% WITH HIP graphs on, -0.7% with graphs off (the decode un-fusion below is the whole difference).
confirms: 2
lifecycle: active
last_seen: 2026-09-09
---
# Gate/up INTERLEAVED is not free on mxfp4 — try the SEPARATED path first, it is one env var
- lever: `SGLANG_USE_AITER_MOE_GU_ITLV=0` (GateMode.SEPARATED). Cheapest possible fast-path trial: no
  overlay, no rebuild, survives the `PYTHONPATH` bundle as plain env.
- apply: add to the server env next to the boot-required `SGLANG_USE_AITER=1` for a native
  `quantization_config.quant_method=mxfp4` checkpoint.
- verify: read the aiter kernel-selection lines in the server log. Interleaved default selects
  `flydsl_moe1_afp8_wfp4_bf16_*_gui_fp8`; engaged SEPARATED selects `flydsl_moe1_afp4_wfp4_bf16_t*_w*`
  (no `_gui_fp8`). Check TTFT, not throughput alone — the whole effect is prefill-side.
- caution: the mechanism is NOT the one the profile suggests, so also verify WHICH phase moved, and
  verify DECODE separately rather than assuming it is untouched. Earlier evidence read decode as
  UNCHANGED (activation dtype does stay bf16: M=64 < `GPTOSS_SWIGLU_MXFP4_BF16_BOUND`(256)), but a
  re-profile of the accepted stack CORRECTS that: the SEPARATED w13 layout also disables the ck_tile
  kind1 FUSED-swiglu instance, so decode stage1 drops to kind3 `MoeSilu` (114.0 -> 111.3 us) and a
  standalone `swiglu_act_and_mul_bias_kernel` (1.47% GPU) plus a doubled `vectorized_elementwise`
  (0.96% -> 1.93%) appear — net decode +123 ms per wave in extra dispatches. HIP/CUDA graphs hide it
  (graphs on +1.92% e2e, graphs off -0.7%), so **also verify cuda-graph capture is actually on for the
  decode path before banking this flag**, and re-check it on any stack where graphs are disabled or
  capture silently falls back.
  Also verify accuracy: SEPARATED runs fp4 ACTIVATIONS on prefill, so it is lossy — gate it (gsm8k 5-shot
  greedy measured 0.9167 vs 0.9250 baseline at n=120, and 0.940 vs 0.950 at n=500, ~1 sigma; accept on
  the band, not on the point estimate).
- neighbouring dead-end worth not re-buying: forcing fp8 DECODE activations on the same stack
  (`AITER_BF16_FP8_MOE_BOUND=1`) engaged cleanly and still lost 6.0% e2e (TPOT 10.07 -> 10.96) — the
  aiter `tuned_fmoe.csv` microbench predicted a win because the per-step activation quantization cost
  sits OUTSIDE the timed kernel. Trust the e2e A/B over a fused-MoE tuning CSV for anything that changes
  activation dtype.
- follow-on lever (UNMEASURED, worth one sweep trial): re-fuse the decode swiglu — keep the fp4 prefill
  rows while restoring the kind1 fused instance — is a concrete ~1.5-3.4% byte-reduction target that
  only exists once this flag is on.
- neighbouring lever on the same stack (UNMEASURED, one env var): the `elementwise`/`fill` overhead
  cluster on native-mxfp4 MoE is NOT a kernel-track target — both symbols are the two halves of ONE
  `at::constant_pad_nd` (empty + `fill_` over the padded shape + `narrow().copy_` over the unpadded one)
  emitted by the MoE method's `F.pad(x, (0, hidden_pad))`, once per layer. There is no bandwidth
  headroom (~3% of roofline, launch-latency-bound) and no rewritable seam, so route it to the CONFIG
  tuner: `SGLANG_AITER_FUSE_RMSNORM_PAD=1` makes the preceding add-rmsnorm emit the already-padded
  tensor (`x_pad_to_multiple`) and the pad is skipped entirely, bit-identical on the unpadded columns.
  Preconditions to check first: mxfp4 quant_method, HIP + `SGLANG_USE_AITER=1`, TP=1. Also verify by
  A/B, not arithmetic — it adds back one output-trim `.contiguous()` per layer, so it is a partial
  retirement of the launches, not a full one.
- source: exp/e2e_*gpt-oss-120b*/ 2026-09-07..08 (gfx950, sglang 0.5.x, aiter c16d44b9) —
  config/sweep_results.json trial `cfg5_gu_separated`, director_e2e_validation.json; second confirm +
  the decode-un-fusion correction from exp/e2e_*gpt-oss-120b*/ 2026-09-09 round-1 re-profile
  (profile/round_config/profile_topN.json vs profile/round_0/).
