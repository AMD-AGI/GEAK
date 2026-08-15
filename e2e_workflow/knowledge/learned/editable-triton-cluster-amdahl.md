---
key: editable Triton op cluster (FLA/mamba; quant-prologue) · gfx942 · prefill-dominated hybrid + decode-bound dense
type: routing
confidence: ★★★
effect: per-kernel iso 1.10–1.18× real, but each ~1–3% GPU → solo e2e below the 0.5% noise band; even a 10.2%-GPU cluster at iso 1.196× (Amdahl ceiling +1.7%) measured only +0.59% with overlapping distributions
confirms: 4
last_seen: 2026-08-14
---
# Editable Triton linear-attn cluster → STACK-and-compound, don't expect a solo e2e pass
- lever: the gated-delta / FLA / mamba Triton kernels (chunk_gated_delta_rule_fwd_h, chunk_fwd_kernel_o,
  causal_conv1d, recompute_w_u_fwd) are the best EDITABLE targets on hybrid gfx942, with large *isolated*
  wins — but each sits at ~1–3% GPU, so by Amdahl no single one moves e2e past noise in the
  prefill regime (where ~80% GPU is dense GEMM). Optimize them as a COMBINED cluster and let the
  Director's final stacked gate decide.
- apply: spend the head/config budget on dense GEMM FIRST; route the whole cluster as carry-forward and
  measure the SUM. Extract seam = modules under `sglang.srt.layers.attention.{fla,mamba}`.
- verify: **pre-dispatch screen** — if `pct_gpu × (1 − 1/plausible_iso) < NOISE_BAND_PCT (~0.5%)`, mark
  carry-forward-only (don't expect a solo gate to pass). Engagement via the overlay banner — see
  [[method-verify-engagement]]. e2e A/B via [[method-e2e-ab-harness]].
- also applies (**decode-bound dense fp8 vLLM**, gfx942/Qwen3-14B-FP8, 2026-08-14): the quant-prologue
  cluster (`_fused_rms_fp8_group_quant` + `_act_mul_and_dynamic_fp8_group_quant` + the two aiter HIP quant
  kernels, 10.2% GPU combined) is the same shape of target — extract it as ONE op at the seam vLLM
  dispatches (the four `torch.ops.vllm.rocm_aiter_*` impls), timed as one decoder-layer prologue
  (2×rms_add_quant N=5120 + act_mul_quant 34816→17408 + group_quant N=5120, the measured live 2:1:1 ratio).
  Screen said `10.2% × (1 − 1/1.196) = +1.7%` ceiling → carry-forward, and the measured +0.59% (overlapping)
  matched. Run the screen with the **deployable** iso, not the authored one, when part of the fusion is
  disabled for live safety (see [[method-cudagraph-safe-integration]]).
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-05 / 06-07 / 06-09;
  /wekafs/test_results/Qwen3_14B_20260813/e2e_Qwen3-14B-FP8_20260813_031549_2866199_26474/overlay/cand_fused_quant_prologue/
