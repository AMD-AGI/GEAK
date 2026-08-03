---
key: activation+fp8-quant fusion · gfx942 · sglang fp8 decode-bound serving
type: lever
confidence: ★★
effect: fuse SiLU act_and_mul WITH the per-group fp8 activation-quant that feeds down_proj into ONE kernel at the MLP producer seam → +0.9076% e2e (verified, non-overlapping, gsm8k 0.90==0.90). The FUSION (elim a separate quant pass) is the win, not a faster standalone act or quant kernel.
last_seen: 2026-08-02
confirms: 2
---
# Fuse SiLU act_and_mul + per-group fp8 quant into the MLP down_proj producer
- lever: on a dense fp8 model, the down_proj input is produced by SiLU(gate)*up THEN separately
  per-group-fp8-quantized (aiter `dynamic_per_group_scaled_quant`). Fuse both into a single authored
  `silu_and_mul_quant` kernel so the activation is quantized in-place at its producer — removes a full
  elementwise read/write pass over the [M,intermediate] activation each decode step.
- apply: patch `Qwen2MLP.forward` (down_proj producer-fusion seam) to call the fused kernel; the aiter
  `dynamic_per_group_scaled_quant` call count for the activation-quant path drops (1120→840 here as the
  down_proj leg folds in). Overlay rebind, CUDA-graph-capture-safe.
- verify: engagement banner (`fused_silu_quant ENGAGED` / `fused_engage=1` on cand, 0 on a clean
  baseline) AND capture succeeded; parity = byte-exact 10/10 + gsm8k parity vs a FRESH no-overlay
  baseline (it's a quant kernel, so byte-exact + task-metric is the correct bar). Amdahl: ~3.44% GPU at
  ~1.41× serving-weighted → ceiling ~+1.0%; measured +0.9076% sits in-band.
- caution: this banks ~all the headroom of THIS fusion. A SECOND independent author of the SAME fused
  op re-engaged and stayed parity-clean but delivered only +0.17% (overlapping) over the incumbent →
  STACK/provisional, not a standalone re-win. Don't re-nominate the same fusion for a fresh accept;
  the fused elementwise/quant surface then drops below the noise band. ALSO verify the ISOLATED
  bake-off still shows a real speedup BEFORE scheduling: once the fp8 a8w8 blockscale GEMM win lands
  (Triton→CK, down_proj ~10× cheaper, GEMM mass shrinks and gate_up_proj becomes the dominant head),
  a fresh author of this fusion measured 0× isolated → dead_end. The fusion pair (dynamic_per_group
  _scaled_quant + act_and_mul, ~7% GPU) still SHOWS in the profile but no longer converts.
- source: /wekafs/test_results/Qwen3_14B_20260730 (round_1 cand_dyn_pergroup_quant +0.9076% ACCEPT;
  round_2 2nd author silu_and_mul_quant re-engaged, +0.17% overlapping → stack);
  /wekafs/test_results/Qwen3_14B_20260802 (round_1, post-CK-GEMM landscape: re-author 0× isolated → dead_end)
