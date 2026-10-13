---
key: kernel fusion (add+rmsnorm+group-quant) · gfx942 · sglang fp8 block-scale dense prefill+decode
type: lever
confidence: ★★
effect: +9.64% e2e output tok/s MEDIAN, non-overlapping single-run A/B (baseline 766.6/789.6/790.7 median 789.58 vs cand 801.1/865.7/866.5 median 865.69; cand_min 801.08 > ref_max 790.65, guaranteed floor +1.32%). Driver TPOT -9.27% (39.46 vs 43.50 ms) — decode launch-bound, so e2e >> the screen's kernel-time Amdahl (~1%). gsm8k flat within noise (0.945 -> 0.925, 2.0pt drop, z=0.81 at n=200). Qwen3-14B-FP8 TP1.
confirms: 1
last_seen: 2026-09-30
---
# fuse add_residual + RMSNorm + per-group-128 fp8 quant at the LayerCommunicator RMSNorm seam
- lever: on gfx942 the input_layernorm(->qkv) / post_attention_layernorm(->gate_up) run as TWO
  kernels: `aiter.rmsnorm2d_fwd_with_add` (bf16 out) then a per-group-128 `dynamic_per_group_scaled_quant`
  inside the fp8 block linear apply. One self-authored Triton kernel does add+RMSNorm+group-quant and
  returns `((fp8, scale), residual_out)`; the block linear consumes the tuple and skips its own quant,
  so the bf16 [M,5120] intermediate never touches HBM. Biggest e2e mover is removing kernel LAUNCHES at
  decode (cuda-graph bs<=4, launch-bound) — size from the e2e A/B, NOT the ~1% kernel-time estimate.
- apply: patch `sglang.srt.layers.layernorm.RMSNorm.forward_aiter` to fire only for RMSNorms tagged by
  a patched `LayerCommunicator.__init__` (`_e02_fuse` on input_layernorm/post_attention_layernorm); gate
  on residual!=None, 2D bf16, hidden<=8192, hidden%128==0. The `(fp8, scale)` tuple is consumed by
  `Fp8LinearMethod.apply` block_quant path (fp8.py, `isinstance(x, tuple)` -> input_scale=x[1]); scale is
  row-major `(M, N//128)` fp32, dtype float8_e4m3fnuz. q_norm/k_norm (residual=None) and final model.norm
  (untagged) stay untouched.
- verify: `[overlay-e02-addrmsnorm-group-quant] ENGAGED` banner on the live server (prefill+decode), plus
  the fused kernel in the reprofile and the separate per-group-quant gone from the qkv/gate_up inputs.
- caution: Qwen3DecoderLayer.forward does `hidden_states.shape[0]` RIGHT AFTER prepare_attn — it
  AttributeErrors on a `(fp8,scale)` tuple during cuda-graph capture (sglang's gfx95 fused-quant path is
  never taken here since prepare_attn is called with quant_format=""). Patch the decoder with a
  tuple-safe token guard (self_attn/mlp already consume the tuple). ALSO verify the existing kernel: aiter
  `add_rmsnorm_quant` is per-token-only in this build (group_size 32/64/128 raise "not support"); aiter
  Triton `fused_rms_fp8_group_quant` (group128) is accurate but only 1.0-1.13x unit and lost to the
  self-authored kernel here — re-measure, don't assume.
- source: /home/yixiongh/geak/0930/qwen35 (e2e run geak-e2e-qwen35-20260930-full-blind, exec e02);
  overlay fusion_overlays/Qwen3-14B-FP8/e02_addrmsnorm_group_quant/, unit fusion/unit_e02/.
