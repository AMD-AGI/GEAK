---
key: kernel fusion (silu_and_mul + group-quant) · gfx942 · sglang fp8 block-scale dense MLP decode
type: lever
effect: +10.62% e2e output tok/s MEDIAN, non-overlapping interleaved A/B (3 reps/leg) vs the accepted e02 overlay (ref combined_e02 819.8/860.7/834.4 median 834.445 vs cand combined_e02_e04 923.1/934.2/878.5 median 923.066; cand_min 878.51 > ref_max 860.70, guaranteed floor +2.07%). Driver TPOT -6.43% (37.12 vs 39.67 ms) — decode launch-bound. gsm8k HIGHER on cand (0.925 -> 0.935, n=200, max-tokens 4096) — no degradation. Qwen3-14B-FP8 TP1, stacked on e02.
confidence: ★★★
confirms: 1
last_seen: 2026-10-01
---
# fuse SiLU-and-mul + per-group-128 fp8 quant at the MLP down_proj input seam
- lever: on gfx942 the Qwen3 dense MLP runs `sgl_kernel.silu_and_mul` (bf16 [M,17408] out) then a
  per-group-128 `dynamic_per_group_scaled_quant` INSIDE `aiter_w8a8_block_fp8_linear` (the down_proj
  apply). One self-authored Triton kernel does silu(gate)*up + group-quant and returns `(fp8, scale)`;
  down_proj consumes the tuple and skips its own quant, so the bf16 [M,17408] intermediate never
  touches HBM. Biggest e2e mover is removing kernel LAUNCHES at decode (cuda-graph bs<=4, launch-bound).
- apply: patch `sglang.srt.models.qwen2.Qwen2MLP.forward` (imported as Qwen3MLP) — wrap `__init__` to tag
  instances `_e04_fuse`, and gate on `down_proj.quant_method.block_quant`, gate_up 2D bf16, half%128==0.
  The `(fp8, scale)` tuple flows through RowParallelLinear (TP1 input_is_parallel) into
  `Fp8LinearMethod.apply` block_quant path (`isinstance(x, tuple)` -> input_scale=x[1]); scale is
  row-major `(M, 17408//128=136)` fp32, dtype float8_e4m3fnuz, NO transpose (gfx942 triton
  `gemm_a8w8_blockscale`). gate_up layout is [gate | up] halves; silu on first half.
- verify: `[overlay-e04-silumul-group-quant] ENGAGED` banner on the live server, and the separate
  silu_and_mul + per-group-quant gone from the down_proj input in the reprofile.
- caution: the tightest existing kernel `aiter.ops.triton.activation.act_mul_and_fp8_group_quant(gate_up,
  "silu", 128)` FITS the seam and is BIT-IDENTICAL numerically, but its kernel is only ~1.5x vs the
  reference chain versus the self-authored ~2.3x (self ~1.6x faster at every decode M) — wire+unit-test
  it, then keep the self-authored one; don't assume the vendor kernel wins. ALSO verify granularity:
  `fused_silu_mul_fp8_per_tensor_static_quant` is per-TENSOR static (wrong for per-group-128 dynamic);
  `fused_reduce_act_mul_fp8_group_quant` adds a cross-token reduce this seam lacks; the
  `silu_and_mul_masked_post_*_quant_fwd` pair are EP-MoE expert kernels, not the dense MLP seam.
- micro-tune (an applied fusion kernel is NOT done): the fused `_e04_kernel` still has a live, cheap
  tile lever. Swapping its tiling for a 2D-tiled shape-adaptive variant (`_e04_kernel_2d`, GEMM swap
  files byte-identical) gave iso 1.656x geomean (director-reproduced) and a further **+2.32% e2e**
  BYTE-EXACT (max_rel_err=0.0 on all shapes -> identical greedy tokens, zero added drift) on top of the
  fusion, single-replica non-overlapping A/B (ref_med 1258.57 -> cand_med 1287.80). Within the 4.45%
  Amdahl ceiling (10.76% head, iso 1.656x) -> ~half transferred. Reprofile: the head collapses
  10.76%/avg 540.7us -> 2.58%/avg 127.8us (~4.2x less GPU time), below the 5% roofline-head bar.
  caution: the roofline prior (latency-bound, `roofline_pct` 0.121, `attainable_speedup` 7.264x,
  conf LOW) grossly over-predicted what a tile tune recovers (measured 1.656x) -> on a latency-bound
  head do NOT size a micro-tune from a low-confidence roofline attainable; use the pct_gpu_time Amdahl
  ceiling instead.
- e2e-gate (FINAL reconcile, 2026-10-01): the e04 fusion AND its 2D-tile micro-tune shipped in the
  Director-validated final stack — run 770.3 → 1256.7 tok/s, Director same-session 1.644× (validated_win),
  output parity PASS (byte-exact, gsm8k flat) → confidence raised ★★ → ★★★ (Director-verified e2e).
- source: exp/e2e_*Qwen3-14B-FP8*/ 2026-09-30..10-01 (sglang TP1 gfx942, exec e04 + e04 micro-tune);
  overlay fusion_overlays/Qwen3-14B-FP8/e04_silumul_group_quant/, unit fusion/unit_e04/.
