---
key: mxfp8_microscale (dense linear + grouped MoE) · gfx950/CDNA4 · vLLM
type: routing
confidence: ★★★
effect: iso tile-sweep 1.5–1.6× prefill, 1.39–1.71× decode (no regress); e2e NOT yet transferred ⚠
confirms: 8
last_seen: 2026-06-19
---
# MXFP8 (E8M0 1×32 microscale) GEMM → only lever is the Triton `tl.dot_scaled` rewrite
- lever: the live MXFP8 path on gfx950 vLLM is the **native CDNA4 Triton `tl.dot_scaled`** kernel —
  both `mxfp8_native_moe.py:_grouped_gemm_mxfp8` (MoE head, ~32% GPU) and
  `kernels/linear/mxfp8/rocm_native.py:_mxfp8_dot_scaled_linear` (dense head, ~22%). It HARDCODES
  `BLOCK_M=64 BLOCK_N=128 BLOCK_K=128 num_warps=8`, no autotune/num_stages/split-K → that is the headroom.
- apply: Tier-C Triton **rewrite/optimize ONLY**. Per-shape autotune BLOCK_M/N/K, num_warps∈{4,8},
  num_stages∈{1,2}, waves_per_eu, matrix_instr_nonkdim. Prefill M=8192 → BM≥256 + ns2 (the big win);
  decode M∈{1,32,128} → small BM + BK=256 + ns2 (a real gain, NOT just no-regress). MUST NOT regress decode.
- verify: synthesized oracle (no `reference_io.pt`) → run the immutable `unittest.py` DIRECTLY as the
  bench; the shared `op_bench.py` dense/blockscale path CANNOT represent E8M0-grouped (raises
  harness_suspect BY DESIGN) → self-repair with a driver wrapping unittest.py.
- dead-end: aiter / FlyDSL / CK have NO MXFP8-E8M0 primitive on this image (aiter = mxfp4 + a8w8-float
  only; flydsl = bf16/a8w8/A4W4). So `AITER_CONFIG_GEMM_BF16` = 0 engagement, no env/flag winner. Skip
  Tier-A/B entirely, go straight to the Triton rewrite.
- dead-end: BM=64 + no-pipeline is the live anti-tune for prefill; do NOT keep it.
- ⚠ integration: vLLM serves decode under cudagraph_mode FULL_AND_PIECEWISE → a self-capturing wrapper
  falls back to eager and only the static tile change survives (prior 1.22× iso netted ~0 e2e). The
  kernel MUST be compile-once/shape-agnostic + PRE-WARMED for all decode buckets. See [[method-cudagraph-safe-integration]].
- source: exp/e2e_*MiniMax-M3-MXFP8*/ bakeoff GPU4/GPU5 runs 2026-06-18 / 06-19 (8 consistent re-confirms; dense-linear N,K∈{2560×6144,6144×2048,6144×6144,6144×3072}, geomean tune_sp 1.577× decode+prefill, max_rel_err≤0.004 vs tol 0.06)
