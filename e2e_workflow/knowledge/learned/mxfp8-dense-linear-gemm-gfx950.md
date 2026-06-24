---
key: mxfp8-dense-linear-gemm · gfx950 · both
type: lever
confidence: ★★
effect: iso ~1.12x geomean (BLOCK_K 128->256, all 12 cases win incl. decode M1/M64); e2e unverified
confirms: 1
last_seen: 2026-06-22
---
# MXFP8 E8M0 DENSE linear GEMM (vLLM native, gfx950) — author/rewrite Triton, tune BLOCK_K/N
- context: live = in-tree Triton `tl.dot_scaled` DENSE-linear kernel `_mxfp8_linear_kernel`
  (`_mxfp8_dot_scaled_linear` seam, rocm_native.py / RocmDotScaledMxfp8LinearKernel.apply_weights).
  Hardcoded BLOCK_M=64,BLOCK_N=128,BLOCK_K=128,num_warps=8, NO autotune. Sibling of the grouped-MoE
  head; this is the qkv/o_proj/shared+dense MLP gate_up/down family (N,K) at M in {1,64,1024}.
  Head op ~25% GPU time on M3-MXFP8. EDITABLE -> Tier-C rewrite, not env/flag.
- lever: BLOCK_K=256 (keep BLOCK_M=64,BLOCK_N=128,warps=8) beats the in-tree default on EVERY case incl.
  both decode (M1,M64) and prefill — geomean 0.0703->0.0630 ms (1.117x); +waves_per_eu=2 -> 0.0626
  (1.124x). No decode regression (qkvM1 .066->.063, qkvM64 .086->.065, mlp_gate_up M1024 .158->.126).
  Real author headroom beyond this: per-(N,K) BLOCK_N + split-K for tiny decode M + per-shape warps.
- apply: author_plan = triton rewrite FIRST (real editable kernel + proven knob headroom), then flydsl
  author (SOTA GEMM DSL but NO drop-in E8M0 dense primitive -> fresh baseline). Decode buckets
  MANDATORY (REQUIRE_DECODE_BUCKET). Keep weight-prep cache COMPACT fp8, NOT bf16 (KV-starvation).
- verify: immutable unittest geomean ms vs live baseline 0.0703; rtol/atol 3e-2, all 12 cases correct
  (live rel_err ~0.007).
- dead-end: BLOCK_N=256, BLOCK_N=64, num_warps=4, BLOCK_K=128+waves=2 all REGRESS geomean here.
- dead-end: aiter DB GEMM tune (AITER_TUNE_GEMM/AITER_CONFIG_GEMM_BF16) does NOT touch this path (tunes
  dense bf16 gemm_a16w16). aiter/flydsl/ck/hipblaslt dense GEMMs are fp32-scale or bf16 = WRONG math for
  E8M0 [1,32] microscale (op_bench blockscale path -> 42.99 rel-err). No env/flag Tier-B; author-only.
- harness: shared op_bench.py MIS-ROUTES (is_blockscale_gemm True) to dense a8w8 fp32-scale path ->
  aiter_blockscale TypeError (dtype= kwarg) + aiter_bpreshuffle 42.99 rel-err = harness mis-route, NOT
  no-win. Repair: drive immutable unittest _build_case/_call directly. Driver: opbench_mxfp8_linear_driver.py
- source: /shared/amdgpu/home/zihaoan2_qle/kernel_agent/v4/GEAK/worktree/deep/exp/e2e_MiniMax-M3-MXFP8_20260622_063601_1857462_27251 (opbench_result_corrected.json, mxfp8_linear_knob_sweep.json)
