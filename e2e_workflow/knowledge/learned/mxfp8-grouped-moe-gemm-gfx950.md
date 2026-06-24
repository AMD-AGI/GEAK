---
key: mxfp8-grouped-moe-gemm · gfx950 · both
type: lever
confidence: ★★
effect: iso ~1.08x geomean (BLOCK_K 128->256, all 6 buckets win incl. decode); e2e unverified
confirms: 1
last_seen: 2026-06-22
---
# MXFP8 E8M0 grouped MoE GEMM (vLLM native, gfx950) — author Triton, tune BLOCK_K/N
- context: live = in-tree Triton `tl.dot_scaled` grouped kernel `_mxfp8_grouped_gemm_kernel`
  (`_grouped_gemm_mxfp8` seam, mxfp8_native_moe.py). Hardcoded BLOCK_N=128, BLOCK_K=128, num_warps=8,
  NO autotune. Head op ~40% GPU time on M3-MXFP8. EDITABLE -> Tier-C rewrite, not env/flag.
- lever: BLOCK_K=256 (keep BLOCK_N=128, warps=8) beats the in-tree default on EVERY bucket incl. both
  decode (T1,T64) and prefill — geomean 0.1289->0.1193 ms, no decode regression. Real author headroom:
  per-(N,K) BLOCK_N + split-K (k_batch) for tiny decode M + per-shape warps will go further.
- apply: route=rewrite, target_language=triton (existing editable impl). flydsl/aiter are AUTHOR-only
  here (no drop-in grouped E8M0 entry; aiter has `gemm_a8w8_bpreshuffle_flydsl` = dense fp32-scale, wrong
  math). Decode buckets MANDATORY — don't regress T=1/T=64.
- verify: immutable unittest geomean ms vs baseline 0.1320; rtol/atol 3e-2 (fp8), all 6 cases correct.
- dead-end: BLOCK_N=256, BLOCK_N=64, num_warps=4, waves_per_eu=2 all REGRESS geomean on these shapes.
- dead-end: aiter DB GEMM tune (AITER_TUNE_GEMM/AITER_CONFIG_GEMM_BF16) does NOT touch this path — it
  tunes dense bf16 `gemm_a16w16`, not this bespoke grouped MXFP8 kernel. Zero engagement. No env Tier-B.
- harness: shared op_bench.py MIS-ROUTES this (is_blockscale_gemm True) to the dense a8w8 blockscale
  path -> TypeError + 41.8 rel-err garbage = harness_suspect, NOT a no-win. Repair: drive the immutable
  unittest's _build_case/_call directly (grouped scatter + E8M0). Driver: see source eval dir.
- source: /shared/amdgpu/home/zihaoan2_qle/kernel_agent/v4/GEAK/worktree/deep/exp/e2e_MiniMax-M3-MXFP8_20260622_063601_1857462_27251 (opbench_result_corrected.json, probe_triton_knobs.py)
