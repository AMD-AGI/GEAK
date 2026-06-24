---
key: mxfp8-e8m0-grouped-moe-gemm · gfx950 · both
type: routing
confidence: ★★
effect: iso baseline only (live=editable Triton dot_scaled, geomean 1.0 on immutable harness); no env/flag win; author route is the lever
confirms: 2
last_seen: 2026-06-21
---
# MXFP8 (1x32 E8M0 microscale) grouped MoE expert GEMM — route via Tier-C author, not env
- lever: the live op (`vllm...mxfp8_native_moe:_grouped_gemm_mxfp8`) is an EDITABLE in-tree Triton
  `tl.dot_scaled` kernel on gfx950 native MX cores. No drop-in env/flag win exists; optimize the code.
- dead-end: aiter bf16 `AITER_TUNE_GEMM`/`AITER_CONFIG_GEMM_BF16` DB is the WRONG lever here — this op is
  dispatched directly by vLLM's Triton kernel (sorted_token_ids + a_div + topk-mul + E8M0 1x32), not by
  aiter `gemm_a16w16`; the bf16 DB has no matching key → 0 engagement.
- dead-end: `scripts/op_bench.py` mis-handles this op — `_is_blockscale_gemm` routes it to the dense
  a8w8 blockscale probe (`aiter:gemm_a8w8_blockscale*`), which is a DIFFERENT op (baseline raised on the
  `dtype` kwarg; bpreshuffle gave max_rel_err≈41.8). Use the task's IMMUTABLE `unittest.py` as the
  authoritative bake-off (it times the real `_grouped_gemm_mxfp8` vs a faithful E8M0 oracle).
- apply: author_plan = [{flydsl, author} FIRST, {triton, rewrite}]. FlyDSL HAS real E8M0 microscale
  grouped-MoE primitives (`aiter.ops.flydsl.flydsl_moe_stage1/stage2`, kernels `mixed_moe_gemm_2stage.py`
  + `silu_and_mul_fq.py`) taking sorted_token_ids/sorted_expert_ids/num_valid_ids/topk + w*_scale, knobs
  tile_m/n/k, k_batch (split-K for decode), persist_m, waves_per_eu, use_async_copy — map 1:1 to this op.
  Triton rewrite headroom: the in-tree kernel hardcodes BLOCK_N=BLOCK_K=128, num_warps=8, no autotune,
  no split-K for decode-M.
- verify: immutable `unittest.py` (geomean over 6 cases, decode T∈{1,64} mandatory; must not regress
  decode-M). Baseline bars (gfx950, TP4): gemm1_w13 prefill M4096 N1536 K6144 = 0.326ms (dominant);
  gemm2_w2 prefill M4096 N6144 K768 = 0.199ms; decode T64 = 0.255/0.152ms.
- source: exp/e2e_MiniMax-M3-MXFP8_20260621_144547_3794_10149 (op _mxfp8_grouped_gemm_kernel, pct_gpu=24.77);
  reconfirmed exp/e2e_MiniMax-M3-MXFP8_20260621_213057_794868_11259 (pct_gpu=32.61, now THE dominant head;
  immutable unittest 6/6 pass geomean 1.0; flydsl_moe_stage1/stage2 + flydsl_preshuffle_gemm_a8 present;
  is_flydsl_available=True. Baseline bars GPU0: gemm1_w13 prefill M8192 0.358ms (dominant) / decode T32
  0.191 / T1 0.056; gemm2_w2 prefill M8192 0.230 / decode T32 0.113 / T1 0.034. best_known_ms=0.358)
