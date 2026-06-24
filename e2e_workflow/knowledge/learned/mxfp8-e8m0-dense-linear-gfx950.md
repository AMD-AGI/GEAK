---
key: mxfp8-e8m0-dense-linear-gemm · gfx950 · both
type: routing
confidence: ★★
effect: iso baseline only (live=editable Triton dot_scaled, geomean 1.0 on immutable harness); no env/flag win; author route is the lever
confirms: 2
last_seen: 2026-06-21
---
# MXFP8 (1x32 E8M0 microscale) DENSE LINEAR GEMM — route via Tier-C author, not env
- op: `_mxfp8_linear_kernel` / wrapper `_mxfp8_dot_scaled_linear` in
  `vllm/model_executor/kernels/linear/mxfp8/rocm_native.py` (qkv/o/dense-gate_up/dense-down/shared-gate_up/
  shared-down projection family, TP=4, pct_gpu≈16.8). Distinct from the grouped-MoE expert GEMM card.
- lever: live op is an EDITABLE in-tree Triton `tl.dot_scaled` kernel on gfx950 native MX cores. No
  drop-in env/flag win; optimize the code. Headroom is structural: kernel HARDCODES BLOCK_M=64,
  BLOCK_N=128, BLOCK_K=128, num_warps=8, NO autotune, NO split-K for decode-M, and re-quantizes the
  activation per call (`mxfp8_e4m3_quantize(x)` prologue).
- dead-end: aiter bf16 `AITER_CONFIG_GEMM_BF16` DB is the WRONG lever — this op is dispatched directly by
  vLLM's Triton kernel (E8M0 1x32 dot_scaled), NOT aiter `gemm_a16w16`; bf16 DB key never matches → 0
  engagement. Same reason kills the flydsl-via-aiter-DB env path (the DB only races bf16 dense, and the
  live seam is vLLM Triton, not aiter).
- dead-end: `scripts/op_bench.py` mis-handles this op — `_is_blockscale_gemm` routes it to the dense a8w8
  blockscale probe (`aiter:gemm_a8w8_blockscale_bpreshuffle` → max_rel_err≈41.95, others raise). That is a
  DIFFERENT op; ignore it. The task's IMMUTABLE `unittest.py` is the authoritative bake-off (runs clean,
  18/18 pass, times real `_mxfp8_dot_scaled_linear` vs faithful E8M0 oracle). harness_suspect was False but
  op_bench numbers are meaningless here — use the unittest baselines.
- apply: author_plan = [{flydsl, author} FIRST, {triton, rewrite}]. FlyDSL is available
  (`is_flydsl_available()`=True) with `flydsl_preshuffle_gemm_a8(XQ,WQ,x_scale,w_scale,Out, tile_m,tile_n,
  tile_k, lds_stage, use_cshuffle_epilog, use_async_copy, waves_per_eu)` + `flydsl_hgemm` + split-K helpers
  — a real fp8 GEMM primitive the author baseline reuses (knobs map to per-shape tuning incl. split-K for
  decode). Triton rewrite headroom = autotune the hardcoded tiles + split-K for decode-M + fuse/kill the
  per-call x-requant; weights are static fp8+uint8 (do NOT re-materialize bf16 → balloons cache, starves KV).
- verify: immutable `unittest.py` (geomean over 18 cases, decode M∈{1,64} MANDATORY; must not regress
  decode-M). Baseline bars (gfx950, TP4, GPU0): dominant dense_gate_up prefill M8192 N6144 K6144 = 0.837ms;
  dense_down prefill = 0.418ms; qkv prefill = 0.357ms; o_proj prefill = 0.301ms; decode M64 ~0.05-0.09ms each.
- source: exp/e2e_MiniMax-M3-MXFP8_20260621_144547_3794_10149 (op _mxfp8_linear_kernel, pct_gpu=16.79);
  reconfirmed exp/e2e_MiniMax-M3-MXFP8_20260621_213057_794868_11259 (pct_gpu=11.2; immutable unittest
  16/16 pass geomean 1.0, max_rel_err<=0.0255<tol0.06; is_flydsl_available=True, flydsl_preshuffle_gemm_a8
  + flydsl_hgemm callable. Baseline bars GPU0: gate_up prefill M8192 N6144 K6144 0.4291ms (dominant) /
  decode M32 0.0499 / M1 0.0483; qkv prefill M8192 0.2361 / decode 0.043; down prefill M8192 0.2336;
  o_proj prefill M8192 0.1794. best_known_ms=0.4291. op_bench sig-mismatch reconfirmed -> unittest is authoritative.)
