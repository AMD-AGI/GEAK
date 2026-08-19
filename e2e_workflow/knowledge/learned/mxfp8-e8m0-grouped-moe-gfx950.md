---
key: mxfp8-e8m0-grouped-moe-gemm · gfx950 · both
type: routing
confidence: ★★★
confirms: 3
effect: no env/flag win exists (iso 1.0× against the immutable harness) — the author route is the only lever. Same seam and same verdict on BOTH the vLLM and sglang stacks.
last_seen: 2026-08-14
---
# MXFP8 (1×32 E8M0) grouped MoE expert GEMM — route to Tier-C author, not env

- path: (1) the live op is an EDITABLE in-tree Triton `tl.dot_scaled` kernel on gfx950 native MX cores
  (`vllm…mxfp8_native_moe:_grouped_gemm_mxfp8`; sglang
  `…moe.mxfp8_moe_amd_gfx95:fused_experts_mxfp8`, same `_mxfp8_grouped_gemm_kernel` design). (2) Skip the
  env lane — no drop-in exists, and two specific dead-ends waste a round (below). (3) Author:
  `author_plan = [{flydsl, author} FIRST, {triton, rewrite}]`.
- expected gain: unquantified — no verified e2e yet. Head share is 14.8–32.6% GPU across the three
  confirms, so it is worth a full author round. Triton headroom is structural: the in-tree kernel
  hardcodes BLOCK_K=128, BLOCK_N 64/128, num_warps 4/8, with NO autotune and NO split-K for decode-M
  (BLOCK_K=256 alone measured ~1.08× iso). FlyDSL headroom is that it has real E8M0 microscale
  grouped-MoE primitives (`aiter.ops.flydsl.flydsl_moe_stage1/stage2`) whose knobs — tile_m/n/k, k_batch
  split-K, persist_m, waves_per_eu, use_async_copy — map 1:1 onto this op.
- dead-end 1: the aiter bf16 `AITER_TUNE_GEMM`/`AITER_CONFIG_GEMM_BF16` DB is the WRONG lever — this op
  is dispatched directly by the framework's Triton kernel, not by aiter `gemm_a16w16`, so the DB has no
  matching key and engagement is 0.
- dead-end 2: `scripts/op_bench.py` misroutes it — `_is_blockscale_gemm` sends it to the dense a8w8
  blockscale probe, a DIFFERENT op (bpreshuffle gave max_rel_err≈41.8). Use the task's IMMUTABLE
  `unittest.py` as the authoritative bake-off; it times the real kernel against a faithful E8M0 oracle.
- verify: immutable `unittest.py`, geomean over 6 cases, decode T∈{1,64} MANDATORY — must not regress
  decode-M. Reference bars (gfx950 TP4): gemm1_w13 prefill M4096 N1536 K6144 = 0.326 ms (dominant),
  gemm2_w2 prefill = 0.199 ms, decode T64 = 0.255/0.152 ms.
- source: exp/e2e_*MiniMax-M3-MXFP8*/ 2026-06-21 ×2 (vLLM, head 24.77% → 32.61%; unittest 6/6 pass,
  geomean 1.0; is_flydsl_available=True) and 2026-08-13 (sglang TP8, head 14.78%; aiter.fused_moe also
  carries native mxfp8 infra — fused_dynamic_mxfp8_quant_moe_sort, fused_moe_2stages).
