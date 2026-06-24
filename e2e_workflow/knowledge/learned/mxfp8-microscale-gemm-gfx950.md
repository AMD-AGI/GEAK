---
key: mxfp8_microscale (dense linear + grouped MoE) · gfx950/CDNA4 · vLLM
type: lever
confidence: ★★★
effect: STATIC tiles → +12.1% e2e VERIFIED (dense+grouped, conc32/osl1024); iso dense 1.39–1.58× / grouped 1.17×
last_seen: 2026-06-20
---
# MXFP8 (E8M0 1×32 microscale) GEMM heads → editable Triton `tl.dot_scaled` tiles
- lever (worth trying first): the live MXFP8 path on gfx950 vLLM is the native CDNA4 Triton
  `tl.dot_scaled` kernel — `mxfp8_native_moe.py:_grouped_gemm_mxfp8` (MoE, ~32% GPU) +
  `kernels/linear/mxfp8/rocm_native.py:_mxfp8_dot_scaled_linear` (dense, ~22%). Both ship
  `BLOCK_M=64 BLOCK_N=128 BLOCK_K=128 nw=8` for ALL shapes = headroom. Still run the full bake-off/sweep;
  this is a strong seed, not the only option.
- apply: a per-(N,K,regime) **STATIC tile config** (a host-int branch on M/N/K — compile-once, cudagraph-safe).
  Tiles that transferred to e2e (verify, don't assume): dense decode(M≤64)→BK=256, dense prefill→BM=128;
  grouped GEMM1(N=1536,K=6144) decode→BN=64+BK=256+nw=4 (1.27–1.43×), grouped prefill + GEMM2→BN=128+BK=256.
- verify: synthesized oracle (no `reference_io.pt`) → run the immutable `unittest.py` DIRECTLY as the bench
  (`op_bench.py` blockscale path can't represent E8M0-grouped → self-repair a driver). Then confirm the
  e2e gate, not just isolated.
- caution: **a host-heavy rewrite can win isolated yet REGRESS e2e on this decode-bound serving.** The
  FlyDSL fused-fp8 dense kernel measured **1.94× isolated but −14.8% e2e** (per-call activation-quant
  fold + data_ptr-cache + FlyDSL dispatch overhead dominates skinny-M decode). So *also* compare a
  zero-host-overhead static-tile variant and let the e2e gate pick — don't ship an isolated win unverified.
- caution: **when STACKING both GEMM heads, pick by COMBINED e2e, not solo.** Measured: FlyDSL dense
  +6.7% SOLO but dense+grouped COMBINED only +9.65%, whereas static-tile dense stacks to ~+12% combined
  (the two heads' gains overlap + FlyDSL's per-call host work eats the shared decode path). Prefer the
  static-tile dense as primary; FlyDSL is a secondary candidate to verify by the combined gate, not solo.
- caution: aiter/FlyDSL/CK have no MXFP8-E8M0 *drop-in* primitive on this image (so an env/flag swap was
  a no-op last time) — worth a quick re-probe each run, but the editable Triton tile path is the reliable seam.
- integration: vLLM serves decode under cudagraph_mode FULL_AND_PIECEWISE; a self-capturing/JIT/host-sync
  wrapper falls back to eager. A pure static-tile change is inherently capture-safe. See [[method-cudagraph-safe-integration]].
- source: GEAK/exp_ab A/B 2026-06-19 (dense+grouped static tiles +10.25%→+12.1% non-overlapping; FlyDSL −14.8%);
  iso sweeps exp/e2e_*MiniMax-M3-MXFP8*/ (8 re-confirms, max_rel_err≤0.004 vs tol 0.06).
- source: 2026-06-20 bakeoff re-confirm exp/e2e_MiniMax-M3-MXFP8_20260620_093154_1073576_26273 — live
  `_grouped_gemm_mxfp8` still ships static BLOCK_N=128/BLOCK_K=128/nw=8 for ALL shapes (headroom intact);
  aiter dir has fp8-blockscale + mxfp4 MoE but NO E8M0-1x32 grouped drop-in (env/flag swap still no-op);
  immutable unittest baseline all-6-pass (err≤0.0069 vs tol 0.06), best-known geomean ms recorded.
- source: 2026-06-20 DENSE head re-confirm (same eval dir, _mxfp8_linear_kernel_task, N=K=6144) — static
  default 64/128/128/nw8 sweep vs immutable oracle (err≤0.0037 vs tol 0.06): per-(M,regime) tiles win
  decode M1→(BM128,BN64,BK256,nw8) 1.33×, M32→(64,64,256,4) 1.29×, M512→(128,256,128,8) 1.33×,
  prefill M8192→(128,128,128,4) 1.64×; geomean iso ~1.39×. aiter E8M0 dense drop-in absent (bake-off
  exhausted 62 entrypoints, none fit) → editable Triton static-tile rewrite is the seam; route=rewrite.
