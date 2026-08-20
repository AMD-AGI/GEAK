---
key: mxfp8_microscale (dense linear + grouped MoE) · gfx950/CDNA4 · vLLM+sglang
type: lever
confidence: ★★★
effect: per-(N,K,regime) STATIC tiles → +12.1% e2e VERIFIED (vLLM dense+grouped stacked); iso dense 1.39–1.58×, grouped 1.17×. sglang leg: same seam, same headroom, e2e still UNMEASURED (contention).
last_seen: 2026-08-17
---
# MXFP8 (E8M0 1×32 microscale) GEMM heads → editable Triton `tl.dot_scaled` tiles

- path: (1) the live seam on both frameworks is the native CDNA4 Triton `tl.dot_scaled` kernel —
  vLLM `mxfp8_native_moe.py:_grouped_gemm_mxfp8` (~32% GPU) + `linear/mxfp8/rocm_native.py`
  (~22%); sglang `mxfp8_amd_gfx95.py:dot_scaled_mxfp8_blockscaled_linear`. (2) Expect no env/flag
  win — aiter/FlyDSL/CK ship no E8M0-1×32 drop-in on this image (probed 62 and 72 entrypoints; worth
  a quick re-probe each run). (3) The lever is that all of them ship ONE tile config
  (`BLOCK_M=64 BLOCK_N=128 BLOCK_K=128 nw=8`) for ALL shapes → author per-(N,K,regime) static tiles.
- expected gain: iso ~1.39× geomean dense, 1.17–1.43× grouped; stacked dense+grouped banked +12.1%
  e2e at ~32%+~22% head share. Tiles that transferred (verify, don't assume): dense decode(M≤64)
  → BK=256, dense prefill → BM=128; grouped GEMM1 (N=1536,K=6144) decode → BN=64+BK=256+nw=4,
  grouped prefill + GEMM2 → BN=128+BK=256.
- apply: a host-int branch on M/N/K — compile-once and cudagraph-safe. vLLM serves decode under
  cudagraph_mode FULL_AND_PIECEWISE, so a self-capturing/JIT/host-sync wrapper falls back to eager;
  a pure static-tile change is inherently capture-safe. See [[method-cudagraph-safe-integration]].
- verify: there is no `reference_io.pt` here and the `op_bench.py` blockscale path cannot represent
  E8M0-grouped (nor resolve symbolic `a_shape=["M",6144]`) → it reports `harness_suspect`. Run the
  immutable `unittest.py` DIRECTLY as the bench instead (endorsed self-repair), then gate on e2e.
- caution: **a host-heavy rewrite can win isolated and REGRESS e2e on this decode-bound mix** — the
  FlyDSL fused-fp8 dense kernel measured 1.94× isolated but −14.8% e2e (per-call activation-quant
  fold + data_ptr cache + dispatch overhead dominates skinny-M decode). And **when stacking both GEMM
  heads, pick by COMBINED e2e, not solo**: FlyDSL dense was +6.7% solo but only +9.65% combined,
  while static-tile dense stacks to ~+12%. Prefer static tiles as primary.
- caution: this head's e2e gate is unmeasurable on a CONTENDED box — a TP8 mxfp8 launch needs
  ~52.6 GB/GPU and OOMs pre-model-load when another namespace pins the GPUs. Confirm GPUs are free
  (<~50 GB used, <~10% GFX) before trusting any delta, and treat a fully-overlapping sub-noise A/B as
  UNMEASURED (re-task), not as a no-win. Rebind banners proving the seam is bound are not a win.
  A `pct_gpu_time=0` reading under contention is a profile-block artifact, not a small head.
- source: GEAK/exp_ab A/B 2026-06-19 (dense+grouped static tiles +12.1% non-overlapping; FlyDSL
  −14.8%); exp/e2e_*MiniMax-M3-MXFP8*/ 2026-06-20 → 2026-08-16 (vLLM + sglang TP8 re-confirms, 8×;
  immutable unittest all-pass, err ≤ 0.0077 vs tol 0.06; head ~31.7% → iso 1.39× ⇒ +9.78% ceiling;
  sglang e2e flagged UNMEASURED — rebind engaged 11/11 workers but the A/B sat at +0.23% inside the
  0.5% band and the Director leg never launched).
