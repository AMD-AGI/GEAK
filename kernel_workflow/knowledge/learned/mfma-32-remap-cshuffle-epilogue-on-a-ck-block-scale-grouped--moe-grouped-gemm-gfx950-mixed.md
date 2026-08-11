---
key: fp8 per-block-scale grouped MoE GEMM built on Composable Kernel, gfx950/MI355, gate+up fused stage
type: lever
confidence: ★★
effect: 1.4655x isolated geomean vs frozen baseline, bit-exact; per-case: small 2-token case 1.30x, 32-token case 1.55x, 64-token case 1.58x — the win grows with tokens-per-expert and the small case caps the geomean
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-11
name: mfma-32-remap-cshuffle-epilogue-on-a-ck-block-scale-grouped--moe-grouped-gemm-gfx950-mixed
description: On CK fp8 block-scale grouped-MoE GEMM (gfx950), remap the pipeline to MFMA 32x32 + CShuffle epilogue: ~1.47x isolated, biggest on large cases
keywords: ['moe', 'grouped-gemm', 'fp8-blockscale', 'composable-kernel', 'mfma', 'cshuffle', 'lds-padding', 'gfx950']
kernels: ['moe_stage1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
---
# MFMA-32 remap + CShuffle epilogue on a CK block-scale grouped MoE GEMM
- lever: Move the CK grouped-GEMM pipeline from MFMA 16x16 to 32x32 (with the matching host shuffle mapping), then add a CShuffle write-out epilogue and one row of LDS pad on the A block.
- apply: The pipeline-version remap lives in the modifiable header that selects the fused gate+up pipeline variant; MFMA-32 tuning only takes effect through that remap. Host-side shuffle dims must be re-paired with the new MFMA tile, epilogue scalar-per-vector 8 with per-shuffle (1,1), A-block LDS extra-M 0->1.
- stack: total 1.4655x isolated (bit-exact, director-verified) = three directions compounded
  - 1. MFMA 16x16->32x32 + host shuffle repair — 1.4441x standalone — carries essentially the whole win
  - 2. CShuffle epilogue write-out — +1.25% on top of (1)
  - 3. A-block LDS pad 0->1 — +0.33% on top of (1,2); real but thin
  - note: attribution is incremental in landing order, not independent.
- verify: Compare against the frozen baseline per case and confirm bit-exactness (err_ratio 0, cosine diff at 1e-8 level); check the emitted ISA actually uses the 32x32 MFMA and that VGPR count did not cross the occupancy-2 boundary.
- pitfall: Perf and correctness runs silently used a stale binary -> the build step deletes the shared object and JIT cache -> compile first inside every measurement run.
Deeper prefetch looked free but a 3rd B buffer spilled -> occupancy dropped to 1 -> ~2x regression; keep the buffer count at the occupancy-2 VGPR budget.
- caution: Also verify the block-M coupling before widening the tile: MPerBlock is tied to the host sort block size, and doubling it makes one block straddle two expert groups and corrupt results. Also verify non-temporal loads case by case — with topk>1 the A operand is L2-reused and marking it non-temporal cost 12-13% here.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11
