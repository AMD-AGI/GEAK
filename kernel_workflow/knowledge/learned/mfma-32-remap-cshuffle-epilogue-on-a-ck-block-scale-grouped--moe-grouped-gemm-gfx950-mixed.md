---
key: fp8 per-block-scale grouped MoE GEMM built on Composable Kernel, gfx950/MI355, gate+up fused stage
type: lever
confidence: ★★
effect: reproduced in a second independent campaign at 1.2605x isolated geomean vs its own frozen baseline (per-case: 2-token 1.23x, 32-token 1.29x, 64-token 1.28x); the first campaign measured 1.4655x bit-exact against a different frozen baseline (1.30x / 1.55x / 1.58x per case) — the win grows with tokens-per-expert and the small case caps the geomean
confirms_cited: 3
confirms_blind: 0
losses: 0
attempts: 9
toolchain: unknown
last_seen: 2026-08-12
name: mfma-32-remap-cshuffle-epilogue-on-a-ck-block-scale-grouped--moe-grouped-gemm-gfx950-mixed
description: On CK fp8 block-scale grouped-MoE GEMM (gfx950), remap the pipeline to MFMA 32x32 + CShuffle epilogue: 1.26-1.47x isolated, reproduced in two campaigns
keywords: ['moe', 'grouped-gemm', 'fp8-blockscale', 'composable-kernel', 'mfma', 'mfma-nonkdim', 'cshuffle', 'lds-padding', 'block-m', 'template-instantiation', 'occupancy', 'isa-check', 'gfx950']
kernels: ['moe_stage1', 'ck_moe_stage1_gemm', 'ck_moe_stage2_gemm']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
---
# MFMA-32 remap + CShuffle epilogue on a CK block-scale grouped MoE GEMM
- lever: Move the CK grouped-GEMM pipeline from MFMA 16x16 to 32x32 on BOTH grouped stages (with the matching host shuffle mapping), then add a CShuffle write-out epilogue and one row of LDS pad on the A block — and treat the per-bucket block-M / instance-routing axis as re-opened by the shape change.
- apply: The pipeline-version remap lives in the modifiable header that selects the fused gate+up pipeline variant; MFMA-32 tuning only takes effect through that remap. Host-side shuffle dims have to be re-paired with the new MFMA tile, epilogue scalar-per-vector 8 with per-shuffle (1,1), A-block LDS extra-M 0->1. A 32-row MFMA cannot be hosted at MPerBlock=16, and the pipeline variant asserting MRepeat>=4 pins the larger block-M back to 16x16 — so the route to 32x32 at the wanted block-M was registering a new V1 twin of the existing tile in the codegen instance list, leaving the pre-existing instances untouched.
- stack: total 1.2605x isolated (director-verified, second independent campaign) = one dominant direction plus two thin ones
  - 1. MFMA 16x16->32x32 on both stages + the block-M / instance-routing re-sweep it re-opens — 1.2619x standalone (round 1, verified); the literal per-XDL edit alone was 1.058x, so roughly two thirds of it is the re-opened routing axis
  - 2. shared-memory reclaim on the ping-pong variant — 1.0438x standalone (round 1, verified) — added ~+0.3% median on top of (1); both levers shrink the same resource
  - 3. host/dispatch squeeze — 1.0079x standalone (round 1, verified) — an honest null on an op that is >99% device-bound
  - note: attribution is incremental in landing order, not independent. First campaign, same lever, different frozen baseline: total 1.4655x = remap 1.4441x, CShuffle epilogue +1.25%, A-block LDS pad +0.33%; those same two follow-ups re-measured at +0.84% and +0.30% in the second campaign — thin, but consistent in sign.
- verify: Compare against the frozen baseline per case and confirm bit-exactness (err_ratio 0, cosine diff at 1e-8 level); check the emitted ISA of the instances actually DISPATCHED (not just the tuning list) uses the 32x32 MFMA, and that VGPR count did not cross the occupancy-2 boundary.
- pitfall: Perf and correctness runs silently used a stale binary -> the build step deletes the shared object and JIT cache -> compile first inside every measurement run, and re-time after moving the pre-round binary aside.
Deeper prefetch looked free but a 3rd B buffer spilled -> occupancy dropped to 1 -> ~2x regression; keep the buffer count at the occupancy-2 VGPR budget.
The second grouped stage became co-equal in time with the first while doing half the FLOPs -> the (1,1) CShuffle epilogue had been applied to stage 1 only, so at 32x32 stage 2's full-block staging buffer doubled and halved its CTAs/CU -> port the symmetric epilogue edit to stage 2 (verified here, but worth under +1%, below that round's improvement gate).
A block-M value elected by a prior sweep stayed elected -> that sweep ran at 16x16, where the smaller block won -> re-sweep block-M and instance routing after any MFMA-shape change; at 32x32 the larger block won on every case.
An unexplained ~6.5% regression appeared on a code path that had not changed -> the harness checked GPU idleness BEFORE the exclusive lock was held, so a sibling's just-exited process was still releasing memory -> gate on idleness after acquiring the lock and retry; the regression disappeared.
A version-control diff taken inside the private per-candidate workspace produced an empty patch -> that copy carries no VCS metadata, so the diff resolved against the enclosing repository -> build the patch by diffing the canonical file against the workspace file and rewriting the a/ b/ prefixes.
- caution: Also verify the block-M coupling before widening the tile: MPerBlock is tied to the host sort block size, and doubling it makes one block straddle two expert groups and corrupt results; doubling block-M on both stages together also measured slower on every case at both pipeline versions, though routing the two stages' instances independently was never isolated. Also verify non-temporal loads case by case — with topk>1 the A operand is L2-reused and marking it non-temporal cost 12-13% here. Also verify what binds AFTER the remap before funding more of the same direction: here the register file and shared memory reached the same CTA/CU ceiling simultaneously, measured residency fell, and the bound class flipped from dependency-stall to issue/occupancy — the two classes take opposite fixes.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11; 2h KB-seeded campaign, run kernel_20_geak_0811_2h_kb_new, 2026-08-12
