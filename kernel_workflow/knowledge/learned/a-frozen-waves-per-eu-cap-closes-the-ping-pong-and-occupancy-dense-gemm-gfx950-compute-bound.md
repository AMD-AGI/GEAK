---
name: a-frozen-waves-per-eu-cap-closes-the-ping-pong-and-occupancy-dense-gemm-gfx950-compute-bound
description: Price a published peak-efficiency GEMM schedule against your shape and a backend-pinned waves-per-EU first: four directions on this axis all returned 1.00x
keywords: [occupancy, waves-per-eu, dense-gemm, compute-bound, mfma, pipeline-stages, lds-tiling, tile-shape, vgpr]
kernels: [_gemm_a16_w16_kernel]
platforms: [gfx950]
kernel_class: dense_gemm
regime: compute-bound
key: ping-pong / occupancy-1 scheduling on a Triton fp16 dense GEMM at skinny-M low-K shapes on gfx950, where the AMD backend pins waves-per-EU in make_llir
lifecycle: active
type: anti-pattern
confidence: ★★
effect: four separate directions on this axis all returned 1.00x - occupancy is hard-locked at 2 waves (the backend writes 'amdgpu-waves-per-eu 2,2' unconditionally at the end of make_llir), the occupancy-1 register-double-buffer variant at VGPR 304 measured ~50% SLOWER, and forcing 8 waves via warps_per_cta=[8,1] collapsed 2 workgroups/CU to 1 and cost ~80% because inter-tile overlap is what hides load latency at this ~1793 FLOP/byte intensity; the incumbent sits at ~65% MFMA efficiency on the two large-M cases and ~38% on the small-M case (tail-limited at ~320 workgroups over ~256 CU) against the ~98.75%-of-peak fp16 ceiling the published schedule attains at 4096^2 x 8192, a regime that scopes skinny/low-K shapes out
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.58h / 44 passes, 2026-08-11
last_seen: 2026-08-11
---
# A frozen waves-per-EU cap closes the ping-pong and occupancy-1 axis on skinny low-K shapes
- lever: Before funding a round on a published peak-efficiency schedule (ping-pong, occupancy-1 register double-buffering, deeper async staging), check what shape it was published at and whether occupancy is yours to move at all - a low arithmetic-intensity, small-M, low-K shape wins from 2 workgroups per CU overlapping, so any schedule that trades workgroups for one fatter workgroup starts ~80% in the hole, and if the backend pins waves-per-EU at the end of LLIR generation no body-level primitive changes it.
- apply: Read the emitted LLIR/ISA for the waves-per-EU attribute and the workgroups-per-CU it implies before writing the schedule, and price the enabling primitive first (buffer_load_to_shared feeding a DotOperand failed to lower in this Triton build, an unrealized_conversion_cast at LLVM translation).
- pitfall: the intermediate LDS round-trip looked like pure waste -> removing it cost coalescing on the global loads -> keep the convert_layout round-trip through LDS; direct dot-layout global loads measured 2.5x slower.
- verify: Confirm the wave count actually changed in the compiled artifact, not just in the requested attribute, and re-time on the same skinny/low-K case the roofline claim was made on.
- caution: Also verify the published schedule's own shape regime before importing its expected gain - the ceiling it demonstrates may be measured on a square, high-K case this workload never reaches.
- source: chuschen 16h time-budget campaign run, 15.58h / 44 passes, 2026-08-11
