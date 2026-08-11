---
name: buy-lds-back-from-the-pipeline-depth-then-spend-it-on-in-bod-dense-gemm-gfx950-compute-bound
description: Shrink pipeline depth to free LDS, then coarsen tile rows in-body under a frozen launch config: 2.11x from the depth cut alone, 2.66x geomean stacked
keywords: [pipeline-stages, lds-tiling, coarsening, tile-shape, vgpr, occupancy, dense-gemm, compute-bound, launch-config]
kernels: [_gemm_a16_w16_kernel]
platforms: [gfx950]
kernel_class: dense_gemm
regime: compute-bound
key: in-body row coarsening plus pipeline-depth reduction on a Triton fp16 dense GEMM on gfx950 whose launch config the harness has frozen, small-M and large-M cases
lifecycle: active
type: lever
confidence: ★★
effect: 2.664x geomean cumulative (best single pass 2.6811x), per-case 1.887x on the small-M case (M=2048) and 3.148x / 3.184x on the two large-M cases; empirical roofline position went 0.510 -> 1.470 over 44 passes
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.58h / 44 passes, 2026-08-11
last_seen: 2026-08-11
---
# Buy LDS back from the pipeline depth, then spend it on in-body row coarsening
- lever: When the harness freezes BLOCK_M or the whole launch config, the tile-row count is still reachable from inside the body - have one program walk 2 tile rows in registers instead of one, and pay for the extra accumulators first by shrinking the software-pipeline depth, which on this generation is the largest LDS consumer and often buys a whole occupancy step on its own.
- apply: Cut stages before widening anything (LDS per workgroup dropped by ~4.5x with the kernel body otherwise unmodified), then raise the coarsening factor; bigger BM was monotonically better right up to the spill edge, and gains concentrate on the large-M cases while the small-M case lags on tail/occupancy.
- stack: total 2.664x geomean = three directions compounded - 1. pipeline stages 3 -> 1, 2.1107x standalone, the bulk of the win; 2. in-body M-coarsening at COARSEN=2 on top of (1), reaching 2.5978x with VGPR 170 -> 216, occupancy 2, zero spills, bit-exact output; 3. a full register-staged wide-K rewrite at BM=256 / BN=128 / KT=64 / instr_shape=[16,16,32] / warps_per_cta=[4,1] / k_width=16, ending at 2.664x. Attribution is incremental in landing order, not independent.
- pitfall: pushing the coarsening factor further stopped paying -> the extra accumulators crossed the register wall -> COARSEN=3 returned exactly 1.00x and a doubled [512,128] fp32-accumulator tile spilled into a 10x regression; check VGPR and spill count on every new body.
- verify: Read VGPR and spill count off the compile for each coarsening step and confirm bit-exact output, then re-time per case against the frozen baseline.
- caution: Also verify the small-M case separately - it is tail- and occupancy-limited, so a factor tuned on the large-M cases can be flat there.
- source: chuschen 16h time-budget campaign run, 15.58h / 44 passes, 2026-08-11
