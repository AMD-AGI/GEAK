---
name: an-fp32-accumulator-that-pins-occupancy-at-two-waves-is-not--moe-grouped-gemm-gfx950-compute-bound
description: When VGPR and AGPR share one pool, moving an fp32 accumulator cannot buy a wave: the escape round returned 0 and a hand-written HIP rewrite measured 1.005x
keywords: [occupancy, vgpr, mfma, moe, dequant, tile-shape, compute-bound, operand-reuse]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: occupancy-escape attempts on a bf16 fused-MoE grouped GEMM on gfx950/CDNA4, where ArchVGPR+AGPR are one shared pool and the fp32 accumulator sets the register floor
lifecycle: active
type: anti-pattern
confidence: ★★
effect: closed axis - on this part occupancy = 512 / (ArchVGPR + AGPR), a sum and not a max, so an fp32 [512,64] accumulator costs 64 registers per lane in either file and lands at 242 total, 2 waves regardless; the direction that budgeted 5.8x for an AGPR-accumulator 3-wave escape plus warp-specialised dequant returned no candidate, a hand-written HIP / inline-asm rewrite of the same inner GEMM measured 1.00517x against generated code already at 204 / 186 registers, 2 waves and zero spill on the two large batch buckets, a ground-up occupancy-tolerant rewrite returned 1.0115x (its only surviving win a warp-count re-sweep on the small bucket), and staging the A operand through LDS to buy occupancy returned nothing; the batch-64 case sits at ~29% of dense bf16 peak with the A operand L2-resident
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.74h / 32 passes, 2026-08-11
last_seen: 2026-08-11
---
# An fp32 accumulator that pins occupancy at two waves is not escapable by moving it to the other register file
- lever: Before funding an occupancy-escape round on an MFMA GEMM, price the accumulator - if the register floor is set by the accumulator shape and the two register files are a shared pool on this architecture, no placement of that accumulator changes the wave count, and the remaining occupancy story is tile height.
- apply: Read the compiled artifact for register count and spill first, then spend the round on tile shape or on how much dequant each tile amortises instead.
- pitfall: a raw-HIP / inline-asm rewrite of the inner loop was expected to recover the schedule -> the generated code was already at zero spill with full software pipelining, leaving no slack -> expect ~1.0x from such a rewrite and do not spend several rounds on it.
- verify: Compute occupancy from the summed register files (not the larger of the two) and confirm the wave count in the code object before and after.
- caution: Also verify the pool really is shared on the target part - the same accumulator move can buy a wave on an architecture where the files are budgeted independently.
- source: chuschen 16h time-budget campaign run, 15.74h / 32 passes, 2026-08-11
