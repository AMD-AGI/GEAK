---
key: occupancy-2 register-tight int4 MoE grouped GEMM on gfx950/CDNA4 where the fp32 accumulator, not scheduling, sets the ceiling
type: anti-pattern
confidence: ★★
effect: 0 of 4 occupancy-escape directions beat the frozen baseline on any case: AGPR-accumulator occ3 and warp-specialized dequant 0x (never reached a passing build/measurement), hand-written inline-asm inner GEMM 1.005x (inside the ~1-2% noise band), operand-staging-in-LDS for the batch-64 case 0x. The large-batch buckets sit at 186/204 registers with zero spill; occupancy 2 is arithmetic, not a tuning miss.
confirms_cited: 3
confirms_blind: 0
losses: 2
attempts: 10
toolchain: unknown
last_seen: 2026-08-12
name: accumulator-set-occupancy-floor-on-a-unified-vgpr-agpr-pool--moe-grouped-gemm-gfx950-both
description: On CDNA4 the VGPR and AGPR files share one budget, so a large fp32 accumulator pins occupancy at 2; four occupancy-escape rewrites all measured non-positive.
keywords: ['occupancy', 'vgpr', 'agpr', 'accumulator', 'raw-hip', 'warp-specialization', 'register-pressure', 'moe', 'grouped-gemm', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
verified_on: 2026-08-11
---
# Accumulator-set occupancy floor on a unified VGPR+AGPR pool closes the occupancy-escape lane
- lever: Before spending rounds on an occupancy escape, compute the floor: on this arch occupancy is 512/(ArchVGPR+AGPR) with the two files SUMMED, so an fp32 [BLOCK_M,64] MFMA accumulator costs 64 registers per lane in either file and the sum already decides the answer.
- apply: Read the register counts and spill count out of the compiled ISA for each bucket; if the accumulator alone puts the sum over the occupancy-3 threshold, the escape is arithmetically unavailable and the round is better spent on tile shape or data movement.
- verify: An escape claim is only real if the ISA shows the register total actually dropped below the next occupancy step AND the isolated A/B moves more than the noise band; a rewrite that lands at 1.005x has changed nothing.
- pitfall: A hand-written low-level rewrite of the inner GEMM looked like free headroom -> the compiler-scheduled version was already spill-free and software-pipelined at the same occupancy, so there was no register or scheduling slack to reclaim -> treat a zero-spill, already-pipelined inner loop as evidence against the rewrite lane rather than for it.
- caution: Also verify the pool semantics on YOUR arch before reusing this: the summed-file behaviour is what closes the lane here, and an arch with independent accumulator registers can still make the escape pay.
- source: 16h single-kernel time-budget campaign, run id chuschen16h, 32 passes, 2026-08-11
