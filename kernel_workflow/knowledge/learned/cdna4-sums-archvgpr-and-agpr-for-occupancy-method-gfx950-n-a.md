---
key: occupancy arithmetic for fp32-accumulator MFMA GEMMs on gfx950/CDNA4, where CDNA3-era AGPR-offload tricks are assumed to still buy a wave
type: anti-pattern
confidence: ★★
effect: 1.00x: a raw-HIP AGPR-accumulator rewrite aimed at occupancy 3 returned no advance on either large-M case, and occupancy stayed 2 because waves = 512/(ArchVGPR+AGPR) with a 64-register-per-lane accumulator putting the total at ~242
confirms_cited: 1
confirms_blind: 0
losses: 1
attempts: 3
toolchain: unknown
last_seen: 2026-08-17
name: cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a
description: On gfx950/CDNA4 occupancy divides one summed ArchVGPR+AGPR pool, so an AGPR-accumulator occupancy escape on an fp32-accum MFMA GEMM cannot exist
keywords: ['occupancy', 'agpr', 'vgpr', 'mfma', 'accumulator', 'raw-hip', 'anti-pattern', 'gfx950', 'grouped-gemm']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
levers: ['compute.occupancy', 'compute.agpr-accumulator']
---
# CDNA4 sums ArchVGPR and AGPR for occupancy
- lever: Before planning an occupancy escape by moving an MFMA accumulator into AGPRs, do the arithmetic for this arch: the two files share one budget here, so the move is register-neutral.
- apply: Compute waves = 512/(ArchVGPR+AGPR) with the accumulator counted once, whichever file holds it; if that lands on the same wave count you already have, the whole rewrite lane is a no-op and a hand-written HIP variant will not change it.
- verify: Confirm from the compiled ISA that the register total, not scheduling, is the binder: zero spill plus a software-pipelined schedule means the compiler already found the slack a rewrite would be looking for.
- pitfall: A large expected gain was assigned to this direction from CDNA3 intuition → the max-of-two-files occupancy model no longer holds → derive the wave count from the summed pool before budgeting the round.
- caution: Also verify the accumulator dtype: the register floor here came from an fp32 accumulator over a large M tile, so a narrower accumulator or a smaller M tile changes the arithmetic and may reopen the lane.
- source: 16h per-kernel time-budget campaign, run fused_moe_kernel_gptq_awq-ch16h, 2026-08-12
