---
key: int4-dequant feed path of a group-quantized MoE grouped GEMM already pinned at occupancy 2 by its fp32 accumulator, gfx950/CDNA4, Triton with a hand-written register pipeline
type: anti-pattern
confidence: ★★
effect: closed axis, same per-case set each time: magic-number bitcast dequant 0.99x (tiny-M case 0.978x) though bit-exact and ISA-confirmed, Triton num_stages>=2 0.85-0.74x at BLOCK_M 256/512 and ns=3 over the LDS cap, '.cg' cache modifier 0.83x/0.89x on the two large-M cases, raw-HIP inner GEMM 1.005x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed
description: Four dequant/feed-path directions all returned ~1.00x or worse on an occupancy-2 VGPR-floored int4 MoE GEMM: that axis is closed on gfx950
keywords: ['moe', 'grouped-gemm', 'int4', 'dequant', 'num-stages', 'cache-modifier', 'raw-hip', 'occupancy', 'anti-pattern', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
levers: ['compute.dequant-opcount', 'mem.cache-modifier', 'compute.pipeliner']
origin_kernels: ['fused_moe_kernel_gptq_awq']
---
# Dequant op-count is off the critical path once the GEMM is occupancy-pinned
- lever: When an int4-weight GEMM is already at occupancy 2 with zero spill, budget the round elsewhere than the dequant feed path: its VALU work already overlaps the MFMA pipeline.
- apply: Cheap discriminator before spending a round: check spill count and occupancy from the compiled ISA. Zero spill at occupancy 2 with the accumulator at the register ceiling means op-count, cache hints and pipeliner settings have no slack to recover.
- verify: Each of these fails loudly rather than silently, which is what makes them trustworthy negatives: the ISA shows the conversion ops actually disappeared, and the '.cg' regression proves the edit landed.
- pitfall: Layering the Triton auto-pipeliner on a loop that already has a manual register double-buffer regressed hard → it multi-buffers the dequantized operands through LDS on top of the manual one → keep the manual pipeline and leave num_stages at its default; ns=1 only disables the LDS auto-stage and reads as noise.
- caution: Also verify any small win here survives a repeat: a ~0.6% flicker on one bucket did not, run-to-run spread on that bucket being ~1-2%.
- source: 16h per-kernel time-budget campaign, run fused_moe_kernel_gptq_awq-ch16h, 2026-08-12
