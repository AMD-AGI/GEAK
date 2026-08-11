---
name: a-spill-free-restructure-can-still-lose-the-win-by-stepping--quantized-gemm-gfx950-compute-bound
description: Price a manual loop restructure in VGPRs against the next occupancy step, not in spills: two spill-free hoists on a quantized GEMM lost ~10-20%
keywords: [vgpr, occupancy, pipeline-stages, quantized-gemm, isa-check, compute-bound, operand-reuse, lds-tiling]
kernels: [_w8a8_triton_block_scaled_mm]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: manual software-pipelining / loop restructuring in the k loop of a Triton block-scaled quantized GEMM on gfx950 whose body already sits at its register ceiling
lifecycle: active
type: anti-pattern
confidence: ★★
effect: hand-hoisting the next k-block's operand load, and then load+convert, across the outer loop was briefed at 11.2x and measured 0 -- both forms regressed the 10.66x seed on the same three-case shape set, ~10% slower for loads only and ~20% for loads+convert, with spill_count=0 throughout and bit-exact output. The whole loss was VGPR 154 / occupancy 3 -> VGPR 202 / occupancy 2, against a measured occupancy-3 ceiling of <=170 VGPRs (~16 VGPRs of headroom at the seed). The same cliff closed a second lane: replacing the LDS transpose with a no-transpose reshape landed at VGPR 211 / occupancy 2 and ~25% slower, while the transposed path compiled to byte-identical ISA whether the transpose sat on the integer load or the wide tile. A third lane confirmed the level below is also unreachable and irrelevant: occupancy 4 -> 5 would not break, and occupancy was never the limiter -- an earlier edit went 4 -> 3 and still gained +20%.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.73h / 31 passes, 2026-08-11
last_seen: 2026-08-11
---
# A spill-free restructure can still lose the win by stepping down one occupancy level
- lever: before funding a manual software-pipelining or loop-restructuring lane on a body already at its register ceiling, price it in VGPRs against the next occupancy step rather than in spills. Absence of spilling is not evidence the restructure was free: loop-carried prefetch state and prologue/epilogue duplication buy scheduling freedom with registers, and one step down in occupancy can cost more than the whole scheduling win. Two cheap screens retire most of this family: measure how many VGPRs separate the current build from the next occupancy step, and check whether the backend pipeliner (num_stages>1) is already prefetching the raw loads the manual hoist would duplicate -- if it is, the hoist buys nothing and pays registers.
- apply: compile the candidate and read VGPR count, occupancy and spill count before timing it; treat crossing the occupancy step as a rejection reason on its own.
- verify: when a restructure regresses with zero spills, attribute it from the static ISA occupancy line rather than re-tuning the schedule -- at the wall clock the two look identical.
- caution: also verify occupancy as a budget constraint at the current working point, not as a score to maximise: an occupancy drop that comes with a working-set change was still a large win here, and an earlier edit lost a level and gained +20%.
- source: chuschen 16h time-budget campaign run, 15.73h / 31 passes, 2026-08-11
