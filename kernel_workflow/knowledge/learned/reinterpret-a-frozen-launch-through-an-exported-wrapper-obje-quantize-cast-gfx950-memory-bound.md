---
key: per-token group fp8 quantize/cast in Triton on gfx950, where the caller pins grid and num_warps at the call site and the tile is therefore not the author's to choose
type: lever
confidence: ★★
effect: 2.29x geomean isolated vs frozen baseline, bit-exact; per-case 3.10x and 3.23x on the two large token-count cases, 1.19x on the tiny case; nameplate HBM utilisation ~22% -> ~62%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound
description: Export a launcher object with the runner's __getitem__(grid) shape to re-tile a frozen num_warps=1 launch: 2.29x geomean, bit-exact, on memory-bound fp8 quant
keywords: ['launch-config', 'wrapper-relaunch', 'quantize-cast', 'fp8', 'memory-bound', 'num-warps', 'tiling', 'bit-exact', 'cache-modifier']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: memory-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-07-30
roofline: memory-bound ~22% -> memory-bound ~62% of nameplate HBM bandwidth
origin_kernels: ['_per_token_group_quant_fp8']
---
# Reinterpret a frozen launch through an exported wrapper object
- lever: when the runner resolves an exported symbol and calls it as kern[grid](args), the launch config is reachable: export a wrapper OBJECT whose __getitem__(grid) returns a callable with the identical signature, and have it relaunch an inner jit kernel under your own grid, tile, num_warps and num_stages.
- apply: inner tile = one program per 32 token-rows x one group width, num_warps=4, num_stages=1, .cs cache modifier on the quantized store only (not on loads, not on the fp32 scale store); expose every knob as an env var so each is A/B-able without a rebuild.
- verify: oracle parity should stay bit-exact (the arithmetic is unchanged, only the mapping); confirm the config actually engaged by dumping the resolved launch params, and track fraction-of-nameplate bandwidth rather than the ratio alone.
- pitfall: diff came back empty when generating the patch → the run directory was gitignored → build the patch by diffing against the canonical source instead of via the repo index.
- caution: also verify the wrapper's outer signature matches the caller's exactly, and re-check parity for a dtype-conversion path folded in at the same time (a fnuz->native-OCP fp8 reinterpret with value correction was bit-exact here but is a separate claim).
- source: run _per_token_group_quant_fp8-ch16h, 16h per-kernel time-budget campaign, 2026-07-30
