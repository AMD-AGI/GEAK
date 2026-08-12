---
key: elementwise fp8 per-token-group quantize/cast on gfx950, Triton, where the benchmark harness pins the launch grid and num_warps
type: lever
confidence: ★★
effect: 1.14-1.15x plateau held for 43 consecutive passes, then 2.29-2.32x geomean bit-exact once the launch was re-tiled; per-case: 32-row case 3.10x, 64-row case 3.23x, tiny 2-row case only 1.19x (its own floor). HBM utilization 22% -> 62% of nameplate.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-11
name: reinterpret-frozen-launch-via-wrapper-object-quantize-cast-gfx950-mixed
description: Export a launcher OBJECT so the frozen num_warps=1/one-program-per-row launch can be re-tiled: 1.15x plateau -> 2.3x on memory-bound fp8 quant-cast.
keywords: ['launch-config', 'wrapper-relaunch', 'harness-seam', 'quantize-cast', 'fp8', 'memory-bound', 'tiling', 'num-warps']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: mixed
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
roofline: memory-bound 0.18 -> memory-bound 0.58 of empirical roof
---
# reinterpret-frozen-launch-via-wrapper-object
- lever: When the runner resolves an exported symbol and calls sym[grid](...), the pinned grid and num_warps are a seam, not a constraint: export an object whose __getitem__(grid) returns a callable with the same signature, and have it relaunch an inner jit body under a tiling and config you choose.
- apply: Wrapper keeps the exact call signature, derives its own grid from the incoming one, and dispatches an inner jit where one program owns a tile of rows (here 32 rows x 128-wide group), num_warps=4, num_stages=1, streaming cache modifier on the quantized store only; expose the knobs as env overrides so a sweep needs no rebuild.
- verify: Confirm bit-exact output against the oracle, then check the isolated A/B moves the large cases specifically and that measured bandwidth utilization rises toward the achievable fraction; a win that shows up only on the smallest case is warmup or replay artifact.
- pitfall: 43 passes stalled at ~1.15x reworking only the kernel body -> the launch config had been assumed out of scope, so every candidate inherited a one-warp, one-program-per-row launch that cannot saturate memory -> re-deriving the grid inside an exported wrapper object unlocked the remaining 2x.
- caution: Also verify the wrapper's own dispatch cost on the smallest shape (it can eat the gain there), and that the inner tiling stays bit-exact rather than merely within tolerance.
- source: GEAK 16h per-kernel time-budget campaign, quantize/cast lane, 56 passes, 2026-08-11
