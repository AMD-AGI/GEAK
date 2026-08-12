---
key: tiny int64 index-scatter memory-movement op on gfx950, Triton launched from Python, where host dispatch dominates the measured window
type: lever
confidence: ★★
effect: 2.61x geomean vs frozen baseline, per-case 2.92x / 2.60x / 2.35x at the small / mid / large batch cases; non-overlapping same-session A/B; lands at ~0.55-0.63 of the empty-launch bracket ceiling
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: raw-driver-launch-for-a-dispatch-bound-copy-op-memory-movement-gfx950-launch-bound
description: Dispatch-bound tiny index-scatter: raw ctypes hipModuleLaunchKernel with pre-packed params replaces the Triton Python launcher, ~2.6x per-case
keywords: ['launch-overhead', 'dispatch-bound', 'host-submit', 'tiny-kernel', 'memory-movement', 'hip-graph', 'ctypes']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: memory_movement
regime: launch-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-07-30
---
# Raw driver launch for a dispatch-bound copy op
- lever: When the measured window is mostly host submit, call the already-compiled module through a raw ctypes driver launch instead of the framework's Python launch wrapper.
- apply: Cache the compiled binary and a ctypes kernelParams buffer packed once at setup; per call, do one driver launch with a fixed grid, no argument re-marshalling and no autotuner/hook re-entry.
- verify: Check the grid and bit-exact integer output against the golden reference, then confirm the gain is roughly flat across batch cases - a per-call constant, not a shape effect.
- pitfall: graph capture/replay looked like the answer and returned only ~1.32x -> replay still pays the Python bracket around it -> the raw module launch on the same seed reached 2.61x and superseded it.
- caution: The win is bounded by the timing bracket itself; also verify what fraction of the measured window an empty launch already costs before attributing more headroom to further host work.
- source: run write_req_to_token_pool_triton-ch16h, 2026-07-30, 16h budget, gfx950
