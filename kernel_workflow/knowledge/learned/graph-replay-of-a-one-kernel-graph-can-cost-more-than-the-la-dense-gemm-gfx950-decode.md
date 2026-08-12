---
key: graph capture/replay as a launch-overhead lever for a single-kernel dense linear op on gfx950 ROCm, where one launch is the whole workload
type: anti-pattern
confidence: ★★
effect: replay ~2x slower than eager, robust across 3 signatures x 3 repeats; the wrapper's own signature cache costs ~0.9-1.0x of a launch on the host critical path so even the eager-fallback wrapper nets 0.88-0.96x per-case (batch 2 0.88x, batch 32 0.91x, batch 64 0.96x). Captured output passed parity, so the loss is timing, not capture.
confirms_cited: 1
confirms_blind: 0
losses: 2
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: graph-replay-of-a-one-kernel-graph-can-cost-more-than-the-la-dense-gemm-gfx950-decode
description: Graph-capturing one tiny kernel to beat the launch floor: replay measured ~2x slower than eager, and the signature-cache wrapper alone nets 0.88-0.96x.
keywords: ['hip-graph', 'launch-overhead', 'dispatch-floor', 'decode', 'skinny-m', 'host-runtime']
kernels: ['wvSplitK']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
---
# Graph replay of a one-kernel graph can cost more than the launch it replaces
- lever: treat graph capture as a candidate for multi-kernel sequences; for a single-launch op, price the replay path against eager in the same session before investing a round in wiring it.
- apply: capture the one launch, then A/B replay vs eager inside the same wrapper, with the timing region placed exactly where the harness places it (all wrapper Python is inside it).
- verify: interleave replay and eager in one process across every input signature; a robust ratio across repeats separates a real ordering from clock drift.
- pitfall: the wrapper added to dispatch the graph became the regression itself -> a per-call pointer/shape/dtype signature tuple plus closure dispatch runs inside the timed region -> even the eager fallback path lost to the unwrapped baseline, so the graph experiment cannot be read without measuring the empty wrapper first.
- caution: also verify the kernel count under capture: the conclusion is about a one-launch graph, and a sequence amortising several launches per replay is a different measurement worth taking.
- source: chuschen 16h per-kernel time-budget campaign, 18 passes, 2026-08-11
