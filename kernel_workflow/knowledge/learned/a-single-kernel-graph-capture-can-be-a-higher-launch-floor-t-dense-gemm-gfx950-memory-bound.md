---
name: a-single-kernel-graph-capture-can-be-a-higher-launch-floor-t-dense-gemm-gfx950-memory-bound
description: Measure graph replay against eager before building on it: for a one-kernel capture replay was ~2x SLOWER, and the shim alone net-regressed to 0.88-0.96x
keywords: [graph-capture, launch-overhead, dispatch-floor, measurement-method, interleaved-ab, control-experiment, dense-gemm, memory-bound]
kernels: [_gemm_a16_w16_kernel]
platforms: [gfx950]
kernel_class: dense_gemm
regime: memory-bound
key: HIP graph capture/replay of a single small dense-GEMM dispatch on gfx950, graded by an event-bracketed per-call harness that also times the Python selection shim
lifecycle: active
type: anti-pattern
confidence: ★★
effect: in-wrapper same-session A/B on every case - graph replay measured ~2x SLOWER than eager, robust across 3 signatures x 3 correctness repeats, with captured output numerically correct (Correctness PASS), so a genuine timing dead end rather than a capture bug; separately the Python signature-cache wrapper needed to dispatch the graph (a data_ptr/shape/dtype tuple plus a closure) added ~15% onto the host critical path (about a seventh of it), so even the eager-fallback arm net-regressed to 0.88-0.96x per case against the raw unwrapped baseline. No config shipped.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: time-budget-16h campaign run, 2026-07-30
last_seen: 2026-08-11
---
# A single-kernel graph capture can be a higher launch floor than the direct dispatch it replaces
- lever: When the plan for a launch-floored op is 'capture it in a graph and replay', measure the replay path against eager before building around it - for a graph containing one tiny kernel the hipGraphLaunch enqueue can cost more than a direct torch -> C++ -> hipLaunchKernel dispatch, so the collapse target becomes the new floor.
- apply: Budget the harness-visible cost of the dispatch shim as part of the lever, not as free scaffolding: when the harness times the callable between event records, any Python that selects or keys the graph sits on the measured path and can be a meaningful fraction of a call this short.
- verify: Same-session interleaved replay-vs-eager medians on every case, plus an eager-fallback-through-the-wrapper arm to separate the graph cost from the shim cost.
- caution: Also verify at higher kernel counts per capture before generalising - this measured one kernel per graph, where there is no launch batching left to amortise the replay entry over.
- source: time-budget-16h campaign run, 2026-07-30
