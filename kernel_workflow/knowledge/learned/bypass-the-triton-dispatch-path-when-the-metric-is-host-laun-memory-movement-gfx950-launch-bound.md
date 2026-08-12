---
key: collapsing the Python/Triton dispatch path of a tiny paged-KV copy kernel on gfx950, graded per call by an event-pair harness across a 32x batch-size range
type: lever
confidence: ★★
effect: 2.24x on its own and 2.35x cumulative (director-verified) on a tiny paged copy whose per-call time profiled as ~100% host dispatch, and it held uniformly at every batch case (2.32-2.43x per case across a 32x range in batch size) precisely because the body was never the cost; host enqueue itself fell ~2.9x
confirms_cited: 2
confirms_blind: 1
losses: 1
attempts: 12
toolchain: rocm7.2 / triton3.6.0 / torch2.11
last_seen: 2026-08-12
name: bypass-the-triton-dispatch-path-when-the-metric-is-host-laun-memory-movement-gfx950-launch-bound
description: Replace the graded symbol with a cached direct-launcher closure when the per-call time is host dispatch: 2.24x alone, 2.35x cumulative, uniform across batch
keywords: ['launch-overhead', 'dispatch-floor', 'launch-bound', 'measurement-method', 'control-experiment', 'graph-capture', 'memory-movement', 'interleaved-ab']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: memory_movement
regime: launch-bound
lifecycle: active
---
# Bypass the Triton dispatch path when the metric is host-launch bound
- lever: When the graded per-call time is dominated by Python/Triton dispatch - the tell is that an empty kernel through the stock path measures as slow as, or slower than, the real one - replace the graded symbol with an object whose __getitem__(grid) returns a per-(grid, signature) cached closure calling the compiled kernel's backend launcher directly, hoisting signature binding, specialization-key construction, cache lookup, option packing and the stream query out of the per-call path.
- apply: Keep the call form byte-identical to the stock one so the perf and correctness paths both still resolve it; forward tensors untouched each call (never memoise pointers), revalidate every dtype and constexpr that is part of the compile key, and fall back to the stock jit_fn[grid](*args) on any deviation, unhashable or callable grid, or runtime-API mismatch.
- stack: total 2.49x isolated (director-verified) on a re-run of the same op = two directions compounded
  - 1. host.launch-overhead - the dispatch-path bypass, ~2.34x standalone (round 1, verified) - the bulk of the win
  - 2. algo.chain-shortening - replacing an O(program-id) chain of DEPENDENT scalar loads with a masked block load + a block sum, +3.1% geomean on top of (1) (round 2, verified) - it looks small on the geomean but it is the entire batch-slope: the largest-minus-smallest case gap fell ~4x and device time went flat in batch
  - note: attribution is incremental in landing order; (2) was never isolated against the stock path, and its device-side saving was discounted ~3x because it had been hiding behind the dispatch it overlapped
- pitfall: the device work a chain-shortening removes can be worth several times less on the graded metric than in device time -> under a launch floor the two overlap -> grade a scaling-term direction on the SCALING curve (does the case-to-case gap collapse?), not only on the geomean
- pitfall: a wrapper gated on 'hook is not None' reported a speedup while silently no-opping -> on triton 3.6 the launch hooks are an empty-but-non-None HookChain -> gate on 'not hook.calls' instead.
- verify: Time host enqueue separately from the graded metric and re-run the exact-match correctness gate through the new object.
- caution: Also verify each follow-on shaving step against the harness floor before spending a round on it - here four further mechanisms in the same lane (raw ctypes launch, int-pointer packing, a native C++ shim, a dedicated deep round) removed nearly half of the remaining host enqueue time for zero graded change, and graph capture/replay cost roughly 2.7x the host time of the closure it would have replaced.
- source: run kernel_20_geak_0808_4h 2026-08-08
- source: run kernel_20_geak_0811_2h, 3-round re-run on the same op and box, 2026-08-12
