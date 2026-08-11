---
key: collapsing the Python/Triton dispatch path of a tiny paged-KV copy kernel on gfx950, graded per call by an event-pair harness across a 32x batch-size range
type: lever
confidence: ★★
effect: 2.24x on its own and 2.35x cumulative (director-verified) on a tiny paged copy whose per-call time profiled as ~100% host dispatch, and it held uniformly at every batch case (2.32-2.43x per case across a 32x range in batch size) precisely because the body was never the cost; host enqueue itself fell ~2.9x
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm7.2 / triton3.6.0 / torch2.11
last_seen: 2026-08-08
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
- pitfall: a wrapper gated on 'hook is not None' reported a speedup while silently no-opping -> on triton 3.6 the launch hooks are an empty-but-non-None HookChain -> gate on 'not hook.calls' instead.
- verify: Time host enqueue separately from the graded metric and re-run the exact-match correctness gate through the new object.
- caution: Also verify each follow-on shaving step against the harness floor before spending a round on it - here four further mechanisms in the same lane (raw ctypes launch, int-pointer packing, a native C++ shim, a dedicated deep round) removed nearly half of the remaining host enqueue time for zero graded change, and graph capture/replay cost roughly 2.7x the host time of the closure it would have replaced.
- source: run kernel_20_geak_0808_4h 2026-08-08
