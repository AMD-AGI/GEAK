---
key: memory movement · gfx950 · launch-bound
type: lever
confidence: ★★
effect: 2.24x on its own and 2.35x cumulative (director-verified) on a tiny paged copy whose per-call time profiled as ~100% host dispatch; it held uniformly at every batch case (2.32-2.43x per case across a 32x range in batch size) precisely because the body was never the cost. Host enqueue 12.8 -> 4.4 us.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm7.2 / triton3.6.0 / torch2.11
last_seen: 2026-08-08
---
# Bypass the Triton dispatch path when the metric is host-launch bound
- lever: When the graded per-call time is dominated by Python/Triton dispatch - the tell is that an empty kernel through the stock path measures as slow as, or slower than, the real one - replace the graded symbol with an object whose __getitem__(grid) returns a per-(grid, signature) cached closure calling the compiled kernel's backend launcher directly, hoisting signature binding, specialization-key construction, cache lookup, option packing and the stream query out of the per-call path.
- apply: Keep the call form byte-identical to the stock one so the perf and correctness paths both still resolve it; forward tensors untouched each call (never memoise pointers), revalidate every dtype and constexpr that is part of the compile key, and fall back to the stock jit_fn[grid](*args) on any deviation, unhashable or callable grid, or runtime-API mismatch.
- verify: Time host enqueue separately from the graded metric and re-run the exact-match correctness gate through the new object; on triton 3.6 the launch hooks are an empty-but-non-None HookChain, so a wrapper gated on 'hook is not None' silently no-ops while still reporting a speedup - gate on 'not hook.calls' instead.
- caution: Also verify each follow-on shaving step against the harness floor before spending a round on it: here four further mechanisms in the same lane (raw ctypes launch, int-pointer packing, a native C++ shim, a dedicated deep round) removed up to 2.06 us of real host time for 0.000 ms of graded change, and graph capture/replay measured ~12 us of host time, slower than the closure it would have replaced.
- source: run kernel_20_geak_0808_4h 2026-08-08
