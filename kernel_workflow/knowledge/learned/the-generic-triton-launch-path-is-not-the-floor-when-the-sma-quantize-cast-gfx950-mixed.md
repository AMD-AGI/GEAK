---
key: per-token quantize/cast on gfx950 driven through the generic Triton Python launch path, where the small shape is host-launch-bound and the large ones are bandwidth-bound
type: lever
confidence: ★★
effect: Director-verified per-case: 2.77x on the small launch-bound case vs 4.76x and 4.00x on the two large memory-bound cases (3.75x geomean). The launch-path work is what moved the small case - in-run it went 1.15x -> 2.76x while the two large cases did not move at all, so expect ~1.0x wherever the kernel is already bandwidth-bound. Host CPU per call fell 3.0x, from ~2.7x the kernel's own device time down to ~0.9x it.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm 7.2 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: the-generic-triton-launch-path-is-not-the-floor-when-the-sma-quantize-cast-gfx950-mixed
description: Memoize the compiled kernel and call its low-level entry directly on a launch-bound quant/cast shape: 2.77x there, ~1.0x once the case is bandwidth-bound
keywords: ['launch-overhead', 'dispatch-floor', 'measurement-method', 'launch-bound', 'kernel-cache', 'quant']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: mixed
lifecycle: active
---
# The generic Triton launch path is not the floor when the small case is launch-bound
- lever: When the small-shape case is launch-bound (host time per call at or above kernel time per call), the generic Python launch path is skippable rather than a floor: the compilation key is constant per shape class, so memoize the CompiledKernel and call its low-level entry directly instead of walking the JIT wrapper's specialization each call.
- apply: Cache (CompiledKernel.run, function, packed_metadata) plus a pre-built positional arg tuple on a hand-made shape key (dtypes, contiguity, constexpr values); pass cached integer data_ptr()s instead of tensors, which skips a data_ptr() call and a driver pointer query per pointer per launch; declare any new constexpr knob as a module-level global instead of threading it into the hand-built tuple; and keep a degrade ladder that peels one tier per compile failure and clears used_global_vals, or a stripped hint can never be retried.
- verify: Measure host time per call with an in-process no-sync probe and compare it against the kernel's own device time - the win exists only while host exceeds device - and confirm every call still dispatches, because a launcher that quietly stops launching reads as an enormous speedup.
- pitfall: a caller alternating between two buffer sets ran 1.35x slower once the last-call fast path was armed -> the fast path armed on the FIRST sighting of a key, so the alternation re-armed and mispredicted every call -> arm only once the same key repeats, which restored parity (caught before shipping, so also verify the alternating-caller case before shipping any last-call fast path).
- caution: Also verify how much room is left underneath before funding more host work - here the residual was a single driver launch call worth roughly half the kernel's own device time, and a ctypes bypass replacing that call came out slower rather than faster, by roughly a quarter of that residual call, while doing less work.
- source: run kernel_20_geak_0808_4h 2026-08-08
