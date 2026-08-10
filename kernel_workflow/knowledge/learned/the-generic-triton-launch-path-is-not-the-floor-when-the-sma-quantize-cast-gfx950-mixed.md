---
key: quantize / cast · gfx950 · mixed
type: lever
confidence: ★★
effect: Director-verified per-case: 2.77x on the small launch-bound case vs 4.76x and 4.00x on the large memory-bound cases (3.75x geomean). The launch-path work is what moved the small case - in-run it went 1.15x -> 2.76x while the two large cases did not move at all, so expect ~1.0x wherever the kernel is already bandwidth-bound. Host CPU 13.4 -> 4.45 us/call against a ~5 us kernel.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.2 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
---
# The generic Triton launch path is not the floor when the small case is launch-bound
- lever: When the small-shape case is launch-bound (host us/call at or above kernel us/call), the generic Python launch path is skippable rather than a floor: the compilation key is constant per shape class, so memoize the CompiledKernel and call its low-level entry directly instead of walking the JIT wrapper's specialization each call.
- apply: Cache (CompiledKernel.run, function, packed_metadata) plus a pre-built positional arg tuple on a hand-made shape key (dtypes, contiguity, constexpr values); pass cached integer data_ptr()s instead of tensors, which skips a data_ptr() call and a driver pointer query per pointer per launch; declare any new constexpr knob as a module-level global instead of threading it into the hand-built tuple; and keep a degrade ladder that peels one tier per compile failure and clears used_global_vals, or a stripped hint can never be retried.
- verify: Measure CPU us/call with an in-process no-sync probe and compare it against the kernel's own device time - the win exists only while host exceeds device - and confirm every call still dispatches, because a launcher that quietly stops launching reads as an enormous speedup.
- caution: Also verify the alternating-caller anti-pattern before shipping any last-call fast path: arming on FIRST sighting made a caller alternating between two buffer sets 1.35x slower, while requiring the key to repeat before arming restored parity. Also verify how much room is left underneath - here the residual was a ~2.3 us driver launch call, and a ctypes bypass came out 0.64 us slower than the stock C launcher while doing less work.
- source: run kernel_20_geak_0808_4h 2026-08-08
