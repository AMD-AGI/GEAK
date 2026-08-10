---
key: attention · gfx950 · decode
type: method
confidence: ★★
effect: An instrument, not a speedup: gutting a small dispatch's whole body to a trivial copy (deliberately wrong output) measured 4.00 us min vs 4.04 us stock — zero work was not one nanosecond faster, so that dispatch was 100% workgroup-dispatch floor. The prediction held: really optimizing that same body then measured 0% per-kernel and 0.9-5.2% WORSE wall on all three decode cases, so one ~40 s measurement stood in for a whole direction.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.2.3 / torch 2.11.0 / hip (AOT hipcc, template-codegen op)
last_seen: 2026-08-08
---
# Null the kernel body before optimizing it
- lever: Before spending a direction on a kernel body, build a deliberately-incorrect null version of it (write a trivial value and return) and measure that as the lane's lower bound. If the null is not faster, the body is not the cost, and the entire lane is priced at zero in one measurement.
- apply: Keep launch geometry, arguments and dispatch count identical and gut only the body, so the difference isolates body work from the dispatch/occupancy floor. The same idiom sizes a fusion or coherence lever: a fence-free (incorrect) build is the zero-coherence-cost ceiling for any cross-workgroup reduction, and if that ceiling is ~1.0x no correct variant can beat it.
- verify: Confirm the null really ran wrong (the correctness metric collapses) — a null that still passes numerics did not replace the body. Read per-dispatch device time round-robin interleaved against stock rather than sequentially; single-shot per-dispatch readings carried +-3.5% spread here and reversed three verdicts.
- caution: Also verify the case is not host-bound before turning any device-time saving into an expected wall gain: here the two dispatches summed to ~30 us of device time against a 44 us wall on the largest decode case, so even a free dispatch would have bought ~0 wall.
- source: run kernel_20_geak_0808_4h 2026-08-08
