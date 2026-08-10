---
key: moe grouped gemm · gfx950 · mixed
type: lever
confidence: ★★
effect: director-verified 1.49x geomean end state, per-case 1.38x / 1.57x / 1.54x from the smallest to the largest token count; this axis contributed ~+11% cumulative in two steps — the live-prefix renumber paid +7.7% at the small token count, where the weight stream IS the cost, and +4.9% / +4.3% on the large ones; then halving the chunk count paid a further +2.9% on every case with an essentially byte-identical code object.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm7.2 / torch2.11.0 / hip (CK C++ templates, JIT-compiled)
last_seen: 2026-08-08
---
# Renumber workgroups for weight-slice contiguity, live prefix only
- lever: When consecutive tiles of one group/expert all stream the same multi-MB weight slice, the natural round-robin of the flat workgroup id scatters those tiles across every L2, so each cache holds a fraction of the available reuse; renumber the id so consecutive workgroups form long contiguous runs over one weight slice. The currency is the LENGTH of the contiguous run rather than affinity to any particular cache, so once the renumber is in, also sweep the chunk count downward (fewer, longer chunks).
- apply: Write it as a pure index bijection in the prologue with compile-time-foldable divisors, leaving the loop body, the accumulation order and the output bytes identical so that only placement changes; and restrict the bijection to the LIVE prefix of the grid, derived from the already-loaded token or tile count, so ids beyond it keep their natural value and fall into the existing early exit.
- verify: Gate on correctness before believing any timing — a mapping that is not a bijection silently skips tiles and measures FASTER (fast-and-wrong is the signature: one such attempt read as +3% at an error ratio of 0.22). Then diff the inner-loop ISA, where instruction, memory-class and register counts should all be unchanged, so the entire delta is placement. A real locality win also tends to be the lowest-variance build, since the jitter source is cache contention.
- caution: Also verify the grid's dead tail: applied over the FULL grid the same renumber packs a padded grid's dead workgroups into whole chunks and idles them — it flipped the small-token case from -7.7% to +9.2% while still delivering the large-case wins, so tiles that are free under the default round-robin are not free under any locality remap. Also verify chunk counts at powers of two before non-power-of-2 ones (one non-power-of-2 count measured 46% slower than its neighbour), and price ordering ideas with a bit-exact permutation of the live tiles first: that ~90 s screen showed a random scatter costs +90-94% and a de-interleave +21-33%, i.e. the default ordering already banks most of the prize.
- source: run kernel_20_geak_0808_4h 2026-08-08
