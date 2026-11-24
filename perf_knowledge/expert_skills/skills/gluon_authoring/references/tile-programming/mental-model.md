# Tile Programming Mental Model

Arch convention for every `tile-programming/` page: gfx950 (CDNA4) first, gfx942 (CDNA3) as "gfx942
downgrade" notes; API spellings are upstream Triton 3.8.0. Dataflow names used across pages: **AC** async
global→LDS (`gl.amd.cdna4.async_copy.*`), **GR** global→reg, **LW** reg→LDS, **LR** LDS→reg (`ds_read`,
`ds_read_b64_tr`), **DOT** (`gl.amd.cdna4.mfma` / `mfma_scaled`). On Gluon the overlap is authored
(`pipeline.md ## Where the overlap comes from, and it is not the same question per tier`); `num_stages` has no
effect on the Gluon path (3.6.0–3.8.0 compiler source).

## Reduction & parallelization structure (decide before tiles)

Enumerate the op's reductions: one on the parallel axis is clean; one on another axis pays recompute, atomic,
materialize-and-reduce, or **defuse** (split the fused kernel per output axis, e.g. attention bwd → dkdv + dq).
Atomics cost write volume, which scheduling does not reduce — the levers are fewer contributors or a narrower
atomic dtype (packed bf16 atomics tied at decode-scale shapes and paid at large M only with relaxed, cleanly
packed atomics, gfx950 / 3.8.0; A/B it;
`memory-path.md ## Output / reduction path (atomics)`).

## Latency vs throughput (the key distinction)

Latency is hidden by prefetch, throughput by interleave. A lever that removes exposed serial-chain latency
pays most at low occupancy; at 2+ resident waves the same change was often neutral-to-negative, though taking a
dependency off the MFMA critical chain has also paid on a 2-wave prefill loop — waves/CU is worth reading
before attributing a gain or closing the direction.

## Porting a technique across architectures

gfx950/gfx942 are occupancy-first (static per-wave registers, no warp specialization): techniques from
asynchrony-first targets that spend registers on pipeline state (producer/consumer warps, register-buffered
multi-stage pipelines, dynamic register handoff) tend to remove the waves doing the hiding — they regressed in
most recorded cases (one critical-path-bound attention shape won), so a floor probe is a cheap first read. What
has ported: direct-to-LDS DMA, the scale fold, operand prefetch. gfx1250 / RDNA: out of scope.

## The hand-asm ceiling

When a production hand-asm kernel exists (aiter / CK), benchmarking it on the same boundary gives a practical
ceiling reference.
Some of its wins are not compiler-expressible: a free-running `s_setprio`-only two-group offset relies on
cycle-counted buffer disjointness, while with runtime buffer indices (`k % nBuffers`) a correct compiler must
keep a cross-wave LDS fence that re-syncs the groups every iteration. Such a gap is a scoped abstraction
ceiling rather than an open layer.
