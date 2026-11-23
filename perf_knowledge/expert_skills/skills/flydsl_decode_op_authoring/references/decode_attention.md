# Decode attention over a paged KV cache (GQA / MHA, sliding window, sinks, split-K, block-sparse)

Single-token decode attention, optionally with the current token's KV write folded in and the model's
output gate applied. Maturity: the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over
18 shapes; gfx950 and tensor parallelism above 1 are untested. It **has** timings against the deployed
kernel: ahead on long contexts, behind on short ones and on the block-sparse variant (table below).

## What the design covers

| covered | not covered |
|---|---|
| one query token per decode row | more than one query token per row (speculative / multi-token verify) |
| a paged KV cache in one allocation addressed through a block table, any layout expressible as element strides; also K and V in separate allocations and a token-addressed table (one cache row per token) | a shuffled or interleaved page layout; int8, 4-bit or per-token-scaled caches |
| KV in the arch's fp8 or in bf16; GQA and MHA | latent attention (see [`mla_decode.md`](mla_decode.md)) |
| q / k and v head dims that are multiples of 64 | other head dims (refuse rather than approximate) |
| a sliding window, attention sinks, split-K with per-segment maxima, an fp8-rounded query and numerator | softcap, ALiBi, non-causal or chunked windows |
| **fold**: the op also receives the current token's k / v and writes them to the cache | a model constant applied to k / v inside the op; a V scale on the output |
| an output gate (`attention output x sigmoid(gate)`, optionally behind a weightless per-head RMS norm) | other gate activations |
| a **block-sparse selection**: attend only the blocks an index list names | a position-granular selection (that is the MLA index-list form) |

## The softmax is the backend's, and that shapes the code

An exact softmax is more accurate than the deployed kernel and still fails a parity tolerance calibrated
against it. The tile length, the segment count and the rounding of query and numerator are parity
facts; take them from `native_decode_numerics`. What matters structurally:

- Tiles and segments are laid out over **absolute positions** `0 .. L-1`. A window only masks earlier
  positions; it does not restart the tiling.
- **Each segment keeps its own running maximum**, denominator and accumulator. A sink belongs to
  segment 0 only.
- **An empty segment must drop out.** Combining segments one by one from minus infinity forms
  `exp(-inf - -inf)` and returns NaN for a window that starts past segment 0. Accumulate every tile in
  the frame of the running maximum over all tiles seen instead.
- **A masked position is not addressed.** The block-table entry of a tile's tail past `L - 1` may not
  exist. Clamp the *addressed* position into `[first kept, L - 1]`; an unclamped version faulted.
- **Fold:** the last position is the current token, taken from registers and rounded exactly as the
  cache stores it. Reading the slot back from the cache is a race with the task that writes it.

## Task decomposition and ownership

1. **Base: one task is one kv head's query heads over all rows** (up to 16 heads, the matrix core's
   column count; a larger group is cut into tasks of its largest divisor up to 16). The kv head's K / V
   are read once for all of them. The task writes all its heads' outputs once, after all rows.
2. **Split over tasks** (full context, more than one segment): a task is `(kv head, head chunk, split)`.
   The backend's segments are dealt to the split tasks as whole segments on absolute positions. Each
   non-owner task stores, per row and query head, its unnormalised accumulator, frame maximum and
   denominator in fp32 to a workspace and posts a flag. The **owner is the highest task index of its
   set**, waits for the other flags, merges the records in split order (largest frame maximum, every
   part rescaled to it, one division) and stores. Only the owner writes outputs.
3. **Narrower head groups**: when a split op still has fewer than 64 tasks, cut the head groups until it
   reaches 64. Each group re-reads its kv head's K / V; L2 absorbs that.

Defaults that were measured to be right: split = segment count for a full-context op that has fewer than
64 tasks otherwise, capped at 256 tasks; **no split for a windowed op** (its tiles sit in the last one
or two segments, so a split only adds a merge). With fold, only the task holding position `L - 1` writes
the cache; the others clamp their addresses into their own tiles. Task `t` belongs to workgroup
`t mod n_blocks`, and an owner waits only for lower indices, so any workgroup count is deadlock-free.

## Lane and wave layout, and why

- **All waves of the workgroup compute.** A round deals out tiles: wave `w` takes `max(tile, 64)`
  consecutive positions, as blocks of 16 positions.
- **Scores are one matrix-core chain per 16-position block**, `S^T = K . Q^T` (16 positions x 16
  query-head columns). Lane `l` loads, for position `l % 16`, a quarter of its K row with 16-byte loads,
  and holds the query operand of head `l % 16` for the whole row. The result leaves lane `l` with head
  `l % 16`'s scores for four positions of every block, **kept in registers for the second pass**, so K
  is read once per tile.
- Operand format: the fp8 16x16x32 instruction when the query is rounded to the cache's fp8, otherwise
  the bf16 16x16x16 instruction on exactly widened values. Every product is exact; only the fp32
  summation order differs from the backend.
- **The block-table entry of a page is read once per wave chunk** (by its first position, then
  permuted to the other lanes), one round ahead; the first round's entries go out right after the
  sequence lengths.
- A tile's per-head maximum is one in-lane max and two cross-lane steps. The round's tile maxima meet
  in LDS (**one workgroup barrier per round**), and each tile is normalised by its own segment's running
  maximum through that tile, which is exactly the sequential kernel's value.
- **`P . V` is a second chain per tile**, `O^T = V^T . P^T`. The probabilities are already in the operand
  lanes, so there is no transpose through LDS; lane `l` loads `v_dim / 16` contiguous v dims of its four
  positions. An unrounded fp32 probability is split exactly into three bf16 parts (three instructions).
- At the end of a row the waves' sums meet in LDS in a fixed wave order, and wave `w` finishes heads
  `w, w + waves, ...`.
- **Above 256 threads** a wave has at most 256 registers, so it takes 32 positions per round and loads
  V only after the K operands are consumed.

## Synchronisation

One workgroup barrier per round (the tile maxima), one LDS combine per row, and with a split the flag /
workspace / owner protocol. No task waits for the cache write.

## Resource budget

- Registers: **172 to 339 at 256 threads**, because every K and V load of a round is in flight before
  the matrix-core work; that means one wave per SIMD. In the 512-thread form 114 to 237, no scratch.
- LDS 1 to 33 KB (queries, tile maxima, end-of-row sums); a split adds the frame maxima.
- Split workspace: `rows x heads x (split - 1) x (v_dim + 4) + 2 x tasks` 32-bit words, about 124 KB for
  16 heads on 16 segments at one row.
- Block-sparse with 128-position blocks: about 486 registers at 256 threads (fits one wave per SIMD) and
  about 456 in the 512-thread form, which **spills**: 2.4x slower there. Use at most 256 threads.

## Where the time goes

Speed relative to the deployed kernel (above 1 = this design is faster). Full-context shape: a
decoder layer at 4 ranks, 1 kv head and 16 query heads per rank, fp8 cache with fold, compared
against vLLM's Triton unified attention including its cache write.

| design step | 1k context | 20k | 131k |
|---|---|---|---|
| serial: one wave, a position at a time | about 0.005x overall | | |
| lane = position, matrix cores, all waves, K / V shared by the group | about 0.22x overall | | |
| split over 16 tasks | 0.67x | 1.13x | 1.48x |
| split, head groups of 4 (64 tasks) | 0.83x | 1.34x | 1.85x |

- Windowed attention (window 128, sinks, 2 kv heads): 1.06x; the serial version was 0.31x.
- Block-sparse (16 query heads on 1 kv head, 16 selected 128-position blocks, fold): **0.47x** at 256
  threads and 64 tasks, 0.60x with 64-thread workgroups and one head per task. The deployed side is two
  kernels (attend, merge) whose KV write happens in an earlier kernel and is not counted; this design's
  time includes the fold.
- The serial version's time was proportional to the window, flat in the context, and unchanged by the
  segment count or the thread count (under 2 %): a per-position latency chain on one wave, not
  arithmetic and not empty segments.
- What is left is latency: the dependent cold chain sequence length, block table, K / V pages; one
  barrier per round; the end-of-row combine; the owner's wait and merge. At a 1k context a segment is
  2 tiles, so a task uses 2 of its 8 waves.
- A layout that stores V dim-major is loaded element by element and is the slowest layout, about 2x to
  4.5x the time of the contiguous layouts at the same scale.

## Measured dead ends and non-levers

| tried or observed | result | mechanism |
|---|---|---|
| changing segment count or thread count on the serial design | under 2 % | the bound was the per-position chain |
| a split on a windowed op | no gain | its tiles sit in one or two segments; the split only adds a merge |
| 512-thread workgroups for 128-position selected blocks | 2.4x slower | the capped round spills |
| more waves for the block-sparse form | three of four waves idle | one selected block is one wave's round |
| an exact softmax | fails parity | the backend's rounding grid is part of the result |

## What to try next

- Short contexts: more, narrower tasks (head groups), since a split segment is a short latency chain.
- Block-sparse: workgroups of at most 256 threads with one head per task; then shorten the chain
  sequence length, selection entry, block table, K / V, which is what its time is made of.
- Few kv heads per rank (a windowed op with 2 kv heads runs on 2 CUs): its time equals the dependent
  chain plus launch, the same chain the deployed kernels pay; little is left inside the kernel.

## What to check

- Fresh sequences and short contexts, where no averaging hides a rounding or masking mistake; a
  tolerance measured only on long captured steps does not cover them.
- A context past window + one segment without sinks (the NaN case above).
- A block table that is neither the identity nor in order, with poisoned entries past each row's last
  block, so an unclamped address faults in test rather than in serving.
- With fold: compare the whole cache after the call; undoing the write must fail the test.
- A sink tensor is read where and as the model keeps it (often bf16 under a model-specific name);
  running without it because a default name found nothing is a silent wrong answer.
- Block-sparse: only the first `min(list length, blocks in the row)` entries are attended, a padding
  entry is a neutral update (not block 0), and the last selected block is partial.
- Registers at both 256 and 512 threads if the thread count is not yours to choose.
