# Absorbed multi-head latent attention (MLA) decode

One query token per row, the query absorbed through `W_UK`, attention over a paged latent cache, the
current token's latent written by the op itself, and the output either left in latent form or projected
through `W_UV`. Maturity: the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over 6
shapes, all with synthetic weights (no latent-attention checkpoint was at hand); gfx950 is untested. It
**has** timings against two deployed paths (below).

## What the design covers

| covered | not covered |
|---|---|
| inputs `q_nope [rows, heads, nope]`, `q_pe [rows, heads, rope]`, the current token's latent `kv_c [rows, latent]` and `k_pe [rows, rope]`, all bf16 | a packed query, prefill, a multi-token verify block |
| output `[rows, heads * v]` (the op applies `W_UV`) or `[rows, heads, latent]` (a following projection applies it) | an fp32 latent output |
| a paged latent cache, one latent "head" per position, the rope part at a fixed offset after the latent part; bf16 or the arch's fp8 with a scalar scale | separate latent / rope planes, a shuffled page, per-block scaled caches |
| the op writes the current token itself (**fold**) | a cache another op wrote |
| bf16 `W_UK` / `W_UV` of any strides (they are often permuted views of one projection weight) | fp8 or 4-bit batched-GEMM weights |
| a bf16 query into the attention | an fp8 query |
| contiguous positions `0 .. L-1`, **or an index list** (sparse MLA: attend the positions an int32 list names) | an index list of physical slots, a list per head |
| latent and rope dims multiples of 64; a tile of whole 16-position blocks (16 or 32; 64 with an fp8 cache) | other dims; a tile whose staged rows exceed the LDS |

As for every attention, the tile length, the split rule and the rounding of the absorbed query, the
numerator and the read-back cache element are parity facts (`native_decode_numerics`). Several of them
sit below the test tolerance when wrong, so they have to come from the backend source.

## Task decomposition and ownership

Three kinds of task in one grid-stride loop, `heads + rows x groups x chunks + heads` tasks in total. A
task only ever waits for tasks of a **lower index**, so any workgroup count is correct: one workgroup
runs them in order, and with more the lowest unfinished task is always running.

1. **Absorb** (one task per head). Computes the absorbed query of every row for its head and stores it
   in fp32, already rounded as the backend rounds it, into a device workspace; then posts the head's
   flag. Task 0 additionally writes the current tokens into the cache.
2. **Attention** (one task per row, head group and chunk). A head group is up to 16 heads (the matrix
   core's columns) and reads the latent cache **once for all of them**. A chunk is a run of whole
   backend splits; a split is never cut, because the numerator grid needs each split's sequential
   running maximum. The chunk count is the split count capped at 128. The task stores its chunk state
   (maximum, denominator, accumulator per head) to the workspace and counts itself in for its group.
3. **Finish** (one task per head). Loads its slice of `W_UV` (for one row: before the wait), waits until
   all attention tasks of its group have arrived, merges each row's chunks onto the larger maximum,
   normalises, stores the latent output or applies `W_UV`, and clears its flag and counters so the
   workspace is back to zero.

## Lane and wave layout, and why

**Absorb.** A thread owns 8 contiguous latent dims and every n-th `nope` row; its `W_UK` rows go out 16
at a time as 16-byte loads, all in flight, and the subsets' partial sums are added through LDS in a
fixed order.

**Attention.**

- The workgroup's waves **split the latent dims**: wave `w` owns `latent / waves` dims plus its share of
  the rope steps. Each wave therefore holds a smaller operand set, which is what keeps the registers
  within reach.
- The chunk's splits are walked as one flat sequence of **rounds** of `max(32, tile)` positions,
  software-pipelined: block-table entries two rounds ahead, cache words one round ahead. The first
  loads go out *before* the wait for the absorbed query.
- Per round and 16-position block, score lane `l` (position `l % 16`, quarter `l / 16`) loads its span
  with 16-byte loads, **converts it once** to bf16 operands (an fp8 element times the scale, rounded to
  bf16 by integer arithmetic) and **stages it in the wave's LDS rows** for the second product.
- Scores are bf16 matrix-core chains `S^T = K . Q^T` (two accumulation chains). The waves' partial
  scores are summed through LDS, double-buffered, with one workgroup barrier per round, so every wave
  ends up holding the same scores.
- Per tile: the running maximum (two lane exchanges), `P = exp(s - m)`, the rounded `P` as the operand
  of `O^T = V^T . P^T` read back from the staged rows, fp32 denominators per lane. The accumulator
  rescale is skipped when no column's maximum moved (the factor is exactly 1).
- **Current token:** its cache bits are formed from the inputs before the wait and kept in LDS; they
  replace the loaded words in the one round that holds position `L - 1` (a wave-uniform branch). No task
  waits for the cache store.
- **Exactness:** every matrix-core operand is exactly a bf16 value, or is split exactly into three bf16
  parts (an fp32 query, the fp32 read-back of a scaled fp8 element, an fp32 numerator). Products are
  exact; only the fp32 summation order differs from the backend's sequential walk.

**Finish.** Each wave merges a strided subset of the chunks (four chunks' loads in flight per
iteration), then the waves meet in order through LDS. With `W_UV` in the op, the waves share the
16-column groups, one transposed reduction per group and row.

**Index list.** The walk's entries replace positions: per round each score lane loads its entry's list
value and then its block-table entry, both two rounds ahead. An entry outside `[0, L)` attends nothing.
Empty states (a tile, split or chunk that saw only padding) are kept NaN-free: the factor stays 1, two
empty states combine to an empty one, an empty chunk weighs 0 in the merge.

## Synchronisation

Flags (absorb to attention), arrival counters (attention to finish), all in one workspace that is zero
before the first call and zero again after every call. Flags and counters are device-scope atomics; data
goes through coherent stores and loads and is drained before the flag or counter that announces it. A
wait is bounded and traps. Inside a task: one workgroup barrier per round.

## Resource budget

- **LDS about 35 to 52 KB**, so one workgroup per CU. A tile of 64 positions with a bf16 cache does not
  fit and is refused.
- Registers at 256 threads, by operand path: a bf16 cache read as stored 262; a converted cache 361 to
  376. In the 512-thread form only the converted single-part path fits (254); the other two were
  measured to spill there and are refused.
- An index list adds about 10 registers. One shape with a 256-wide value dim reports 506: the one-row
  finish prefetches 256 words of `W_UV` per lane, which runs past 256 into the accumulator file, with no
  scratch.
- Width and context do not enter the register count (rows, rounds, chunks and `W_UV` groups are loops).
- Workspace: about 4.3 MB at 16 heads, 128 chunks and one row.

## Where the time goes

Speed relative to deployed paths, 16 heads, fp8 cache (above 1 = this design is faster):

| compared with | result |
|---|---|
| the Triton MLA decode path, eager split rule (512 positions per split) | about 1.5x to 1.8x faster over 1k to 128k contexts (the source gives ranges, not per-context pairs) |
| aiter's prebuilt assembly MLA decode (`mla_decode_fwd`) | about 0.6x at a 1k context; on par from 8k up |
| the Triton path under a one-split rule (a full decode graph) | about 2x faster, but both are linear in the context there |
| sparse, index list of 2048: the deployed op sequence (absorb product, concatenate, attention kernel, output product) | 0.95x to 0.99x: parity, flat in the context |
| sparse: the deployed attention kernel alone | 0.72x to 0.74x; the difference is the absorb and finish phases |

- The first version walked one position at a time on one wave chain per head and was linear in the
  context: about 200x slower at a 128k context than the current design.
- The parallelism is `rows x head groups x active chunks`, and the chunks are the backend's splits.
  Under a one-split rule one task walks the whole row, and time is linear in the context again.
- A 512-position split is bound by vector arithmetic, not by matrix cores: the fp8-times-scale-to-bf16
  conversion and the exponentials, at one wave per SIMD. The three-part operand paths cost about three
  times the matrix-core work.
- **The fixed cost is about 40 % of the warm time at short contexts**: absorb, flag, attention, counter,
  finish are dependent phases with a coherent round trip each. Cold, that fixed cost is about 1.6x the
  warm one.

## Measured dead ends and limits

| tried or considered | result | mechanism |
|---|---|---|
| cutting a split across workgroups | not possible as designed | a split needs its running maximum before its `P` values; it would take a scores-first pass |
| the bf16-as-stored or three-part operand path at 512 threads | spills | measured; the per-wave operand set does not fit 256 registers (not analysed further) |
| one position at a time per head | linear in the context | serial latency chain |
| a tile of 64 positions on a bf16 cache | refused | the staged rows exceed 64 KiB of LDS |

## What to try next

- A scores-first pass, so that a long split can be cut across workgroups (this is what would lift the
  one-split case off its linear time).
- Shorten the fixed chain for short contexts: the absorb and the finish are separate phases only
  because of the head-group sharing; for one row and few heads a fused path may be cheaper (untested).
- Avoid the three-part paths by choosing the converted single-part path whenever the backend's rounding
  allows it.

## What to check

- The fold: skipping it reads stale cache bytes and is off by tens to hundreds of tolerances; undoing
  the cache write after the call must fail the test.
- `W_UK` / `W_UV` strides come from the tensors the backend holds; the other layout's strides on the
  same storage give plausible garbage.
- A cache scale other than 1 on an fp8 cache.
- An index list with padding inside the walk and a split made only of padding (the NaN-free rules).
- The split count for a sparse row is taken on `min(L, list length)`, not on `L`.
- Workgroup counts 1 and 7: every wait is on a lower task index, so both must pass.
- The first call per width runs eagerly (it allocates and zeroes the workspace).
