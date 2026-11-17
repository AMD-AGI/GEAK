# Sparse-attention decode indexer (score every candidate, select the top-k as an index set)

The indexer of a sparse-attention layer: score every cached candidate (a position, or a block of
positions) of each decode row against the row's index query, and write the top-k as an **index set**.
The attention over the selected entries is a different op: latent attention with an index list
([`mla_decode.md`](mla_decode.md)) or paged attention with a block list
([`decode_attention.md`](decode_attention.md)). Maturity: first version of the design; it passed every
correctness leg on gfx942 with FlyDSL 0.4.x over 5 shapes; gfx950 is untested. It **has** timings
against two frameworks' own indexers and is **slower than both except at the longest context**.

## What the design covers

| covered | not covered |
|---|---|
| a query `[rows, heads, dim]` in bf16 or the arch's fp8, in the same format as the cached keys | query and keys in different formats |
| score form A, **weighted ReLU sum**: per position, the sum over heads of `weight[row, head] x relu(key scale x (q_head . k))`; one selection per row | a scale applied after the head sum that cannot be folded into the weights |
| score form B, **block maximum**: per head, the maximum over a block's valid positions of `q_head . k`; one selection per head | form B with a per-token key scale, or at token granularity |
| candidates at token granularity, or blocks of 4, 8 or a power of two x 16 positions | other block sizes |
| forced candidates: the first few and / or the last few are always selected | more forced candidates than `k` |
| `k` from 1 to 4096; an optional fp32 scores output | rounded scores, more than one index output |
| a paged index-key cache: fp8 pages with per-token fp32 scales (flat, or shuffled in 16-token groups), or bf16 rows without scales | an fp8 cache without scales, per-block scales, a compressed cache |
| the current token's index key already in the cache, **or folded**: the op quantises and writes it | a fold under tensor parallelism above 1; any projection weight (those are other ops) |
| 64 to 512 threads per workgroup | more than 512 threads |

**The output convention is part of the op.** An index set is int32 logical ids (positions or block ids
within the row, not physical slots), selected entries ascending, then `-1` padding. A row with fewer
candidates than `k` lists all of them and pads. An exact tie at the k-th score goes to the lower index.

## Task decomposition and ownership

One grid-stride loop over three kinds of task. A task waits only on **lower** task indices, so any
workgroup count is correct. There is no grid-wide barrier.

1. **Score tasks** `(row, chunk)`: a chunk is 256 positions. Each candidate's score is turned into an
   **order-preserving 32-bit key** (a forced candidate gets the largest key, a candidate past the row's
   end the smallest) and stored to a device workspace; then the task posts its flag, tagged with the
   launch epoch.
2. **Filter tasks** `(row, span)`, only for rows longer than 8192 candidates: the top-k of each
   8192-candidate span, in index order, into workspace lists. This bounds what the select task stages.
3. **Select task** (one per row, or one per head for score form B): waits for its row's score (or
   filter) flags, then selects and writes the output once.

Sizes for orientation: token granularity at a 131k maximum length and one row gives 513 score tasks, 17
filter tasks and 1 select task; block granularity (128-position blocks) gives 257 + 1.

## Lane and wave layout, and why

**Scoring.**

- The waves split a chunk into segments; a 16-position group is one matrix-core chain
  `S^T = K . Q^T` (16 positions x 16 heads; the fp8 16x16x32 instruction or the bf16 16x16x16 one).
- Lane `l` loads whole 16-byte pieces of its position's key row, with the query operands in the same
  dim order, so lane `l` ends up holding head `l % 16` for four positions.
- Form A adds the heads across a 16-lane row with DPP; form B folds positions across lane quarters and
  groups with maxima.
- Why matrix cores here and not in a GEMV: there are 16 positions x up to 16 heads to contract per
  wave, and the products are exact (fp8 x fp8, bf16 x bf16), so only the fp32 summation order differs.

**Folded key write.** Wave 0 quantises the new key into LDS. The one score task whose chunk holds
position `L - 1` stores it to the cache; **every** score task scores position `L - 1` from the LDS bits,
which are the very bits the write stores. No task reads the slot being written.

**Selection, in one workgroup.**

- The flags are polled in parallel by the whole workgroup.
- A row with at most `k` candidates is copied and padded.
- Otherwise a **radix select** over the 32-bit keys: three histogram passes (11, 11 and 10 bits; 2048
  LDS bins) find the k-th key and how many keys equal to it are taken (the lowest indices).
- One **compaction pass in index order** writes every key above the threshold and the first few equal
  to it, each to the place given by a workgroup-wide prefix count. The output is ascending by
  construction; no sort.

## Synchronisation

Workspace plus epoch-tagged flags; nothing in the workspace needs clearing between calls. The select
task is the owner and has the highest index of its row. Inside the select task: workgroup barriers
around each histogram and scan pass.

## Resource budget

- Registers at 256 threads: 109 to 208 over the five shapes, no scratch. Plan for
  `192 + 16-head blocks x (2 matrix-core steps + 8)`.
- LDS about 41 KB: the 8 KB histogram, the scans, the folded key, and 32 KB for the keys of a selection
  staged in LDS. One workgroup per CU.
- Workspace: one 32-bit key per candidate per selection, one 64-bit flag per score / filter task, plus
  the filter lists; about 0.8 MB at a 131k maximum length and one row.

## Where the time goes

Kernel time relative to the framework's own indexer, which is two kernels (scores, then top-k). Above 1
means this design is faster.

| shape | 1k context | 20k | 131k |
|---|---|---|---|
| token granularity, weighted ReLU sum, fp8 keys, top 2048 | 0.71x | 0.64x | 1.06x |
| block granularity (128), block maximum, bf16 keys, top 16 | 0.88x | 0.51x | 0.79x |

Under graph replay of one call, launch included, the picture is the same: 0.6x to 0.9x, and 1.1x for
the token-granular shape at 131k.

- The scoring scales well: it is spread over hundreds of tasks.
- **The selection is the problem.** It runs in one workgroup: radix passes and a compaction, each a few
  global round trips and workgroup barriers. That is a fixed serial tail which makes up most of the
  kernel time at a 1k context, and which the frameworks' dedicated top-k kernels (one a 512-thread
  per-row top-k, one a fused Triton top-k) do not pay at small sizes.
- Both frameworks' selections, judged against this design's reference, had no violations at any
  context, so the comparison is like for like.

## Measured dead ends

None recorded: this is a first version, and no optimisation round has been spent on the selection.

## What to try next

1. A wider selection: more than one workgroup, or fewer passes (for a small `k` such as 16, a
   tournament in registers instead of a radix select).
2. For short contexts, skip the workspace round trip: when a row's candidates fit one workgroup, score
   and select in the same task.
3. Only after that, tune the scoring chunk size.

Both suggestions are untested.

## What to check

- **Stale keys in a reused workspace.** Keys of segments past a row's end were stale from an earlier
  call until keys past the selection size were masked. Reproduce by running a long context and then a
  short one on the same workspace.
- A stale read of the current token: fill the slot being folded with NaN bytes before the call; a
  kernel that scores it from the cache then shows a NaN score.
- Forced candidates are present and are not counted against a score comparison.
- The canonical order, distinctness, range, and `min(k, candidates)` entries.
- Two candidates may legitimately trade places only when their scores are within the fp32 uncertainty
  of the dot product (a near tie) or equal (an exact tie). A test that demands the exact set of a
  float64 reference without such a band will fail a correct kernel.
- The scores output of a forced place is a sentinel, not its score.
- With the fold, compare the written cache bytes exactly; undoing the write must fail.
