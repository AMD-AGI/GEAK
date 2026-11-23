# RMSNorm (plain, fused add, gated residual, per head, fp8 group-quant output)

Maturity: the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over 15 shapes; gfx950
is untested; **the design itself has no timing against a deployed kernel**. It is deliberately simple
and redundant, and it is shaped for a slot inside a larger kernel. For a standalone add + RMSNorm
launch, two kernel lanes measured a different geometry against the deployed kernel: read
[Measured in a kernel lane](#measured-in-a-kernel-lane) first.

## What the design covers

| covered | not covered |
|---|---|
| `[rows, hidden]` bf16 in and out, one weight of `hidden` elements; the input or the normed output may be fp32 | fp16 activations |
| a **fused add**: `x` + `residual` in, `normed` + `new_residual` out | a fused add that writes the new residual over its input and returns nothing (check what the oracle expects) |
| a **gated residual**: `alpha * stream + beta * norm(branch)`, alone or followed by the next norm of the new stream (a whole residual boundary) | gate coefficients computed at run time; a plain `stream + norm(branch)` (cut it as a norm and an add) |
| **per head**: `[rows, heads, head_dim]`, each `(row, head)` normalised over its `head_dim` (a q / k norm) | a per-head norm with a fused add or an fp8 output; `head_dim` not a multiple of 16 |
| **fp8 output**: the normed value quantised per row and group, plus an fp32 scale per row and group | int8, per-token or static quant outputs |
| weight in bf16 or fp32, multiplied as `w`, as `1 + w`, as `w + constant`, after the statistic or before it; or no weight | LayerNorm (mean, bias), activation-gated norms |
| any `hidden` that is a multiple of 16; widths within the register bound below (rows are unrolled at build time; measured at 1 to 4 rows) | a norm whose input is still being reduced across ranks |

Where the backend rounds (the sum before the variance, the normalised value before the weight, the
value before quantisation, the maximum the scale maps to) is invisible in the op signature and differs
between backend families. Those switches belong to `native_decode_numerics`. Note that for a bf16
output a wrong rounding switch usually stays *inside* the test tolerance, so a passing test does not
prove the switch; with an fp8 output the scale output is tight and does show it.

## Task decomposition and ownership

- **One task is one output unit over all rows**: a 16-column tile (rows form), one head (per-head
  form), or one quantisation group with its scale column (fp8 form).
- Task `t` belongs to workgroup `t mod n_blocks` (grid-stride): correct for any workgroup count.
- Inside a 16-column tile, thread `T` owns column `T % 16` of rows `T // 16`, `T // 16 + threads / 16`,
  and so on.

## Lane and wave layout, and why

- **The statistic is computed redundantly by every wave of every workgroup.** Lane `L` loads 4 bf16
  values with one 8-byte load at column `4 (L + 64 k)` (a guarded tail), squares and sums them in fp32,
  and the 64 partials meet in an XOR butterfly across the wave. Per-head: the same reduction inside the
  task, over the head's `head_dim` columns.
- Why redundant: the row statistic needs the whole row, while a task owns only 16 columns. Recomputing
  it in every wave needs no LDS, no workgroup barrier and no cross-workgroup traffic, so the op is
  trivially correct for any grid. It is the cheapest correct design, not the fastest one.
- **Each thread recomputes what it stores from its own loads**, with the same operations the reduction
  used. So the value it quantises is the value the maximum saw, and the value it stores as the new
  residual is the value it normalised. No value crosses threads.
- fp8 output: after the row statistic every wave reduces the group's `max |y|` per row with the same
  vector loads and a max butterfly, so every thread holds the scale; thread 0 stores it, every thread
  keeps its reciprocal.
- Gated residual: the tile phase reads the stream next to the branch and stores the blend. With a next
  norm, a second per-wave pass over each row reduces the squares of the *stored* new stream, and the
  tile phase normalises the value it just stored.
- fp32 inputs are read 4 elements per lane with two 8-byte loads and stored 4 bytes per element.

## Synchronisation

None. No LDS, no barrier, no flags. That is the point of the redundant statistic.

## Resource budget

- Registers per wave: at most `32 + 12 x rows`; add `16 + 4 x rows` for an fp8 output, `16 + 4 x rows`
  for a gated residual, and `8 x rows` more with the next norm.
- Measured: 25 to 67 without those options; 64 for a 3-row gated shape, 48 for a 2-row residual
  boundary, 77 for a 4-row fp32-input norm. The count is not linear in the row count, because the
  unrolled per-row reductions are scheduled together.
- LDS 0, scratch 0.

## Where the time goes

- The whole input is read **once per wave per workgroup** for the statistic, then again per tile, and
  once more per group for the fp8 maximum. With up to 304 workgroups of 4 waves that is over a thousand
  reads of a row where one would do. Most of them are L2 hits, which is why the design is still usable.
- The store phase uses 16 threads per row and 2-byte (1-byte fp8) accesses; per-head and fp8 tasks loop
  over their 16-column sub-tiles.
- For scale: a naive first RMSNorm kernel (before this design) was about 8x slower than the native op,
  and a naive fused-add norm 21x to 37x slower.

## Measured in a kernel lane

Two kernel lanes each authored a FlyDSL
fused add + RMSNorm on gfx942 against the deployed HIP kernel: 2880 columns, bf16, input rows with a
stride wider than the row, 11 timed cases (1, 2, 4 and 8 rows under graph replay, seven wide cases from
400 to 16384 rows). Every figure below is *measured in kernel lanes*, one run each, one optimisation
round each: the weighted results were re-measured independently, the per-move shares are the lanes' own
same-process A/B figures. Differences under about 1.5 % between single runs are noise. Do not carry the
figures to another operator, shape or GPU generation.

**Result.** Weighted speedup over the deployed kernel: 1.30x in one lane and 1.26x
in the other. Neither used the 16-column-tile design
above; at this size a row is one or two workgroups.

**Read the oracle's case weights first.** The 4-row case carried 78 % of the weight and the four decode
cases 94 %; a 1 % cut in the 4-row case was worth about 0.77 % of the score. The seven wide cases
together carried about 6 %.

**Measure the floor first.** An empty kernel, timed the way the oracle times (graph replay after a
cache eviction), cost about 72 % of the deployed kernel's launch. Both lanes ended 7 % to 10 % above
that floor; reaching it would have added 8 % to 9 % to the score.

**What paid on the decode cases.**

| move | measured |
|---|---|
| lean arithmetic after the load: the hardware reciprocal square root instead of a precise square root plus a divide, an integer round-to-nearest-even pack without NaN checks, a DPP wave reduce instead of an LDS shuffle chain, packed FMA | 13.5 % of the launch time in one lane (vector instructions 183 to 104), about 19 % in the other (instructions 446 to 204), where matching the deployed kernel's rounding points also made decode bit-exact |
| 5 raw pointers as arguments, row strides and epsilon as build constants, so one scalar load precedes the first buffer load | inside the figures above (argument block 112 to 44 bytes) |
| one 256-thread workgroup per row with 12 elements per lane (a 16-byte and an 8-byte slot), instead of 16 with dead out-of-range work | inside the figures above |
| two workgroups per row, each reducing the full-row sum of squares and storing its own half | 2.3 % at 4 rows, 3.7 % at 8 rows; **not alias-safe**, see below |
| a device-scope cache policy on **both** decode stores | 2.7 % in one lane; 2.9 % at 4 rows and 2.2 % at 8 rows in the other |
| the residual store and the weight unpack in the shadow of the single barrier | 0.4 % to 0.6 % |

**The wide cases need their own kernel.**

- One lane wrote a second kernel for more than 256 rows (384 threads, one 16-byte chunk per thread,
  non-temporal loads and stores) and dispatched on the row count: 1.15x to 1.63x in device time, and
  1.28x, 1.32x and 1.21x on the three widest cases; the 400-row case went from 0.83x to 1.10x and one
  800-row case stayed at 0.96x.
- The other lane kept one kernel and only switched the cache policy per row-count class: 1.17x and
  1.08x at 400 and 800 rows, but 1.08x, 1.04x and 1.05x on the three widest.
- The eager (no-graph) wide cases are host-bound. Caching the validation and launch plan per layout cut
  the host path by 19 % and 44 %, and moved the two smallest eager cases from about 1.0x to 1.11x to
  1.26x.

**Dead ends** (both lanes unless noted).

| tried | result |
|---|---|
| kernarg preload | 0.5 % to 1.8 % slower |
| non-temporal loads on the decode path | 10 % to 11 % slower |
| the store policy on one of the two stores only | slower than the default policy (3 % in one lane) |
| three or more workgroups per row | 0.4 % to 2.7 % slower |
| 512 threads per row, 128-thread workgroups, other lane widths | slower (512 threads: 4 %) |
| device-scope stores on the wide cases (one lane) | 1 % to 14 % slower |
| row offsets in the buffer's scalar-offset field | invalid on gfx942: the range check includes it, so rows are dropped. It looked like a large speedup because output buffers were shared between variants; poison the outputs per variant |
| profiler kernel durations under graph replay (one lane) | about 1.7x the true time for both kernels; useless for splitting the budget |
| private benches for the 400- to 800-row cases (one lane) | disagreed with the oracle (above 1.0x privately, 0.94x to 0.96x on the oracle); rank variants with the oracle's own timing path |

**Hazard of the two-workgroup split.** Each workgroup reads the other's half while that one stores it.
With an output aliasing an input, results were wrong in up to 50 of 50 trials at 4 rows. The deployed
call is out of place, so the oracle does not see it; before any in-place use, send aliased calls to
the one-workgroup kernel.


## What to try next (in this order)

1. **Remove the redundant statistic.** Either one wave reduces a row and broadcasts the inverse RMS
   through LDS (within a workgroup) or through a small coherent workspace (across workgroups, with the
   last-arrival or owner pattern of [`flydsl_idioms_and_resources.md`](flydsl_idioms_and_resources.md#cross-workgroup-patterns)),
   or a two-phase op computes the statistic first and the tiles second.
2. **Wider stores.** Replace the 2-byte per-thread stores with vector stores over wider tiles. This
   changes the output unit, so decide it together with step 1.
3. Only then tune the thread count.

Untested suggestion: for a single-row decode norm on a small `hidden`, also consider the opposite
extreme of one workgroup, one pass, statistic in LDS. The op is small enough that launch and fixed
costs may dominate; measure before building the cross-workgroup version.

## What to check

- The in-place convention: a fused norm-and-quant backend may return the new residual in place over the
  residual input. Match what the unit test compares.
- The per-head form must be declared on the per-head view; normalising over a flattened row is a
  different statistic (off by tens of tolerances).
- The epsilon matters only for small-magnitude rows or heads; include such a case when testing.
- A zero quantisation group: its scale is the floor of the maximum, not zero.
- Register count at the largest width, not at width 1.
- The fp8 values are judged at one fp8 step, so they catch gross errors only; the scale output is what
  pins the arithmetic.
