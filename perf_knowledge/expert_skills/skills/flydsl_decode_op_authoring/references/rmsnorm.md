# RMSNorm (plain, fused add, gated residual, per head, fp8 group-quant output)

Maturity: the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over 15 shapes; gfx950
is untested; **there is no timing against a deployed kernel**. The standalone op is faster than a
vectorised fp32 torch norm, which is a weak denominator. The design is deliberately simple and
redundant; its first optimisation is known and listed below.

## What the design covers

| covered | not covered |
|---|---|
| `[rows, hidden]` bf16 in and out, one weight of `hidden` elements; the input or the normed output may be fp32 | fp16 activations |
| a **fused add**: `x` + `residual` in, `normed` + `new_residual` out | a fused add that writes the new residual over its input and returns nothing (check what the oracle expects) |
| a **gated residual**: `alpha * stream + beta * norm(branch)`, alone or followed by the next norm of the new stream (a whole residual boundary) | gate coefficients computed at run time; a plain `stream + norm(branch)` (cut it as a norm and an add) |
| **per head**: `[rows, heads, head_dim]`, each `(row, head)` normalised over its `head_dim` (a q / k norm) | a per-head norm with a fused add or an fp8 output; `head_dim` not a multiple of 16 |
| **fp8 output**: the normed value quantised per row and group, plus an fp32 scale per row and group | int8, per-token or static quant outputs |
| weight in bf16 or fp32, multiplied as `w`, as `1 + w`, as `w + constant`, after the statistic or before it; or no weight | LayerNorm (mean, bias), activation-gated norms |
| any `hidden` that is a multiple of 16, any width | a norm whose input is still being reduced across ranks |

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

- Registers: at most `32 + 12 x rows`; add `16 + 4 x rows` for an fp8 output, `16 + 4 x rows` for a
  gated residual, and `8 x rows` more with the next norm.
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

## Measured dead ends

None were recorded for this op: no optimisation round has been spent on it. Do not read the absence of
dead ends as the design being near its floor.

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
