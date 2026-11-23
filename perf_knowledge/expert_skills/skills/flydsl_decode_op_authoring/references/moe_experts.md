# Routed MoE experts (gate/up product, activation, down product, weighted top-k sum)

Everything after the router: for each row and each of its `k` picks, the gate / up product on the
picked expert, the activation, the down product, the routing weight and the sum over picks. Maturity:
the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over 16 shapes (in separate runs,
never all 16 in one); gfx950 is untested, including for the weight formats that are deployed there;
**there is no timing against a deployed kernel**.

## What the design covers

| covered | not covered |
|---|---|
| hidden rows `[rows, hidden]` bf16 in, `topk_ids` int32 and `topk_weights` `[rows, k]` in, `[rows, hidden]` bf16 out | an op that also routes (see [`router_topk.md`](router_topk.md)), dispatches tokens to other ranks or sorts them by expert |
| expert weights in the 4-bit block-scaled format (MXFP4: two 4-bit values per byte, one power-of-two scale per 32 values), in the arch's fp8 with fp32 block scales, or in bf16; plain strides or the tile shuffle a library kernel wants | other 4-bit or 8-bit schemes; a scale layout specific to another GPU generation |
| optional quantisation of **both** product inputs at run time (fp8 per group, or 4-bit per 32 values) | a quantisation of only one of them; static activation scales |
| gate and up rows interleaved or concatenated in one tensor | gate and up as two tensors |
| SiLU-and-multiply and its clamped / shifted variants | GELU and other activations |
| optional fp32 biases; an optional RMSNorm before the experts; an optional per-pick norm after them; a shared-expert addend; shared experts appended as extra picks; an output scale | a per-expert norm weight; a post-expert norm under tensor parallelism |
| `hidden` a multiple of 32 and at least 64, the intermediate size a multiple of 32, `k <= min(experts, 64)` | other sizes (refuse rather than pad) |

This op has more backend-specific rounding points than any other (where the first product's output is
rounded, whether the activated gate is rounded, how the picks are summed, the activation quant). Most
of them are single bf16 roundings that a tolerance test **cannot** distinguish, so they must be read
from the backend: see `native_decode_numerics`. The two that are structural and easy to get wrong are in
"What to check" below.

## Task decomposition and ownership

- **One task is one (row, pick, slice of intermediate columns).** A slice is 32 intermediate columns, or
  one activation-quant group when the second product's input is quantised per group (a task then needs
  the whole group).
- Task count = `rows x k x intermediate / slice`. At one row, `k = 8` and an intermediate size of 512
  that is 128 tasks, so a single decode row spreads over 128 CUs.
- Task `t` belongs to workgroup `t mod n_blocks`. No task waits for another workgroup.

What a task does:

1. Stages the bf16 row of `x` in LDS. With an activation quant, one wave per group quantises it in
   place and keeps the group scales next to the row.
2. Runs the first product over the slice's gate rows and their up rows.
3. Applies the activation and (if the backend does) quantises the slice's result in LDS.
4. Multiplies the slice into the matching groups of **every** row of the down weight and adds the
   partial sums into the fp32 accumulator row of its (row, pick) in a device workspace, with atomic
   float adds.

Who finishes:

- The **last of a (row, pick)'s tasks to arrive** (an arrival counter, device-scope atomic; the data
  read with coherent loads) adds the bias, applies the routing weight, rounds, and stores the expert row
  back in place.
- The **last expert row of a row to finish** sums the `k` rows in pick order, stores the output row,
  and zeroes the accumulators and counters for the next call.

## Lane and wave layout, and why

- In the first product the lanes of a lane group **split k**: each lane reads one 32-value group per
  step. That is 16 bytes plus one scale byte for 4-bit weights, 32 bytes plus the block's fp32 scale
  for fp8, 64 bytes for bf16. The group is decoded in registers.
- Eight weight rows per lane group; a reduce-scatter leaves each row's sum in one lane with **its
  partner row (gate / up) one exchange away**, so the activation needs no LDS.
- Why 32 values per lane step: it is the scale granularity of the 4-bit format, and it is exactly one
  shuffle tile along k of the tile-shuffled layouts. A shuffled weight therefore costs one divide per
  row and nothing per group.
- The 4-bit decode is **software** on gfx942 (the arch has no 4-bit convert): the magnitude bits are
  shifted into an fp32 bit pattern. fp8 uses the arch's convert. bf16 is a shift into the top half of
  an fp32.
- The second product spreads each slice's result over all rows of the down weight, which is why its
  accumulation crosses tasks and needs the workspace.

## Synchronisation

The last-arrival pattern, twice (per pick, then per row). Nobody waits for another workgroup, so there
is no deadlock at any workgroup count and no grid-wide barrier. The price:

- the fp32 accumulators are added with atomics, so the order of the partial sums varies from run to
  run. Results are inside the tolerance but **not bit-reproducible**;
- the workspace (`4 x rows x k x (hidden + 1) + 4 x rows` bytes) is zero before the first call and left
  zero by every call. The first call per width must run eagerly, and calls that share it must not
  overlap;
- an aborted launch can leave a counter non-zero; that row then never finishes until the workspace is
  zeroed again.

## Resource budget

| weight format | registers per wave, measured | plan for |
|---|---|---|
| 4-bit block-scaled, bf16 activations | 110 to 191 | 200 |
| fp8 with activation quant (keeps the slice's 128 quantised values in registers) | 312 | 320 |
| bf16 | 166 | 192 |
| bf16 with a norm before and after the experts | 326 | add about 160 for the norm before |
| 4-bit weights and 4-bit activations, tiled weights and shuffled scales | 150 to 429, erratic, accumulator-file assisted, no scratch | 432 |

- LDS: `2 x hidden + 4 x slice + 16` bytes, plus the group scales with an activation quant.
- A pre-expert norm inside the op costs about 160 registers. If the budget is under about 330, compute
  that norm as its own op in front; it is numerically the same.
- No scratch on any measured shape.

## Where the time goes

- **The split is everything.** A first version that ran the whole op in one workgroup was **36x slower**
  than this design at one row on a 256-expert, top-8, hidden 4096, intermediate 512 shape.
- On that shape one row reads 24 MiB of expert weights. The time is bound by the **software 4-bit
  decode** on `k x intermediate / 32 = 128` CUs, not by memory bandwidth: it is far from the
  weight-byte floor.
- With an activation quant the slice is the quant group (128), so an fp8 row with a 256-wide
  intermediate spreads over only `k x intermediate / 128 = 16` CUs.
- The second product re-reads nothing but touches every down-weight row per slice.

## Measured dead ends and rejected designs

| tried or considered | result | mechanism |
|---|---|---|
| the whole op in one workgroup | measured: 36x slower | one CU streams 24 MiB; no parallelism over picks or slices |
| a pre-expert norm inside the op on the tiled 4-bit path | over a 512-register budget | 432 + 160; take the norm out |
| waiting for sibling tasks instead of last-arrival | design choice, not measured | with fewer resident workgroups than tasks a cross-workgroup wait can stall; last-arrival cannot |

## What to try next (in the order they are likely to pay)

1. A K split of the first product, so that a row uses more CUs when `k x intermediate / slice` is small
   (the fp8-with-quant case above).
2. Matrix-core dot products for the fp8 path.
3. Software pipelining of the weight loads (issue the next group's loads during this group's decode).
4. A faster 4-bit decode; on an architecture with scaled 4-bit matrix instructions this is a different
   kernel, not a tuning of this one.

## What to check

- **The weight layout.** A library kernel may have shuffled the fp8 or bf16 expert weights *in place*:
  the tensor still reports its original shape with dense strides, but the bytes are in tile order.
  Reading it through the strides alone gives garbage that still looks like random numbers (tens of
  tolerances off). Whether the weight is shuffled is a backend fact, not something the tensor shows.
- **Gate / up order**: interleaved versus concatenated is wrong by order 1 everywhere.
- The fp8 format as loaded (FNUZ on gfx942): the other format halves or doubles every weight.
- An expert id outside this rank's range (expert parallelism) is not skipped by this design.
- Inputs for which each quant group's maximum sits on a column the weights ignore: that is the case
  that makes a missing activation quant visible; on random inputs it hides inside the tolerance.
- Registers at the real width and weight format; the estimate is erratic for the tiled 4-bit path.
- Repeated runs: expect tolerance-level, not bit-level, reproducibility.
