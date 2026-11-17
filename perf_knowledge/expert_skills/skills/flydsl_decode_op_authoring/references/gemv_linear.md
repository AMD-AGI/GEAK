# Decode GEMV / linear projection

`y[rows, N] = x[rows, K] . W[N, K]^T (+ bias)` at 1 to 8 rows, with a bf16 weight or an fp8 block-scaled
weight. Maturity: the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over 27 shapes;
gfx950 is untested; **there is no timing against a deployed kernel** (only internal A/B ratios below).

## What the design covers

| covered | not covered |
|---|---|
| one `[rows, K]` bf16 input, `K` and `N` multiples of 16 | two inputs, a fused residual |
| one `[rows, N]` output; or up to 4 outputs that cut `N` contiguously (a fused QKV projection cut at its parts, a latent-attention projection that yields `q` and `kv` together); or outputs that read the weight rows through a **row map** (a packed projection de-interleaved by offsets or per-head runs) | an output whose width is not a multiple of 16 unless it is given as `[rows, heads, dim]` |
| an **activation epilogue** on a merged gate / up weight `[2I, K]`: SiLU-and-multiply, tanh-GELU-and-multiply, the clamped variants; gate and up rows concatenated or interleaved | the erf GELU; gate and up as two separate weight tensors (refuse it, or cut it as two projections plus an activation op) |
| bf16 weight, row-major, rows possibly padded | a transposed `[K, N]` weight; fp16, int8, int4, 4-bit block-scaled weights |
| fp8 weight in the arch's own format with an fp32 block scale, row- or column-major | per-tensor / per-channel fp8 scales; an fp8 format the arch does not convert natively |
| the backend's dynamic per-group activation quantisation to fp8 ahead of the product, or none | a static activation scale |
| a weight stored tile-shuffled for a library matrix-core GEMM (only the addressing changes) | a shuffle without stated geometry |
| a column-parallel shard, or a row-parallel shard whose partial is reduced elsewhere | a cross-workgroup reduction for tensor parallelism inside this op |

Which rounding points and quantisation steps the backend has (activation quant, bias before or after
rounding, rounding of gate / up before the activation) are parity facts: take them from
`native_decode_numerics`. They change the output, not the structure below.

## Task decomposition and ownership

- **One task is one output tile over all rows**: 16 columns of a `[rows, n]` output, or one head of a
  `[rows, heads, dim]` output. Tasks are numbered output by output, so no task straddles two outputs.
- Task `t` belongs to workgroup `t mod n_blocks` (grid-stride): correct for any workgroup count.
- Task count = `N / 16`. `N = 4096` gives 256 tasks, close to the 304 CUs.
- **A head-shaped output costs parallelism.** The same projection cut into head outputs gives one task
  per head: in one measured layout 18 workgroups did the work that 212 did when the output was one
  packed `[rows, N]` row. Prefer a packed output when the consumer can take it.
- With an epilogue, a wave's output column owns its gate row and its up row; both accumulate in the
  same K loop.

## Lane and wave layout, and why

- Each wave of the workgroup owns `16 / (threads / 64)` output columns of the tile (4 columns per wave
  at 256 threads).
- **The 64 lanes stride over K.** Lane `L` takes 8 consecutive elements at `512 c + 8 L` (a guarded
  tail when `K` is not a multiple of 512). A weight row is therefore read exactly once, with 8-byte
  loads (8 fp8 values, or 4 + 4 bf16), and each 8-element activation chunk is loaded once per wave and
  reused for all of the wave's columns and rows.
- Why lanes along K: at 1 to 4 rows a GEMV is bound by the weight read, so the lanes should issue
  contiguous, coalesced loads along the weight row. Plain FMAs keep up with that stream.
- Why **no matrix cores**: a matrix-core tile needs 16 rows of work per wave and a preshuffled weight;
  at decode widths neither is there. A matrix-core GEMV tried in the router was about 2x slower at one
  row ([`router_topk.md`](router_topk.md)). Library matrix-core GEMMs start to pay at larger widths.
- With an activation quant, the `group / 8` adjacent lanes holding one group find its absolute maximum
  with an XOR butterfly across those lanes, form the scale, and quantise. A block's per-lane partial
  dot is multiplied by (activation scale x weight block scale) before it is accumulated.
- Epilogue: one more butterfly sums the 64 lanes; lane 0 adds the bias, rounds and stores. With an
  activation epilogue lane 0 holds both sums, applies the activation in fp32 (sigmoid through `exp2`)
  and stores once.
- **No LDS, no workgroup barrier, no cross-workgroup traffic** in the base design.

## Synchronisation

None in the base design. The optional **K split** (below) adds one: every segment task stores its fp32
partials to a small zeroed workspace and arrives on the tile's counter; the last arrival adds the
segment partials in segment order `0, 1, ...`, adds the bias, applies the epilogue, stores, and resets
the counter. Because the order is fixed, the result is bit-identical across workgroup counts (checked
at 1, 7 and 64 workgroups over 20 repeats). No atomic float add is used.

## Resource budget

- Registers: an upper envelope of `72 + 4 x rows x columns_per_wave` (doubled with an epilogue).
  Measured 77 to 112 at 256 threads without an epilogue; the activation quant and the bias cost more
  than the accumulators, so treat the formula as a ceiling, not a model.
- 16-element lane steps add about 16 per (column x row): measured 108 at 4 columns, 256 with an
  epilogue, which is the cap of a 512-thread workgroup.
- LDS 0, scratch 0.

## Where the time goes

The kernel is **latency-bound per K step**: one dependent memory round trip per loop trip. That is why
the two performance parameters below work, and why changing the wave count does not.

| variant | measured (256 threads, width 1 unless noted) |
|---|---|
| 16 elements per lane per step instead of 8 (half the K steps) | 1.29x, 1.21x and 1.18x faster on three fp8 `K = 4096` shapes (QKV-like, down-projection, gate/up with SiLU); no gain on a bf16 `K = 2048` shape |
| K split by 4 on a long-K, few-tile shape (`K` 7168, `N` 2048, 2 rows, 128 tiles) | 1.25x faster |
| K split on a short-K small shape (`K` 2048, `N` 512) | 0.93x (slower) |
| K split when the tile count is already about the CU count (232 to 256 tiles) | slower: each workgroup then runs two tasks back to back |

Both variants only reorder fp32 additions of exact products, so they do not move the result outside a
tolerance that already allows an fp32 summation order.

Selection rule: candidates are the default, 16-element steps, and a split of 2 / 4 / 8 while
`tiles x split <= 304`, each also with 16-element steps. A fitted time model ranked these candidates
with a median error of about 12 % (worst case 42 % on a small 32-tile shape), which is good enough to
order them and not good enough to skip measuring.

## Measured dead ends

| tried | result | mechanism |
|---|---|---|
| more waves per workgroup (512 or 1024 threads) | slower | consistent with the bound being the dependent round trip per K step, which more waves do not shorten; not isolated further |
| fewer waves (64 or 128 threads) | slower | result recorded only; 256 threads was the best of the five sizes tried |
| unrolling the K step with the loads hoisted | about 14 % better for 8-element steps, but 16-element steps alone are better, and both together spill into the accumulator file | register pressure |
| K split on shapes with tiles close to the CU count | slower | two tasks per workgroup, plus the merge |
| matrix cores at 1 to 4 rows | not adopted | weight-read bound; needs a preshuffled weight |

## What to try next

- The block scale is loaded per lane per step although all lanes of a group share one value: broadcast
  it through LDS or a scalar.
- A tile-shuffled weight is currently read in place by the FMA loop (one 8-byte word per load), so its
  speed is the row-major speed. A matrix-core path over that layout is the natural next design for
  widths above 4.
- For few-tile shapes with a long `K`, the K split; for many-tile shapes, 16-element steps only.

## What to check

- `K` tail: a `K` that is not a multiple of 512 must take the guarded path; test such a shape.
- A head whose `dim` is not a multiple of 16 masks its last chunk (clamped weight rows, no store).
- Weight row stride and scale strides are read from the loaded tensors (padded rows are common on fp8
  weights; some backends keep the scale column-major).
- With a row map, the bias and the block-scale row follow the **mapped weight row**, not the output
  column. A wrong map is off by hundreds of tolerances, so it is easy to see once tested on a shape
  where the map is not the identity.
- A bias under tensor parallelism is added by one rank only when the native row-parallel layer does so.
- ISA: registers under the cap for your thread count, scratch 0.
