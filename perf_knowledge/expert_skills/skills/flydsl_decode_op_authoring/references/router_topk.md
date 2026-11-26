# MoE router: gate GEMV, scoring, (grouped) top-k, renormalisation

One op from the hidden row to `topk_ids` and `topk_weights`. Maturity: the design passed every
correctness leg on gfx942 with FlyDSL 0.4.x over 14 shapes; gfx950 is untested. It **has** timings
against the deployed two-kernel sequence: faster on five shapes, slightly to clearly slower on three
(table below).

## What the design covers

| covered | not covered |
|---|---|
| hidden rows `[rows, hidden]` bf16 in; `topk_weights` (fp32 or bf16) and `topk_ids` (int32) `[rows, k]` out | a router that also emits expert counts, a sorted token list or a dispatch layout |
| a gate weight `[experts, hidden]`, row-major, bf16 or fp32 | a quantised or transposed gate weight, a bias inside the gate product |
| softmax, sigmoid or sqrt-softplus scoring; an optional per-expert bias used **for selection only** | other scoring functions, a learned temperature |
| plain top-k, or grouped top-k (keep the best groups, then top-k among them) | deeper hierarchies, placement-aware routing |
| a **table selection**: the expert ids are read from an integer table at the row's token id, and only those experts are scored | a table selection combined with softmax, a bias or grouping |
| up to 512 experts, `k <= min(experts, 64)`, `hidden` a multiple of 512 (bf16 gate) or 256 (fp32 gate) | more than 512 experts (needs a two-wave selection: a redesign, not a parameter) |
| renormalisation on or off, a routed scale, either tie rule, ids in pick order or sorted by id | an unsorted top-k whose order is an implementation accident |

The weights of a router are smooth; its ids are not: one swapped expert is a failed test. What decides
the ids is not in the op signature (whether the logits are rounded to bf16 before scoring, the tie
rule, the id order, bias for selection but not for weights). Those are parity facts; take them from
`native_decode_numerics`. A stricter or more accurate computation picks different experts.

## Task decomposition and ownership

- **A row is `ceil(experts / 4)` tasks**: task `t` is row `t // slices`, expert slice `t % slices`, four
  consecutive experts per slice. Workgroup `t mod n_blocks` runs it.
- One decode row therefore spreads over `experts / 4` CUs (64 for 256 experts) instead of one.
- **The last of a row's slices to arrive routes the row** and writes both outputs, once.
- With the table selection the slices are the table's columns, so a row has `ceil(k / 4)` tasks and
  streams `k` gate rows instead of all of them (a `k / experts` share of the bytes).

Why the split is inside the op rather than "a GEMV op followed by a select-only op": a generic linear
op writes bf16 outputs and reads bf16 or fp8 weights, so it can neither produce the fp32 logits that
most routers score on nor read an fp32 gate. It would also add a hand-off between two kernels. The
in-op split keeps every logits format working and keeps the output unit at one row.

## Lane and wave layout, and why

**Gate GEMV of a slice.**

- The task issues its first step's gate loads, then stages the row of `x` in LDS (`2 x hidden` bytes).
- Every wave takes a share of the hidden dim, in chunks of 64 lanes x 16 bytes, for all four experts of
  the slice. Lane `L` reads contiguous gate elements with one 16-byte load, coalesced along the row,
  and the matching `x` from LDS.
- The loop is software-pipelined: the next step's gate words are in flight during this step's FMAs.
- A reduce-scatter leaves each expert's wave partial in one lane; the partials go to LDS and thread `u`
  adds expert `u`'s partials in wave order. The logit is stored to the workspace with a coherent store.
- FMAs, not matrix cores: see the dead ends.

**Arrival.** The wave that stored the slice drains its stores; lane 0 adds 1 to the row's counter with
one device-scope atomic. The verdict "I am last" is carried as SSA values (the result of the `if`, then
a `readlane`). Do **not** pass it through an LDS flag: that was miscompiled (the other lanes read before
lane 0's store under the divergent branch).

**Selection, in wave 0 alone, from registers.**

- Lane `l` loads the logits (and bias) of experts `l + 64 i` straight from the workspace, all before
  the first use.
- Softmax is a wave max and a wave sum; sigmoid is per lane. Wave-wide reductions are DPP steps over
  16-lane rows plus four `readlane`s.
- Grouped: the biased scores go to LDS, lane `g` scores group `g`, and the kept groups are those whose
  rank is below the keep count, gathered with a ballot.
- Top-k is `k` rounds of a wave max over **packed 64-bit keys**: the fp32 score mapped to an
  order-preserving unsigned word, above the expert id or its complement. The key order is then exactly
  "larger score first, then the tie rule", and each winner leaves the race.
- Lanes `0 .. k-1` each store one pick; with ids sorted by expert, a pick's column is the number of
  picks with a lower id.

Why wave 0 alone: the selection is a serial chain of rounds; the other waves have nothing to add.

## Synchronisation

The last-arrival pattern, once per row. Nobody waits for another workgroup; any number of resident
workgroups makes progress. The workspace holds fp32 logits `[rows, experts]` and one counter per row
(`4 x rows x (experts + 1)` bytes); it is zero before the first call, every call leaves the counters
zero, and the first call per width must run eagerly.

## Resource budget

- Registers per wave: 70 to 178 at 256 threads; one fp32-gate, hidden 8192 shape took 272 (16 of them
  in the accumulator file, no scratch). At 512 threads every shape fits 256 with no scratch. Plan for
  `240 + 4 x ceil(experts / 64)`.
- LDS: `2 x hidden + 4 x waves x 4` bytes, plus `256 x ceil(experts / 64)` when grouped.

## Where the time goes

Before the split, one workgroup streamed a row's whole gate through one CU: time proportional to the
bytes per row, flat in the width, at one CU's load rate (roughly 1 to 2 % of the device's memory
bandwidth).

| measurement | result |
|---|---|
| kernel time, one-workgroup version over the split design | 5.1x, 5.8x and 11x less on a 2 MiB bf16 gate, a 3.5 MiB bf16 gate and an 8 MiB fp32 gate |
| versus the deployed gate GEMM + top-k (two launches): five shapes with fp32 logits or an fp32 gate | **1.5x to 2.4x faster** |
| two 128-expert softmax shapes with bf16 logits | 0.92x and 0.98x |
| one grouped 256-expert shape with bf16 logits | **0.81x** |
| a real router in a serving layer, versus aiter's top-k path | 1.30x to 1.32x faster |

- What is left after the split is a **fixed chain**, not the gate's bytes: launch, the `x` row, one
  GEMV step, the logits' store and drain, the arrival atomic, the logits' reload, then the selection.
- The selection costs about 2.5 to 4.5 % of the op's time per top-k round, instruction-bound in one
  wave.
- On the 0.81x shape the deployed GEMM and the deployed top-k are each individually cheaper than this
  design's fixed chain. No in-kernel tuning fixes that; only a shorter chain would.
- An fp32 gate doubles the bytes per row.

## Measured dead ends

| tried | result | mechanism |
|---|---|---|
| a matrix-core GEMV (bf16 16x16x16) | about 2x slower at one row | consistent with the weight stream being the bound at one row; a matrix core has nothing to tile over |
| 8 or 2 experts per slice instead of 4 | slower | result recorded only (my reading: CU spread against per-task fixed cost) |
| a deeper unroll (4 or 8 chunks) | no gain | result recorded only (my reading: the loop is load-latency bound) |
| the GEMV as a separate linear op | not possible for fp32 logits or an fp32 gate | output and weight formats of a generic linear op |
| "the row's task waits for its slices" | replaced by last-arrival | a wait can stall under a small workgroup count; the counter cannot |

## What to try next

- Shorten the fixed chain: the logits make a round trip through the workspace (store, drain, atomic,
  reload) before the selection.
- At 8 rows and 256 experts there are 512 tasks, more than the CUs; workgroups then take several
  slices and the slices of one row may run in sequence. A coarser slice at high widths is untested.
- More than 512 experts: a two-wave selection.

## What to check

- **Exact ties are not rare**: bf16 logits produce them, and so do experts with identical gate rows.
  Test with duplicated gate rows, with both tie rules.
- Bias for selection, not for weights: using the biased score as the weight passes every id check and
  fails every weight.
- Renormalisation happens over the `k` picked weights, before the routed scale.
- A zero-sum row (all picked scores 0): the backend's guard decides between 0 and NaN.
- A near tie cannot be made deterministic against a captured reference; expect the oracle to skip or
  accept either routing for such tokens.
- An aborted launch can leave a row's counter non-zero; that row never routes until the workspace is
  zeroed.
- The table form: ids in the table's column order, a repeated expert counted twice, integer width of
  table and token id as the backend holds them.
