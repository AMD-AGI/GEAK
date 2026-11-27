# Hand-off levers: waits, tags, the all-reduce, and when loads start

In a tuned whole-layer kernel the waves spent about 57% of their time waiting on dependencies, 26%
issuing loads and under 20% computing (*measured*, FlyDSL reference kernel). Hand-offs are where the time
is, and where a mistake becomes a hang or a rare wrong answer: the state you finally save after a change
here gets the correctness check three times (section 6); one run per step is enough while iterating.

## 1. Per-tile hand-offs instead of stage-wide barriers

**When.** The timeline shows gaps between consecutive stages, or most waiting is inside a device-wide
barrier. Such a barrier makes every workgroup wait for the slowest one of the previous stage.

**Recipe.**
1. Split the producer's output into tiles (row blocks of a GEMV output, heads of attention). After
   writing a tile, its producer publishes the tile's tag with one 64-bit release store of tag and
   payload, so the data is visible once the tag is.
2. A consumer polls only the tiles it needs. A GEMV consumer that needs the whole input vector waits on
   all input tiles, but starts loading its own weights before the wait.
3. Keep a device-wide barrier only where every workgroup truly needs everything (before an all-reduce
   that reads the whole partial, or where the epoch must be closed), and number those per launch.
4. Reserve contiguous hand-off index ranges per hand-off in one table, with the same layout for every
   width (the slot count is fixed when the peer memory is created).

The limit of this lever: no device-wide barrier inside the layer at all.

**Already applied if** there is no barrier over the whole grid between stages and every wait is on
per-tile or per-task tags.

**Pitfalls.** Tags come from the launch epoch only, never from a host step counter, so replays get fresh
tags. Every wait has a deadline and its own site id, and reports on failure. Never poll in a bare loop
around a plain load: it hangs forever when a peer dies. A consumer on a different XCD sees a tile only
after the release: keep the producer's store a release and do not cache the data in LDS before the tag.
**Check.** Correctness (graph replay across widths and the sequence check exercise the tags); a stalled
rank must end in the deadline error, not a hang.

**Evidence (measured).** One device-wide barrier cost about 4–5% of a tuned layer's time; a layer with
ten stage-wide barriers would spend 40–50% of that time in them (extrapolated). Measure one barrier
with a microbench on your device and multiply by the barriers per layer before deciding.

## 2. Tagged pairs, one epoch per step, batched polls, bounded waits

**When.** Stages hand off through barriers or through flags that are cleared every step; gaps between
stages or layers; short-context steps slower than the baseline.

**Recipe.**
1. Pack the datum and its tag into one 64-bit word published by one atomic store. The consumer polls
   until the tag matches; the value is then this step's by construction (for example tag = step × L +
   layer + 1 with L ≥ the number of layers). Nothing has to be zeroed between steps.
2. Wait only for what you consume: a stage's tail then delays only its dependants.
3. Advance the step once per step, not once per layer. A per-layer completion counter is an atomic on
   every workgroup per layer and shows up as the layer-to-layer gap; let the last workgroup of the last
   layer bump the step.
4. Poll in batches: one tag per round trip can make short-context steps slower than native. Poll many
   tags per lane and reissue only the stale ones.
5. Scopes: device-scope relaxed atomics inside a GPU; system scope and IPC-shared peer buffers across
   GPUs. Peer buffers are uncached. Waits that clear flags need acquire semantics at system scope.
6. Bounded waits: every cross-workgroup or cross-rank wait has a deadline (seconds); on expiry set the error
   word and stop the launch (skip the dependent work or trap, as the kernel's own helper does) — never continue
   with stale data. A host thread may read the error word only in a relaxed graph-capture mode (else capture breaks).

**Already applied if** the header comment or helpers speak of tagged pairs, an epoch word is advanced by
the last workgroup of the step's last launch, nothing is zeroed per step, and the poll helper takes a
range of tags.

**Pitfalls.** Tag wrap: a 32-bit tag wraps after 2^32 / L steps; size the tag for the longest serving
session or clear the work and peer areas at a known point between two cross-rank rounds, and test the
wrap either way. Every buffer keeps a fixed device address so the launch can be captured in a graph. A
trapped process may write a GPU core dump and not exit unless core dumps on exception are disabled.
Every rank must take the same path each step, or system-scope waits hang until the deadline. **Check.**
Layer-to-layer gap (target: a few percent of the layer); a short-context step not slower than the
baseline; random parity on every rank; a forced timeout ends in the deadline error — no hang, no stale data.

**Evidence (measured).** One epoch per step cut the layer-to-layer gap by about two thirds (from about
8% of the layer to about 3%) yet moved serving P90 by only −0.3% in that round. Batched polling was one
item of a first rewrite that took a layer to about 57% of its first working version (together with MFMA
GEMV / attention and prefetch; not isolated).

## 3. Epoch flags: the lower-level form

A lower-level form of the tagged mailbox, which adds a deadline and an error word. Use it only where the
timeline shows the mailbox itself costs time — and keep the deadline.

**Inside a GPU.**
1. Producer: plain stores with a device-scope cache policy (write through the per-XCD L2), drain them
   (`s_waitcnt vmcnt(0)`), a workgroup barrier, then one thread stores flag[task] := epoch.
2. Consumer: the threads of a workgroup poll the flags they need in parallel at device scope until they
   equal the epoch, then a workgroup barrier. Data loads after the wait use the same cache policy.
3. Granularity: waiting for every task of the previous stage is the simple form (overlap then comes from
   placement and prefetch). Finer waits pay where a consumer reads few tiles — attention waiting only
   for its own (request, KV head) tiles of the QKV output.
4. Small values hand off as 8-byte (value, epoch) pairs instead of data + flag: no drain, no flag.
5. Batch the polls: issue every poll load together (16 bytes = 2 pairs each) and reissue while any tag
   is stale, so the wait is one round trip once the producers have landed.

**Across GPUs.** Each rank stores (value, epoch) pairs at system scope straight into every rank's peer
buffer, double-buffered by epoch parity so step n + 1 never overwrites what step n still reads. The
owner polls the tags and sums in rank order in fp32. Peer base addresses loaded from memory are per-lane
values: move them to scalar registers (first-lane read of both halves), or every peer access becomes a
waterfall loop.

**Epoch.** A device word starting at 1 (flags start at −1, so nothing stale matches). Every workgroup
reads it at entry; the last one to finish advances it. Nothing is cleared and every argument is a fixed
pointer, so the launch stays graph-capturable.

**Pitfalls.** Without a deadline a lost publish hangs the GPU. Bulk loads in flight delay a poll's
result under in-order memory wait counting: issue polls before bulk loads, or poll from a separate wave.
Ballots right after a divergent spin-poll can be miscompiled: put a workgroup barrier between them and
check the result in the unit test.

**Evidence (measured, FlyDSL reference kernel, which uses this form; its waits had no deadline — keep
every bound a kernel has, and treat adding bounds as a separate, declared change).** Waiting per
(request, KV head) instead of on the whole QKV stage, with the old K/V and sink loaded before that wait
and one LDS exchange instead of an 8-round merge: −52% of the sliding-window attention stage. Flags
polled before the weights, with the down weights coalesced through an LDS transpose: −15% of the down
stage, partly lost again by the end. Scalar peer bases and (value, tag) pairs from router to expert
union were items of a round worth 1.19× in total.

## 4. Fuse the all-reduce into its neighbours

**When.** The all-reduce (after o_proj, after the MLP / expert down projection) is a serial block: the
GEMM finishes, then a device-wide barrier, a cross-rank barrier, the sum, and only then the next stage.

**Recipe.**
1. Write partials from the GEMM epilogue: each GEMV workgroup stores its output rows straight into this
   rank's partial slot (a double-buffered half chosen by epoch parity), not into an output tensor.
2. One cross-rank synchronization per reduction. Two reductions cannot share one when the second needs
   the first's result on every rank (the MLP needs the attention result), but the device-wide barrier in
   front of the cross-rank one can often become per-tile hand-offs into the reducing workgroup.
3. Reduce in the consumer's layout: each workgroup sums only the slice of the hidden vector it consumes
   next (reduce-scatter style), reading that slice from every rank's partial in rank order.
4. Residual add and the next RMSNorm in the reduce epilogue: each workgroup computes the sum of squares
   of its slice and publishes it; every workgroup combines the slices in a fixed order; the normed slice
   feeds the next GEMV without a round trip through memory.
5. Barrier-free variant: each workgroup stores its output channels as (value, tag) pairs straight into
   every rank's uncached peer buffer at system scope; the owner of a slice polls the tags and sums in
   rank order. Poll with several lanes in parallel rather than lane 0 alone.

**Already applied if** no collective-library call and no separate all-reduce launch sits between
stages, GEMV epilogues store into peer buffers, and the post-attention stage does sum + residual + norm.

**Pitfalls.** Bit-identical ranks: the sum over ranks is fp32 in rank order; every rank adds the same
slices in the same order, or the ranks drift apart and a sequence check fails later. The same holds for
the sum of squares: combine slice sums in slice order, never arrival order. Two reductions use separate
partial buffers and indices; a half's capacity is max width × hidden and identical for every width that
shares the buffer. Peers read partials over IPC: release the writes before a peer reads. Keep the
baseline's rounding of the stored sum (aiter's custom all-reduce truncates the bf16 sum on gfx942 — *verified by
a real-op probe* on two ranks; `native_decode_numerics`, TPA-09). **Check.** Random parity on every rank catches order
drift as a mismatch between ranks.

**Evidence (measured).** Expert down fused with the final sum over all experts (fp32 slot-order sum in
LDS) and pushed straight into the second all-reduce: −36% of those stages (with the later −15% on the
down stage, the net at the end of the lane was −40%). Polling all-reduce pairs with parallel lanes:
neutral on serving P90 (+0.1%).

## 5. Issue the next stage's loads before its inputs arrive

**When.** A stage's span starts with a load ramp, or a stage waits on data whose address was known long
before. Candidates, roughly in the order they usually pay:

| issue early | when | note |
|---|---|---|
| this workgroup's rows of the first GEMV's weights (QKV) | at kernel start / during the input norm | weights never depend on the step's data |
| first K tile of the workgroup's attention task | at kernel start | its address follows from the block table and sequence lengths |
| the next GEMV's weights (o_proj) | during attention | attention is latency-bound; the memory system has room |
| down / second expert GEMM weights | with gate/up, once routing is known | competes with gate/up for bandwidth: *measured* twice, a loss or no gain |
| sliding window: the first V rounds of the window | at kernel start | the window is fixed by the position |
| small inputs before weights; block scales before the weights they scale | per stage | the math waits on the scale, not on the bulk |
| the next task's weights and inputs | while computing the current task | stages with several tasks per workgroup |

**Already applied if** stage functions issue their weight loads before the wait on input tags
("load-first" passes) and the first K tile is issued at kernel entry.

**Pitfalls.** Every early load holds VGPRs across the wait; a spill turns a win into a loss: check the
ISA for scratch after every prefetch. Stores in flight (the KV write) delay the waits of later loads
under in-order wait counting: move such stores after the loop that waits. Gains shrink fast: do the big
ones (next GEMV's weights, first K tile) first. **Check.** Per-stage start-to-first-use gap in the
timeline; no scratch in the ISA.

**Evidence (measured).** On serving P90 each early-issue tweak was worth at most 2–3%, even when a
standalone bench improved: first K tile at kernel start −2.3%; three later early-issue rounds −1.2%, 0
and −0.3%. Inside one stage it can be large: the next task's weights prefetched, with the activation
hoisted to LDS, −19% of the expert gate/up stage. Dead ends: deeper prefetch in that stage — noise (the
per-task data wait was already negligible; the stage was bound by its in-CU dependency chain); down
weights carried in gate/up's last prefetch slots — gate/up lost about 1.5× what down gained, because
memory was already saturated; down issued during gate/up in the HIP design — reverted, no gain;
touch-ahead with registers before a wait — slower; touch-ahead with direct global → LDS loads — the
compiler put a full memory wait before the next LDS read; gating the o_proj prefetch — neutral.

## 6. Pitfalls of a persistent kernel (the oracle sees these only sometimes)

- **LDS races after a stage.** A missing barrier at the end of a stage that reuses LDS (or between
  staging a KV block and the current-token write) shows only now and then. On the state you finally save after a synchronization or LDS change, repeat the
  correctness check three times and say how many you ran (one run per step while iterating): a race
  that shows once in N runs needs about 3N runs to be caught with 95% confidence.
- **Spills.** Every lever that holds more state (prefetch, shared weight reads, next-round loads) can
  hit the VGPR cap: check scratch in the ISA. **LDS alignment:** arrays that are not 16-byte aligned cost
  wide LDS accesses (*measured*: one misaligned array cost about a tenth of the layer).
- **Uncached peers.** Peer buffers and flags must be uncached; waits that clear flags need system-scope
  acquire. Words written by several workgroups each get a cache line (128 bytes) of their own.
- **Tag wrap** (section 2) and **divergent paths between ranks** (they hang until the deadline).
