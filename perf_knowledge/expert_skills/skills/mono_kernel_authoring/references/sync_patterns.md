# Synchronization patterns for a persistent layer kernel

Open this for step 4 of the procedure in [`../skill.md`](../skill.md), and whenever a wait hangs or a hand-off
reads stale data.

**Status.** The internal mono-kernel generation workflow (not public) used one shared synchronization runtime for
every kernel. It had its own GPU tests on two ranks and carried the exercised kernels on four ranks. **That
library is not in this repository.** This page describes the patterns and their pitfalls so that you can review a
runtime vendored into your task, or write one. Primitive names below are descriptive, not an API.

## 1. One launch, in outline

```
seq      = read the launch epoch from the control region     # the same value in every block
deadline = device clock + tick budget from the host          # one deadline per launch
for stage in contract order:
    for tile in the tiles this block owns:
        wait(input tiles, tag == seq, deadline, site of the port)   # on failure: skip the work
        compute; store hand-off data with coherent stores
        publish(tile, tag = seq)
all-reduce: write partial[rank][seq & 1]; grid barrier; cross-rank barrier; sum in rank order
last block out: launch counter = seq
```

## 2. Control region

- One per rank: device memory allocated **uncached** and exported to the other ranks of the group through IPC.
- Every slot with more than one writer sits on its own 128-byte line.
- Contents: header; sticky error word; launch counter (the epoch); one slot per rank; one release word per XCD
  (polling then mostly hits a local cache); grid-barrier counters; a cross-rank "done" word; then the tagged
  mailboxes, 8 bytes each.
- Create it, and the peer buffers, once: on the first eligible call, outside any graph capture, collectively on
  every rank of the group at the same decode width.

## 3. Launch epoch from the device

- At kernel entry every block reads the same epoch from the control region; at exit the last block to finish
  stores it back as the launch counter.
- **Every tag, barrier target and double-buffer half derives from the epoch.** Never pass a step number from the
  host: a graph replay runs with the arguments frozen at capture, so a host value never advances. With a device
  epoch no flag or buffer has to be zeroed between steps.
- The launch counter is also the proof of effect: it advances only when the kernel really ran.
- Pitfall, tag width: a narrow tag wraps (a 32-bit tag advanced once per layer per step wraps after 2^32 ÷ layers
  steps). Size it for the longest serving session, or clear the areas at a known point, and test the wrap.
- Pitfall, instances: a model runs many instances of a layer kind in one step. The internal adapter gave every
  instance its own control region. If instances share one, the tag must also identify the instance (for example
  step × L + layer + 1, with L at least the number of layers). *(No check of this ran; the multi-instance check
  in [`review_criteria.md`](review_criteria.md) was designed only.)*

## 4. Bounded waits

- Every wait on memory that another block or another GPU writes has a deadline, computed once at kernel entry
  from the device clock plus a tick budget the host passes in.
- A wait ends on an accepted value, on the deadline, or as soon as the local error word is non-zero (so one
  failure stops everybody quickly).
- After a failed wait, skip the dependent work (or trap, if a hard stop is preferred). Never continue with stale
  data. A trapped process writes a GPU core dump and does not exit until core dumps on exception are disabled
  *(read from the internal notes)*.
- No bare spin loop anywhere outside the runtime. The review reads the sources for them.
- The budget must exceed the slowest legitimate launch. *(Exercised: with slow first-version stage bodies a
  fixed deadline expired with no fault present. The fault tests then used the larger of the configured deadline
  and 4× the slowest rank's layer time.)*

## 5. Sticky error word

- One word in each rank's control region holding cause, site and rank. The first error wins and stays.
- The failing waiter writes it into its own region and into every peer's region, so all ranks stop.
- The host reads it only off the hot path: in tests, at teardown, behind a debug flag. Never per step: a host
  sync inside the step breaks graph capture and puts back the latency the kernel removes. A watchdog thread that
  reads it must use a capture mode that tolerates it, or it breaks capture at start-up.
- Reset it only in tests and teardown. After a reset a clean launch must succeed.

## 6. One deadline site per hand-off port

- Give every hand-off port its own site number. Every wait on that port's tiles reports that number; no two
  ports share one. Each all-reduce has its own pair (grid barrier, cross-rank barrier).
- Keep a table: site → consuming stages, port, producing stage. The error word then reads "rank R: stage S
  waiting for port P, published by Q". A value read by several stages has one site, and the producer is the one
  to inspect: does it publish under every block count and for every tile?
- Pitfall *(exercised)*: all waits of a launch share one deadline, so the word may name a downstream waiter. A
  publish dropped in the RoPE stage was recorded as the output-projection stage waiting for the attention
  output. After a timeout, list the tiles whose mailbox does not carry this launch's tag and **look first at the
  producer stage of the most upstream such port**: not at the stage the word names, and not at the stage that
  reads the port.

## 7. Per-tile mailboxes

- Producer: write the data, then publish with **one 64-bit store that carries the tag (the epoch) and a 32-bit
  payload**. Data first, tag last.
- Consumer: poll only the mailbox of the tile it needs until the tag equals this launch's epoch. The data is then
  this launch's by construction; a tag left by an earlier launch never matches.
- Wait only for what you consume. A stage's tail then delays only its dependants, which is why the layouts need
  no grid barrier inside the layer.
- Poll in bulk: all threads of a block poll the tiles they need in parallel, then one block barrier. *(Measured
  in the copy-body run with ordinary release / acquire, before the cheap discipline of section 8: about 30×
  cheaper than polling tile by tile; relaxed polling with a single final acquire saved about another 20%.)*
- A mailbox belongs to its destination: to hand a value to a peer rank, publish into the peer's control region.
- Every tile is published exactly once per launch, and read only after its wait. Test both: fill the hand-off
  inputs with NaN and let the wait copy the tile in (a read before the wait, or outside the waited range, then
  shows up as NaN), and count publishes per tile.
- Fewer mailboxes is not a goal in itself, but every tile costs one publish and one poll.

## 8. Cheap hand-offs on gfx942 (measured)

- With ordinary release / acquire at agent or system scope, each release emits a cache write-back and each
  acquire a cache invalidate. With 304 blocks doing this at once the operations serialize inside each XCD.
- Cheap discipline: **publish** = let the block's outstanding stores complete, a block barrier, then a relaxed
  store of the tag; **wait** = polling only, with no cache maintenance.
- Effect: the overhead floor of a whole layer fell about 13×. In a single-GPU micro-benchmark a publish became
  about 30× cheaper, a wait about 50×, a grid barrier about 20×.
- Price: **hand-off data must be read and written with coherent (cache-bypassing, `sc1`) accesses.** In the
  micro-benchmark a plain load after the wait returned stale data almost every time; a coherent load never did.
- Check in the ISA that a stage with hand-off inputs has coherent loads and a stage with hand-off outputs has
  coherent stores. This finds a stage that has none. It does not find a single missed tensor; a data-flow check
  for that was not built.

## 9. Barriers

- **Grid barrier.** All blocks of the launch meet, so all must be resident at once: at most one persistent
  block per CU. Targets are monotonic values derived from the epoch and the barrier's index within the launch,
  so nothing is reset. Agent scope inside one GPU; system scope when peers will read what was written before it.
- **Cross-rank barrier.** After a system-scope grid barrier, block 0 of every rank meets the peers through the
  per-rank slots; the other blocks wait on the cross-rank "done" word.
- A first correct kernel may put a grid barrier between stages (every block does each stage in turn). That is
  correct and slow; per-tile waits replace it later.

## 10. In-kernel all-reduce

- Each rank owns a partial buffer: uncached, IPC-shared, **double-buffered**. The half is chosen by the parity
  of the epoch.
- Sequence: write this rank's partial → one system-scope grid barrier → one cross-rank barrier → every rank
  sums all partials **in fp32, in rank order** → store with the native backend's rounding of the sum.
- Rank-order fp32 summation makes every rank's result bit-identical. *(Measured: bit-identical to the
  framework's collective library on two ranks.)* Whether the stored sum is rounded to nearest or truncated must
  match the native all-reduce: see skill `native_decode_numerics`.
- **Slot capacity** *(real bug)*: capacity = (largest decode width that shares the buffer) × hidden elements,
  and **every width passes that same capacity**. Buffer bytes = 2 × capacity × bytes per element. Example: widths
  1 and 4, hidden 4096, bf16 (2 bytes): capacity 4 × 4096 = 16384 elements for both widths; buffer 2 × 16384 × 2 =
  65536 bytes. With per-width capacities the halves of different widths overlap.
- **Peer address** *(real bug)*: the framework's caching allocator sub-allocates, and an IPC handle covers the
  whole underlying allocation. A peer must address the buffer as opened base + offset, with the base taken
  from the allocation's address range. Otherwise the reduce reads other memory and returns wrong numbers with
  no error.
- Two reductions in a layer (attention and MLP) use separate buffers and separate barrier indices.
- In a stage that ends with the all-reduce, the stage body writes only its rank's partial. Summing and
  publishing is one shared piece of code.
- A transient failure to open a peer's IPC handle was also seen in native code: treat it as no verdict and run
  again, not as a kernel failure.

## 11. Stage scratch and task flags

- Some stages exchange partial results between their own tasks: split-K attention (one KV head's context divided
  over tasks and merged by one of them), an MoE reduction across blocks. They need a stage-owned global buffer
  and per-task "done" flags, not a grid barrier.
- The flag is the epoch, written after the data. The owner waits, with the deadline, until the flag equals the
  epoch. Flags of earlier launches hold older epochs, so the buffer is not zeroed between launches or replays.
  A reset restarts the epoch, so it must zero the control region and every such buffer together. Allocate the
  buffer once per width, outside capture.
- **No-deadlock rule.** A block runs its tasks in increasing task number and all blocks are resident, so a task
  may wait only for tasks with a smaller number. The owner is the highest-numbered task of the group it merges.
  The stage is then correct for any block count, including 1.
- Not to be confused with register spill, which the ISA also calls scratch and which must be 0.

## 12. Graph capture and the adapter

- Fixed device addresses for every buffer; nothing zeroed between launches. Per-call addresses (KV cache,
  attention metadata) are kernel arguments re-read on every call: the cache can be reallocated after profiling.
- **Every rank takes the same path each step.** If one rank falls back to the native layer while another
  launches the kernel, both sides wait until the deadline.
- The framework's collective library can be unstable during capture (its watchdog queries events of collectives
  issued before the capture). The internal harness settled briefly after warm-up, agreed on capture success
  across ranks, retried up to three times, and otherwise fell back to eager timing on all ranks together.
  Native-path collectives in your adapter need the same care.
- FlyDSL's compile cache is keyed on the launcher's source and scalar closure values, not on helper modules.
  After editing the runtime or a stage helper, change the launcher name (put a hash of the sources into it) or
  run once with the cache disabled ([debugging card](../../../../languages/flydsl/debugging.md)). Each decode
  width is its own build with its own launcher name.

## 13. Self-tests before a layer depends on a runtime

1. A chain of hand-offs across blocks; repeated launches are bit-identical.
2. An orphan wait (no producer) ends in the timeout error with the right site, and every rank stops.
3. A rank barrier across the group.
4. The all-reduce: bit-identical on every rank; on two ranks also equal to the framework's collective library
   (with more ranks, or against a custom native all-reduce, only the tolerance applies).
5. A 16-byte vector tile handed off with coherent accesses; every load in the ISA is coherent.
6. After a timeout and a reset, a clean launch succeeds.
7. Many producers and consumers; many replays never accept a tag of an earlier launch *(planned, not built)*.

*(The internal runtime had items 1–5 as GPU tests; item 6 was checked in the layer-level copy-body run.)*
