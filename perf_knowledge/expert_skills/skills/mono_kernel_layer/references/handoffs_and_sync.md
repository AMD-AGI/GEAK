# Hand-offs and synchronization inside a persistent layer kernel

Levers for the gaps between stages and between layers: per-tile hand-offs instead of grid barriers, three
ways to tag a hand-off, fusing the TP all-reduce into its neighbours, and issuing loads early. The
primitives these build on are in [runtime_primitives.md](runtime_primitives.md); measure the gaps first
([measuring.md](measuring.md)).

Contents: [Choosing a hand-off](#choosing-a-hand-off) ·
[Per-tile hand-offs](#per-tile-hand-offs-instead-of-stage-wide-barriers) ·
[Tagged (value, tag) hand-offs](#tagged-value-tag-hand-offs) ·
[Epoch-flag hand-offs](#epoch-flag-hand-offs-in-flydsl-inside-a-gpu-and-across-gpus) ·
[All-reduce fusion](#fuse-the-tp-all-reduce-into-its-neighbours) ·
[Early-issue prefetch](#issue-the-next-stages-loads-before-its-inputs-arrive)

## Choosing a hand-off

Three variants of one idea (publish the data with a tag; the consumer waits for the tag, not for a
barrier). They differ in how the tag is packed and whether the wait is bounded.

| variant | what is published | tag source | wait | use it when |
|---|---|---|---|---|
| runtime mailbox (`mailbox_put` / `mailbox_get`) | data written first, then one 64-bit release store of tag + 32-bit payload | launch epoch `seq` | bounded by a deadline, errors reported | default per-tile hand-off; gaps between stages, barriers dominate the timeline |
| tagged (value, tag) pair | the value and its tag in one 64-bit word, one atomic store | step × L + layer + 1, one epoch per step | bounded by a deadline, error word, trap | layer-to-layer gaps, flags cleared every step; the limit of per-tile hand-offs: no grid barrier in the layer |
| epoch flag | stores with a cache policy, drain, then `flag[task] := epoch` (small values: 8-byte (value, epoch) pairs) | a device-side epoch word | no deadline of its own: keep the runtime's bounded wait | only where the timeline shows the mailbox itself costs time |

## Per-tile hand-offs instead of stage-wide barriers

**When.** The timeline shows gaps between consecutive stages, or most of the waiting is inside
`grid_barrier`. A grid barrier makes every workgroup wait for the slowest one of the previous stage; with
~300 workgroups and global atomics at ~116–202 ns core-to-core
([`hardware/cdna3_mi300/xcd_chiplet.md`](../../../../hardware/cdna3_mi300/xcd_chiplet.md)), each barrier
costs several µs of pure latency plus the skew.

**Recipe.**
1. Split the producer's output into tiles (e.g. row blocks of the GEMV output, heads of attention). After
   writing a tile, its producer workgroup does `mailbox_put(region, tile_idx, seq, payload)`: one 64-bit
   release store of tag and payload, so the data is visible once the tag is.
2. A consumer polls only the tiles it needs: `mailbox_get(region, tile_idx, seq, deadline=...)`. A GEMV
   consumer that needs the whole input vector waits on all input tiles, but it can start loading its own
   weights before the wait (weights do not depend on the producer).
3. Keep `grid_barrier` only where every workgroup truly needs everything (before an all-reduce that reads
   the whole partial, or where the epoch must be closed), and number those with `phase`.
4. Size: `n_mailboxes` in `PeerGroup` is fixed at creation; reserve contiguous index ranges per hand-off
   in one table, with the same layout for every width.

**Pitfalls.**
- Tags come from `seq` only (never a host step counter), so replays get fresh tags.
- Every `mailbox_get` has a deadline and a unique `site`, and `report_unless` on failure.
- Do not poll in a bare loop around a plain load: it hangs forever when a peer dies, which a
  fault-injection run exposes ([acceptance_gates.md](acceptance_gates.md#checking-gates-2-3-and-4-outside-the-unittest)).
  Use `mailbox_get` / `wait_until`.
- A consumer reading a tile through the L2 of another XCD sees it only after the release; keep the
  producer's store a release (`mailbox_put` does it) and do not cache the data in LDS before the tag.

**Check.** The oracle (graph replay across widths and the sequence check exercise the tags), then the
gates. A fault-injection run (a stalled rank) must end in the deadline error, not a hang.

**Cost model.** Measure one device-wide grid barrier on your device with a microbench (expect a few µs on
MI300X) and multiply by the barriers per layer: with ~10 of them they can take a large share of a layer
whose weights stream in tens of µs. The limit of this lever is no grid barrier inside the layer at all,
with every wait on the tiles it consumes ([tagged hand-offs](#tagged-value-tag-hand-offs)).

## Tagged (value, tag) hand-offs

Tagged hand-offs instead of barriers and zeroed flags.

**When.** Stages hand off through grid barriers or through flags that must be cleared every step; the
timeline shows gaps between stages or between layers, or short-context steps are slower than native.

**Recipe.**
1. **Pack the datum and its tag in one 64-bit word**, published by one 64-bit atomic store; the consumer
   polls until the tag matches and then the value is this step's by construction (e.g. tag = step × L +
   layer + 1 with L ≥ the number of layers). No buffer has to be zeroed between steps.
2. **Wait only for what you consume.** A CTA waits for the shards it needs, not for the whole previous
   stage, so a stage's tail delays only its dependants (no grid barrier anywhere in the layer).
3. **Advance the step once per step, not once per layer.** A per-layer completion counter is an atomic on
   every CTA per layer and shows up as the layer-to-layer gap; let the last CTA of the last layer bump the
   step number.
4. **Poll in batches.** Polling tags one by one costs a round trip each and can make short-context steps
   slower than native; poll many tags per lane and reissue only the stale ones.
5. **Scopes:** agent-scope relaxed atomics inside a GPU; system scope + IPC mailboxes (`PeerBuffer`)
   across GPUs. Peer buffers uncached. Waits that clear flags need ACQUIRE at system scope.
6. **Bounded waits.** Every cross-CTA / cross-rank wait has a deadline (seconds); on expiry set an error
   word and `trap` — never continue with stale data. A host thread can read the error word as a second
   check, but it must run in relaxed graph-capture mode (see the watchdog pitfall in
   [apply_mono_kernel_to_framework](../../apply_mono_kernel_to_framework/skill.md#knobs--pitfalls)).

**Pitfalls.**
- **Tag wrap.** A 32-bit tag wraps after 2^32 / L steps. Either size the tag for the longest serving
  session, or clear the work and peer areas at a known point between two cross-rank rounds; test the wrap
  either way.
- Fixed device addresses for every buffer, so the kernel can be captured in a CUDA graph and replayed.
- A trapped process writes a GPU core dump and does not exit unless `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`.
- Every rank must take the same path each step, or the system-scope waits hang until the deadline (the
  native-path decision of gate 6 in [acceptance_gates.md](acceptance_gates.md#hard-gates) is one case of
  this).

**Check.** Layer-to-layer gap in the timeline (target: a few µs at most); a short-context step not slower
than native; the oracle's random parity on every rank; a forced-timeout test that the kernel traps instead
of hanging.

## Epoch-flag hand-offs in FlyDSL (inside a GPU and across GPUs)

A lower-level form of the runtime's mailboxes (above), which add deadlines and an error word. Use it only
where the timeline shows the mailbox itself costs time, and keep the deadline.

**Inside a GPU.**
1. Producer: plain buffer stores with a device-scope cache policy (SC1: write through the per-XCD L2),
   then `s_waitcnt vmcnt(0)`, a CTA barrier, and one thread stores `flag[task] := epoch`.
2. Consumer: the threads of a CTA poll the flags they need in parallel at device scope until == epoch,
   then a CTA barrier. Data loads after the wait use the same cache policy.
3. Granularity: waiting for every task of the previous stage is the simple form; overlap then comes from
   placement and prefetch ([task tables](layer_structure.md#task_tables)). Finer waits pay where a
   consumer reads few tiles (e.g. attention waiting only for its own (request, KV head) tiles of the QKV
   output).
4. Small values can hand off as 8-byte (value, epoch) pairs instead of data + flag: no drain, no flag, one
   load tells both.
5. Batch polls: issue every poll load together (16 B = 2 pairs each) and reissue while any tag is stale,
   so the wait is one round trip once the producers have landed.

**Across GPUs (all-reduce, all-gather).**
- Each rank stores (value, epoch) pairs at system scope straight into every rank's peer buffer,
  double-buffered by epoch parity so step n + 1 never overwrites what step n still reads. The owner polls
  the tags and sums in rank order in fp32.
- Peer base addresses loaded from memory are VGPR values: move them to SGPRs (readfirstlane of both
  halves), otherwise every peer access gets a waterfall loop.

**Epoch.** A device word holds the epoch, starting at 1 (flags start at −1, so no stale match). Every CTA
reads it at entry; the last CTA to finish advances it. Nothing is cleared and every kernel argument is a
fixed pointer, so the launch stays graph-capturable.

**Pitfalls.**
- Without a deadline a lost publish hangs the GPU. Keep the runtime's bounded waits unless the timeline
  proves their cost matters.
- Bulk loads in flight delay a poll's result under in-order `vmcnt`. Issue polls before bulk loads, or
  poll from a separate wave.
- Ballots right after a divergent spin-poll can be miscompiled: put a CTA barrier between them and check
  the result in the unittest.

## Fuse the TP all-reduce into its neighbours

**When.** The timeline shows the all-reduce (after o_proj, after the MLP down projection) as a serial
block: the GEMM finishes, then a grid barrier, a cross-rank barrier, then the sum, then the next stage.

**Recipe.**
1. **Write partials from the GEMM epilogue.** Each GEMV workgroup writes its output rows straight into
   this rank's `partial_slot(partials[rank], seq, slot_elems, dtype)` (the double-buffered half chosen by
   `seq & 1`) instead of a separate output tensor.
2. **One cross-rank synchronization per reduction.** `allreduce_sum` does one grid barrier
   (`scope=SYSTEM`) plus one `xrank_barrier`. If two reductions are back to back with nothing in between
   that needs the first result on every rank, they cannot share it (the MLP needs the attention result);
   but the `grid_barrier` before the cross-rank one can often be replaced by per-tile mailboxes into
   workgroup 0 of the reduce ([per-tile hand-offs](#per-tile-hand-offs-instead-of-stage-wide-barriers)).
3. **Reduce in the consumer's layout.** Each workgroup sums only the slice of the hidden vector it will
   consume next (reduce-scatter style), reading that slice from every rank's partial in rank order.
4. **Residual add + next RMSNorm in the reduce epilogue.** RMSNorm needs the sum of squares of the full
   hidden row: each workgroup computes it for its slice, publishes it through a mailbox, and every
   workgroup combines the slices in a fixed order. Then the normed slice feeds the next GEMV without a
   round trip through HBM.

**Pitfalls.**
- **Bit-identical ranks.** The sum over ranks is fp32 in rank order; every rank must add in the same order
  and the same slices, or the ranks drift apart and the sequence check fails later.
- The fixed-order rule holds for the sum of squares too: combine slice sums in slice order, not in arrival
  order.
- Two reductions use separate partial buffers and `xrank_index`es; `slot_elems` is the capacity of one
  half (max width × hidden) and identical for every width sharing the buffer.
- Partials are read by peers over IPC: their writes must be released (the barrier or mailbox does it)
  before the peer reads.

**Check.** The oracle (random parity on every rank catches order drift as a mismatch between ranks), then
the gates.

**Variants to measure.**
- Barrier-free form: each workgroup stores its output channels as (value, tag) pairs straight into every
  rank's uncached peer buffer at system scope; the owner of a slice polls the tags and sums in rank order
  ([tagged hand-offs](#tagged-value-tag-hand-offs)). No grid barrier, no separate cross-rank barrier.
- Poll the pairs of a wave with several lanes in parallel rather than lane 0 alone.

## Issue the next stage's loads before its inputs arrive

Complements [weight streaming](stage_kernels.md#stream-the-weights-at-hbm-speed) (how to stream a GEMV's
weights) with *when* to start the streams. Draft: measure each item.

**When.** A stage's span starts with a load ramp (first bytes arrive several µs after its inputs are
ready), or a stage waits on data whose address was known long before.

**Recipe — candidates, roughly in the order they usually pay.**

| issue early | when | note |
|---|---|---|
| this block's rows of the first GEMV's weights (e.g. QKV) | at kernel start / during the input norm | weights never depend on the step's data |
| first K tile of the block's attention task | at kernel start | its address follows from the block table and `seq_lens` |
| the next GEMV's weights (e.g. o_proj) | during attention | attention is latency-bound; the memory system has room |
| down / second expert GEMM weights | together with gate/up, once routing is known | may compete with gate/up for bandwidth: measure |
| sliding window: the first V rounds of the window | at kernel start | the window is fixed by the position |
| small inputs before weights; block scales before the weights they scale | per stage | the math waits on the scale, not the bulk |
| next task's weights and inputs | while computing the current task | for stages with several tasks per block |

**Pitfalls.**
- Each early load holds VGPRs across the wait. A spill turns a win into a loss: check the ISA for scratch
  after every prefetch ([kernel pitfalls](flydsl_idioms_and_pitfalls.md#kernel-pitfalls)).
- Stores in flight (e.g. the KV write) delay the waits of later loads under in-order `vmcnt`; move such
  stores after the loop that waits.
- Gains shrink fast and a standalone bench gain may not show at the serving metric. Do the big ones
  (weights of the next GEMV, first K tile) first.

**Check.** Per-stage start-to-first-use gap in the timeline; standalone layer time; ISA has no scratch.

## Sources

- Hardware figures: [`hardware/cdna3_mi300/xcd_chiplet.md`](../../../../hardware/cdna3_mi300/xcd_chiplet.md).
