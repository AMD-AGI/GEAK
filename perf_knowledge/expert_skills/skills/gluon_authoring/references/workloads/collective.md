# Multi-GPU From Inside The Kernel (gfx950 / gfx942)

For the case where the communication is *inside* the kernel — a fused allreduce-norm-quant, a
decode step that consumes a peer's KV shard, a routing pass that publishes to peers as it goes.
Not a replacement for RCCL: if the collective stands alone as its own launch, use the library.
This page is for when fusing it into a kernel is the point, and the library's launch boundary is
the thing you are trying to delete.

The primitives below are plain-Triton builtins re-exported into `gl`, so the mechanics are not
Gluon-specific. The page lives here because deciding *where in a hand-authored pipeline* the
publish and the wait go is an explicit-tile decision, and because the failure modes are the kind
that pass every small-shape test.

Everything here is **source-proven against 3.8.0 / 3.7.1 / 3.7.0 / 3.6.0** (3.8.0 spelling first;
older-tag differences are called out where they exist) and unprobed on hardware, gfx950 first with
gfx942 identical unless stated. Nothing on this page has been timed.

## Before the mechanism: what selects the algorithm, and what a ceiling can mean here

Two decisions sit in front of every line below, and both are commonly made by accident rather than
chosen.

**The selection variable is bytes per peer, not the calling convention.** One-shot — every rank
reads all peers and reduces locally — pulls `(TP-1)` copies of the full message into each rank.
Reduce-scatter-then-gather moves `2*(TP-1)/TP` of it and pays a second rendezvous for the
privilege. The ratio between them is `TP/2`: identical traffic at TP=2, twice at TP=4, four times
at TP=8, which is why one-shot wins while the message is small enough for round trips to dominate
and loses once bytes do. That crossover is a **message size**, so it belongs in a host-side
threshold table keyed on the element count about to move.

What it must not be keyed on is whether the caller happened to pass a peer-addressable workspace.
An ABI-driven selection means the same shape takes a different algorithm depending on how it was
called, so the faster arm is unreachable for any caller that did not know to ask — and the
threshold, never having been measured, is not recorded anywhere to be wrong. If the presence of a
workspace decides, that is a **fallback path**, and the selection rule still has to exist above it.

**Bytes are not the only thing the split buys, and the second reason is the one people reach for
without knowing it is gated.** An all-reduce cannot be overlapped with compute while it is still an
all-reduce: the one production path that overlaps a collective with a GEMM does so only after the
`all_reduce` has been rewritten as `reduce_scatter` + `all_gather`, because it is the
`mm -> reduce_scatter` edge that has anything to pipeline at all. With that rewrite disabled, the
fusion pass that is supposed to build the overlap is a no-op. So reduce-scatter is both the
byte-cheaper arm at large messages *and* the only arm that is decomposable.

**Do not read that as a direction for this workload.** Across four independent production
implementations of TP all-reduce, every one chose **fusion** — folding the reduce together with the
residual add and the RMSNorm, often with an FP8 quantization, into one kernel — and none chose
overlap. Where overlap exists in those stacks at all it is default-off and it is on MoE
all-to-all, not on TP all-reduce. Together with the missing ceiling below, that makes "overlap the
collective with compute" the one direction this page declines to recommend: it needs timing
stability to pay, and this is the section that cannot price timing. The available direction is the
next one — same algorithm, fewer bytes. **Falsifiable:** a production implementation that puts TP
all-reduce on a side stream overlapped with a GEMM refutes this.

**A percent-of-peak claim is not available here, and the substitute is not a number you already
have.** `index.md ## collective` states the model gap: the six priced resources are single-GPU, so
a local ceiling quoted for a kernel whose critical path is a peer wait is worse than no ceiling.
The interconnect's own figures are not in the pack either — `perf_knowledge/hardware/data/sku.json` and
`perf_knowledge/hardware/data/hw_constants.json` carry no peer bandwidth and no hop latency. The one fabric number
that exists, `arch.*.l2_fabric_latency_high_cyc`, is a **device-internal** L2-to-fabric read
latency used by bound classification; it does not describe a peer hop and must not be borrowed as
one. Both stay `unknown` rather than being substituted.

Four arguments remain. They are self-referential on purpose, and that is what makes them valid:

| The claim you want | The form that is actually available |
| --- | --- |
| this algorithm moves less than that one | the byte ratio above, evaluated at your `TP` and message size. Needs no ceiling at all — it compares two candidates to each other |
| the local half is efficient | a ceiling scoped to the local work, **with the scope stated in the record**. `index.md ## collective` asks for the statement, not merely the scoping |
| this change helped | same-session A/B: both arms in one process, same buffers, same world size. Across sessions the peer arrival order is not held fixed, so the comparison has a free variable |
| the communication itself costs this much | a floor probe — the same launch with the reduction removed and correctness knowingly invalid, read only as a self-referential floor for the arms above |

The fourth is the only way to price the wait, and it prices it for **this protocol at this world
size on this box**. It is not a hardware ceiling and does not transfer, so record it with the world
size beside it or it will be re-read later as one.

### Sending fewer bytes without changing the algorithm

The byte ratio above compares two algorithms moving the same payload. There is a second axis: make
the payload itself smaller on the wire, losslessly, so nothing about the reduction or the numerics
changes. The shape that does this is **separating the payload by bit position rather than by
element** — a BF16 value is one sign bit, eight exponent bits and seven mantissa bits, and those
three fields have very different entropy across a normalized activation tensor. Packed element-wise
they interleave, so a generic byte-oriented scheme sees noise; regrouped so that like fields are
adjacent, the low-entropy field becomes compressible while the high-entropy one is left alone. The
receiver reconstructs the original bits exactly, which is what makes it available at all here:
**the oracle for a lossless transform is bit equality**, a cheaper and stricter gate than any of
the numeric ones this page's other levers need.

Treat the size of the win as unknown until measured, because it is a property of your data rather
than of the format:

1. **Measure it offline first, with no kernel change at all.** Dump the payload a real rank would
   publish, regroup it by field, and compare the packed size against the raw size. This is the step
   that decides whether anything else is worth doing, and it costs no GPU time and no protocol risk.
2. **Only then price the pack/unpack.** Both sides pay VALU work to regroup and restore, so this
   trades fabric bytes for local compute. It can only pay where the peer wait is what you are
   waiting on — and per the table above you cannot establish that with a percent-of-peak, so the
   decision rests on a same-session A/B between the packed and unpacked arms.
3. **Keep the format out of the protocol's invariants.** The packed representation changes what the
   payload buffer contains, not who may write it or when — so the four invariants in
   `## A rendezvous is four invariants, and the atomic is not one of them` apply unchanged, and a
   length that now varies with the data is a new reason to state the buffer's ownership and sizing
   explicitly.

Where the payload is already FP8 or otherwise quantized, expect less: the exponent field the
regrouping exploits is most of what quantization already removed.

## The one primitive: an atomic carrying sem and scope

`gl.atomic_add` / `atomic_cas` / `atomic_xchg` / `atomic_or` / `atomic_and` / `atomic_max` /
`atomic_min` / `atomic_xor` are `builtin(tl_core.atomic_*)` — the plain builtin, unchanged, on all
four versions. Each takes `sem=` and `scope=`:

| Argument | Accepted values | Default |
| --- | --- | --- |
| `sem` | `"relaxed"`, `"acquire"`, `"release"`, `"acq_rel"` | `"acq_rel"` |
| `scope` | `"cta"`, `"gpu"`, `"sys"` | `"gpu"` |

Identical on all four versions, including the defaults.

> **The default scope is the wrong one for this page, and it fails silently.** `scope="gpu"` orders
> against other waves on *this* device. A peer GPU reading the same memory is outside that scope,
> so a publish that looks correct and is correct single-device becomes a race the moment TP>1 —
> with no diagnostic, no compile error, and a small-shape test that passes because the peer was
> slow enough anyway. Every cross-device atomic on this page carries `scope="sys"` explicitly.

`atomic_scatter_add` / `atomic_scatter_max` are a different mechanism despite the name: they are
**methods on the shared-memory descriptor** (LDS atomics), and they arrived in **3.8.0**. They do
not participate in anything on this page — CTA scope cannot publish to a peer, and the scatter form
takes an index tensor rather than a pointer, so neither the `sem` / `scope` arguments above nor the
rendezvous invariants apply to them. They are owned by
`../gluon/smem-lds-reference.md ## Atomic RMW in LDS — the atomic_scatter_* family`.

**The converse of the rule above is the part that gets mis-audited, so state it: `scope` follows
the reader set, not the buffer.** A word can sit physically inside a peer-visible lock buffer and
still be correct at `scope="gpu"`, because its only readers are CTAs on this device. **Two scopes
inside one data structure is the normal shape, not an inconsistency** — in a surveyed four-word
lock, words 0 and 1 are always `sys` and word 2 is always `gpu`, the latter being a rank-local
counter whose result is then republished at `sys` scope. Across surveyed collective kernels the
`gpu` sites are a small minority and **every one of them is a rank-local counter of exactly that
kind** — that, not the proportion, is the usable fact. So auditing a kernel by promoting each
`gpu` you find to `sys` produces false positives
and a system-scope writeback nobody needed.

The same rule read from the other side explains an absence rather than a presence. When the work is
device-local **and** the payload has no reader inside this launch at all — a device-local counter
published across a kernel-launch boundary, `workloads/moe.md` — leaning on the default scope is
correct and omitting `scope=` is a deliberate choice, not an oversight. Cross-device, write it;
device-local, the default is the answer. Decide it from who reads the word.

## Publish and consume: let the atomic emit the sequence

A release atomic at system scope is not just a flag write. On this target the backend expands it
to the cache maintenance the flag implies:

| What you write | What the backend emits around it |
| --- | --- |
| `sem="release", scope="sys"` | `buffer_wbl2 sc0 sc1`, then `s_waitcnt vmcnt(0)` — dirty L2 lines written back and the writeback awaited, *before* the flag becomes visible |
| `sem="release", scope="gpu"` | `buffer_wbl2 sc1` + the same wait — agent scope only |
| `sem="acquire", scope="sys"` | `buffer_inv sc0 sc1` after the flag read, so the payload load that follows cannot be served from a stale line |
| `sem="acquire", scope="gpu"` | `buffer_inv sc1` |

The table is the gfx950 sequence, and it holds unchanged on the gfx942 downgrade: both are in the
gfx940-family memory model and use the same `sc0` / `sc1` cache-policy bits. Confirm it on your
build by the disassembly check in `## How to verify any of this`, not by this table.

So the ordinary publish/consume pair needs no inline asm at all:

```python
# producer: payload with ordinary stores, then one release atomic as the flag
gl.store(out_ptr + off, payload)
gl.atomic_xchg(flag_ptr, seq, sem="release", scope="sys")

# consumer: acquire on the flag, then ordinary loads of the payload
seen = gl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
vals = gl.load(peer_ptr + off)
```

### Who issues the maintenance

The question to ask is **not** "is my signal an atomic". It is "did I write a `gl.atomic_*` whose
`sem=` and `scope=` ask the backend for this". The maintenance is attached to the intrinsic and its
arguments, so an operation that is atomic in every other sense can still get nothing.

**Ask it as two orthogonal questions, because that is what the two arguments answer.** Sorting by
*how the signal is spelled* mixes them together and mis-files the legitimate cases:

> **(1) Where are the readers? → that fixes `scope`.**
> **(2) Who completes the payload's writeback? → that fixes `sem`.**

| (1) readers | (2) who completes the writeback | What to write, and what is emitted |
| --- | --- | --- |
| another device | the backend, from the release | `sem="release"` (or `"acq_rel"`) with `scope="sys"`. Adding `buffer_wbl2` by hand gets you a second writeback on top of the emitted one. |
| another device | **nobody — nothing asked** | This is the failure. A real `gl.atomic_*` at `sem="relaxed"`, or one left on the default `scope="gpu"`: `sc1` alone does not reach a peer. Both read as safe and are not. The fix is an argument, not an instruction. |
| this device only | the backend | `scope="gpu"` — which is the default, so omitting it is a choice you are entitled to (`## The one primitive`). |
| **no reader inside this launch at all** | the **launch boundary** | **Keep `sem="relaxed"`.** The payload is consumed by a later launch, so the ordering has already been supplied; raising `sem` buys one `buffer_wbl2 sc1` and one `s_waitcnt vmcnt(0)` that nothing needs (`moe.md ### The launch boundary is the grid-level synchronization`). |
| whoever you published to | **you, by another mechanism** | `relaxed` publish whose writeback is completed by the output store's own cache policy plus an explicit `s_waitcnt` / `gl.barrier()`. Legitimate — but the cache policy on the store is now coupled to the cost of the release, and **that coupling is not established here**: whether a given write-through policy supplies the bits a release would have is not traced in this pack. If you rely on it, verify it on your build; if it does not hold, you are in row 2 with the symptom that hides longest. |
| — | the compiler never saw an atomic | **You.** A hand-written atomic in inline asm, or an ordinary store used as the flag, emits no sequence around one. `../gluon/inline-asm-reference.md ## Class 4 — synchronization` for the mnemonics; `## Class 4b — protocol blocks` when the asm body contains a label, an `s_cbranch` or an EXEC save/restore — that is a program, and the per-instruction advice does not reach it. |

The principle underneath all six rows: **`sem` is not a safety condition, it is the entry point for
asking the compiler to issue maintenance on your behalf.** Where the maintenance is genuinely
supplied by something else — a launch boundary, a cache policy plus an explicit wait — `relaxed` is
the right answer and raising it is pure cost. The price you pay for that is that the reason is no
longer written on the line: a reader auditing the atomic sees `relaxed` and cannot tell rows 2, 4
and 5 apart. Record which one it is next to the publish.

## Spinning, and the hoist that will eat your wait

A wait is a `while` loop, and `while` works: Gluon and plain Triton share one code generator, so
`visit_While` behaves identically under `@gluon.jit`.

The problem is that a spin loop is, to the optimizer, a loop whose condition never changes — so
the load feeding it is loop-invariant and gets hoisted out, leaving a loop that spins forever on a
value read once. Two things prevent it, and **neither is in the `gl` namespace on any of the four
versions**:

```python
import triton.language as tl          # the symbols below are not re-exported into gl

while tl.condition(flag != expected, disable_licm=True):
    flag = gl.load(flag_ptr, volatile=True)
```

- **`tl.condition(cond, disable_licm=True)`** wraps the loop condition and makes the code
  generator annotate the emitted `scf.while` with `llvm.loop_annotation`, disabling LICM for that
  loop. Exported from `triton.language` on all four versions; **absent from `gl.*` on all four**.
  Naming it through an imported `triton.language` module is legal inside a `@gluon.jit` body — the
  frontend's global-access gate admits module objects outright — and it is the same move the
  layout factories already rely on
  (`../tile-programming/layout-recipes.md ## Compute the layout instead of tabulating it (constexpr_function)`).
- **`volatile=True`** on the load. `gl.load` is `builtin(tl_core.load)`, so the parameter is
  there on all four versions even though the Gluon docs never mention it.

Spinning on an **atomic** read instead sidesteps both, because an atomic is side-effecting and
cannot be hoisted. That is the more robust spelling, and it is what the publish/consume pair above
uses. Prefer it; reach for `disable_licm` only when the wait genuinely must be a plain load.

> **Every spinning wave is occupying a CU while computing nothing.** A spin is not a free wait like
> a `s_waitcnt` — it holds its registers, its LDS allocation, and its occupancy slot for the whole
> wait. Two consequences: keep the spin short enough that the slot is cheaper than a relaunch, and
> never spin in a wave that is also holding a large accumulator you could have stored first.

## Co-residency is a launch option, not an assumption

The deadlock this page invites: block A spins on a flag that block B will publish, and block B was
never scheduled, because the grid was larger than the device holds. There is no forward-progress
guarantee between blocks of an ordinary launch.

The AMD backend exposes the one **language-level** answer as a launch option, on all four versions:

```python
kernel[grid](args..., launch_cooperative_grid=True)
```

It routes the launch through `hipModuleLaunchCooperativeKernel`, whose contract is that every
block is resident simultaneously — which also caps your grid at what actually fits. Two details
worth knowing before you rely on it:

- **It is a device capability, and 3.6.0 does not check it for you.** From 3.7.0 the driver
  asserts `device_properties['cooperativeLaunch']` with a clear message; on 3.6.0 the request goes
  straight through and an unsupported device fails at the HIP call instead.
- **It constrains the grid, so it interacts with every occupancy decision you already made.**
  Registers and LDS per workgroup now set a *ceiling on how many blocks may participate*, not just
  a throughput figure. Re-read the occupancy budget after turning it on, not before.

This is a launch **mode**, and it is unrelated to the NVIDIA thread-block-cluster / DSMEM mechanism
that `../hardware/bound-class-signals.md ### NV/NCU levers with NO AMD counterpart — do NOT port`
rules out. That row is not an argument against this option, and the option is not a way to recover
what that row says is absent.

### The option is the only guarantee, and neither production stack takes it

Everything above establishes that the option works. What it does not establish is that anyone
reaches for it. Two independent production stacks were scanned for it and for every synonym —
`cooperative` case-insensitively, the dashed CLI spelling, `hipLaunchCooperativeKernel`, `grid.sync`:

- In a **Gluon kernel-pack tree** (454 `.py` files under its kernel-pack root),
  `launch_cooperative_grid` appears **zero times**. Every other `cooperative` hit is a helper name
  about threads sharing a load — `_cooperative_weight`, `_rmsnorm_cooperative` — never a launch.
- In a **ROCm operator library's `csrc/`**, the exact token appears **twice, and neither is a kernel
  launch**: both sit in an ahead-of-time Gluon compile wrapper that plumbs the flag through as a
  `bool = False` field and a `--launch-cooperative-grid` switch. Nothing in the tree passes it.
  Repo-wide, the only other occurrence is a **commented-out** `# launch_cooperative_grid=True`
  still sitting in the argument list of a live Triton launch.

So the accurate statement is not "do not use it". It is: **the option exists, it is the only
forward-progress guarantee the language offers, it was reached for at least once and backed out of,
and both stacks shipped structures that do not need it.** One of them records why beside the
two-kernel split it chose instead — a source comment noting that using cooperative groups to sync
the grid "results in `hipErrorCooperativeLaunchTooLarge`". The guarantee is not only a constraint
you accept; it is a constraint a real shape can fail outright.

`moe.md ### The launch boundary is the grid-level synchronization` reaches the same result from the
other direction on an unrelated body of source — across surveyed fused-MoE kernels, spin loops,
`atomic_cas`, flag polling and cooperative-grid launches are all absent. Two workload pages, two
independent surveys, one answer. Read the rest of this section as what those stacks wrote
**instead**.

### Four structures that do not need the guarantee

Each of these is load-bearing in one of the two surveyed stacks. They are ordered by how much of
the problem they **restate** rather than work around, and the last one is the strongest.

**1. Put the cross-block wait in a dedicated one-block kernel.** The prefill all-reduce files launch
their synchronization as separate `grid=(1,)` kernels at `num_warps=1` — `_begin`, `_published`,
`_end`, five such launches per file across six files; a decode-shaped sibling uses a single
`_epoch` launch at the same `grid=(1,)` and `num_warps=1`. A grid of one block is trivially
co-resident, so the wait is legal
with no launch option and no capability check, and the payload kernels around it stay ordinary
launches whose grid nothing constrains. What it costs is launch count, priced by the launch-fusion
law in `../hardware/roofline-models.md ## Kernel time decomposition` — exposed while the dispatch is
eager, already amortized under a graph, which is the opposite of the intuitive reading and is why
this route is cheaper than it looks in a graph-captured deployment.
(The `num_warps=1` here is on a **1-D single-block** grid. The recorded hazard in
`../hardware/capability-matrix.md` is a *2-D multi-block* grid at `num_warps=1`, which this is not.)

**2. Write the residency requirement down, in the source, as a number.** Where a kernel genuinely
does depend on its blocks being co-resident, production states the dependency instead of assuming
it. Two forms, and what separates them is whether anything can check it:

- as a **source assertion beside the grid computation** — one file carries the comment "At most four
  512-thread CTAs must fit on one CU for forward progress" directly above the line that *computes*
  the grid to keep that true, clamping the block count rather than deriving it from the shape;
- as a **host-side refusal** — the other stack raises before the launch when the requested block
  count exceeds the device's CU count, in a message that names the grid-wide barrier and the
  spin-wait deadlock the surplus blocks would cause.

The discipline is that the assertion names the **number and the resource** — four, 512 threads, one
CU — because that is exactly what a later occupancy change invalidates. "Needs co-residency" in a
comment is not checkable by anyone, including its author. Where the number comes from is the
occupancy model, which this page does not own: `../hardware/planning-constants.md`. Note also what
this does to the `saturation = grid_tiles / CUs` ratio the SKU pages carry
(`../hardware/amd-cdna4-skus.md`): there it reads as a utilization figure, and here a value above
one is a correctness statement instead.

**3. Elect one waiter, and let the others leave.** Three module docstrings and two in-body comments
in the Gluon stack name a **last-reader election** and state the consequence in the same breath:
"Only the last reader CTA waits for peers, so progress does not require grid residency";
"last-reader election ... without requiring simultaneous residency of all row CTAs". The shape is
that every CTA publishes its arrival with an atomic, exactly one of them — whichever RMW returns the
final ticket — does the waiting, and the rest exit. Only one block is ever blocked, and it is
blocked on peers whose arrival it has already observed, so a sibling that was never scheduled cannot
deadlock it.

**The precondition is that the non-elected blocks *exit*, not spin.** The same election written the
other way buys none of this: a persistent-grid kernel in the other stack elects a last arriver to
run a serial prefix-sum phase while "the rest spin on the release flag" — and that kernel is exactly
the one carrying the host-side CU-count refusal in route 2, because its non-elected blocks are now
waiting on another block. Election removes the residency requirement **only when losing the election
ends the block's participation.**

**4. Replicate the rendezvous per block, so no block ever waits on another block.** This is the one
that is not a workaround. A C++ all-reduce in the operator library indexes every signal word by
`[blockIdx.x][rank]`: block *i* of each rank publishes into block *i* of every peer and waits only
for block *i* of every peer. No word is shared by two blocks of the same grid. Block 7 can arrive,
rendezvous and retire before block 3 is ever scheduled, so there is no sibling whose absence can
deadlock anything.

What that buys is categorical rather than incremental: **the grid is unconstrained.** No cap, no
capability check, no launch option — and, unlike every route above, the occupancy budget is
untouched, so the second caveat never applies at all. What it costs is a property of the
*decomposition* rather than of the synchronization: the protocol has to be expressible as
`ngpus`-way agreement replicated per block, which means block *i* on every rank must own the same
slice of the problem. Where the work partitions that way this is the route to take; where it does
not, the first question is whether it can be **made** to — that question is upstream of all three
routes above, and it is the same decision
`../tile-programming/mental-model.md ## Reduction & parallelization structure (decide before tiles)`
already owns.

### Which one to reach for

**Ask for the guarantee** when all three of these hold: the algorithm needs a real grid-wide barrier
that cannot be replicated per block — a serial phase over state every block wrote; the grid you want
is already smaller than what fits, so the cap costs you nothing you wanted; and you are willing to
freeze the occupancy budget underneath it. On 3.6.0 there is a fourth, because the driver will not
ask it for you: check `cooperativeLaunch` yourself.

**Change the structure** when any one of these is true — and the first two are the common case:

- the grid is sized by the **problem** rather than by the device, so the cap is a correctness
  landmine that the first larger shape finds;
- you would be picking a tile, a register budget or an LDS layout to satisfy a residency cap rather
  than to satisfy the kernel. An occupancy decision taken to serve a launch mode inverts the reading
  in `../hardware/roofline-models.md ### Occupancy is a lever only for latency/memory-bound work`,
  and inverting it is a thing to do deliberately or not at all;
- the "barrier" is really `ngpus`-way agreement that could be replicated per block (route 4);
- the serial phase is small enough to be its own one-block launch (route 1).

If you cannot take the co-residency guarantee — and the default assumption is that you are not
taking it — do not write a spin that waits on an arbitrary other block of your own grid. Split the
kernel at that point and let the launch boundary be the synchronization. That is the route the
surveyed HIP kernel took after cooperative groups refused its grid outright, and it is the same
conclusion `moe.md ### The launch boundary is the grid-level synchronization` reaches for a workload
that never had a cross-block wait to remove.

## A rendezvous is four invariants, and the atomic is not one of them

There is deliberately no vetted rendezvous skeleton on this page. The atomic spelling is the part
the sections above already settle, and it is the part that tends to be right; what separates a
working rendezvous from one that passes every test and then corrupts under load is the surrounding
state discipline. A skeleton would hand you the easy half and hide the four questions below, each
of which has to be answered for your buffer layout rather than copied.

Check all four before running a protocol, and again after any change to the buffers:

**1. Aliasing — does the flag share storage with anything?**
The publish orders *stores that precede it* against *loads that follow the matching acquire*. It
says nothing about two protocols that happen to use the same word, or the same cache line. Three
ways this breaks: a flag packed into the payload buffer (the consumer's payload load and its flag
load are then the same line, and the acquire's invalidate throws away the payload it just
validated); two rounds or two directions sharing one flag word; and a flag array whose stride puts
two peers' flags in one line, so a peer's unrelated publish drags your line around. Give flags their
own allocation and their own line.

This one is worth stating as a **cross-implementation invariant rather than as advice**: two
production all-reduce stacks written in different languages by different teams both do it
literally, and both spend *more* than a line — one aligns its per-block arrival counters to 128 B
(`alignas(128)`), the other spaces its ready flags 128 B apart and, in its larger-tile kernel,
several hundred words apart. Read "its own line" as the floor, not the target. The one legitimate
violation seen in surveyed production source is a **lifetime** split rather than a sharing one: a claim bitset
whose storage is later overwritten by reduced data, with a single global confirmation cutting the
two lifetimes apart. That is only safe because nothing reads the bitset after the cut, which is a
property you have to be able to state — not a licence to pack a live flag into a payload.

**2. Barriers — who is allowed to publish, and what has to be done first?**
The release atomic orders the *publishing wave's* stores. If the payload was written by the whole
workgroup and one wave flips the flag, the other waves' stores are not covered — a workgroup barrier
has to sit between the payload stores and the publish, and a second one after the acquire before
the other waves read. This is an intra-device ordering obligation that `scope="sys"` does not
discharge; it is `gl.barrier` (`gl.thread_barrier` on 3.6.0 — `../gluon/pipeline-reference.md`), and its
absence is invisible at small shapes because one wave usually finishes anyway.

**3. Wraparound — how does the consumer tell round `n` from round `n-1`?**
A flag reused across iterations has to distinguish *not yet arrived* from *left over*. A toggled bit
cannot, once a fast peer laps a slow one. A monotone sequence number can, and it is why the publish
example above writes `seq` rather than `1`: the consumer waits for `seen >= expected` rather than
`seen != 0`. If you do reset flags between rounds, the reset is itself a publish and needs the same
ordering as the payload — a "cheap" `gl.store` of zero is exactly the row-4 case in
`### Who issues the maintenance`. Also fix the wrap: a sequence in a 32-bit word wraps, and the
comparison has to be written so that it still works when it does.

**4. Ownership — who zeroes it, who frees it, and what world size was it built for?**
Every buffer in the protocol needs exactly one owner for initialization, and that owner has to run
before the first launch that reads it, not inside it. The same applies to the peer-pointer table
(`## Peer pointers arrive as integers`) and to any bitset. State the world size the buffers were
sized for next to the allocation: a table built for one world size and reused at another is the
failure that presents as an illegal access with no indication of which entry was wrong.

None of the four is checkable by the compiler, and none produces a wrong answer on a run where the
peers happened to arrive in a convenient order. Treat the checklist as part of the acceptance
criteria in `## How to verify any of this`, not as design advice.

## Your flags outlive the graph

The four invariants above are stated per launch, and a captured graph breaks that framing in two
ways at once: **a replay does not re-run whatever initialized the protocol state, and any host
scalar the protocol read at launch is baked into the graph at capture time.** Both are invisible in
every eager test, because the failure needs a *second* replay to appear. Two independent production
stacks met this and answered it in opposite directions, which is what makes the constraint the
invariant rather than either answer:

- **Advance the state on the device.** One ROCm C++ all-reduce keeps its per-block flag *color* in
  device memory (`uint32_t* d_flag_color`), reads it at kernel entry and writes it back at exit,
  and says why in the source: a host scalar would be baked in, so every replay would reuse one
  color and invariant 3's wraparound question would be answered wrongly from replay two onward.
  Its initial color is `1`, not `0`, so it cannot collide with the just-memset flag buffer — that
  is invariant 3 and invariant 4 landing on the same word once a graph overlaps their lifetimes.
- **Forbid the reset.** A Gluon TP8 collective adapter states it as a module-level contract: the
  opaque synchronization state is initialized once and must never be reset between layers or graph
  replays. Invariant 4's owner therefore runs **outside** capture, once per process — not per layer
  and not per replay.

The same constraint reaches the peer-pointer table (`## Peer pointers arrive as integers`). A graph
kernel node retains the *address* of the row it was captured with and re-reads that row's contents
on every replay, so the table has to be a stable allocation whose rows are filled in **after**
capture — IPC handle registration is not legal inside capture — rather than a pointer value handed
to the launch.

Three questions to add to the checklist above whenever this kernel may be captured. None of them is
a performance question:

1. **Is any protocol constant a host scalar at launch?** Under capture it is now a constant of the
   graph. Move it to device memory, or make it a stable pointer whose contents are re-read.
2. **Does anything reset flags between rounds?** A reset that ran per launch does not run per
   replay. Either advance monotonically on the device, or declare the state process-lifetime and
   never reset it.
3. **Was the peer-pointer table populated before capture?** If registration is illegal inside
   capture it cannot have been, so the row must *exist* at capture and be filled later.

Whether this kernel is capture-safe is a property of the kernel, not of the harness, so record it
beside the world size (invariant 4) — `unknown` is an acceptable value and an absent one is not.

## Claiming work instead of barriering

Where the peers are producing at different rates, a barrier costs you the slowest one on every
step. A claim protocol costs you nothing when the work is already balanced:

```python
# each claimer takes a tile by flipping its bit; the RMW returns the OLD word
old = gl.atomic_or(bitset_ptr + word, 1 << bit, sem="acq_rel", scope="sys")
mine = (old & (1 << bit)) == 0          # I flipped it from 0 -> 1, so it is mine
```

The property that makes this work is that every atomic RMW **returns the value before the
operation**, so the claim and the test are one round trip and cannot interleave. `atomic_cas` is
the version for a claim that carries a value rather than a bit.

Three rules keep it honest:

1. **A failed claim must not retry the same slot forever.** Advance to the next candidate. A retry
   loop on a contended slot is a spin with all of the costs above and none of the progress.
2. **`sem="acq_rel"` on the claim itself**, because the claim both publishes (I took this) and
   consumes (what was there). This is the one place on the page where the default `sem` is right
   and only the `scope` needs changing.
3. **The bitset is shared mutable state across devices**, so it needs the same lifetime discipline
   as the payload buffers — zeroed by someone, exactly once, before the launch that reads it.

## Peer pointers arrive as integers

The peer base addresses come from the host as an ordinary integer tensor and become dereferenceable
through `gl.pointer_type` — the mechanics, and the three ways the cast bites, are in
`../gluon/imports-and-launching.md ## Pointers as data (gl.pointer_type)`. Two things are specific to the
multi-device case:

- **A peer address is opaque to every check you have.** It is not in this process's allocation
  table; masks, bounds, and `gl.max_contiguous` say nothing about it. A stale entry — a peer that
  reallocated, a table built for a different world size — faults at the dereference, deep inside a
  kernel, with a message about an illegal access and no indication which entry was wrong.
- **Build the table once on the host and treat it as read-only in the kernel.** A table the kernel
  also writes needs the full publish/consume discipline above applied to the table itself, which
  is a second collective protocol wrapped around the first.

## How to verify any of this

Correct on every launch you ran is the expected appearance of a broken protocol here, so the
checks are structural rather than numeric.

### Reviewing someone's maintenance: two steps, and the first one is not a grep

Auditing this page's rules is itself a place to get a wrong answer, in **both** directions, and
the reason is that the source and the disassembly answer different questions.

1. **Ask what the source wrote as the signal, before counting anything.** A publish through
   `gl.atomic_*` with `sem=` and `scope=` asking for the maintenance is a publish where the
   backend emits it and the kernel author writes nothing — so **zero explicit `buffer_wbl2` in the
   source is the correct shape there, not a finding.** Grepping a tree for maintenance
   instructions and reporting the files with none will flag every correct compiler-path
   implementation, and "fixing" one adds a second writeback on top of the emitted sequence. The
   count is only a defect signal once step 1 has established that the signal was hand-written in
   an asm block, or was not an atomic at all — rows 3 and 4 of `### Who issues the maintenance`.
2. **Then read the disassembly, where the sequence has to be there either way.** A
   `sem="release", scope="sys"` publish must show `buffer_wbl2 sc0 sc1` and the following
   `s_waitcnt`; a consume must show `buffer_inv sc0 sc1`. Absence here *is* a finding whichever
   path produced it: on the intrinsic path it means the scope silently defaulted somewhere, and on
   the hand-written path it means nobody supplied what the compiler was never asked for.

The asymmetry is worth stating because the two failure directions are not equally visible. A
missing maintenance sequence produces a rare, ordering-dependent wrong answer; a redundant one
produces correct results and a slower kernel, so it survives review and is never attributed.

Then the structural checks:

- **Race-test, do not smoke-test**
  (`../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`).
  Vary the peer arrival order if you can — the bug class here is an ordering assumption that holds
  whenever one side happens to be slower. **A green race-test says no race fired, not that the
  ordering is sufficient**: an absent acquire changes the outcome of no interleaving that occurred,
  it only admits ones that did not — which is why step 2 above reads the disassembly instead of
  accumulating passes.
- **Test at a world size you did not develop at.** Most of the failures above are latent at TP=2
  and immediate at TP=8, because two peers rarely expose an ordering assumption that four do.
- **Record what you could not check.** Nothing on this page has been run on hardware; a claim you
  promote from source-proven to measured belongs in the build's record with the probe beside it
  (`../method/triage.md`).
