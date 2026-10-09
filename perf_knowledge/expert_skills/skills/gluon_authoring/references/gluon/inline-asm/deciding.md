# Do I need inline asm here? — four evidence lanes

Part of `../inline-asm-reference.md`. The table in that page's `## Before you reach for it` rules
out the temptations the language already covers. This is the question after that one: *given this
kernel, at this shape, with this evidence, is an asm site the right answer, and which one.*

It is four layers deep on purpose. **Lane 0 carries the decision**; Lanes 1–3 sharpen a Lane-0
hypothesis or tell you which of two candidate sites to write. Lane 0 is tooling-free except for
L0.7, whose four numbers come out of the static `.amdgcn` and still need no GPU.

> A Lane-1/2/3 signal with no Lane-0 rule behind it is **not** a reason to write asm. It is a reason
> to go looking for the structural fact you have not found yet.
>
> **L0.7 is the one rule whose signal is a compiled-artifact reading, and it is gated on L0.0 for
> exactly that reason.** Pressure readings are the easiest signal on this page to over-read: the
> usual cause of register pressure is a headroom problem the kernel created for itself, and asm
> spent there buys back a hole you dug. L0.0 is what separates the two cases.

Every rule is written in the same five parts: **signal → hypothesis → the (M, I) cell it points at →
what to write → how you would know you were wrong.** The last part is not decoration: an asm site
whose falsifier you cannot state is a site you cannot remove later, and the delete-safety rule in
`costs.md` depends on you having written the alternative down.

## What this page can localize, and where it stops

Before the lanes: know which question they answer. Localization runs four levels deep, and **the
third one is where instruments stop.**

```
L1 which kernel  ->  L2 which loop / stage  ->  L3 which line, which value  ->  L4 which class of asm
     reliably            usually                 rarely                          often
```

**The break at L3 is structural, not a tooling gap.** The commonest shape in this whole area is an
empty asm string, which compiles to **zero instructions** — so every PC-attributing instrument has
no address there to sample. A profile sees the *consequence* (where pressure piles up, where the
spill lands, which matrix instruction is waiting) and never the *lever* (the def-use edge you have
to cut).

One exception, and it is the case worth knowing: **when the consequence itself emits instructions,
L3 is reachable.** A spill emits `scratch_*`, those carry source attribution, and aggregating them
by line puts you inside the right basic block — L1.4's route, and the only lane that walks the
whole way down.

Two corollaries, both of which cost time when they are not stated:

> **No hotspot at a site is not evidence that there is nothing to do there.** It is the expected
> appearance of the commonest shape on these pages.

> **The hottest source line is usually the *consumer*.** A matrix instruction is where pressure is
> *spent*, not where it is *created*. The site you want sits on the edge between a producer and its
> consumer, which is a data-flow fact and not a profile reading.

So read the lanes this way: **Lane 0 and Lane 1 put you in the right basic block; data-flow
reasoning puts you on the right edge.** Where a rule below needs that second half, it says so.

## Lane 0 — structural facts, readable from the kernel alone

No compiler, no profiler, no audit. These are properties of the kernel's contract, and they are the
layer that decides. Work down the list; the first one that fires is your answer.

**L0.0 — the headroom test. A whole-kernel precondition; run it before any site-level rule, and
run it again before you believe a pressure signal.** *Signal:* does this kernel still have room to
move work around? Three questions, none of which needs a profiler:

1. **Is a launch parameter holding the kernel under a cliff it does not need to be under?** An
   unroll factor, a stage count, a tile size or a warp count can pin occupancy or force a spill all
   by itself. Sweep the parameter and re-read the spill count before you conclude the kernel needs
   more registers than it has.
2. **Is anything round-tripping through global or shared memory that could be register-resident?**
   An accumulator that is written out and re-read once per tile, while the register file is half
   empty, is headroom lying on the floor.
3. **Does the kernel schedule its own memory pipeline at all** — a commit/wait group structure, a
   rotating shared-memory index, or a hand-written wait — **or neither?**

*Hypothesis:* where any of the three has slack, a pressure or schedule symptom is usually a
self-inflicted one, and **placement is only purchasable where there is a pipeline to place things
in.** *Cell:* gates **I-D** only. *Write:* nothing — this rule removes candidates and never adds
one. *Wrong if:* you sweep the parameter and the spill does not move, the round-trip turns out to
be load-bearing across waves, or the kernel schedules itself through a construct question 3 does
not name. Widen the question before trusting a negative.

**What it gates, and what it does not.** I-C (a missing instruction) and I-A / I-B (an answer or an
ordering) are **not** subject to this rule — a wave-collective is missing whether or not there is
headroom anywhere. **Only the "nothing visible" population is gated**, and L0.7 below will not fire
until this rule has been run.

**What this rule costs, stated so "no headroom" is not declared cheaply.** It is three questions, but
answering them is a search, not a lookup. A rebuild that closes its gap this way can spend many
variants across many rounds, several of them losing, before the parameter that was causing the
spill falls out. A single sweep of one knob that does not move the spill count is
**not** this rule returning "no headroom" — it is question 1 answered for one knob. Budget for the
search, and if you cannot afford it, record that you skipped the rule rather than recording that it
returned nothing. The asymmetry is what makes this worth paying for: declaring "no headroom" too
early buys a site that patches a hole you dug, and that site then has to be owned, shape-binned and
carried through every upgrade.

**Why it is first.** Blind rebuilds — strip the asm, allow only upstream constructs back — split
into two outcomes, and the split is what this rule detects. **Where headroom existed, the gap closed
with no asm at all**: a launch parameter was itself causing the spill the asm was patching, or an
accumulator was being staged through memory while the register file sat half empty. **Where no
headroom was left** — already at one wave per SIMD, an ablation decomposing additively with no
overlap to recover, and every independent route to more occupancy measuring slower — asm was the
only lever remaining. Both outcomes occur; assuming the second without testing for the first is how
a site gets written to patch a hole you dug.

**L0.1 — the order-versus-data test.** *Signal:* two operations whose **relative order** is part of
correctness, with no value flowing from one to the other in the Gluon source. *Hypothesis:* the
compiler is under no obligation to keep them apart, because dependence is the only thing it tracks.
*Cell:* **I-B**, mechanism M0-tie or M1. *Write:* first look for the binding — `gl.barrier()`,
`commit_group` / `wait_group` (`../pipeline-reference.md` ## Getting `wait_group` right). Reach for
asm only when the order you need is *inside* one wave and *below* a group boundary. The canonical
case is draining this wave's LDS **readers** before a buffer slot is refilled: copy completion and
reader completion are distinct obligations, and `wait_group` covers only the producer side.
*Wrong if:* a value does flow between the two operations. Then rebinding the result is the whole
fix, the site is I-D at best, and deleting it changes nothing you can observe.

**L0.2 — the cross-CTA / cross-device test.** *Signal:* correctness depends on memory written by
another CTA or another device, and the handshake needs either a **loop** (poll until a peer's
counter moves) or a cache scope the binding will not emit. *Hypothesis:* you need a program, not an
instruction. *Cell:* **M3 / I-A** — protocol block. *Write:* the acquire/release skeleton and its
preconditions (`classes-sync.md`). *Wrong if:* `gl.atomic_*(sem=, scope="sys")` expresses both
halves — settle that against `../../workloads/collective.md ## Publish and consume` before writing a
line of asm, because that page enumerates the cases and this one does not. **A protocol block
written where an atomic would have done is the most expensive thing on this page to own.**

**L0.3 — the machine-mode test.** *Signal:* the numeric answer depends on a **wave-persistent mode
bit** — on this target, the `HW_REG_MODE` bit selecting narrow-float conversion behaviour.
*Hypothesis:* there is no Gluon surface for it and there never will be. *Cell:* **I-A / M1**, always
as a bracket. *Write:* the save/restore pair, each half anchored to a live value (`classes.md`).
*Wrong if:* the numeric difference you are chasing can be produced by arithmetic — a clamp, a
`gl.where`, a different rounding mode on a conversion you already call. **Mode state is not a way to
spell arithmetic you did not want to write.**

**L0.4 — the wave-algorithm test.** *Signal:* the algorithm is defined in terms of **lanes** —
ballot a predicate, elect the lowest set lane, compute a lane's rank, move a value to or from a
named lane, exchange a register-index bit with a lane-index bit. *Hypothesis:* there is no
tensor-level spelling on either tier. *Cell:* **I-C ∩ cross-lane** (`classes-lane-register.md`).
*Write:* the construction that matches — A for a ballot/elect bijection, B for a predicate operand,
C for a packed swap. *Wrong if:* the thing you want is a reduction over a tensor axis and you can
choose a layout that maps that axis onto lanes. Then `gl.max` / `gl.sum` over the axis is the answer
and the asm is a lateral move with preconditions attached. **The two are not exclusive:** the common
shape is a tensor-level reduction over the axis first, dropping into a lane ladder only for the
cross-lane remainder.

**L0.5 — the register-file / adjacency test.** *Signal:* a **downstream instruction form** you need
requires an operand in a particular register file, or requires two values in adjacent registers —
scalar-base-plus-vector-offset addressing, a 64-bit mnemonic, a packed-pair FMA. *Hypothesis:*
nothing at the tensor level arranges that. *Cell:* **I-C ∩ M0** (`classes-lane-register.md`).
*Write:* `"=s,0"` for the file, a 64-bit `dtype` with a tie for the pair. *Wrong if:* **you cannot
name the *proof* that the value is wave-uniform.** `"=s,…"` is an assertion the compiler cannot
check; trace the operand back to a `program_id`, a `gl.constexpr`, or a `static_range` index. If it
traces back to a `gl.arange`, the pin is a lie and the wave silently takes lane 0.

**L0.6 — the instruction-gap test.** *Signal:* the silicon has an instruction and the binding does
not expose it — the checkable form is `hasattr(rocdl, '<name>')` returning `False` while the
mnemonic assembles for your `-mcpu` (`../../hardware/cdna3-gfx942.md`). *Hypothesis:* a wrapper gap,
not a silicon one. *Cell:* **I-C / M1**. *Write:* the one instruction, `is_pure=True`, **and the
`else:` arm that computes the same thing in plain Gluon, in the same commit.** *Wrong if:* the
instruction carries a caller-side numerical contract you have not reproduced — base-2
transcendentals, an unrefined reciprocal, a comparator borrowed from another type. `classes.md`
lists the recurring ones; **the failure is silent and numeric.**

**L0.7 — the live-range test. Conditional: it does not fire until L0.0 has been run and found no
headroom.** *Signal:* three readings out of the static `.amdgcn`, no GPU and no profiler, **in this
order** — (a) the spill-slot count and private-segment size, plus a census of `scratch_*`
instructions: nonzero is unambiguous and needs no interpretation; (b) the accumulation-register
count going from zero to positive while the matrix-instruction count is unchanged, which is the
allocator moving a value into the other register file rather than the kernel asking for more;
(c) only then the register-demand figure, and **only when the spill count is zero** — a kernel that
spilled has had its demand clamped to a budget, so the number you are reading is the cap and not the
demand. **Do not read an occupancy step off the descriptor's allocation field here**; that
derivation has rounding, an LDS term and a waves-per-workgroup conversion attached to it, and it
lives in `../../hardware/planning-constants.md ## VGPR / occupancy thresholds`. *Hypothesis:* a value is live longer than it needs to be and the allocator is
paying for it — in accumulation registers, in scratch, or in an occupancy step. *Cell:* **I-D ∩
M0-tie**. *Write:* one empty tied identity, and **pin the longest-lived value rather than the
hottest instruction** — structurally this is almost always the accumulator carried across the loop
back-edge and consumed by `dot` / `mfma`. Its position is not a choice: **after that value's last
update and before its next use.** Derive `pack` and the slot count with the arity law rather than
copying them (`shape-keying.md ### The arity law`). Then ask the guard question: on the first or
last iteration, does the pinned value still have a later use? If not, put the site under a
compile-time guard — no profile reading corresponds to that guard, and omitting it buys a copy you
did not want. *Wrong if:* the normalized-disassembly hash is unchanged against the arm without the
site, which means the edit never reached machine code and the site is inert rather than subtle
(`costs.md ## Verifying that it worked`, step 1.5). **Or** the spill count moves the wrong way: a
tie that pins a whole fragment live is itself a way to *create* pressure, so **removing** a tie is
as legitimate a response to this signal as adding one.

**Three warnings, because each reverses the naive reading.**

- **The binding may already own this.** Before writing a tie to steer an accumulator's register
  file, check whether the matrix builtin on your build takes a register-class argument. Where it
  does, that is the supported spelling and this rule does not apply.
- **Do not go to the hottest line.** The instrument points at the consumer; the site sits on the
  producer side of the edge.
- **Do not size the edit from one site.** Where several ties share one budget, the counter tells
  you the budget is blown and **cannot tell you which claimant to evict**
  (`classifying.md ## BUDGET-GROUP`).

**Why the L0.0 gate is part of the rule and not advice.** The signal here is a compiled-artifact
reading, which makes this look like a Lane-1 rule, and the population it leads to is the largest on
this page. Ungated, "there is pressure" becomes a licence, and the usual cause of pressure is
headroom the kernel gave away somewhere else. Gated, the rule only fires where moving work around
has already been tried and found not to be available.

**L0.8 — nothing above fired.** Then do not write asm. This is the common case and it stays the
common case. A useful calibration: among kernels that *do* use inline asm, a large share of the
sites emit **no instruction at all** — they exist only to constrain the compiler. That is a
statement about how narrow the remaining population is, not a licence to add to it.

**Record the four things that make the decision re-checkable**, next to the kernel, at the moment
you take it: the **target**, the **compiler version**, the **launch geometry** (`num_warps`, wave
size) and the **shape bin** it was derived for. Every derivation in `shape-keying.md` is a function
of at least one of those four, and **none of them is visible in the asm text.**

## Lane 1 — the compiled artifact

Dump first, then read: `kernel_workflow/scripts/kernel_tools/dump_ir.sh --variant <name> --out <ir_dir>` (pack shim: `scripts/dump_ir.sh`) writes
`<variant>.{ttgir,llir,amdgcn,s}`, and `TRITON_ALWAYS_COMPILE=1` is what you need when a
`@gluon.jit` kernel comes back as a cache hit. The hot-loop reading procedure — relaxed versus
full-drain waits, barriers per iteration, `s_nop` census, VGPR/AGPR/spill from the kernel descriptor
— is owned by `../../tile-programming/compiler-contract.md`. Do not re-derive it here; the rules
below say only what each of its signals means *for this page*.

**L1.1 — a full drain where you expected a relaxed one.** *Signal:* a wait classified as `cnt == 0`
at a point where you know only some of the traffic matters, or an `s_waitcnt` sitting *before* the
loads it is supposed to cover. *Hypothesis:* the wait-count pass cannot see a dependence you know
exists, so it is being conservative. *Cell:* **I-D / M0-tie** (make the dependence visible) or
**I-B / M1** (`s_waitcnt vmcnt(N)`, hand-counted). *Write:* **prefer the M0 form** — tie the result
of the read through an empty asm with `~{memory}` and let the backend place the wait for you. It
needs no count, so it cannot go stale. *Wrong if:* the dump after the change shows the same wait, in
the same place, with the same count. Then your model of the dependence was wrong; revert rather than
escalating to a hand-counted wait.

**L1.2 — a memory operation that has crossed a point you care about.** *Signal:* the source order
and the `.s` order disagree about a copy issue, a store, or an LDS read relative to a barrier or a
wait. *Hypothesis:* nothing in the IR forbids the motion. *Cell:* **I-D / M0 untied** — the
`("", "=v,~{memory}", [])` marker. *Write:* the marker at the point, with nothing tied, because you
are constraining memory motion and not liveness. *Wrong if:* the motion you saw was a *legal
consequence of a value dependence* you had already broken — then the marker will hold it and you
have bought a schedule constraint you did not need. **And note the coupling:** if a hand-counted
`vmcnt(N)` elsewhere depends on copies being issued where you think they are, this marker is what
protects that count, and removing it turns a correct `vmcnt(4)` into an incorrect one. **The pair is
the hazard, not either site.**

**L1.3 — a wave-uniform value living in a VGPR.** *Signal:* a value you can prove is uniform appears
as a vector operand — a `global_load` with a vector base instead of a scalar base plus a vector
offset, or a literal re-materialised into every instruction. *Hypothesis:* the compiler had no
reason to believe it was uniform. *Cell:* **I-C ∩ M0** (`"=s,0"`) or **I-C / M1**
(`v_readfirstlane_b32`, if the value must actually be *moved* rather than asserted). *Write:* the
pin next to the proof of uniformity, not next to the use. *Wrong if:* after the pin the dump still
shows a vector base, or shows a `v_readfirstlane` the compiler inserted for you. Either means the
value was not uniform and **the pin was silently wrong all along.**

**L1.4 — spills around a large accumulator.** *Signal:* `scratch_*` in the hot loop, or
`next_free_vgpr` / the `.amdgpu_metadata` spill counts moving the wrong way. *Hypothesis:* live
ranges overlap where they need not. *Cell:* **I-D / M0-tie**, widened to the fragment. *Write:* one
tied identity covering the whole per-thread fragment at the point where it must be live (the
generator idiom, `classes-lane-register.md`). ***And the symmetric move is equally available:*** a
tie that pins a whole fragment live is itself a way to create pressure, so **removing** a tie is as
legitimate a response to a spill signal as adding one. *Wrong if:* the spill count does not move, or
moves the wrong way. **This lane gives you a direction to try, never a conclusion.**

**Why a direction and not a conclusion — and the one case where it is nearly both.** The counters
that fire here are whole-kernel numbers and carry no line. What *does* carry a line is the spill
traffic itself: aggregate `scratch_load` / `scratch_store` by source line in the per-instruction
trace, and the peak names the basic block, usually within a few lines of the site. **That is the
only route on this page where an instrument reaches L3**, and it works only because a spill emits
instructions. Where the pressure has not turned into scratch — an accumulator that merely sank into
accumulation registers, say — there is nothing to attribute, and you are on L0.7's data-flow route
instead.

**L1.5 — the instruction you wrote is not in the dump.** That is not an entry signal, it is a
verification failure. Go to `costs.md ## Verifying that it worked`, step 2.

## Lane 2 — counters and timing: our procedure, and its limits

> **Read this before using this lane.** Nothing on these pages establishes that a mechanism pays on
> *your* kernel; every causal claim is `not established` until you produce the measurement named
> beside it. What one A/B campaign over an asm-carrying production pack does license is a **prior**,
> and the prior is worth knowing before you spend a window:
>
> **Most sites move nothing.** Across stub arms of every site in that pack, well under half moved a
> clock past the noise floor at all, and a large minority produced a **byte-identical binary** when
> the site was removed. **Ordered by how often a class carried any measurable effect, commonest
> first: register-class pins and instruction selection, then everything else, with mode writes,
> drains and priority brackets clustered at the bottom.** That ordering is the useful part; the
> fractions are one pack on one box and yours will differ.
>
> The operational consequence is step 0 below, not a change to any rule on this page.

**When this lane is admissible at all.** Only for **I-D** sites, and only for the choice *between*
two already-correct implementations. For I-A, I-B and M3 the question is never "is it better" but
"is it right", and the answer comes from `costs.md ## Verifying that it worked` and from
`../../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`
— not from a counter.

> **A profile that says an I-B site is unnecessary is a profile answering a question you did not
> ask.**

**The procedure.**

0. **Hash before you measure.** Run `costs.md ## Verifying that it worked` step 1.5 on the two
   arms first. It needs no GPU, and an unchanged hash means the edit never reached machine code —
   there is nothing to time, and the site should be removed rather than measured. This step also
   produces the control arm you need in step 6.
1. **Preflight the profiler once**, do not assume PMC is visible — the probe and the fallback are in
   `../../method/profile.md`. Zero dispatches in the kernel trace means PMC is blind on that box and
   you are back to Lane 1, which is a complete answer, not a degraded one.
2. **Build the A/B into the source.** The asm arm and the `else:` arm behind one `gl.constexpr`, so
   the two variants differ by exactly the site and nothing else. This is the same `else:` arm that
   L0.6 told you to write, doing its second job. Without it you are comparing two commits, and a
   commit is not a variable.
3. **Interleave the two arms in one clean window** and record the window's spread alongside the
   numbers (`../../method/benchmark-hygiene.md`). One number from each arm, taken at different times, is
   not a comparison.
4. **Prefer the clock-insensitive discriminators** the pack names for this — MFMA utilisation,
   VALU %, memory-unit stall, issue/dependency wait — over absolute time on a box you do not own
   (`../../method/profile.md`). Counter *names* vary by build; probe them rather than hardcoding
   them (`../../hardware/bound-class-signals.md ## rocprof-compute block-ID cheat-sheet`).
5. **Close the loop in Lane 1.** The standing rule applies unchanged: a change that shows in timing
   but not in the instruction mix has not been attributed yet
   (`../../tile-programming/instruction-scheduling.md ## Acceptance`). For an M0 site there is no
   mnemonic to count, so the Lane-1 leg is the *register* evidence — operand class, spill counts,
   where the backend placed the wait.
6. **Build the noise floor out of an arm whose normalized disassembly is byte-identical to the
   baseline**, not out of repeat count. A comment-only arm has a true delta of zero, so its measured
   spread *is* your resolution limit. It is per kernel and per box; inheriting someone else's floor
   is how a null result gets reported as a win.
7. **A budget-sharing group cannot be measured one site at a time.** Where a resource reading sits a
   few units over a threshold, build the all-off arm and at least one grouped arm as well. Single-
   site arms alone will read a non-monotone response as a monotone one
   (`classifying.md ## BUDGET-GROUP`).
8. **Re-measure across the shapes you actually dispatch**, then ask the harder question: **is the
   axis that decides the benefit even in the dispatcher's signature?** The same construct can be
   worth a few per cent at one end of a range and nothing at the other, keyed on something — a
   context length, a loop trip count — that the shape key selecting the kernel never sees. When that
   happens **no amount of shape specialization can route to it**, and the honest record is the range
   rather than the best cell.

**What this lane can and cannot conclude.** It can conclude "these two correct implementations
differ on this box, this build, this shape, by this much, with this spread". It cannot conclude that
a mechanism is good, or that a cell of the M × I table is worth using. Those remain
**`not established`**; what would settle them is exactly the A/B above, run on the kernel in front
of you, recorded with its four context facts.

## Lane 3 — a source-level audit of *your* kernel

`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py` (pack shim `scripts/asm_loop_audit.py`) reads the compiled `.s`. This lane is the other end: an AST pass over
your own kernel's Python that enumerates its `inline_asm` call sites and their parameters. The
report is what matters, not any particular command, so the rules below are written against **report
contents**.

If you are building the report yourself: match on `inline_asm`, **not** on `gl.inline_asm` — the
helper is reachable through more than one namespace. Keep argument text as expressions rather than
literals. Both shortcuts produce wrong counts rather than errors.

| Report field | What it means | What to do |
| --- | --- | --- |
| `asm` or `constraints` is an **expression**, not a literal | the site is **Mx** — it cannot be classified, and the text differs per instantiation | resolve it at its defining line, **at the shape you are building**, then re-classify. Expect some bodies to be assembled from several interpolated fragments |
| output-slot count ≠ the arity law | `pack`, `dtype` and the constraint string disagree | **stop.** This is the one report line that is a hard error rather than a prompt (`shape-keying.md ## Deriving the parameters`) |
| tie digits are not exactly `0..n-1` in order | a transcription error, or a construct with no established precedent | re-derive |
| `is_pure=True` **and** `~{memory}` present | contradictory intent: purity permits exactly the motion the clobber forbids | one of the two is wrong |
| `is_pure=True` and the result is not rebound | the block is deletable and will be deleted | rebind it, or set `is_pure=False` if the point is its position |
| body names `vcc` / `scc` (or uses an instruction that writes SCC) and the clobber list omits it | a live predicate will be corrupted | add `~{vcc}` / `~{scc}` |
| a braced physical register appears (`=&{v0}`, `~{s0}`) | absolute assignment; portability is now per-kernel | flag it. It survives no change in `pack`, operand count or register budget |
| the site is under a compile-time guard whose test names a tile or shape constant | it is a **shape-binned** site | route to `shape-keying.md ## Deciding the bins` |
| the site is in an `if`-body whose `If` has a non-empty `else` | **delete candidate** — but only a candidate | the mechanical prefilter's precision is *not* 1. Confirm in `costs.md ## Durability and rollback` |
| the site has no `PAIRED-WITH` but the intent rule returned **I-B** | a wait with no issue, or half a bracket | find the partner before you believe anything the site says about what it guarantees |
| no `gl.static_assert` and no `convert_layout(..., assert_trivial=...)` nearby | the preconditions are unasserted | assert them. Widespread practice is *not* to, and that is a gap to close rather than a convention to copy |

## The flow, top to bottom

```
START: I think this kernel needs inline asm.

1. Does the table in `## Before you reach for it` already cover it?
   YES -> use the binding. STOP.
   NO  -> continue.

2. LANE 0, in order. First hit decides.
   2.0 HEADROOM (whole-kernel; gates the I-D population only).
       Is a launch parameter pinning occupancy / forcing the spill?
       Is anything round-tripping through memory that could stay in registers?
       Does the kernel schedule its own pipeline at all?
       ANY slack -> spend it FIRST. An asm site here patches a hole you dug.
       No slack   -> 2.7 is unlocked. Either way continue to 2.1.
   2.1 Order matters, no data dependence between the two ops?
       -> binding first (barrier / wait_group). Still uncovered, and the order is
          intra-wave and sub-group?           -> I-B, M0-tie or M1.        go to 3.
   2.2 Cross-CTA / cross-device state, needing a poll loop or a cache scope?
       -> settle against `../../workloads/collective.md` first.
          Still uncovered?                    -> I-A, M3 protocol block.   go to 3.
   2.3 Answer depends on a wave-persistent mode bit?
       -> I-A, M1, ALWAYS a bracket.                                       go to 3.
   2.4 Algorithm is defined over lanes (ballot / elect / rank / lane exchange)?
       -> I-C, cross-lane. Pick construction A, B or C.                    go to 3.
   2.5 A downstream instruction form needs a register file or adjacency?
       -> can you NAME the proof of uniformity?
          NO  -> do not pin. STOP.
          YES -> I-C, M0.                                                  go to 3.
   2.6 hasattr(rocdl, mnemonic) is False and the mnemonic assembles?
       -> I-C, M1. WRITE THE else: ARM IN THE SAME COMMIT.                 go to 3.
   2.7 LIVE RANGE -- only if 2.0 found no slack.
       Static .amdgcn: next_free_vgpr over a step / agpr_count 0->+ /
       spill count / scratch_* count.
       -> does the matrix builtin already take a register-class argument?
          YES -> use it. STOP.
          NO  -> I-D, M0-tie. Pin the LONGEST-LIVED value (the loop-carried
                 accumulator), not the hottest line. Position is forced:
                 after its last update, before its next use.          go to 3.
   2.8 Nothing fired -> DO NOT WRITE ASM. STOP.

3. Record: target, compiler version, launch geometry, shape bin.

4. DERIVE (never copy):
   pack  = fragment_elements / (num_warps * wave_size)     + static_assert
   S     = pack                     if sizeof(dtype) >= 4
         = ceil(pack*sizeof/4)      if sizeof(dtype) <  4
   outputs = sum S over the dtype tuple
   inputs  = sum S over args
   ties    = 0..outputs-1, in order
   $n counts SLOTS, outputs first
   is_pure = False unless the block is purely a function of its inputs AND
             you accept CSE + hoist + sink + delete-when-unused
   ~{memory} iff the body touches memory or motion is the point
             (NEVER together with is_pure=True)
   ~{scc}/~{vcc} iff the body names or implicitly writes them
   =&  iff any output is written before the last read of any input

5. BIN: re-run step 4 at both extreme shapes.
   Any derived value differs, or any precondition (wave-private / log2(tile)
   constant / copy count / lane bijection) differs?
   YES -> two bins. Vary the least invasive thing; keep ONE derivation helper;
          static_assert the legal values; test at both boundary shapes.
   NO  -> one implementation.

6. VERIFY: trace -> TRITON_ALWAYS_COMPILE=1 + dump_ir.sh
   -> HASH the normalized disassembly against the arm without the site.
      IDENTICAL -> the edit never reached machine code. DELETE IT. STOP.
   -> count the mnemonic
   -> check placement between the two instructions you named in advance
   -> for M0, diff the register/wait/spill evidence against the else: arm
   -> deletion probe -> intent-specific correctness test (I-A state case /
      I-B race-test / I-C numeric diff vs else: / I-D bit-identical).
   Not there?        -> is_pure, rebinding, or the bin. Fix and re-dump.
   There but floated -> the anchor is missing (tie / clobber / consumption).
   I-D but not bit-identical -> YOUR CLASSIFICATION IS WRONG. Re-classify.

7. LANE 2 (measurement) is admissible ONLY for I-D, only between two already
   correct arms, only as an interleaved A/B in one window, and only with the
   Lane-1 leg closed. Noise floor comes from a byte-identical control arm.
   Sites sharing one budget need an all-off arm too, never single-site arms.

8. RECORD the removal path: the else: arm, and the mnemonic count as a CI check.
```
