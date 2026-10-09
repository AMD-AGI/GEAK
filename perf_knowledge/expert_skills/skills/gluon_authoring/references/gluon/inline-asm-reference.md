# Gluon Inline Assembly (gfx950 / gfx942)

## Three jobs, one syntax

`gl.inline_asm_elementwise` is a single call used for three unrelated purposes. They have different
removal tests, different evidence standards, and different failure modes, and the commonest way to
misread a site is to assume it was written for the purpose you came looking for. Decide which one
you are holding before anything else; `inline-asm/classifying.md ## Axis I — intent` decides it from
the site itself, by one question — *delete it, rebind its output to its input, and ask what changes.*

| purpose | what the site buys | what breaks if you delete it | intent bins |
| --- | --- | --- | --- |
| **Vocabulary** — the ISA can say something the language cannot | an instruction with no Gluon spelling | same answer, different and usually worse instructions | I-C |
| **Semantics and synchronization** — correctness, not speed | an ordering obligation, a machine-state bit, a coherence qualifier | **the answer changes**, or a race appears intermittently | I-A, I-B |
| **Placement** — a constraint on the register allocator and the scheduler | a compiler *decision*; the site itself emits no instruction | nothing visible — same answer, same guarantees, different allocation and schedule | I-D |

Three consequences worth carrying:

- **Only the first is an escape hatch.** Vocabulary sites keep an `else:` arm in plain Gluon more
  often than not, and become removable the day the binding grows the missing surface.
- **The second is not optional and not a tuning knob.** A drain whose removal changes the numbers is
  load-bearing at every shape. Do not A/B it for speed; verify it for correctness.
- **The third is the one that looks like nothing.** An empty body (`asm=""`) computes the identity
  and compiles to zero instructions, so every PC-attributing instrument reads nothing at its
  address — the effect lands on its neighbours. It is also the only one of the three whose value is
  a property of the *surrounding code* rather than of the site.

Two pages already send readers here for the first purpose without saying how — the gfx942 int4
dequant binding gap (`../hardware/cdna3-gfx942.md`) and the same gap restated in
`../tile-programming/low-precision.md` — and a language ceiling recorded in
`../hardware/capability-matrix.md` is only a ceiling if this page has no answer for it.

Read `../hardware/isa-mechanisms.md` for what the instructions do. This page is about getting them
into a Gluon kernel and about the far more common case where you should not.

**This reference carries no performance number for your kernel, and you should not add one.** The
scope of that disclaimer is *your* kernel, not the placement purpose itself: a placement site has no
semantics to justify it, so a measurement is the only thing that can, and the flow in
`inline-asm/deciding.md` is built to produce one. What is ruled out is importing a number from
somewhere else — gain on a placement site is a property of site × register budget × surrounding
code, and the same site has been measured at four materially different values under four contexts.
Every "purpose" stated here comes from instruction semantics, from stated intent, or from
control-flow structure. Where a question needs measurement it is written `not established` together
with the measurement that would settle it. One measured prior — how often each mechanism class
carried any effect at all — is stated, with its scope attached, in `inline-asm/deciding.md` Lane 2;
it exists to tell you what to expect before you spend a measurement window, and it settles nothing.

## Where to go

This reference is split. The material behind the class headings below lives in `inline-asm/`.

| Read this | When |
| --- | --- |
| `inline-asm/deciding.md` | **You are deciding whether to write one.** Four evidence lanes, and the top-to-bottom flow. Start here |
| `inline-asm/classifying.md` | **You are holding one someone else wrote.** The mechanism × intent axes, `PAIRED-WITH`, the two mechanical cross-checks |
| `inline-asm/shape-keying.md` | **You are writing the call.** Parameter derivations — `pack`, the arity law, `is_pure`, `dtype`, constraint letters — and how to decide bin boundaries |
| `inline-asm/classes.md` | Instruction selection, scheduling control, machine state |
| `inline-asm/classes-sync.md` | Synchronization, and protocol blocks |
| `inline-asm/classes-lane-register.md` | Cross-lane collectives, and register-class control |
| `inline-asm/field-guide.md` | Reverse index, ranked traps, idiom catalogue. **Use when reading unknown asm** |
| `inline-asm/misreading.md` | **A measurement disagrees with what you expected.** Readings that invert — resource counts, saturation rates, occupancy, contention — and which source to trust for registers, spill and accumulation registers |
| `inline-asm/costs.md` | What it costs, the verification loop, a worked derivation, durability and the removal test |

**How to audit a tree for these sites.** Parse, do not grep. Walk `ast.parse` and select each
`ast.Call` whose callee name *contains* `inline_asm` — the helper can be reached through either the
Gluon or the Triton namespace, so a callee-qualified pattern like `gl.inline_asm` silently drops the
`tl.`-spelled ones. Two further traps, both of which produce wrong counts rather than errors:
`ast.walk` also visits a chained `.to(...)` call whose unparsed callee *contains* the inner call's
text, which double-counts; and a call may be reached through a multi-line `import`, which a
line-oriented pattern reports as a site.

The six `## Class N` headings are cited by heading text from other pages in this pack. They are kept
here on purpose as routing entries; the material itself is in the chapter files.

## Before you reach for it: what the language already emits

Most reasons to want inline asm on this target are already expressed, and the emitted sequence is
more likely to be right than a hand-written one:

| The temptation | Already expressed as | Note |
| --- | --- | --- |
| cache-scope bits on an atomic (`sc0` / `sc1`) | `gl.atomic_*(..., sem=, scope=)` | `scope="sys"` emits the full system-scope sequence including the L2 writeback and the wait |
| an L2 writeback before publishing a flag | the `sem="release"` atomic that publishes it | bare `buffer_wbl2` is only for a publish the **compiler does not see as a release** — which is not the same as "not an atomic" |
| priority toggling around an MFMA block | `warp_pipeline_stage` cluster markers | lowers to `s_setprio` + `sched_barrier`, 3.7.0+ (`pipeline-reference.md`) |
| waiting for outstanding async copies | `commit_group` / `wait_group` | hand-written `s_waitcnt` loses the compiler's dependency checking (`## Class 4 — synchronization`) |
| skipping work on a warp-uniform condition | `gl.warp_predicate` where the build has it | vendor-fork only upstream-absent; see `../hardware/capability-matrix.md` |
| an ordering guarantee between LDS reads and the next copy | `gl.barrier()` + `wait_group` | an empty asm with `~{memory}` is **not** this (`## Class 2 — scheduling control`) |
| a *drain* of this wave's LDS readers before a slot is refilled | there is no binding call — this one is real | copy completion and reader completion are distinct obligations; `wait_group` covers the producer only |

Use this reference when the answer above is genuinely "the language has no spelling for this", not
when it is "I do not know the spelling yet". The routing question that comes *after* this table —
given this kernel, at this shape, is an asm site the right answer and which one — is
`inline-asm/deciding.md`.

### The complement: gaps that blind rebuilds actually hit

The table above is the false alarms. This one is the opposite list — capabilities that three blind
rebuilds of asm-carrying kernels, allowed only upstream Gluon constructs, could not reconstruct.
Short on purpose: **two of those three rebuilds needed nothing from this list**, and closed their
gap by recovering headroom instead (`inline-asm/deciding.md` L0.0). Two later rebuilds on a
different archetype contributed the last row, and one of them beat its incumbent outright while
needing none of the others — so treat this table as the residue after headroom is exhausted, not as
a list of things you will need.

| The gap | Why the language does not reach it |
| --- | --- |
| **Split-K across warps** | the matrix layout's warp partition divides the *output*, and its product is pinned to `num_warps` — so there is no spelling that gives two warps the same output tile and different slices of the reduction axis |
| **Instruction order within a wave** | no scheduling barrier, no scheduling-group primitive, no issue-priority surface. Note `schedule_hint` is **not** this: on 3.8.0 it is a declaration with no effect, and its own source says so |
| **An LDS-free fused matrix epilogue store** | coalescing the result of a matrix instruction into wide stores needs a lane permutation, and the tensor level cannot express one without a shared-memory round trip |
| **Asserting a value is wave-uniform** | there is no construct that pins an SSA value into a scalar register, and no equivalent of an assumption hint the compiler will propagate |
| **Keeping a long-lived value out of the accumulation registers** | register class is an allocator decision with no surface: when unified demand crosses an occupancy step the allocator may sink a long live range into AGPRs, and there is no spelling for "arch registers only". This is the gap the **placement** purpose fills, and two independent blind rebuilds named it unprompted |

**Read the second column, not the first.** Each row is a *structural* reason, so it survives until
the binding grows the missing surface — at which point the site becomes removable and the `else:`
arm you kept is already there.

## The only door: gl.inline_asm_elementwise

```python
out = gl.inline_asm_elementwise(
    asm="...",              # target assembly text, $0.. operand placeholders
    constraints="=v,v",     # LLVM constraint string: outputs first, then inputs
    args=[x],               # input tensors, implicitly broadcast to a common shape
    dtype=gl.float32,       # element type(s) of the result; a tuple gives a tuple of tensors
    is_pure=True,           # "this block has no side effects"
    pack=1,                 # elements handled per asm instance
)
```

Available on all four versions (checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0), re-exported into `gl` from
Triton core, so it behaves exactly as the plain-Triton builtin does — and that is literally true,
not a figure of speech: the same idiom appears byte-identically in plain-Triton files spelled
`tl.inline_asm_elementwise`. "The only door" is a statement about *what the door is*, not about
which namespace you spell it in.

**And it is a statement about the versions above, with an expiry date.** After 3.8.0 upstream added
a **second, Gluon-only** entry point (`gl.inline_asm` / `ttg.inline_asm`) with a deliberately
different contract: **one invocation per thread** rather than per packed group — so no `pack` and no
arity law — **results are optional**, explicit distributed layouts are required for any results, the
asm string may itself be generated at trace time, and memory-descriptor operands are allowed but
must be impure. Neither docstring cross-references the other. **On the versions this pack checks,
everything below stands unchanged; on a newer build, the dummy-result idiom and the arity law are
properties of *this* door rather than of inline asm in general** — so if what you need is
zero-result, side-effecting, per-thread asm, check whether your build has the newer door before
importing an idiom from this page into a call spelled the other way.

Six properties decide whether your asm is usable here, and three of them surprise people:

- **It is elementwise, and it is the only door.** There is no scalar or statement-level inline asm
  in Gluon. Everything in this reference — including the machine-state and synchronization classes
  that have nothing to do with per-element data — has to be smuggled through a tensor map.
- **It must return at least one tensor.** Empty `dtype` is rejected. For a side-effect-only block
  you return a dummy tensor of arbitrary type and ignore it; upstream documents this as the
  **official workaround** — its own words — and notes an unused result should not cost anything.
  **Read "official", not "intended":** the restriction falls out of the op carrying MLIR's
  elementwise trait, the docstring says the op does not *currently* support what you want, and a
  patch lifting it was written and reviewed with the maintainer stating he preferred lifting it. It
  was dropped for unrelated reasons, not because anyone argued the dummy result was better.
  **Lifting it would be backward compatible**, so it can disappear without being called a breaking
  change — and the newer sibling door above already does not have it.
- **Which elements share an asm instance is unspecified.** `pack` says how many elements one
  instance handles, not *which* ones. Never write asm that assumes a particular lane, a particular
  neighbour, or any cross-element relationship.
  **One exception, and it is load-bearing:** the *bit ordering within* a pack is **not** unspecified
  — lower element indices occupy the lower bits of the packed value, and code relies on it. Upstream
  states this in exactly one issue comment and in no docstring, op description or doc page, so treat
  it as true-but-undocumented: safe to depend on, not safe to assume the next reader knows.
  **There is a legal way to get the lane anyway, and it is the difference between assuming and
  passing.** Materialize the lane index as a *tensor* — `gl.arange` under a layout that makes element
  order equal lane order — and feed it in as an operand to be used as a predicate. Nothing is
  assumed about the mapping, because the mapping is now a value the compiler produced. That is what
  makes the cross-lane class writable at all; its preconditions turn the layout assumption into
  something `gl.static_assert` can refuse at compile time
  (`## Class 5 — cross-lane and wave-collective`).
- **`is_pure=True` is safe only where the result is consumed downstream.** The test is *whether
  anything reads the result*, not which class the block belongs to — put that question first,
  because the condition is what decides, and reading it as a per-class rule gets both directions
  wrong. The mechanism: `is_pure=True` promises the optimizer the block has no side effects, which
  lets it be CSE'd, sunk, hoisted, or deleted once its result is unused. So a block written for its
  effect and carrying a dummy result nobody reads is simply deleted — which is why most classes want
  `is_pure=False`. But a block whose result *is* consumed can legitimately be `is_pure=True` even
  when it was written to shape codegen.
- **Inputs narrower than 4 bytes are packed into 4-byte registers.** Write the unpack into the asm
  rather than assuming an `i8` arrives in its own register. This is not only a body-writing note: it
  is half of the arity law that decides how many `=` slots your constraint string needs
  (`inline-asm/shape-keying.md ## Deriving the parameters — compute them, never copy them`).
- **`$n` numbering counts outputs first, and `dtype` decides how many outputs there are.** A tuple
  `dtype` gives a tuple of results: `dtype=(gl.uint64, gl.uint32, gl.int32)` means `$0`,`$1`,`$2` are
  outputs and **the first input is `$3`**. This is the single most common transcription error, and
  it has three further traps on top of it. First, `$n` counts **slots**, not tensors — a `pack=8`
  `bfloat16` output is *four* slots, so the first input is `$4`. Second, the tie digit is positional
  in the *constraint list* but refers to an *output index*, so reordering the list silently renumbers
  every `$n`. Third, an argument may legally appear **twice**, once as a vector operand and once as a
  scalar, because the body needs it in both register classes — a tidy-up that de-duplicates the
  argument list breaks the asm.

> **The dummy-result idiom is load-bearing, not a wart.** A side-effecting block needs a result so it
> can exist, and it needs that result *consumed* so it lands where you meant. Feeding the dummy into
> a value the following code actually uses — a tied operand — is what gives the block an ordering
> anchor. Without one you have written an instruction whose placement is at the scheduler's
> discretion, which for an `s_waitcnt` or an `s_setreg` is the whole thing you were trying to
> control.

## Classifying a site: mechanism × intent

Two axes — **M** (read the two strings) × **I** (delete it and rebind: what changes) — replacing a
single-axis reading that is not reproducible. Both are decidable, so two readers land in the same
bin. Also covers `PAIRED-WITH`, the delete-safety prefilter, and the no-claim rule.

→ **`inline-asm/classifying.md`**, and `inline-asm/deciding.md` for the four evidence lanes that
decide whether to write one at all.

## Class 1 — instruction selection

The silicon has the instruction; the binding does not expose it. Intent I-C, usually genuinely pure
— and the instruction is rarely the whole primitive: the caller-side numerical contract lives in
Python and is invisible in the asm text.

→ **`inline-asm/classes.md ## Class 1 — instruction selection`**

## Class 2 — scheduling control

An instruction placed at a specific point for its timing effect. The empty-asm tied form is the
commonest shape in this whole area — and **it is not a fence**.

→ **`inline-asm/classes.md ## Class 2 — scheduling control`**

## Class 3 — machine state

The MODE register. No Gluon surface, wave-persistent, sticky, and `is_pure=False` always.

→ **`inline-asm/classes.md ## Class 3 — machine state`**

## Class 4 — synchronization

A wait or a publish whose granularity the abstraction does not offer: the reader-drain marker,
`s_waitcnt vmcnt(N)`, `buffer_wbl2`, and the dependency-carried drain.

→ **`inline-asm/classes-sync.md ## Class 4 — synchronization`**

## Class 4b — protocol blocks

The body contains a branch, a label, or an EXEC role-split. It is a *program*, not a missing
instruction, and the per-instruction advice elsewhere does not reach it.

→ **`inline-asm/classes-sync.md ## Class 4b — protocol blocks`**

## Class 5 — cross-lane and wave-collective

Ballot, elect, rank, lane exchange. The preconditions are the whole content of the class — and they
attach to the *construction*, not to the class.

→ **`inline-asm/classes-lane-register.md ## Class 5 — cross-lane and wave-collective`**

## Class 6 — register-class control

A value in a particular register file, in adjacent registers, or in a named physical register. Read
this class narrowly: **the mechanism is far more common than the intent.**

→ **`inline-asm/classes-lane-register.md ## Class 6 — register-class control`**

## Shape-keying: inline asm is a per-M specialization

Asm text, opcode, constraint string, `pack`, clobber list and comparison polarity can each be keyed
on a tile constant — sometimes three at once in one call. Holds the parameter derivations, including
the arity law, and the bin-boundary procedure.

→ **`inline-asm/shape-keying.md`**

## Reverse index: I see X in an unknown kernel

By constraint string and by mnemonic: what the author was doing, what must already be true, and the
one thing to go and check.

→ **`inline-asm/field-guide.md ## Reverse index: I see X in an unknown kernel`**

## Traps, ranked by how likely you are to hit one

Twenty, ranked by frequency × silence of the failure × how natural the wrong move is. The first six
are near-certain on any real editing pass.

→ **`inline-asm/field-guide.md ## Traps, ranked by how likely you are to hit one`**

## Idiom catalogue

Twenty-two recurring shapes, one canonical name each, with the aliases they travel under.

→ **`inline-asm/field-guide.md ## Idiom catalogue`**

## What it costs you, on every class

The optimizer stops seeing through the block; portability is per-mnemonic and fails late; durability
is per cell. Includes the verification loop, a worked derivation, the durability table, the one
mechanical removal test, and the open questions.

→ **`inline-asm/costs.md`**

## Readings that point the wrong way

Resource counters answer *does it fit*, never *does it overlap* — so a variant can be smaller on
every axis and slower. Also: which source to trust for registers, spill and accumulation registers,
and why a zero-instruction site is invisible to every PC-attributing instrument.

→ **`inline-asm/misreading.md`**

### Deciding the bins

→ `inline-asm/shape-keying.md ## Deciding the bins — a procedure`

### Durability and rollback — what survives an upgrade

→ `inline-asm/costs.md ## Durability and rollback — what survives an upgrade`

### Verifying that it worked — write, compile, dump, check

→ `inline-asm/costs.md ## Verifying that it worked — write, compile, dump, check`

### Deriving the parameters — compute them, never copy them

→ `inline-asm/shape-keying.md ## Deriving the parameters — compute them, never copy them`

### Do I need inline asm here? — four evidence lanes

→ `inline-asm/deciding.md`
