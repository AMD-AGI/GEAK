# What it costs, how to verify it, and how to get rid of it

Part of `../inline-asm-reference.md`.

## What it costs you, on every class

- **The optimizer stops seeing through the block.** Values entering it are materialized; the
  scheduler works around it instead of through it. On a hot loop the barrier effect can exceed the
  instruction you were trying to reach.
- **Portability is per-mnemonic, and failure is late.** Nothing in the type system knows your asm is
  target-specific.
- **Version durability is per-class.** Instruction selection outlives compiler upgrades; scheduling
  control rarely does. Weigh a scheduling hack accordingly **before** it becomes load-bearing. The
  per-cell version of this is `### Durability and rollback` below.
- **Record the constraint where it will be read.** A kernel that only works on one target, one
  compiler, or one mode setting has a precondition; write it next to the kernel
  (`../../method/triage.md` when the gap is the language's rather than yours).
- **A site is often not auditable alone.** A substantial share of real sites are one half of a pair
  — issue/wait, save/restore, prefetch/finish, release/acquire — and the halves are frequently
  linked by nothing but a tied constraint. Budget for reading the partner, and record `PAIRED-WITH`
  when you write one.
- **The asm is a per-shape artifact like everything else in this stack.** Asm text, opcode,
  constraint arity, `pack`, clobber list and comparison polarity can each be keyed on a tile
  constant. **A site that is correct at one shape is not thereby correct at another, and nothing in
  the type system notices** (`shape-keying.md`).

## Verifying that it worked — write, compile, dump, check

Every class ends with a `**Verify:**` line. This is the loop those lines are instances of. The step
most often skipped is step 2, which is also the one that catches the commonest failure: **a block
that was deleted because nothing consumed its result.**

**Step 0 — does it trace?** Free, and it catches a real class of error before you spend a compile: an
`n` constraint with a runtime value fails here, and so does every `gl.static_assert` you wrote in
`shape-keying.md ## Deriving the parameters`. **If step 0 is silent and you wrote no assertions, you
have not gained information — you have only learned that Python ran.**

**Step 1 — compile and dump.** `kernel_workflow/scripts/kernel_tools/dump_ir.sh --variant <name> --out <ir_dir>` (pack shim: `scripts/dump_ir.sh`) gives you
`<variant>.{ttgir,llir,amdgcn,s}`. For a `@gluon.jit` kernel, **`TRITON_ALWAYS_COMPILE=1`**, or you
will be reading a cache hit from before your edit — **the single most common way to verify the wrong
binary.** Pin one Triton build for the whole comparison
(`../../tile-programming/compiler-contract.md`).

**Step 1.5 — the free step, and it is a proof rather than a hint.** Before reading anything, hash the
*normalized* disassembly of the arm with your site and the arm without it. No GPU, and it costs
about what a directory listing costs.

- **Hashes equal** ⇒ **the edit never reached machine code.** Not weak evidence — decisive, on one
  precondition you check once on your build: that an independent recompile of unchanged source
  reproduces a byte-identical normalized dump. A site whose removal changes nothing in the binary is
  **inert, not subtle**. Delete it and stop; do not spend a measurement window on it.
- **Hashes differ** ⇒ every byte of the difference is attributable to this one edit, which is what
  makes the later readings worth taking.

**Normalization is where this goes wrong, and it fails towards a false positive** — a no-op edit
looking like a real ISA change. Stripping `.loc` directives is not enough, and neither is also
stripping the call-line and declaration-line debug fields. Debug metadata embeds **the arm's own
source filename and path as a string**, and string tables are referenced by byte offset, so an arm
file whose name is one character longer shifts every subsequent offset in the section: two
byte-for-byte identical kernels then hash differently purely because of what they were called.

**The recipe that survives that**, two hashes rather than one:

1. **Drop every debug section, not a chosen list of directives.** At the object level, an
   objcopy-style remove-section by regex over `.debug_*` is a strict superset and provably excludes
   no executed instruction. At the text level, drop all `.debug_*` blocks plus `.file` / `.loc` /
   `.cfi_*` / local temp labels, trailing comments and blank lines.
2. **Independently hash instruction lines only** — no directives, no labels, no comments,
   whitespace normalized — so the two hashes corroborate each other.

**Validate the instrument before you trust one verdict.** Take one arm, make a byte-identical copy
under a different-length filename, compile both from a cleared compiler cache, and confirm the naive
whole-file hash **disagrees** while both filtered hashes **agree**. Cheap insurance that closes the
whole class twice: have the runner copy every variant to one fixed path before compiling, so the
compiler sees a constant filename across arms.

Then confirm a comment-only arm hashes identically to the baseline — **that arm is also your noise
floor**, so you need it either way.

**Run this before Lane 2, always.** A large minority of sites in any long-lived kernel turn out to
be inert here, and each one is a measurement window you do not have to spend.

**Step 2 — is it there at all?** Count your mnemonic in the `.s`. The expected count is
(sites) × (instances of the enclosing region), and **compute it before you look.**

- **Count is zero.** The block was deleted. Three causes, in the order to check them:
  1. `is_pure=True` and the result is not read. You promised the optimizer no effects, then made the
     effect unused. This is the intended behaviour of the flag and it is silent.
  2. The result was never rebound — a bare expression statement where the value was supposed to
     replace the original binding (`acc = gl.inline_asm_elementwise(...)`). **Every M0 site depends
     on the rebinding.**
  3. The enclosing constexpr branch folded away at this shape. Check the bin, not the asm.
- **Count is lower than expected.** Some instances CSE'd — again a purity consequence.
- **Count is higher than expected.** The block was cloned into both arms of something, or hoisted and
  duplicated. Usually harmless; **for an I-A bracket it is not**, because a mode write that happens
  twice and is undone once leaves the wave in the wrong mode.

**Step 3 — is it in the right place?** Presence is not placement. Identify, *before* you dump, the
two instructions your site is supposed to sit between, and find them in the `.s`.

**Present but floated means the anchor is missing, not that the compiler misbehaved.**

| Symptom in the dump | Missing anchor |
| --- | --- |
| the block drifted across a memory operation | `~{memory}` |
| the block drifted across arithmetic that should depend on it | no tie, or the tied result is not the value consumed downstream |
| a `s_setreg` sank past the conversions it governs | the mode write has no operand the compiler can see; mint a token or thread a live value through it (`classes.md ## Class 3`) |
| a `s_setprio` landed elsewhere than written | it is pinned weakly — see the five pinnings, strongest first, in `classes.md ## Class 2` |
| a wait moved relative to the loads it covers | the wait's *operand list* is the ordering statement; check nobody removed an "unused" argument |

**Step 4 — the M0 case, where there is no mnemonic to count.** An empty asm emits nothing, so steps 2
and 3 have nothing to look at. Verify by **effect, and always as a diff against the same kernel with
the site removed** — which is why the `else:` arm exists:

- **Register file.** Read the consuming instruction's operand: a scalar base (`s[...]`) plus a vector
  offset, versus a vector base. That is the `"=s,0"` pin taking or not taking.
- **The wait the backend inserted.** For a dependency-carried drain, the evidence is an
  `s_waitcnt lgkmcnt` appearing immediately before your (invisible) block and the next copy issue
  appearing after it. **If both sit on the same side, the tie did not do what you wanted** — check
  the rebinding first, then check that the slot count covers the *whole* read rather than a prefix.
- **Liveness and allocation.** `next_free_vgpr`, the `.amdgpu_metadata` counts, and any hot-loop
  `scratch_*` — read with `kernel_workflow/scripts/kernel_tools/asm_loop_audit.py` (pack shim: `scripts/asm_loop_audit.py`). A tie that was supposed to bound a live range
  and moved the spill count the wrong way is a tie to **remove**, not to widen.
- **A class pin that silently did not take looks identical in source and in behaviour until something
  spills.** That is why this step is not optional.

**Step 5 — the deletion probe, run deliberately.** Build the variant with your site's result unused
and confirm from the dump that the site disappears. That tells you positively that the *consumption*
is the anchor — the invariant every M0 site rests on, and the one nothing in the type system checks.
Then put it back.

**Step 6 — verify the intent, not the instruction.** The right correctness test is a function of the
I cell, and using the wrong one is how synchronization bugs get shipped:

| Cell | What proves it |
| --- | --- |
| **I-A** (wrong answer) | a case that *exercises the state* — an input that actually overflows, a value that actually saturates, a rank that actually races. **A clean run on benign input proves nothing about a mode bit** |
| **I-B** (lost ordering) | a determinism race-test, not a single correct run: fixed seed, same input tensors, `N >= ~40` launches, assert `max\|out_i - out_0\| == 0` (`../../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`). Then repeat at the largest shape in the bin |
| **I-C** (selection) | a numeric diff against the `else:` arm, on data that exercises the edges the instruction is approximate about — infinities for a reciprocal, an empty mask for a find-first, a denormal for a borrowed comparator |
| **I-D** (nothing visible) | the answer must be **bit-identical**. If it is not, your classification was wrong and the site is I-A, I-B or I-C — go back and re-classify before doing anything else |

**Step 7 — record what makes it re-checkable.** The target, the compiler version, the launch
geometry, the shape bin, the expected mnemonic count, and the dump excerpt showing the placement.
**The mnemonic count is also your regression test:** a one-line check that survives into CI, and it
fails exactly when a compiler upgrade deletes or relocates the block.

## Worked derivation — a shared-memory reader drain at a new shape

A kernel: a bf16 GEMM, `num_warps=4`, wave64, a single LDS slot per operand that this wave both
fills and reads, `BM=32`, and a per-lane operand fragment of 16 bf16 elements. It is correct, and
Lane 1 says something is wrong with the schedule.

**Lane 1 signal.** The `.s` shows the next `buffer_load_to_shared` for the slot issuing before the
`ds_read_b128`s that consume the previous contents have retired — there is no `s_waitcnt lgkmcnt`
between them at all.

**Lane 0, rule L0.1.** Two operations whose relative order is correctness (reader completion, then
refill) with **no value flowing between them** — the copy does not read the value the reads produced.
So the compiler owes nothing. Binding first: `wait_group` retires the *producer*, and what is at
stake here is the *consumer*. Different obligation, no binding. **Cell: I-B.**

**Mechanism choice.** Two candidates:

- **M1** — an explicit `"s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0"` with `"=v,~{memory}"`, `args=[]`.
- **M0-tie** — tie the *result of the LDS read* through an empty asm with `~{memory}` and let the
  backend's wait-count pass place the wait.

Take M0-tie, for a reason that is derivable rather than aesthetic: **it carries no count, so it
cannot go stale** at the next tile change or compiler bump. It has a precondition M1 does not: the
slot must be **wave-private**, or a `gl.barrier()` must also be present. Our kernel fills and reads
the slot in the same wave, so that holds — **assert it.**

**Derive the parameters.**

- `pack`: the fragment is the whole LDS read, 16 bf16 per lane. `pack = 16`.
- `dtype`: bitcast to `gl.uint16` before the tie, so nothing is FP-interpreted across the identity.
- `S`: `sizeof(uint16) = 2 < 4`, so `S = ceil(16 × 2 / 4) = 8`. **Eight output slots, not sixteen.**
  *This is the step a copied call gets wrong, and it is the one that fails silently.*
- constraint string: eight `=v`, then ties `0..7`, then `~{memory}`.
- input slots: one argument, `S = 8` → eight input slots, exactly matching the eight tie digits.
  Consistent.
- `is_pure`: the block's product is its *position*, so **`False`**. And `~{memory}` is present, so
  `True` would have been the contradiction the lint rule forbids.
- `~{scc}` / `~{vcc}`: the body is empty. None.
- `=&`: no output is written before an input is read — the body writes nothing. Not needed.

**Write it, with the rebinding and the alternative arm:**

```python
if DEPENDENCY_FENCE:
    words = gl.inline_asm_elementwise(
        "", "=v,=v,=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,6,7,~{memory}",
        [value.to(gl.uint16, bitcast=True)], dtype=gl.uint16,
        is_pure=False, pack=16,
    )
    value = words.to(value.dtype, bitcast=True)      # the rebinding IS the anchor
else:
    gl.inline_asm_elementwise(                        # the FALLBACK arm — also asm, see below
        "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
        constraints="=v,~{memory}", args=[], dtype=gl.int32,
        is_pure=False, pack=1,
    )
```

**Be honest about what this `else:` arm is and is not.** It is a *fallback* — a second, more
conservative way to discharge the same obligation — and that is all. Three things it is **not**,
each of which this reference states elsewhere and which are easy to conflate:

- **It is not an instance of the removal test.** That test requires an `else:` arm in **plain
  Gluon**. Both arms here are asm, so `## Durability and rollback` does not apply and neither arm is
  "safe to delete".
- **It is not a rollback to not-having-asm.** Turning `DEPENDENCY_FENCE` off still leaves you with an
  asm site; the obligation is I-B and something must discharge it.
- **It is not a Lane-2 A/B.** The cell is I-B, and `deciding.md` admits measurement only between two
  already-correct **I-D** arms. Comparing these two arms on a counter answers a question you did not
  ask; what they are for is the **step-4 dump diff**, which is a correctness check.

The genuine plain-Gluon alternative for this obligation is a `gl.barrier()`, which is heavier and
which the wave-private precondition is what lets you avoid. If you want a removal path, that is the
arm to keep alive — not this one.

**Verify.** Step 2 has nothing to count — M0 emits no instruction — so go straight to step 4 and diff
the two arms' dumps. What must be true in the `DEPENDENCY_FENCE` arm: an `s_waitcnt lgkmcnt` appears
after the last `ds_read_b128` and **before** the next `buffer_load_to_shared`. If it appears on the
wrong side, the tie did not reach the value the copy would overwrite — check that `value` was rebound
and that the slot count covers the *whole* read and not a prefix. Step 5: build with `words` unused
and confirm the drain disappears, proving the consumption is the anchor. Step 6: the cell is I-B, so
the test is a determinism race-test at the largest shape in the bin, not a single run.

**Bin.** Re-derive at the other end of the range. At `num_warps > 1` sharing the slot, the
wave-private precondition fails and the M0 arm is no longer sufficient — a bin boundary from step 2
of `shape-keying.md ## Deciding the bins`, not from timing. At a fragment of eight elements,
`pack = 8` and `S = 4`, so the constraint string changes — a second boundary, from step 1. Two
derived boundaries, one helper, one `gl.static_assert(PACK == 8 or PACK == 16)` making the set of
legal bins exhaustive.

## Durability and rollback — what survives an upgrade

Durability is per **cell**, and the distinction that matters is not how long a site lasts but
**whether it tells you when it stops working.**

| Cell | Survives a compiler upgrade? | Survives a target change? | Fails loudly? | Re-test trigger |
| --- | --- | --- | --- | --- |
| **I-A / M1** — machine state | yes; the semantics are the ISA's | **no** | no — wrong numerics, no missing instruction | target change; any lift into a new kernel |
| **I-A / M3** — protocol block | yes | **no** | sometimes — a hang is loud, a race is not | target change, world-size change, record-layout change |
| **I-C / M1** — instruction selection | yes | per-mnemonic | no | target change; and on a binding upgrade, check whether it became **removable** |
| **I-B / M1** — hand-counted `vmcnt(N)` | **no** | n/a | **no.** A wrong count is a wrong answer at a shape you have not run | **every** compiler bump and every tile change |
| **I-B / M0** — dependency-carried drain | needs re-checking; it delegates placement to the backend's wait-count pass | n/a | no | every compiler bump — verify via step 4 above |
| **I-D / M0** — dependency shaping | **no**, and it gives no signal at all | n/a | no, and there is nothing to lose but a preference | opportunistically; re-derive at every re-tile |
| any cell with `phys` | **no** | no | no | any change to the kernel, including ones that look unrelated |

**The asymmetry to internalise:** the cells that survive upgrades are the ones that break on *target*
changes, and the cells that break on upgrades are mostly the ones whose failure you cannot observe.
Both halves argue for the same discipline — **keep the alternative arm, and keep the mnemonic count
from step 2 as a check that runs.**

**After a Triton version bump, re-check these five, in this order:**

1. **The site still emits.** Step 2, as a regression check. Cheap, and it catches deletions a numeric
   test will not — a deleted I-D block changes no answers.

   **Treat this as pruning, not only as regression checking.** A site that stopped emitting is a
   site to *remove*, and removal is a result. Unpruned, a kernel accumulates sites that do nothing:
   in one surveyed production pack a large minority of sites produced a byte-identical binary when
   stubbed, concentrated in the mode-write and drain classes. That is what a process which adds
   sites and never re-checks them looks like after a few compiler bumps, and step 1.5 makes the
   check cost nothing — so there is no reason to carry the residue.
2. **Any hand-counted wait depth.** The count is a property of the instruction sequence the compiler
   emitted, and nothing binds it to the source. Re-derive from the copy widths and re-assert.
3. **Dependency-carried drains.** Their whole mechanism is an obligation on the backend's wait-count
   pass. Confirm the wait is still inserted where you expect (step 4).
4. **Whether the binding grew a spelling.** If `hasattr(rocdl, '<name>')` is now `True`, or a `gl.*`
   call now covers the case, your I-C site became removable — and the `else:` arm is already sitting
   there. **Removing a site is a result, not a regression.**
5. **Generated constraint strings.** `constexpr_function` evaluation is a trace-time behaviour; a
   generator that silently produced a different arity would be caught by the arity-law check and by
   nothing else.

The API itself has been stable across all four versions this pack checks (3.6.0 / 3.7.0 / 3.7.1 /
3.8.0), so this list is about what the *compiler* does with the call, not about the call.

### The one mechanical removal test — prefilter, then confirm

> **If the same file contains an `else:` branch computing the same thing in plain Gluon, the site is
> a selection site (I-C) and deleting it is safe for correctness.**

This is the only mechanical "can I remove this?" test available. **It has a mechanical prefilter and
the prefilter is not the test.** "The site sits in an `if`-body whose `If` has a non-empty `else`" is
decidable by an AST pass, and that set is not the answer: it also contains machine-state writes under
an unrelated guard, whose deletion is a silently wrong answer. Use the prefilter to build the
candidate list, then confirm each with three checks, **all** of which must pass:

1. **The `else:` arm computes the same values**, in plain Gluon, and binds the same names. An `else:`
   that does something *else* is a role split, not an alternative — and **an `else:` that is itself
   inline asm is not an instance of this rule at all.**
2. **The intent rules return I-C.** In particular rule 1 must not fire: no `s_setreg*`, no
   coherence-qualified or atomic memory operation, no cache maintenance in the body
   (`classifying.md`).
3. **`PAIRED-WITH` is empty.** Half a bracket has no intent of its own, and the `else:` arm of one
   half tells you nothing about the other.

**What the test does not say.** Safe for correctness is not safe for the schedule: an I-C site may
also have been carrying a liveness or ordering property as a side effect, which you will see in
Lane 1 and not in a numeric test. And it says nothing at all about a site with no `else:` — which is
most of them, and is why `deciding.md` tells you to write the `else:` arm at the moment you write the
site.

> **The `else:` arm is the rollback.** It costs nothing at run time (it is a constexpr branch), and
> it is simultaneously the A/B switch for Lane 2, the oracle for step 6, and the only mechanical
> removal test that exists.

## Questions this page deliberately leaves open

Recorded so nobody re-derives them and nobody assumes they were settled. Each names what would
establish it.

| Open question | What would settle it |
| --- | --- |
| Whether `is_pure=True` vs `False` on otherwise-identical `"=v,0"` fences produces different code | an ISA or TTGIR diff of the two forms — which is step 1.5 above, so this is a cheap experiment on your build rather than a question to cite |
| Whether the five `s_setprio` pinning mechanisms differ in effect | the generated ISA showing where each `s_setprio` landed. One measured null exists: across every priority site in one surveyed production pack, timed singly and in pairs, none moved a clock past the noise floor — on kernels short enough that the floor was around a per cent. **Read that as undetermined rather than zero**, and note it says nothing about *which* pinning any of them used |
| Whether omitting `~{memory}` on a self-waiting in-loop scalar load is safe | an ISA diff of the two forms |
| Whether a `"=s"`-only `s_waitcnt` with no clobber is intentional | a comment, or a disassembly showing the wait landing where intended |
| Whether `bound_ctrl` omitted vs `bound_ctrl:0` is an intended difference | a lane-boundary correctness argument in the surrounding code |
| The exact hardware semantics of the `HW_REG_MODE` narrow-float bit on this target | the target's ISA MODE register table |
| The field encoding of a hand-built buffer descriptor word | the target's buffer-resource layout spec |
| Why any particular `vmcnt` constant was chosen | the copy-issue count at that program point, i.e. the generated ISA |
| Whether any I-D site is worth having at all | the Lane-2 A/B on the kernel in front of you — and the answer stays per kernel **and per shape**, because one construct can measure a large win at one end of a shape range and nothing at the other. `deciding.md` Lane 2 now carries a prior for how often such a site moves anything at all; it does not remove the measurement |
| The upper bound on `pack` / on operands in one asm block | compile at increasing `pack` on your build until the backend refuses |
| Where to put a bin boundary when **no** derived quantity and no precondition changes | nothing on this page — that boundary is a measurement question, and must be recorded as `not established` with its A/B |
| Whether an M0 site is load-bearing from **another wave's** point of view | not decidable at the site; find the `PAIRED-WITH` partner, or the barrier doing the real work |
| Whether a resolved `Mx` body is the same text at every bin | resolve the constexprs per bin — a helper can flip comparison polarity with the shape |
| Whether "widen the tie" or "remove the tie" answers a spill signal | a before/after spill-count diff per edit; Lane 1 gives a direction and never a conclusion |

Three limits bound this whole reference:

1. **These procedures decide *what to write* and *whether it is correct*. They do not decide whether
   it was worth writing.** Every "is this better" question routes to Lane 2, and Lane 2 is a
   measurement procedure on your hardware. The prior stated there ranks mechanism classes by **how
   often they carried any effect at all**, which is a different question from how much yours will
   carry — **a reader who wants a ranking that substitutes for measuring their own kernel is asking
   for something no evidence here supports.**
2. **The arity law and the `is_pure` rule are decidable; real-world choices are not always
   explicable.** Visually identical fences get spelled both ways with no comment; one instruction
   gets pinned five different ways; sibling kernels differ on `bound_ctrl`. **The rules above are
   what you should follow; they are not a reconstruction of what every author was thinking**, and
   where the two diverge it is marked `not established`.
3. **A site is frequently not decidable alone.** A large minority are one half of a pair, and a
   meaningful share have a computed body that differs per instantiation. **Resolve first, decide
   second**; an audit of half a pair, or of an unresolved `Mx` site, returns the wrong answer
   confidently.

**And the standing one:** nothing here is a performance claim *about your kernel*. Every "purpose" on
these pages comes from instruction semantics, from stated intent, or from control-flow structure —
never from a measurement taken somewhere else. Where a measured prior exists it is stated as a prior,
with its scope attached, in `deciding.md` Lane 2; it tells you what to expect before you spend a
window, and it settles nothing.
