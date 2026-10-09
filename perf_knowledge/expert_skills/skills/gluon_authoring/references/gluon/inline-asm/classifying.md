# Classifying an inline-asm site: mechanism × intent

Part of `../inline-asm-reference.md`. Read this before `deciding.md` if you are holding a site
someone else wrote; read `deciding.md` first if you are deciding whether to write one.

The six `Class N` names later in this reference are *intent* names. Read as a single axis they are
not reproducible: the same set of sites can be binned three very different ways, and the readers are
not disagreeing about the kernels. They are applying three different filters to one word, because a
single-axis name mixes a *mechanism* (a tied identity; a constraint string) with an *intent* label.

Use two axes. Each is decidable from material in front of you, so two readers land in the same bin.

## Axis M — mechanism: what the call is made of

Read the `asm` string and the `constraints` string. Nothing else. First match wins.

Bins are listed in the order you are most likely to meet them, commonest first — M1 and M0 dominate,
M3 is rare but is the one that will cost you a week if you misread it.

| bin | test |
| --- | --- |
| **M0 — empty asm** | `asm` is `""` or whitespace. The constraint string is the entire instruction. Not an edge case — this is one of the most common shapes you will meet |
| **M1 — single instruction** | one mnemonic, **ignoring a trailing `v_mov` / `s_mov` that exists only to fill a mandatory output** |
| **M2 — straight-line block** | two or more real mnemonics, no label and no branch |
| **M3 — protocol block** | the body contains `s_cbranch*`, a numeric label (`1:` / `1b` / `2f`), a `${:uid}` label, or `s_*saveexec_b64`. This is a *program*, not a missing instruction |
| Mx | `asm` is a computed expression that did not fold — **not a bin, a work item.** Resolve it at its defining line, at the shape you are building, then re-bin |

**The fill-`mov` discount in the M1 row is the whole ballgame for reproducibility.** A body whose
second mnemonic exists only to satisfy the mandatory output is a one-instruction site; an audit that
forgets this moves a large block of sites from M1 into M2 and no two audits agree again. Write the
rule down before you count, and say which convention your numbers used.

**`Mx` is a lower bound on every other bin,** not a small residue you can round away. A body built by
a `@constexpr_function` is nameable but unfolded, and it is usually the *most* shape-sensitive site
in the kernel. Until you fold it at your shape, you do not know which bin it is in.

### Six modifiers, each an independent boolean

Each is readable off the same two strings. Listed commonest first — `tie` and `mem` are everywhere,
`phys` is rare and is the one that makes a kernel unportable:

| modifier | test | what it tells you |
| --- | --- | --- |
| `tie` | a bare digit constraint (`0`, `1`, …) | an operand is pinned to an output register |
| `mem` | `~{memory}` present | compiler motion of memory operations across the block is constrained |
| `imm` | an `n` constraint letter | the operand must survive as a literal immediate |
| `phys` | a braced register name (`=&{v0}`, `~{s0}`) | absolute register assignment; portability is now per-kernel |
| `gen` | the constraint string is computed, not a literal | arity is derived — see `shape-keying.md` |
| `pure` | `is_pure=True` | the block may be CSE'd, hoisted, sunk, and deleted when unused |

**`pure` and `mem` are independent as axes and contradictory as a pair.** `is_pure=True` promises the
optimizer exactly the motion `~{memory}` forbids. A call carrying both is not a subtle trade-off —
one of the two is wrong. Treat it as a lint rule.

**`~{scc}` / `~{vcc}` is not a modifier but an obligation.** If the body names those registers, or
uses an instruction that writes them, and the clobber list omits them, a live predicate will be
corrupted. Check it every time; the symptom appears far from the asm.

## Axis I — intent: what the site is bought for

One question. **Delete the site, rebind its output name to its input, and ask what changes.**

| bin | what changes | how to see it from the site |
| --- | --- | --- |
| **I-A — wrong answer** | computed values change, deterministically | the body writes machine state, or does a coherence-qualified or atomic memory op |
| **I-B — lost ordering** | right sometimes; a race appears | the body waits, spins, or brackets EXEC, and the thing waited for was issued somewhere else |
| **I-C — instruction selection** | same answer, different instructions | an expressible alternative exists — often literally on the sibling `else:` branch |
| **I-D — nothing visible** | same answer, same guarantees; only schedule, allocation and live ranges move | M0, or `s_setprio`, or a body that is only a fill-`mov` |

Apply in order; the first rule that fires decides:

1. body has `s_setreg*`, or a memory op carrying `sc0` / `sc1` / `glc`, or any `*_atomic_*`, or
   `buffer_inv` / `buffer_wbl2` → **I-A**
2. body **issues a memory operation and waits for that same operation** → **I-C**
3. body has `s_waitcnt`, a backward branch, or `s_*saveexec_b64` → **I-B**
4. body issues any instruction that is not a fill-`mov`, an `s_setprio` or an `s_nop` → **I-C**
5. otherwise → **I-D**

### Why rule 2 is written down

Rule 2 is a tie-break for sites that both issue and wait, and it is stated explicitly because the
choice moves large numbers of sites between bins.

The argument for classifying by the **issue** rather than the wait is that the same idiom gets
spelled both ways. A scalar metadata load may appear as one call that issues `s_load_dwordx2` and
waits for it in the same body — or as two calls, one that issues with no wait and a second, bodiless
one whose only content is a tie back to the first call's outputs. Classify by the wait and one idiom
lands in two different bins depending only on whether its author inlined it. Classify by the issue
and all three forms read correctly:

> **Delete the issue and nothing was ever loaded. Delete the wait and the load still happened —
> only its ordering is gone.**

## Two mechanical cross-checks

### The delete-safety rule — a prefilter and a confirmation, never a one-step test

If the same file contains an `else:` branch computing the same thing in plain Gluon, the site is a
selection site (I-C) and deleting it is safe *for correctness*. This is the only mechanical "can I
remove this?" test available.

**The structural form of it — the site sits in an `if`-body whose `If` node has a non-empty
`orelse` — is decidable by an AST pass, and its precision is not 1.** It will happily hand you a
machine-state write guarded by an `if` whose `else` does something entirely unrelated, and deleting
that one produces a silently wrong answer. It will also hand you sites where *both* arms are inline
asm, which the rule does not cover at all.

**Use the structural form only to build a candidate list,** then confirm each with the three checks
in `costs.md ## Durability and rollback`.

Note what the rule does **not** say: safe for correctness is not safe for the schedule, and it says
nothing at all about a site with no `else:`.

### The no-claim rule

`is_pure=True` **and** no `~{memory}` means the site makes no ordering claim about memory
whatsoever. If you are leaning on such a site for ordering, you are leaning on nothing.

## PAIRED-WITH — a field on every site, and a gate

A substantial fraction of real sites are **one half of a pair**: issue/wait, save/restore,
prefetch/finish, release/acquire. A per-site record cannot express this, so carry a `PAIRED-WITH`
field — and treat it as a gate:

> **A site with a non-empty `PAIRED-WITH` has no intent of its own. Classify the pair, and give both
> halves the pair's intent.** Auditing either half alone returns the wrong answer about what it
> guarantees.

| shape | first half | second half | the link between them |
| --- | --- | --- | --- |
| issue / wait | issues a load, no wait | bodiless, ties back to the first call's outputs | **the tie only** — nothing else connects them |
| save / restore | reads machine state into a value | writes it back | helper-pair naming; nothing enforces it |
| prefetch / finish | starts the transfer | consumes it | a snapshot value passed to both |
| release / acquire | writeback before publishing | invalidate before reading | the peer memory location |

`s_setprio 1` … `s_setprio 0` is a fifth shape, but a **bracket** rather than a pair: it must balance
on every path. Be careful with automated "fix the imbalance" edits — a bracket may be deliberately
unbalanced across a shape branch, and restoring symmetry silently changes what the kernel does.

## BUDGET-GROUP — when several sites claim one resource

`PAIRED-WITH` covers sites that are two halves of one thing. There is a second multi-site structure,
and reading it as a set of independent sites produces a confidently wrong answer.

> A **`BUDGET-GROUP`** is a set of sites with no relationship to each other **except that they all
> draw on one finite resource** — the register budget, the LDS budget, an occupancy step. Each is
> individually removable. Their effects are **not additive and not even monotone**.

The shape that makes this concrete: several empty tied identities in one kernel, all claiming the
same register file, with an occupancy step just above what the kernel allocates.

**Removing a subset of the group can be worse than removing the whole group.** The sites that look
least important are often exactly what holds the allocation under the step once the site that looks
most important is gone: take that one out alone and the allocation crosses the step, halving
residency; take them all out and it lands back under it. **The response is non-monotone in the
number of sites removed**, so the ordering you would infer from any single-site measurement is not
the real ordering.

A counter reporting "over budget" is correct and is **not** an instruction about which claimant to
evict — it cannot be, because nothing tells it the group exists.

**How to recognise one before it bites.** Whenever a resource reading sits a few units over a
threshold, assume a budget group until you have shown otherwise. The disassembly names the group
when no counter will: the pressure tends to surface as cross-lane move instructions appearing in
exactly the arms where one wide tie is absent — the compiler rematerializing what that tie had been
holding, a quantity that appears in no counter at all.

**The gate.** A site with a non-empty `BUDGET-GROUP` **cannot be evaluated by a single-site A/B.**
Build the all-off arm and at least one grouped arm, or do not report a direction. This is the same
discipline `deciding.md` Lane 2 asks for; it is stated here because the recognition happens while
you are classifying, not while you are measuring.

## Reading the two axes together

Mechanism tells you what to copy and what to check. Intent tells you what happens if you stop. The
cell you are in names the failure you should be defending against:

| | I-A wrong answer | I-B lost ordering | I-C selection | I-D nothing visible |
| --- | --- | --- | --- | --- |
| **M0** | — | dependency-carried drain (`classes-sync.md`) | register-file / adjacency pin (`classes-lane-register.md`) | dependency shaping — the largest population here |
| **M1** | MODE write (`classes.md`) | drain marker (`classes-sync.md`) | the instruction you came for (`classes.md`, `classes-lane-register.md`) | `s_setprio` (`classes.md`) |
| **M2** | cache maintenance, atomics | tied drain | fused scalar load, codec chains | — |
| **M3** | protocol block (`classes-sync.md`) | — | — | — |
