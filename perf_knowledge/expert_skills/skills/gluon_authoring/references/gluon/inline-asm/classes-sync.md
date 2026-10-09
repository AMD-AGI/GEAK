# Classes 4 and 4b: synchronization, and protocol blocks

Part of `../inline-asm-reference.md`. Classes 1–3 are in `classes.md`; classes 5 and 6 in
`classes-lane-register.md`.

## Class 4 — synchronization

**When it is the only route:** a wait or a publish whose granularity the abstraction does not offer.

The shapes below are ordered by how often you will need them, commonest first.

### The drain marker — a different obligation from `wait_group`

The most common class-4 shape is a one-line body whose only content is a wait, plus a throwaway
output:

```python
gl.inline_asm_elementwise("s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                          "=v,~{memory}", [], dtype=gl.int32,
                          is_pure=False, pack=1)    # result deliberately discarded
```

**Copy completion and LDS-reader completion are distinct obligations.** `wait_group` retires the
**producer**; this retires the **consumer**, so the next `buffer_load_to_shared` cannot overwrite a
shared-memory slot whose readers are still in flight. **Removing it is a wrong answer, not a
schedule change.**

It belongs between the last relaxed shared-memory read and the next copy, and the `~{memory}`
clobber is what keeps it there. The full sequence is `wait_group(0)` → `barrier()` → the relaxed
loads → **this block** → `barrier()`. Where a barrier is elided, say why in a comment — a one-wave
region can elide one, and the next reader cannot tell that from an omission.

### `s_waitcnt vmcnt(N)` — staged partial drains

A real technique with a real cost, and the cost decides the use: **it is a pacing lever, not a
readiness lever.** Written up with what you give up in
`../../tile-programming/pipeline.md ### Draining below the group: bare s_waitcnt instead of wait_group`.

The floor you are stepping below is `wait_group`'s own semantics — its `num_outstanding` counts
commit groups, not copies (`../pipeline-reference.md` ## Getting `wait_group` right). **Settle there
first** that the drain you want genuinely has no group boundary to name: a `commit_group` placed
differently reaches many of the granularities this mnemonic gets reached for, and costs none of the
dependency checking.

**The depth does not have to be a literal.** `"s_waitcnt vmcnt($1)"` with
`constraints="=v,n,~{memory}"` passes the count in as an operand. `n` is the immediate-only
constraint, so the value is *required* to fold to a compile-time constant and is materialized into
the instruction rather than into a register. Without it, a staged drain can only be written as an
`if`/`elif` ladder over literals, which cannot express a depth computed from the ring's own
constexprs. **A text search keyed on `vmcnt(<digits>)` is structurally blind to this form** — if you
are auditing, match the `n` constraint too.

**The count is a per-panel instruction count. Derive it every time, and assert it.**

```python
# each wave issues one 128-bit A copy and one 128-bit B copy per panel
gl.static_assert(0 <= NEWER_PANELS <= MAX_PANELS)
gl.inline_asm_elementwise("s_waitcnt vmcnt($1)", "=v,n,~{memory}",
                          [2 * NEWER_PANELS], dtype=gl.int32,
                          is_pure=False, pack=1)
```

A differently-shaped panel pair may issue four VMEM instructions rather than two; the arithmetic is
kernel-specific and there is no default worth memorising.

**Note what protects the count:** a class-2 empty-asm marker preventing the compiler from sinking
copy issues across the point. Remove that marker and a correct `vmcnt(4)` becomes an incorrect one.
**The coupling is the hazard, not either site alone.**

### The wait's operand list can be the entire ordering statement

`s_waitcnt` takes no operands at the ISA level, so the *asm block's* operand list is available for
creating dependencies the wait-count pass must respect. A one-instruction `s_waitcnt vmcnt(0)` may
legitimately carry a long list of bitcast payload anchors, purely so the backend can track read
retirement and avoid a false wait afterwards. A softer form passes an input with **no matching
output slot** at all, so that a completion address depends on a cross-wave reduction and reuses that
reduction's synchronization instead of issuing another barrier.

> **A source-level tidy-up that removes "unused arguments" destroys the ordering and compiles
> clean.** If you are about to delete an operand you cannot explain, find its purpose first.

### The dependency-carried drain — a wait with no `s_waitcnt` in it

Tie the result of a shared-memory read through an empty asm with `~{memory}`, and the backend's
wait-count pass is *obliged* to insert the `lgkmcnt` wait before it; the clobber then holds the next
async copy after it. Net effect: a reader drain expressed purely as a data dependency, with no
explicit wait and no barrier.

**This is the one place where an M0 site carries a real ordering obligation** rather than a
scheduling preference, and its preconditions are strict:

1. the slot must be **wave-private** — one wave owns both slots;
2. the tied result must **replace** the original value downstream;
3. the tie must cover the **whole** read — an arity question, not a `pack` question, and exactly
   where a narrow `dtype` silently fences half the fragment (`shape-keying.md`).

Keep the explicit `s_waitcnt lgkmcnt(0)` available on the other arm of a compile-time flag. It is
the fallback and the A/B.

### `buffer_wbl2` — L2 writeback before a publish

Needed when you publish data written with ordinary stores and signal it with something the compiler
does not recognize as a release. The documented sequences are `buffer_wbl2 sc1` then
`s_waitcnt vmcnt(0)` for agent scope, and `buffer_wbl2 sc0 sc1` then
`s_waitcnt lgkmcnt(0) & vmcnt(0)` for system scope.

**The exemption belongs to the intrinsic, not to the operation.** What emits the maintenance for you
is a `gl.atomic_*` call whose `sem=` and `scope=` ask for it — *not* the fact that your signal
happens to be atomic. An atomic you hand-wrote in an asm block is still an atomic and still gets
nothing emitted around it; so is a `gl.atomic_*` left on the default `scope="gpu"` when the reader is
a peer device. Settle it against `../../workloads/collective.md ## Publish and consume`, which
enumerates the cases, rather than by asking whether an atomic was involved. **Getting this backwards
is silent: the publish compiles, runs, and passes at small shapes.**

### Release and acquire are a pair, and both halves are asm

`buffer_wbl2 sc0 sc1` (+ a drain) publishes; `buffer_inv sc0 sc1` immediately after a successful
poll acquires. Neither has a binding-level spelling in the polled-loop form.

**An asymmetry between two structurally similar sites may be semantic, not sloppiness.** A
*terminal* wait with no payload reads after it correctly omits `buffer_inv` — only the last
consumer's first wave waits, and nothing follows that needs the invalidate. "Fixing" that asymmetry
in either direction is a bug. Leave a comment saying which case a site is.

### A gate on the value cannot see a missing acquire

Everything in this class is normally accepted by re-running and comparing outputs, and there is
one failure here that no amount of that reaches.

> **A missing acquire is not a formula computed wrongly. It is a window this machine did not
> happen to open.** Gates that check the *value* — bit-identical repeats, a tolerance against an
> oracle, a determinism race-test — sample the interleavings that actually occurred. A dropped
> barrier changes the result of no single interleaving; it only makes further interleavings
> **possible**. On memory ordering such a gate is therefore **not weak evidence, it is zero
> evidence**, however many green runs it accumulates.
>
> **The operative corollary: a minimal synchronization sequence that passes the value gates has
> established the arithmetic and nothing about the ordering.** Sufficiency is argued
> **semantically** — which release pairs with which acquire, at which scope — or read out of the
> ISA, where the maintenance either is or is not present. It cannot be inferred from "it came out
> right many times".

Now run that backwards, because that is the direction that costs you: **if a speedup came from
relaxing the ordering, a green run is not evidence that the relaxation is safe.** Removing
synchronization is a common direction with real wins, and its failure mode is exactly the one the
standard gates have no channel for. Ship such a result with the pairing argument or the
disassembly beside it, or ship it labelled unverified.

### Half a pair guarantees nothing

Many class-4 waits are the *completion* half of a split-phase primitive whose issue half is a
different call site with no wait in it, linked only by a tie. Record `PAIRED-WITH` and audit the
pair (`classifying.md ## PAIRED-WITH`).

### `s_and_saveexec_b64` — listed here to be ruled out, then narrowed

It is what `gl.warp_predicate` lowers to where that fork extension exists
(`../pipeline-reference.md`), so it reads like the obvious way to recover the missing capability on
an upstream build. **It is not.** The door takes tensors and returns tensors; there is no way to hand
it the region you wanted skipped, so narrowing exec inside one asm block skips nothing but that
block's own instructions. Writing the mnemonic gets you the ISA without the branch — a language
ceiling, recorded as one (`../../hardware/capability-matrix.md`).

**Read that ceiling at its real width: it is about skipping *Gluon-level* code.** The other shape is
real and is what Class 4b is about — put the whole region you wanted skipped **inside the asm text**
and use exec to hand different waves different roles. The strongest form retires an already-ready
peer out of `exec` and spins until `exec` is empty, so a single `v_cmpx` replaces a cross-lane
reduction per iteration. Both statements hold: **you cannot use exec to skip code the Gluon layer
emitted, and you can write a role-split whose branches are entirely asm text.**

**Verify:** a determinism race-test, not a single correct run
(`../../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`).
Synchronization bugs pass at small shapes and on quiet machines. The failure mode of a wrong `vmcnt`
is a race that survives hundreds of launches and then does not. Bound what that test buys before
you quote it — `### A gate on the value cannot see a missing acquire`.

**Risk:** this class **removes** the compiler's dependency checking rather than adding to it. A
`wait_group` that is wrong is usually a compile error; an `s_waitcnt` that is wrong is a wrong answer
at a shape you have not run yet.

## Class 4b — protocol blocks

**When it is the only route:** the thing you need is not an instruction the binding is missing. It
is a *program* — a loop, a branch, a label, an EXEC role-split — that the binding cannot express at
any granularity.

These cluster in multi-GPU collective kernels, and they are the only sites where removing the asm
produces a genuine **data race** rather than a different schedule.

**The mechanical test:** the body contains `s_cbranch*`, a numeric local label (`1:` / `1b` / `2f`),
a `${:uid}` label, or `s_*saveexec_b64`. If any of those is present, it is a protocol block and the
rest of this reference's per-instruction advice does not reach it.

The canonical body is an acquire:

```python
gl.inline_asm_elementwise("""
    v_cmp_gt_u32_e32 vcc, 64, $3          ; only wave 0 participates
    s_cbranch_vccz .Ldone${:uid}          ; skip the WHOLE body otherwise
  .Lpoll${:uid}:
    global_load_dword $0, $1, off sc0 sc1 ; system-coherent, or the poll never terminates
    s_waitcnt vmcnt(0)
    v_subrev_u32_e32 $0, $2, $0           ; signed difference: wrap-tolerant
    v_cmp_gt_i32_e32 vcc, 0, $0
    s_cbranch_vccnz .Lpoll${:uid}
    buffer_inv sc0 sc1                    ; the ACQUIRE
  .Ldone${:uid}:
""", constraints="=&v,v,v,v,~{vcc},~{memory}", args=[ptr, target, pred],
    dtype=gl.uint32, is_pure=False, pack=1)
```

The matching release is `buffer_wbl2 sc0 sc1` + a drain + a `global_atomic_add … sc1`, usually
EXEC-masked to the participating lanes. The largest blocks combine everything — EXEC save/restore, a
scalar load into named physical SGPRs, L2 writeback, a `vmcnt(0) lgkmcnt(0) expcnt(0)` drain, peer
atomics, the spin, the invalidate and a progress publication — in a few dozen instructions.

### Seven preconditions, every one load-bearing

1. **Lane and wave predication is passed in as an operand, never assumed.** Build a `threads` /
   `lanes` tensor from an explicit layout — `gl.arange(0, 256, layout=gl.BlockedLayout([1], [64],
   [4], [0]))` — and elect inside the body with `s_cmp_lt_u32 $0, 64` or
   `v_cmp_lt_u32 vcc, $2, 64`. **The layout is part of the contract.**
2. **The predicate-and-skip wrapper is not an optimization.** Branch *over the entire body* when the
   lane is not a participant, because the body touches peer memory that non-participants must not
   touch.
3. **`sc0 sc1` (or `glc`) on the poll load is mandatory.** A cached poll never observes the peer's
   update — the loop does not terminate.
4. **The comparison is a signed difference.** `s_sub_u32` / `v_subrev_u32` then `v_cmp_ge_i32 …, 0`
   is a wrap-tolerant `>=`. A direct unsigned compare on raw counters **deadlocks on rollover**.
5. **EXEC must be saved and restored on every exit path.** `s_and_saveexec_b64 $0, vcc` saves the old
   mask into `$0` and replaces EXEC; `s_or_saveexec_b64 $0, $0` restores it. **`$0` is an `=&s` save
   slot, not data.** Drop it, tie it, or skip the restore and the wave's EXEC is corrupted for
   everything that follows. This is why `s_cbranch_execz` targets the label *before* the restore,
   not past it — restore a non-empty mask before cache acquisition, even when the final peer retires.
6. **`~{scc}`, `~{vcc}` and `~{memory}` must all be declared**, and `=&` early-clobber on every
   output the block writes before reading all its inputs. The `~{memory}` is what stops the compiler
   hoisting the peer loads out of the loop.
7. **Prefer `${:uid}` labels.** Numeric local labels (`1:` / `1b` / `2f`) work until two such blocks
   are inlined into one function, at which point **the branches silently retarget.**

### Two further shapes worth recognising

- An **assembler-level** conditional keyed on an `n` operand — `.if $8 == 0 / buffer_wbl2 / .endif`
  — compiles two different bodies with no runtime branch.
- A **raw buffer descriptor** built in named physical SGPRs, to reach an instruction form the
  compiler will not emit (for example a `buffer_atomic_* … offen sc1` where only CTA lane zero has an
  in-bounds offset, avoiding both a generic CAS loop and an unsupported intrinsic conversion). If you
  do this, the descriptor's field encoding is a hardware-spec question — cite the target's
  buffer-resource layout, and do not copy a magic word you cannot decode.

**Verify:** a determinism race-test across ranks, and an ISA read confirming the labels, the EXEC
bracket and the cache maintenance survived. **A protocol block that assembled is not a protocol
block that is correct.**

**Risk:** the highest-consequence material in this reference and the least portable. Such a block
hard-codes a world size, a record layout (offsets recurring across separate asm bodies that must
agree), a wave size, and sometimes physical registers. **None of that is visible to any type
system.**
