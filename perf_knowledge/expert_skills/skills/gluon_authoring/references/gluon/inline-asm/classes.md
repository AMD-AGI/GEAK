# Classes 1–3: instruction selection, scheduling control, machine state

Part of `../inline-asm-reference.md`. Classes 4 and 4b are in `classes-sync.md`; classes 5 and 6 in
`classes-lane-register.md`.

## Class 1 — instruction selection

**When it is the only route:** the silicon has the instruction, the Triton/LLVM binding does not
expose it. The checkable form is `hasattr(rocdl, '<name>')` returning `False` while the mnemonic
assembles for your `-mcpu` — a wrapper gap, not a silicon one. `v_cvt_off_f32_i4` on gfx942 is the
worked case (`../../hardware/cdna3-gfx942.md`).

This is intent **I-C**, and it is the one intent that is usually genuinely pure: a value in, a value
out, no state touched. Use `is_pure=True`, let it be CSE'd, and check that the vectorization you
wanted survived.

### The instruction is rarely the whole primitive

A class-1 site almost always carries a *caller-side numerical contract* that lives in Python and is
invisible in the asm text. **Lifting the asm line alone is the single most reliable way to get a
silently wrong answer.** The recurring ones, in the order you are most likely to hit them:

- **`v_exp_f32` and `v_log_f32` are base 2.** Every correct use pre-multiplies the argument by
  log2(e) — or by log2(e)/2 where the caller works in half-range. Nothing in the mnemonic hints at
  this, and the result is plausible-looking but wrong without it.
- **`v_rcp_f32` is approximate, does not handle infinity, and may need range-shifting.** The
  refinement and the guard are part of the primitive, not optional polish: a Newton step
  (`r = fma(r, fma(-x, r, 1.0), r)`) and an explicit `where(x == inf, 0.0, r)`. Where the final
  result can be subnormal, the operand is scaled up before the reciprocal and the result scaled back
  — with **the Newton step applied after the rescale**, so that the native reciprocal stays in
  normal range throughout. It is normal to see two or three correction layers on one asm line.
- **`v_div_fixup_f32 $0,$1,$2,$3` restores the IEEE edge cases** (0/0, x/0, inf, NaN, signed zero) to
  an rcp-based divide. `$1` is the already-computed quotient, `$2` the denominator, `$3` the
  numerator — **the order is not commutative.**
- **Integer comparators are borrowed, and the borrowing carries a range precondition.** Applying
  `v_max_f64` / `v_min_f64` to `uint64` payloads gives exact unsigned ordering *only* on
  positive-normal mantissa encodings. A denormal, NaN, zero or sign bit anywhere in a key breaks the
  ordering silently. The same shape appears when reducing float *bit patterns* with `v_max_u32`,
  which is correct only after the sign has been stripped. **Assert the range; nothing else will.**

### Before you write one, write the `else:`

Keep the expressible path alive on the other arm of a compile-time branch (`if FAST_EXP:`,
`if REFINE:`, `if USE_U24:`). That is not decoration:

- it is what makes the site removable later,
- it is the A/B switch that `deciding.md` Lane 2 requires,
- it is the arm the delete-safety confirmation in `costs.md` compares against.

Writing it is cheap. Recovering it later is not.

**Verify:** the mnemonic appears in the disassembly at the expected count per loop iteration, plus a
numeric check against the unfused path. A conversion that silently falls back to the generic
sequence still produces right answers, so a correctness test alone will not tell you it landed.

**Risk:** an instruction that exists on one target and not another turns a portable kernel into a
target-specific one with no diagnostic until someone runs it elsewhere. Record the target constraint
next to the kernel, not only in your head.

## Class 2 — scheduling control

**When it is the only route:** you need an instruction placed at a specific point for its timing
effect, and no marker expresses that point. `s_nop` deliberately placed at a stage head is the
canonical case — a phase lever rather than a hazard fix, and *not* the exposed hazard the same
mnemonic usually signals (`../../method/climb.md ## 2. Reversed-intuition traps`, entry 9).

The **empty-asm form is the important idiom here, and it is the single most common shape you will
meet**. An `asm=""` block with a tied operand emits nothing and exists purely to create a data
dependency the scheduler must respect.

### It is not a fence

Read this before you rely on one. An empty tied block with `~{memory}` is an **identity scheduling
boundary**: the tied operand is returned unchanged, no arithmetic happens, no memory is accessed and
no synchronization is performed. `~{memory}` forbids the **compiler** from moving memory operations
across the point. It issues nothing, waits for nothing, and orders nothing at the hardware level.

Shared-memory ownership still requires explicit waits and barriers. Write that down in a comment
next to every one you place, because **the failure mode runs in the expensive direction**: a reader
sees a "fence" next to two `gl.barrier()` calls, concludes the barriers are redundant, removes them,
and has deleted the only real synchronization in the region.

**Its effect is not local, and that is the part that surprises people.** The clobber changes what the
backend's wait-count pass decides **elsewhere**, not only at the point you wrote. A single-line drain
carrying a memory clobber has been measured to double the wait count in the final disassembly, with
the extra waits inserted around a matrix instruction in a different region — and removing the block
*reduced* the wait count while making the kernel slower.

> **Counting `s_waitcnt` occurrences is therefore not a reliable way to judge synchronization cost.**
> The count can move in the opposite direction from the time.

### Three things do change when you place one

And they are worth placing one for:

1. the value is materialized in a register of the stated class at that exact point;
2. its live range provably covers the point;
3. later memory operations cannot be hoisted above it by the compiler.

That is a liveness and allocation tool. See `classes-lane-register.md` for the mechanism and
`shape-keying.md ## Deriving the parameters` for its arity.

### `s_setprio` — the non-empty member of this class

The same instruction gets pinned in place at least five different ways. Listed from strongest hold
to weakest:

1. a tied **whole fragment** — one block consumes the complete per-thread accumulator, so the
   priority change cannot move relative to any part of it;
2. a tied live input (`"=s,0"` with a program-id-like argument);
3. a `~{memory}` clobber;
4. a dummy `s_mov_b32 $0, 0` with no clobber;
5. nothing at all but the `=s` output.

The last two give the compiler materially more freedom than the first three. **Whether that
difference is deliberate is `not established`** — settling it needs the generated ISA showing where
the `s_setprio` actually landed under each form. Prefer form 1 or 2 when the placement matters.

Two further facts that bite:

- **The output is often never written.** A site spelled `asm="s_setprio 1"`, `constraints="=s"`
  declares an output the body never writes. That is benign *as written*, because nothing consumes
  the result — but a copier who *uses* the return value gets an uninitialized SGPR. The safer
  spelling appends `s_mov_b32 $0, 0`; the reason it exists at all is that the helper requires a
  non-empty result.
- **Brackets may be shape-keyed, so "balance them" is not a safe edit.** A kernel may raise priority
  unconditionally, lower it only under a shape guard, raise again under the same guard, and lower
  unconditionally — so that at one end of the shape range there is exactly one raise and one lower
  spanning a much larger region than at the other. That asymmetry is deliberate. A sibling kernel in
  the same family may use four symmetric unconditional brackets for the same work.

### `s_nop` is in this section by mnemonic and usually does not belong to it

A bare `s_nop N` inside a multi-line body is almost always an **ISA hazard cover**, not pacing: this
target requires spacing between a VALU producer and its DPP reader, and a correct DPP ladder carries
one between every step. **Deleting it is a correctness bug, not a round.** Only a *standalone*
`s_nop` site is the pacing lever this section describes.

There is a third case that looks like neither: a block may carry **zero** `s_nop`s because it
interleaves three or four independent register streams, which supplies the same spacing. Removing a
stream from such a block makes it incorrect — the hazard cover was the interleaving.

**Verify:** phase effects are non-monotone. Sweep the placement; do not bisect it
(`../../tile-programming/llir-codesign.md ## Phase pacing: moving a burst instead of removing work`).
Read the `s_nop` and `s_waitcnt` census before and after, and be sure you are not looking at a hazard
the compiler inserted, which is a different problem with the opposite fix
(`../../hardware/bound-class-signals.md`).

**Risk:** the highest ratio of effort to durability on this page. Scheduling wins are pinned to one
compiler version and evaporate on upgrade, and nothing warns you — the asm still assembles.

## Class 3 — machine state

**When it is the only route:** always, for this class. There is no Gluon surface for the MODE
register.

`s_setreg_imm32_b32` writes hardware register fields. The field worth knowing on this target is the
`HW_REG_MODE` bit that selects **saturating versus IEEE behaviour on narrow-float conversion and
arithmetic overflow** — the kind of thing that turns an `inf` into a saturated value several hundred
instructions after the write.

The bit is documented elsewhere under an FP16 name. **Read that as the *field's* name, not as the
scope of what it affects:** in practice it is written on FP8 paths, and the helper naming around it
(begin/end native-FP8 brackets, an IEEE-mode write of 0 against a store-FP8 write of 1, an
`if EMIT_FP8:` gate) is all about FP8 conversion behaviour. **The exact hardware semantics on this
target are `not established` here** — settling it needs the target's ISA MODE register table.

Three rules make this class survivable:

1. **It is wave state, not a scoped setting.** It persists from the write to the end of the wave or
   to an opposing write. There is no block scope and no restore-on-exit.
2. **Restore it yourself**, immediately after the region that needs it, or accept that the rest of
   the kernel runs in the modified mode.
3. **`is_pure=False`, always.** A pure block with an unused result is deleted, and the deletion is
   silent — the kernel computes the wrong numerics with no trace of the missing instruction.

Two more, both learned the hard way:

### Anchor the write to a live value, or it floats

An `s_setreg` has no operands the compiler can see, so never write one bare. Two spellings, both
sound:

```python
# (a) mint a dependency token and route it into whatever must not float above the write
mode_token = gl.inline_asm_elementwise(
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\nv_mov_b32 $0, 0",
    "=v,~{memory}", [], dtype=gl.int32, is_pure=False, pack=1)

# (b) thread a real value through, so the restore cannot sink past the arithmetic
acc, _ = gl.inline_asm_elementwise(
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 0), 0",
    "=v,=v,0,1,~{memory}", [acc, scale], dtype=(gl.float32, gl.float32),
    is_pure=False, pack=1)
```

**Note the `pack=1` in form (b), and do not raise it without re-deriving the string.** The `dtype` is
a 2-tuple, so by the arity law the output-slot count is `Σ S` over *both* entries — `2 × pack`, not
`pack`. At `pack=1` that is the two `=v` slots shown. At `pack=4` it would be **eight**, with ties
`0..7`, and the string above would tie a prefix and silently leave the rest of the fragment
unanchored. This is an I-A site, so that failure is wrong numerics with nothing missing from the
disassembly. See `shape-keying.md ### The arity law`.

The `v_mov_b32 $0, 0` in form (a) is **not arithmetic**: it exists because the backend cannot lower
a zero-result asm. Say so in a comment, or the next reader will try to delete it.

### It is half of a pair — and the unpaired case is the most copy-hostile site there is

The matched `…, 0` / `…, 1` bracket is the normal shape. But a kernel may legitimately clear the bit
for one class of CTA and **never restore it**, with correctness resting entirely on a launch
partition (`if pid < router_tiles:`) guaranteeing that no CTA which took that branch later wants
saturating conversion.

**Lift such a helper into a kernel whose CTAs later do narrow-float work and you get silently wrong
numerics — no crash, and no missing instruction to find.** This is also the counterexample that
stops the `else:` delete-safety prefilter from being a test on its own: the site sits in an `if`-body
with an `else`, and deleting it is a wrong answer.

**Verify:** the numeric behaviour you were after, on a case that actually exercises it (an input
that really overflows), plus the `s_setreg` count in the disassembly. **Mode changes cannot be
verified by shape, or by a clean run on benign input.**
