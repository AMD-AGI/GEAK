# Classes 5 and 6: cross-lane collectives, and register-class control

Part of `../inline-asm-reference.md`. Classes 1–3 are in `classes.md`; classes 4 and 4b in
`classes-sync.md`.

## Class 5 — cross-lane and wave-collective

**When it is the only route:** the algorithm is a *wave* algorithm. Ballot and its consumers —
find-first-set over a mask, `v_readlane` / `v_writelane` to move a value to or from a named lane,
`v_mbcnt` for a lane's prefix position, `v_bcnt` for a population count. None of these has a
tensor-level spelling on either tier, and they are what a register-resident top-k, a leader
election, or a prefix-compaction is actually built out of.

**The preconditions are the whole content of this class — but they attach to the *construction*, not
to the class.** Class 5 asks the compiler for a guarantee it does not otherwise owe you: **that
element order *is* lane order**, in the specific way the asm body assumes. There is more than one
way to earn that guarantee.

### Construction A — the 64-element ballot tensor

The strict one. When you are building a ballot / elect / prefix-rank primitive, this is the
construction to use. All five preconditions must hold at once, and all five are properties of the
layout and the launch rather than of the asm:

| # | Precondition | Why |
| --- | --- | --- |
| 1 | the tensor's length is exactly the wave size | one wave and no more, so there is no second wave to alias into |
| 2 | `size_per_thread = 1` | one element per lane, so the element index *is* a lane index |
| 3 | `warps_per_cta = 1` **and** `num_warps = 1` | a second warp would repeat the element range |
| 4 | `pack = 1` | packing lets one asm instance cover several elements, which breaks the bijection |
| 5 | element order equals lane order under the chosen layout | the mapping the asm depends on |

With those five, a `BlockedLayout([1], [64], [1], [0])` over a 64-element tensor gives an injective
element-to-lane map — **the thing you may not assume.** Do not assume it, assert it:

```python
gl.static_assert(x.shape[0] == 64)   # trace-time failure, not a wrong answer at run time
local = gl.convert_layout(local, gl.BlockedLayout([1], [64], [1], [0]), assert_trivial=True)
```

`assert_trivial=True` is the load-bearing half: a compile-time check that the conversion is a no-op,
i.e. that the tensor *already* had the mapping the asm assumes. **Dropping it changes no generated
code and removes the only guard on the idiom's central assumption.** Pin the launch at `num_warps=1`
in the same place.

The failure this prevents is the expensive kind: violating any of the five compiles cleanly and
yields a wrong answer at a shape you did not run.

### Construction B — a wider predicate tensor, used only as an operand

Preconditions 1 and 3 exist to guarantee a *bijection*, and you do not need a bijection when the
tensor is passed in as a **predicate** rather than read back per-lane. Building
`gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))` across four waves and electing
inside the body with `s_cmp_lt_u32 $0, 64` is correct, because the value is *computed by the
compiler under a declared layout* and merely compared — nothing reads a particular lane.

Use this shape for protocol-block predication (`classes-sync.md`).

### Construction C — a packed cross-lane op, where `pack > 1` is the point

Precondition 4 forbids packing because packing breaks the element↔lane bijection. A register/lane
**bit swap** is an operation on that very structure, so it needs the opposite: `permlane`-family
swaps run with `pack > 1` and a matching number of tied operands.

What replaces precondition 4 here is an **explicit software half**: a matching `reshape → permute`
plus a `gl.DistributedLinearLayout` and `gl.convert_layout(..., assert_trivial=True)` immediately
after. **The permute is the software half of the hardware swap and they must match exactly.**

### Invariants the asm cannot check for you

These apply in all three constructions.

- **Hazard spacing between cross-lane steps.** This target needs spacing between a VALU producer and
  its DPP reader — `s_nop 1` between DPP steps, and before *and* after a `permlane` swap. **The
  exception is a block that interleaves several independent streams, which supplies the spacing
  itself; remove a stream from one of those and it is incorrect.**
- **Step count encodes the group width.** 3 steps = 8 lanes, 4 = 16; a full 64-lane ladder is
  `row_shr:8/4/2/1` + `row_bcast:15` + `row_bcast:31`, terminated by `v_readlane_b32 $0, $1, 63`.
- **`row_bcast:15` must carry `row_mask:0xa` and `row_bcast:31` must carry `row_mask:0xc`.** These
  are **not** `0xf` and must not be "normalized".
- **`bound_ctrl:1` bakes in an identity of 0.** Legal for an unsigned max, or for a max seeded with a
  small positive floor (either a `v_max_f32` against a tiny constant before the ladder, or a
  `gl.maximum(lane_max, 1e-10)` after it). **Wrong for signed data.** Whether an unset `bound_ctrl`
  in an otherwise identical sibling kernel is intentional is `not established` — decide it for your
  own data, do not copy it.
- **`"=v,0"` on a *non-empty* body means the chain reads and writes one register.** Rewriting it to
  `"=v,v"` requires rewriting the first body line to read `$1`. Silent garbage otherwise.
- **`s_ff1_i32_b64` returns −1 on an empty mask, and there are two incompatible guard conventions.**
  Inside the body with `s_and_b32 $1, $1, 63`, or outside with `gl.minimum(first_lane, 63)` plus a
  re-check. **Copying an inside-guarded body into an outside-guarded call site double-guards; the
  reverse reads lane −1.** Know which convention the body you are copying uses.
- **A ballot result is a per-lane broadcast and must be collapsed** — follow the asm with
  `gl.sum(gl.gather(x, zero, 0), 0)`. Skipping it leaves one copy per lane.
- **Electing the lowest set lane is a correctness device, not an optimization.** A find-first-set
  after a ballot is what makes a top-k tie-break deterministic; without it, equal scores resolve
  non-deterministically and routing is unstable run to run.

**Verify:** the mnemonics in the disassembly, plus a case whose answer depends on the lane mapping
being what you claimed — **a ballot over a mask with a known population, not a uniform input.**

## Class 6 — register-class control

**When it is the only route:** you need a value in a particular register *file*, or in *adjacent*
registers, or in a *named physical* register, and no tensor-level construct expresses that. The asm
text here is usually empty — **the constraint string is the whole instruction.**

> **Read the scope of this class narrowly, because the mechanism is far more common than the
> intent.** Empty tied blocks are everywhere, and only a small minority are doing register
> *placement*; the rest are doing dependency shaping and belong to `classes.md ## Class 2`. That one
> conflation is the reason a single-axis reading of these classes is not reproducible. Use the
> two-axis test (`classifying.md`) and put a site here **only when deleting it would change which
> instructions get selected downstream** — not merely when it would change the schedule.

### The three placements that qualify

- **Register file.** `"=s,0"` on an empty asm asserts *this value is wave-uniform, keep it in an
  SGPR*; `"=v,0"` is the VGPR form. The point is downstream instruction form: without it, a
  following VMEM uses a vector base instead of scalar-base-plus-vector-offset. The same intent can
  be spelled with one real instruction (`s_mov_b64 $0, $1` with `"=s,s"`) — a zero-instruction
  versus one-instruction choice for one semantic, and both arms being asm means such a site is **not
  an instance of the plain-Gluon `else:` delete-safety rule.**
  **Check the binding before you write one of these for a matrix accumulator.** The technique is no
  longer only a user-level trick: the AMD backend now wraps each matrix-tile accumulator in an empty
  inline asm with a tied register-class constraint to steer it into a chosen register file, and
  documents the mechanism in a source comment saying exactly what this section relies on — *no
  instruction is emitted, but the value has to sit in the given register class*. Where your build
  exposes that as an argument on the matrix builtin, **that is the supported spelling and it is the
  one to use.** Two limits on how far this validates the hand-written form: upstream constructs it
  inside the compiler as a raw LLVM inline-asm op, **not** through the elementwise door; and the
  user-facing knob is a validated enum, marked experimental, rather than a free-form constraint
  string. Reaching the same mechanism through `constraints=` on the elementwise door is exercised by
  **no upstream test**, so a regression on that path is yours to catch — which is the argument for
  step 1.5 and for keeping the mnemonic check that survives into CI.
- **Adjacency.** A **64-bit tied operand forces an aligned register pair**, which is what a 64-bit
  mnemonic needs and what nothing at the tensor level will arrange. Pack two FP32 values into a
  `uint64`, pass them through a tied identity, split them back. The `uint64` is a **register-pairing
  device, not integer arithmetic** — say so in a comment. Check the round trip with
  `gl.convert_layout(result, acc.type.layout, assert_trivial=True)`.
- **Absolute physical assignment.** Constraints may name a *specific* physical register (`"=&{v0}"`
  … `"=&{v11}"`) so a hand-written body can address `v[0:3]`, `v[4:7]`, `v[8:11]` in a
  `global_load_dwordx4`. Numbered clobbers (`~{s0}`…`~{s7}` where a body does `s_load_dwordx8
  s[0:7]`) are the related, commoner form. **This is the least portable thing in this reference: it
  survives no change in `pack`, in operand count, or in the surrounding register budget.** Use it
  only when an instruction form requires a specific register group.

### A use-site can assert a class on values it does not return

A site may be `v_mov_b32 $0, 0` with `"=v,s,s,s,s,s,s,s,s,~{memory}"`: the output is a discarded
zero, the asm text is dead, and **the eight `s` input constraints are the entire point** — they force
eight scale values into SGPRs at that program point, while the function returns the *pre-asm* tuple.
Nothing about such a site reads as load-bearing and all of it is. Comment it, or it will be deleted.

### `pack` and the constraint arity are one decision — and the relation is not identity

The shorthand "`pack` equals the number of `=v` slots" is true **only when the element type is at
least four bytes wide.** The general rule is the arity law in `shape-keying.md ### Deriving the
parameters`: with

> `S(T) = pack` when `sizeof(T) >= 4`, else `S(T) = ceil(pack × sizeof(T) / 4)`

the number of output slots is `Σ S` over the `dtype` tuple, the number of input slots is `Σ S` over
`args`, tie digits are `0 .. n-1` in order, and `$n` counts **slots**, not tensors.

On a narrow `dtype` the two numbers come apart: `pack=2` on `gl.bfloat16` needs **one** `=v`, not
two. **The failure is silent in the dangerous direction** — hand a width-parameterized generator
`pack` instead of `S` and the tie covers a prefix of the fragment, the call compiles, the answer is
unchanged, and the liveness or ordering property you thought you had covers half the data.

Widening a tie therefore means changing `pack`, the slot count and the tie digits **together**.
Three disciplines for keeping them in step, in increasing order of safety:

1. **Hand-typed.** Sixteen `=v`, then `0..15`, then `pack=16`, with no assertion. Re-tile and you
   retype all three consistently — and nothing catches you if you do not.
2. **Asserted.** A constexpr picks one of two literal strings and a
   `gl.static_assert(PACK == 8 or PACK == 16)` makes the choice exhaustive. This is the shape a
   narrow `dtype` forces: `pack=8` is *four* slots and `pack=16` is *eight*.
3. **Generated — the safe form.**

```python
@triton.constexpr_function
def _identity_constraints(slots):          # NOTE: slots, not pack
    return ",".join(["=v"] * slots + [str(i) for i in range(slots)] + ["~{memory}"])

acc = gl.inline_asm_elementwise("", _identity_constraints(S), [acc],
                                dtype=gl.float32, is_pure=False, pack=PACK)
```

**Write your generator to take `S`, not `pack`.** A generator that takes `pack` is correct only
while every call site is 4-byte-wide, and it fails silently the first time someone uses it on
`bfloat16`.

**Generators are not interchangeable — read one before you reuse it.** Two that look identical may
differ in whether they append `~{memory}`; one may make the clobber **conditional on a tile flag**,
so that the same call site is a memory-motion barrier under one setting and a pure liveness pin
under another.

**Two arithmetic traps ride along.** An expression like `BM * BN // 256` or `values.numel // 256`
encodes **4 waves × 64 lanes** — a launch assumption written nowhere else, which breaks the moment
`num_warps` changes. And there is an upper bound on operands per asm block worth respecting; capping
with `min(64, …)` is a legitimate concession, not a smell.

**Bitcast around the tie.** Converting f32 → i32 before the tie and back after is deliberate: tying
the float directly is a different thing, and possibly an FP-interpreted one. **Copy the bitcast with
the tie.**

**`pack` is a shape-binned parameter, not a two-way choice.** Compute it — one fence width per
regime, derived from the tile or stepped down as M grows. The constraint string and `pack` may be
switched by the **same** flag:

```python
ties = "=v,=v,0,1" if WIDE_BATCH else "=v,=v,=v,=v,0,1,2,3"
pack = 2 if WIDE_BATCH else 4
```

Treat it as a knob selected per shape like the rest of this layer, and re-derive it when the tile
changes (`shape-keying.md`).

**Verify:** read the register operands in the disassembly. **A class pin that silently did not take
looks identical in source and in behaviour until something spills.**
