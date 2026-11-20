# Shape-keying: inline asm is a per-M specialization

Part of `../inline-asm-reference.md`. This chapter holds the parameter derivations — it is the one
to read before you write a call, and the one to re-read when the tile changes.

An asm site looks like one site. It usually is not: the asm is generated per shape, and **five
different things can be keyed on a tile constant** — sometimes three of them at once, in one call.

| what is keyed | how it looks |
| --- | --- |
| the **asm text** | `instruction: gl.constexpr = "s_waitcnt vmcnt(0)" if COUNT == 0 else "s_waitcnt vmcnt(4)"` |
| the **opcode itself** | `"s_load_dwordx2" if POSITION_TYPE == gl.int64 else "s_load_dword"` |
| the **constraint string** *and* `pack`, together | `"=v,=v,0,1" if WIDE else "=v,=v,=v,=v,0,1,2,3"` with `pack = 2 if WIDE else 4` |
| the **clobber list** | `'~{v8},…,~{v15}' if ROWS == 32 else '~{v8},…,~{v11}'` |
| the **comparison polarity** | `"v_cmp_ne_u32 vcc, 0, $4" if materialize else "v_cmp_eq_u32 …"`, with `materialize` a function of M |

**The worst case couples three at once.** A single site may key, on one tile constant: the asm text
(a shift immediate that is `log2(ROWS)`), the list of clobbered **physical VGPRs** (eight versus
four), and an operand (`64 - ROWS`, consumed to build a participating-lane mask). **All three must
move together; changing one is a silent miscompile of the other two.**

**The polarity case deserves its own warning.** One helper name can compile to `v_cmp_ne_u32` at one
shape and `v_cmp_eq_u32` at another. **Reading one instantiation tells you nothing about the other.**
When you audit an asm site, audit it at the shape you are actually building.

## What this means for the workflow

Inline asm is not a one-off escape hatch bolted onto a kernel. It behaves like every other layer of
the explicit-tile stack — selected per shape, re-derived when the tile changes. Two consequences:

- **Re-derive, do not transplant.** A site copied from one shape's file into another's may compile,
  assemble and produce wrong answers, because the thing that changed is invisible in the asm text.
- **When a site is `Mx` (computed `asm` or computed `constraints`), resolving it is mandatory before
  you classify it.** A body may be assembled from several interpolated fragments; an audit that reads
  only the visible literal misses most of it.

## Deriving the parameters — compute them, never copy them

Six things go into the call and five of them are functions of something else. Copying a working call
from another file transplants the *answers* without the *functions*, which is why a lifted site can
compile, assemble and be wrong. **Derive each one, and assert what the derivation assumed.**

### `pack` — elements per asm instance, per lane

**Definition first, because the name misleads.** `pack` is how many tensor elements one instance of
the asm body handles, *per lane*. It is not a vector width, not a register count, and **not** the
number of `=` slots — the arity law below is the actual relation.

**The derivation.** Name the fragment you want one asm instance to cover, then divide:

```
pack = (elements in the fragment) / (lanes that hold it)
     = tile_elements / (num_warps × wave_size)
```

A recurring expression like `BM * BN // 256` is exactly this with `num_warps = 4` and a 64-lane wave.
**That 256 is a launch assumption written nowhere else in the file.** Launch the same kernel at
`num_warps=8` and the correct divisor is 512; the expression still compiles, still produces an
integer, and now ties half the fragment.

**The divisor changes whenever the launch geometry changes, and the only thing that couples the two
is you.** Assert it:

```python
gl.static_assert(BM * BN % (NUM_WARPS * 64) == 0)
PACK: gl.constexpr = BM * BN // (NUM_WARPS * 64)
```

**A second derivation exists and it is not the same one.** An expression like `BLOCK_M // 2` is not a
tile-over-threads quotient at all; it is the per-lane *height* of that particular accumulator. So the
rule is not "use `tile // 256`" — it is:

> **Derive `pack` from the fragment you are naming, write the derivation in a comment, and assert
> it.**

Most call sites in the wild use a literal — 1 by a wide margin, then 2, 4, 8, 16. A literal is fine
*if* you can still state which fragment it came from.

**What bounds it.** There is an upper bound on operands in one asm block; capping with
`min(64, …)` is a legitimate concession rather than a smell. **Where exactly the limit lies is
`not established`** — what would settle it is a compile at increasing `pack` until the backend
refuses, on your build.

**What `pack` interacts with.** `pack > 1` lets one asm instance cover several elements, which breaks
the element↔lane bijection. That makes it *forbidden* in the strict cross-lane construction and
*required* in the packed-swap construction — the same parameter, opposite obligations, decided by
which construction you are in (`classes-lane-register.md ## Class 5`).

### The arity law — `pack`, `dtype` and the constraint string are one decision

**This is the rule to internalise, and it is not "`pack` equals the number of `=v` slots".** That
shorthand is true only for element types of four bytes or more. The general law:

> Let `S(T) = pack` when `sizeof(T) >= 4`, and `S(T) = ceil(pack × sizeof(T) / 4)` when
> `sizeof(T) < 4`.
>
> - **Output slots** = `Σ S(T_i)` over the `dtype` tuple (a scalar `dtype` is a 1-tuple).
> - **Input slots** (ties plus plain inputs) = `Σ S(T_arg)` over `args`.
> - **Tie digits**, when present, are exactly `0 .. n-1` in order, `n` = number of output slots.
> - **`$n` numbering** runs over *slots*, outputs first — **not** over tensors.

The two halves of `S` are two facts this reference already states, made arithmetic: sub-4-byte
elements are packed into 4-byte registers, and a 64-bit element is *one* operand occupying an aligned
pair.

Worked instances:

| `dtype` | `pack` | `S` | output slots |
| --- | --- | --- | --- |
| `gl.float32` | 4 | 4 | 4 — `"=v,=v,=v,=v,0,1,2,3"` |
| `gl.bfloat16` | 2 | 1 | **1** — `"=v,0"` |
| `gl.bfloat16` | 8 | 4 | **4** |
| `gl.uint16` | 16 | 8 | **8** |
| `gl.uint64` | 1 | 1 | **1** — `"=s,0"`, occupying two registers |
| `(gl.bfloat16,)×3` | 8 | 4 each | **12** |

The shorthand and the law disagree on every narrow-`dtype` row, and the law is the one that holds.

**Work one instance all the way through, because a tuple `dtype` at `pack > 1` is where the law
actually earns its keep** — it is the case the shorthand gets wrong by the largest margin, and the
one most often written from memory:

```
dtype = (gl.float32, gl.bfloat16)      pack = 4      args = [acc, scale]

S(float32)  = 4                 (sizeof 4 >= 4, so S = pack)
S(bfloat16) = ceil(4 * 2 / 4) = 2

output slots = 4 + 2 = 6        -> "=v,=v,=v,=v,=v,=v"
tie digits   = 0,1,2,3,4,5      -> appended in order
input slots  = S(acc) + S(scale) = 4 + 2 = 6     (consistent: 6 ties)
$n numbering : $0..$5 are outputs, the first input slot is $6

constraints = "=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,~{memory}"
```

Two things to notice. The two tensors contribute **different** slot counts, so you cannot get the
total by multiplying anything by two. And the total is six, not eight — writing `2 × pack` because
"there are two tensors" is the same error as writing `pack` because "there is one".

**What breaks when `pack`, `dtype` and the string disagree.** Too many slots is usually a compile
failure — the good direction. **Too few is the dangerous one:** the tie covers a prefix of the
fragment, the call compiles, the answer is unchanged, and the liveness or ordering property you
thought you had covers part of the data. Nothing downstream notices — no warning, no wrong number,
no missing instruction.

This is also the precise failure a `constexpr_function` generator prevents, **and the precise failure
it causes if you hand it `pack` when the law wants `S`.**

### Generating the constraint string — and the trap in the generator

The safe shape:

```python
@triton.constexpr_function
def _tie_constraints(slots, memory=True):        # NOTE: slots, not pack
    c = ["=v"] * slots + [str(i) for i in range(slots)]
    return ",".join(c + ["~{memory}"]) if memory else ",".join(c)
```

A generator written to take **`pack`** is correct only while every call site has a 4-byte element
type, and it fails silently the first time someone uses it on `bfloat16`. **Derive `S` from `pack`
and `dtype` in one place, pass `S`,** and the seam closes.

And the generator's argument and the `pack=` argument must be derived from the **same expression**,
not two expressions that happen to agree today.

Where a narrow `dtype` is involved and you are not generating, the honest fallback is to write the
strings out and make the choice exhaustive:

```python
gl.static_assert(PACK == 8 or PACK == 16)
if PACK == 8:
    constraints: gl.constexpr = "=v,=v,=v,=v,0,1,2,3,~{memory}"          # 8 bf16 -> 4 registers
else:
    constraints: gl.constexpr = "=v,=v,=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,6,7,~{memory}"
```

### Operand numbering — three separate traps in one number

1. **Outputs first, and `dtype` decides how many outputs there are.** A 3-tuple `dtype` makes the
   first input `$3`. Count *slots*, not tensors: a `pack=8` bf16 output is four slots, so the first
   input is `$4`.
2. **A tie digit's position in the constraint list is not its meaning.** In `"=v,s,n,0"` the digit is
   fourth in the list and refers to **output zero**, while matching the third argument. Reorder the
   list and every `$n` in the body silently renumbers.
3. **An argument may legitimately appear twice** — once as a vector operand and once as a scalar —
   because the body needs it in both register classes. **A tidy-up that de-duplicates the argument
   list breaks the asm and compiles clean.**

The cheap mechanical check for all three: recount the slots with the arity law and confirm the
highest `$n` in the body is `total_slots - 1`.

### `is_pure` — a decision rule, not a description

Two questions, in order:

1. **Is the block's only externally visible effect its return value?** If it writes machine state,
   touches memory, waits, brackets EXEC, or if **its position in the instruction stream is the
   product you are buying** — then no. `is_pure=False`.
2. **If yes: is the result consumed downstream?** If not, `is_pure=True` will get the block deleted,
   silently and correctly, because you promised it had no effects and then made its effect unused.

Stated positively:

> **`is_pure=True` means "I would be happy for this block to be CSE'd with an identical call
> elsewhere, hoisted out of this loop, sunk past the next barrier, or deleted when its result stops
> being read."**

Say those four things out loud about your site. An empty tied identity placed for its position fails
all four, and takes `False`.

**The mechanical cross-check.** `is_pure=True` together with `~{memory}` is a contradiction — purity
permits exactly the motion the clobber forbids. If your call has both, one of them is wrong.

**What is `not established`:** visually identical `"=v,0"` fences appear spelled both ways in the
wild, with no stated reason. The *rule* above is decidable regardless; what is unknown is whether the
two forms generate different code, and a TTGIR/ISA diff of the two would settle it.

### `dtype` — three decisions wearing one name

- **How many result tensors** you get (a tuple gives a tuple).
- **The element width**, which feeds `S` and therefore the whole arity law.
- **The interpretation of the bits.** Bitcasting `float32 → int32` before a tie and back after is
  deliberate: tying a float directly is a different thing and possibly an FP-interpreted one. **Copy
  the bitcast with the tie.**

Omitting `dtype` is legal and appears on bare `s_setprio` sites. **Adding one to such a site changes
the output register class** — it is not a tidy-up.

### Register-class letters, `=&`, and the clobber list — when each is obligatory

| Spelling | Obligatory when | What silently breaks without it |
| --- | --- | --- |
| `=v` | the consuming instruction needs a vector operand, or the value must be vector-resident here | nothing visible until the allocator makes the opposite choice |
| `=s` | the instruction requires a scalar operand, **or** the register file *is* the point | the downstream VMEM uses a vector base instead of scalar-base-plus-offset |
| a 64-bit `dtype` with a tie | you need an **aligned register pair** for a 64-bit mnemonic or a packed pair | the halves land in unrelated registers and the mnemonic cannot be formed |
| `=&` (early clobber) | the body writes any output **before** the last read of any input — mechanically: ≥2 instructions and `$0` reused as a temporary, or any loop | the allocator may alias output and input; the second instruction reads a register the first destroyed |
| `=&{v0}` … (physical) | the body must name `v[0:3]` literally, e.g. for `global_load_dwordx4` | nothing — until any of `pack`, operand count or the surrounding register budget changes |
| `~{memory}` | the body performs a memory operation, **or** constraining memory motion is the point | the block may be reordered against memory operations. Practice is inconsistent on the self-waiting scalar-load case; whether it matters there is **`not established`**, and an ISA diff would settle it |
| `~{scc}` / `~{vcc}` | the body names them, **or** uses an instruction that writes them implicitly — `s_cmp*`, `s_cselect*`, `s_ff1*`, `s_sub_u32`, `s_bitcmp*`, any `v_cmp_*_e32` | a live predicate is corrupted between the asm and its consumer |
| `n` | the operand must be folded into the instruction encoding (a `vmcnt` depth, a `writelane` slot) | it will not compile with a runtime value — which is the property you want |

**`~{memory}` is not a fence and this table does not make it one.** It constrains the *compiler*.
Real ordering is `gl.barrier()` / `wait_group` (`classes.md ## Class 2`).

### The assertion each derivation earns

Assertions are rare in the wild. **That is a description of common practice, not a standard** — and
it is the single cheapest gap to close. Each of these converts a silent wrong answer at an unrun
shape into a trace-time failure:

| Derived thing | The assertion |
| --- | --- |
| `pack` from the tile | `gl.static_assert(tile_elements % (NUM_WARPS * 64) == 0)` |
| a legal set of `pack` values | `gl.static_assert(PACK == 8 or PACK == 16)` |
| a hand-counted wait depth | `gl.static_assert(0 <= NEWER_PANELS <= MAX_PANELS)` |
| a lane bijection | `gl.static_assert(x.shape[0] == 64)` + `gl.convert_layout(..., assert_trivial=True)` |
| a layout round trip across a tie | `gl.convert_layout(result, acc.type.layout, assert_trivial=True)` |
| an element type the body assumes | `gl.static_assert(value.dtype == gl.float32)` |

## Deciding the bins — a procedure

How to decide, for a kernel in front of you, whether you need more than one implementation and where
the boundary goes.

**Step 1 — find out whether you already have a bin boundary, before you argue about one.** Run the
derivations above at the two extreme shapes you must support. **If any derived quantity takes a
different value, you have two bins already** — not because anyone chose to, but because `pack`
changed, so `S` changed, so the constraint string changed, so the call changed. There is nothing left
to decide except how to express it.

**Step 2 — check the preconditions, not just the parameters.** A parameter that survives the range
can still be riding a precondition that does not:

| Precondition | Where it breaks |
| --- | --- |
| the shared-memory slot is **wave-private** | as soon as more than one wave reads it — guard on `NUM_WARPS == 1` |
| a structural constant equals a tile constant | a shift that is `log2(ROWS)`, a mask built from `64 - ROWS` |
| the outstanding-copy count per panel | changes with the copy width and the panel geometry; one kernel may need two different derivations in two places |
| the element↔lane bijection | the predicate tensor stops being one wave (construction A versus B, `classes-lane-register.md`) |

**Step 3 — put the boundary where a derived quantity or a precondition changes value, and nowhere
else.** A boundary you cannot tie to one of those is a boundary you cannot re-derive later, and it
will drift. In particular: **a boundary motivated only by a timing difference is outside this
procedure** — record it, mark it `not established`, and name the A/B that produced it
(`deciding.md`, Lane 2).

**Step 4 — vary the least invasive thing that works.** In increasing order of what a bin costs you
to own:

1. `pack` and the constraint arity only (they move together by the arity law);
2. an immediate inside the asm text that is a function of the tile — a shift that is `log2(ROWS)`;
3. the **clobber list**, when physical registers are involved — remember this is three coupled edits
   (asm text, clobber list, operand);
4. the **mechanism** — a different M cell on either side. **A bin that varies the mechanism is two
   kernels, not one kernel with a knob**, and should be read as such. An honest example is a
   compile-time flag selecting an explicit `s_waitcnt lgkmcnt(0)` on one arm and a
   dependency-carried tie on the other: two mechanisms, one obligation.

**Step 5 — keep the bins from diverging in correctness.** This is the part that decays over time:
sibling per-shape files drift until two of them differ in, say, whether `bound_ctrl` is set on the
same step, and nobody can say whether that was intended. Five defences, in the order they pay:

- **One helper, parameters derived** — not one file per shape with its own spelling. The derivation
  lives in one `constexpr_function` and every bin calls it.
- **`gl.static_assert` the legal bin values**, so an unlisted shape fails at trace time rather than
  running a formula outside its domain.
- **Keep the `else:` arm alive in every bin.** It is the oracle for the numeric comparison at each
  boundary, and it is what makes the whole site removable later.
- **Test at the boundary shapes**, both sides, with a numeric comparison against that `else:` arm —
  not at a shape in the middle of a bin.
- **Audit at the shape you are building.** One helper name can compile to two different comparison
  polarities. Reading one instantiation tells you nothing about the other — **the strongest argument
  for resolving `Mx` sites per bin rather than once.**

**How common is binning?** Common enough that it is not an edge case in this layer: a large minority
of real sites sit under a compile-time guard, and a meaningful share of those guards test a tile or
shape constant directly.
