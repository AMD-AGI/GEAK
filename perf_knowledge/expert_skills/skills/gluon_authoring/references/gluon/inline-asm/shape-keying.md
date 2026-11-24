# Shape-keying: inline asm is a per-M specialization

Part of `../inline-asm-reference.md`. Read before writing a call and again when the tile changes.

**Five things can be keyed on a tile constant, sometimes three in one call:** asm text
(`"s_waitcnt vmcnt(0)" if COUNT == 0 else "s_waitcnt vmcnt(4)"`), the opcode
(`"s_load_dwordx2" if POSITION_TYPE == gl.int64 else "s_load_dword"`), constraint string **and** `pack`
together, the clobber list (`'~{v8},…,~{v15}' if ROWS == 32 else '~{v8},…,~{v11}'`), and comparison
polarity (`v_cmp_ne_u32` vs `v_cmp_eq_u32` as a function of M). Coupled keys (a `log2(ROWS)` shift + a
physical-VGPR clobber list + a `64 - ROWS` lane-mask operand) must move together. **Re-derive, never
transplant; audit at the shape you build; resolve computed (`Mx`) strings per bin.**

## Deriving the parameters — compute them, never copy them

### `pack` — elements per asm instance, per lane

`pack = fragment_elements / (num_warps × wave_size)` — not a vector width, not the `=` slot count.
`BM * BN // 256` is this with `num_warps=4`, wave64; at `num_warps=8` it is twice the per-lane fragment, and
on 3.8.0 / gfx950 the call then compiles with `undef` padding operands (compile-only check) — a wrong
derivation that raises no error.

```python
gl.static_assert(BM * BN % (NUM_WARPS * 64) == 0)
PACK: gl.constexpr = BM * BN // (NUM_WARPS * 64)
```

Other fragments give other formulas (`BLOCK_M // 2` = one accumulator's per-lane height): derive from the
fragment you name, comment it, assert it. Operand count per block has an upper bound (`min(64, …)` caps
are legitimate; exact limit: compile at increasing `pack` on your build). `pack > 1` breaks element↔lane
bijection: forbidden in ballot construction A, required in `permlane` construction C (`classes-lane-register.md ## Class 5 — cross-lane and wave-collective`).

### The arity law — `pack`, `dtype` and the constraint string are one decision

> `S(T) = pack` if `sizeof(T) >= 4`, else `S(T) = ceil(pack × sizeof(T) / 4)` (sub-4-byte elements are
> packed into 4-byte registers; a 64-bit element is **one** slot occupying an aligned pair).
> **Output slots** = Σ S over the `dtype` tuple · **input slots** = Σ S over `args` · **tie digits** =
> `0..n-1` in order · **`$n`** numbers slots, outputs first.

| `dtype` | `pack` | output slots |
| --- | --- | --- |
| `gl.float32` | 4 | 4 — `"=v,=v,=v,=v,0,1,2,3"` |
| `gl.bfloat16` | 2 | **1** — `"=v,0"` |
| `gl.bfloat16` | 8 | **4** |
| `gl.uint16` | 16 | **8** |
| `gl.uint64` | 1 | **1** — `"=s,0"`, two registers |
| `(gl.float32, gl.bfloat16)` | 4 | 4 + 2 = **6**; first input `$6`: `"=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,~{memory}"` |

**A slot count that disagrees with Σ S at the given `pack` (too many or too few) aborts the compile** on
Triton 3.8.0 / gfx950 with an LLVM assertion ("number of output constraints does not match") rather than a
clean error; a wrong `pack` with a matching slot count compiles and splits or pads the fragment
(`classes-lane-register.md`). Generators must take `S`, not `pack`,
and `S` and `pack=` must come from the same expression. Without a generator on a narrow `dtype`, make
the choice exhaustive:

```python
gl.static_assert(PACK == 8 or PACK == 16)
if PACK == 8:
    constraints: gl.constexpr = "=v,=v,=v,=v,0,1,2,3,~{memory}"          # 8 bf16 -> 4 registers
else:
    constraints: gl.constexpr = "=v,=v,=v,=v,=v,=v,=v,=v,0,1,2,3,4,5,6,7,~{memory}"
```

**Operand numbering check:** no `$n` in the body may exceed `total_slots - 1`; a body that reads its last
input reaches it exactly. A tie digit's
*position* is not its meaning (`"=v,s,n,0"`: fourth entry, refers to output 0); an argument may appear
twice (vector + scalar) — de-duplicating breaks the asm.

### `is_pure` — a decision rule, not a description

`is_pure=True` means "I accept CSE with an identical call, hoisting out of the loop, sinking past the
next barrier, and deletion when the result is unread." If the block writes machine state, touches
memory, waits, brackets EXEC, or **its position is the product**, use `False`. `True` + `~{memory}` is a
contradiction. (Whether `is_pure` changes codegen on otherwise identical `"=v,0"` ties: not established —
hash the two dumps, `costs.md ## Verifying that it worked — write, compile, dump, check`.)

### `dtype`, register letters, clobbers

`dtype` sets result count, element width (→ `S`) and bit interpretation (bitcast f32→i32 around a tie is
deliberate). Omitted `dtype` (bare `s_setprio`) is legal; adding one changes the output register class.

| Spelling | Obligatory when | Silent failure without it |
| --- | --- | --- |
| `=s` | instruction needs a scalar operand, or the register file is the point | VMEM uses a vector base instead of scalar base + offset |
| 64-bit `dtype` + tie | an aligned register pair is needed | halves land in unrelated registers |
| `=&` | any output written before the last input read (≥2 instructions reusing `$0`, any loop) | output aliases an input |
| `=&{v0}`… | the body names physical registers | breaks on any change of `pack` / operands / budget |
| `~{memory}` | body touches memory, or constraining memory motion is the point | block reordered vs memory ops (still **not a fence**) |
| `~{scc}` / `~{vcc}` | body names them or writes them implicitly (`s_cmp*`, `s_cselect*`, `s_ff1*`, `s_sub_u32`, `s_bitcmp*`, `v_cmp_*_e32`) | live predicate corrupted |
| `n` | value must fold into the encoding (`vmcnt` depth, `writelane` slot) | on 3.8.0 / gfx950 a runtime value is not rejected: it compiles and LLVM silently drops the block — count the mnemonic |

Assertions that turn silent wrong answers into trace-time failures: `pack` divisibility, legal `PACK`
set, `0 <= depth <= MAX`, `x.shape[0] == 64` + `convert_layout(..., assert_trivial=True)` for lane
bijection or a tie round trip, `value.dtype == gl.float32` for a body's type assumption.

## Deciding the bins — a procedure

1. **Run the derivations at both extreme shapes.** Any derived value differs ⇒ you already have two bins.
2. **Check preconditions too:** wave-private LDS slot (breaks at >1 reading wave), a structural constant
   tied to the tile (`log2(ROWS)`, `64 - ROWS`), copies per panel (changes with width/geometry), lane bijection.
3. **Put boundaries only where a derived value or precondition changes.** A timing-only boundary is
   `not established` until its A/B is recorded (`deciding.md`).
4. **Vary the least invasive thing:** `pack` + arity → an immediate in the text → the clobber list (three
   coupled edits) → the mechanism (that is two kernels, not a knob).
5. **Keep bins from diverging:** one derivation helper, `static_assert` the legal bin values, keep the
   `else:` arm alive, test both sides of each boundary numerically against it, audit per instantiation.
