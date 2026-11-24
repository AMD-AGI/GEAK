# Classes 1–3: instruction selection, scheduling control, machine state

Part of `../inline-asm-reference.md`. Classes 4/4b: `classes-sync.md`; 5/6: `classes-lane-register.md`.

## Class 1 — instruction selection

**Class 1 applies when no binding emits the instruction — `hasattr(rocdl, '<name>')` is False and no `gl.*`/`tl.*` op
lowers to it — but the mnemonic assembles for your `-mcpu`** (wrapper gap; worked case `v_cvt_off_f32_i4` on gfx942, `../../hardware/cdna3-gfx942.md`). Intent I-C,
usually genuinely pure: `is_pure=True`, keep a plain-Gluon `else:` arm behind a constexpr (it is the
removal path and the A/B).

### The instruction is rarely the whole primitive

The caller-side numerical contract lives in Python; lifting the asm line alone gives silently wrong answers.

- **`v_exp_f32` / `v_log_f32` are base 2** — caller pre-multiplies by log2(e).
- **`v_rcp_f32` is about 1 ULP and flushes denormal inputs and results** (also under
  `denormal-fp-math-f32=ieee`); alone it returns rcp(±inf)=±0, rcp(±0)=±inf and NaN for NaN (GPU-checked,
  gfx950 / 3.8.0). The Newton step `r = fma(r, fma(-x, r, 1.0), r)` makes normals correctly rounded but turns
  **x=±inf, x=±0 and denormal x** into NaN (`fma(-x, r, 1)` is `inf·0`), so an inf-only guard is not enough:
  after Newton fix up both inf (→ ±0) and zero (→ ±inf), or use `v_div_fixup_f32` / plain division when
  specials can occur. For subnormal operands or results scale the operand (2⁻⁶⁴-style) and apply Newton
  *after* the rescale.
- **`v_div_fixup_f32 $0,$1,$2,$3`**: `$1` quotient, `$2` denominator, `$3` numerator — not commutative
  (example, if aiter is available: `aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_decode_qkvz.py`).
- **Borrowed comparators carry a range precondition**: `v_max_f64`/`v_min_f64` on `uint64` keys order
  correctly only for positive-normal encodings; `v_max_u32` on float bits only after the sign is stripped. Assert it.

**Verify:** mnemonic count per loop iteration in the `.s` + numeric diff vs the `else:` arm (a silent
generic fallback still gives right answers). Record the target constraint next to the kernel.

## Class 2 — scheduling control

`asm=""` with a tied operand — the commonest shape in this area — emits nothing and only creates a data
dependency; standalone `s_nop` at a stage head is a phase lever (not a hazard fix).

### It is not a fence

Empty tied asm + `~{memory}` constrains only the compiler; keep the real `wait_group` / `gl.barrier()`
next to it. **Its effect can be non-local:** in one measured case a one-line drain with a memory clobber raised
the `s_waitcnt` count elsewhere (around an MFMA), and removing it reduced the count while making the
kernel slower — **an `s_waitcnt` count is not a reliable sync-cost metric.**

What it does buy: the value is materialized in the stated register class at that point, its live range
covers the point, and the compiler does not hoist later memory ops above it (LLVM `~{memory}` semantics; liveness tool — arity in
`shape-keying.md ## Deriving the parameters — compute them, never copy them`).

### `s_setprio` — the non-empty member of this class

Pinning, strongest first: (1) tie the whole fragment, (2) tie a live input (`"=s,0"`), (3) `~{memory}`,
(4) dummy `s_mov_b32 $0, 0`, (5) bare `=s` output. Prefer 1–2 when placement matters (whether 4–5 float
in practice: not established). `asm="s_setprio 1"` with `"=s"` does not write `$0` (no instruction in the text does) — append `s_mov_b32 $0, 0`
if anyone consumes it. Brackets may be **asymmetric on purpose** under shape guards — "balancing" is not a safe edit.

### `s_nop` is in this section by mnemonic and usually does not belong to it

`s_nop N` inside a multi-line body is an **ISA hazard cover** (VALU→DPP spacing) — deleting it is a
correctness bug. A body with *zero* `s_nop`s may get its spacing from interleaved independent streams;
removing a stream breaks it. Check required spacing with `amd_isa.py --arch gfx950 hazard`. Phase
effects are non-monotone: sweep placement (`../../tile-programming/llir-codesign.md ## Phase pacing: moving a burst instead of removing work`).
**Risk:** a scheduling win is tied to the compiler version it was measured on and can vanish silently on upgrade.

## Class 3 — machine state

No Gluon surface for MODE on the 3.8.0 wheel. `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0|1` toggles saturating vs
IEEE narrow-float conversion (FP16-named field, used on FP8 paths; exact semantics on gfx950: inferred,
check the ISA MODE table). Sticky for the wave → restore it yourself; **`is_pure=False`** (a pure block with an
unused result is silently deleted — compile-checked on 3.8.0 / gfx950), and anchor the write (below).

### Anchor the write to a live value, or it floats

```python
# (a) mint a token and route it into what must not float above the write
mode_token = gl.inline_asm_elementwise(
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\nv_mov_b32 $0, 0",   # v_mov only fills the mandatory output
    "=v,~{memory}", [], dtype=gl.int32, is_pure=False, pack=1)
# (b) thread real values through so the restore cannot sink past the arithmetic
acc, _ = gl.inline_asm_elementwise(
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1",   # hwreg(reg, offset, WIDTH): width 0 does not assemble (3.8.0 / gfx950)
    "=v,=v,0,1,~{memory}", [acc, scale], dtype=(gl.float32, gl.float32), is_pure=False, pack=1)
```

In (b) the 2-tuple `dtype` gives `2 × pack` output slots: at `pack=4` it needs 8 `=v` and ties `0..7`;
the string above has 2, which on Triton 3.8.0 / gfx950 aborts the compile with an LLVM assertion
("number of output constraints does not match"), not a clean error — `shape-keying.md ### The arity law — pack, dtype and the constraint string are one decision`.

### It is half of a pair — and the unpaired case is the most copy-hostile site there is

A kernel may clear the bit for one CTA class and **never restore it**, relying on a launch partition
(`if pid < router_tiles:`). Lifting that helper into a kernel whose CTAs later do narrow-float work gives
silently wrong numerics. **Verify** on input that actually overflows + `s_setreg` count in the `.s`; a
clean run on benign input proves nothing.
