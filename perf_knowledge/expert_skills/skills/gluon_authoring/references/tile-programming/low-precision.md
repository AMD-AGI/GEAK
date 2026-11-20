# Low-Precision Side Path: a8w8 (BF8) and a4w4 (MXFP4)

Read this when the matrix path is FP8 or MXFP4 and uses scaled MFMA. These reuse
the same backbone (memory path / LDS / pipeline / slicing) plus a **scale side
path**; do not rebuild the whole backbone from scratch.

> **Scope: this page assumes a GEMM-shaped loop** — a pure matrix-to-matrix accumulator chain
> where the scale path's only job is to feed the matrix op in time. If the loop also carries a
> vector stage *between* two matrix phases (attention and its variants), the scale path and that
> stage compete for the same issue slots, and the wave-structure advice here can conflict with
> what the attention structure requires. Read `../workloads/attention-lowprec.md` for that
> intersection before applying this page to a fused/attention kernel.

Verify the dtype/arch gate first (`../hardware/planning-constants.md`): on gfx950 use
native OCP FP8 (`float8_e4m3fn` / `e5m2`); gfx942-oriented `e4m3fnuz` may upcast
and invalidate FP8 conclusions. `build_verification_level >= 2` is required
before adopting a scaled-MFMA pipeline.

**gfx942 downgrade (whole page).** Scaled MFMA (`mfma_scaled`, `v_mfma_scale_*`), MXFP4/fp6 and the
e8m0 scale path do not exist on gfx942, so the a8w8-scaled and a4w4 sections below are gfx950-only.
gfx942's fp8 is the **FNUZ** encoding (`float8_e4m3fnuz` / `float8_e5m2fnuz`) on the regular
`gl.amd.cdna3.mfma`; an OCP-fp8 tensor carried over from a gfx950 config must be converted, and a
block scale is applied as a separate VALU descale rather than inside the matrix op. Per-arch dtype
cells: `../hardware/capability-matrix.md`.

**Correctness-gate precondition (check before choosing a low-precision matmul).** Read
the TYPE of the task's correctness oracle, not just its abs/rel tolerance. A
cosine/angular-similarity gate can be STRICTER than the abs/rel gate and can reject an
fp8/fp4 scaled-MFMA path that passes abs/rel — because reducing the matmul to fp8
shifts the output direction more than its magnitude. If such a gate is present, keep
the matmul in bf16 (dequant -> bf16 MFMA) and put precision reduction only on the
**dequant/scale side path** (e.g. a hardware scaled-upcast fuse), not on the matmul
itself. Decide this before building the scaled-MFMA pipeline, so a tight cosine gate is
not discovered after the rewrite (`../method/close.md`).

## When scaled MFMA is required here, and when it is not

**The constraint is per-`(version, M, N, K)`, not per-dtype.** `v3.8.0`'s MFMA v4
table registers `mfma_f32_16x16x32_fp8_fp8` and `mfma_f32_32x32x16_fp8_fp8` (plus
the fp8/bf8 mixes), so those shapes are reachable from the **regular** matrix op.
FP4 (e2m1) has no entry in that table at all, and neither do the unregistered fp8
`(M, N, K)` combinations — **those** are what `cdna4.mfma_scaled` is required for.
The `no matching matrix core intrinsic ... f8E4M3FN` error is that registry
missing **one shape**, not a dtype-level prohibition: when it fires, look up which
`(M, N, K)` is registered for your target rather than abandoning the dtype. When
there is no explicit descale, pass `None` scales plus the format string
(`mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)`; a unit e8m0 scale is
materialized internally). Plain Triton `tl.dot(fp8)` instead lowers to the
scale-less `tt.dot_scaled`, so a faithful transcription will show this
plain-vs-Gluon instruction gap. Per-dtype `k_width` / `instr_shape` / scale / acc
deltas: `../hardware/capability-matrix.md`.

Copy this scaled-MFMA skeleton for fp4 and for any fp8 `(M, N, K)` the v4 table
does not register, then specialize formats, scales, and accumulator layout:

```python
# For a REGISTERED fp8 (M, N, K), plain cdna4.mfma reaches it; this is the path
# for fp4 and for the shapes the v4 table does not carry.
acc = gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
# Specialize the format and scale operands for the target dtype.
```

## a8w8 (BF8, e5m2)

- Tile: `256 x 256 x 128` (`BLOCK_K=128`, was 64 for FP16).
- Instruction: `gl.amd.cdna4.mfma_scaled(..., "e5m2", ...)`; `v_mfma_scale_f32_
  16x16x128_f8f6f4`, `cbsz/blgp=1/1`. ~32-cyc MFMA -> scheduler interleaves
  ~2 MFMA per mem op (not 4); double `BLOCK_K` if you need 4.
- `k_width = 32`. LDS: a padded shared layout with interval/padding pairs
  `[[1024, 16], [2048, 32]]` (that list is only the first constructor argument —
  `layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`).
- This kernel can pass `None` scale tensors with `"e5m2"` (scaled-MFMA API, no
  active descale tensor in the simplest form).
- Slicing: M+N (same as the FP16 slice-MN design) is the mature design; an N-only design at this
  tile can spill (tutorial: ~111 spills until the M+N fix).

## a4w4 (MXFP4, e2m1)

- Tile: `256 x 256 x 256`. Instruction: `mfma_scaled(a, a_scale, "e2m1", b,
  b_scale, "e2m1", acc)`; `cbsz/blgp=4/4`. ~16-cyc MFMA (same interleave density
  as FP16).
- Scale layout via `get_mfma_scale_layout()`.
- **Scale pipeline — two forms; the direct one is the mature one and the default.**
  - *Direct to LDS* (default): the scales go `buffer_load_to_shared` alongside the tiles, **no
    `ds_write` at all**. An earlier revision here said this was impossible because scales are below
    the per-thread width the async path encodes — that is the wrong test. What matters is the
    per-thread contribution of the *scale load itself*: a scale block that works out to 4 B per
    thread is exactly the 32-bit direct-to-LDS form, and it lowers. Check the arithmetic
    (`scale_bytes / threads`) against the architecture's encoded widths
    (`perf_knowledge/hardware/data/hw_constants.json` `direct_to_lds_bit_widths`; the rule and the
    scattered-run caveat: `../gluon/memory-reference.md ### Minimum per-thread granularity
    (applicability by dtype)`) rather than assuming scales are too small.
  - Going direct buys more than the copy: it removes the `ds_write` trap below, removes the
    four-deep register buffering, puts the scales in the **same commit group** as the tiles so one
    wait count covers both, and takes a `ds_write` out of the hot loop that a scheduling pass
    would otherwise have to reason about.
  - *Register round trip (GR -> LW -> LR)* (fallback, for a scale width that misses the legal set
    or a non-contiguous source run): `buffer_load` into registers, `store` to LDS, then
    `ds_read_tr` to reach the MFMA scale layout. Needs ~4 scale register buffers per operand for
    GR/compute overlap, and it drags in the `ds_write` trap below.
- **ds_write 400-cycle trap** (only on the round-trip form): the scale `ds_write` contends with
  `buffer_load_to_lds` on the LDS write port (~400-cycle stall); place MFMA after
  the `ds_write`. The `llirSched` scheduler that did this automatically when the dependency was
  visible is **fork-only** (absent upstream 3.8.0 —
`non-upstream-reserve.md ## 1. The LLIR-scheduler plugin`); upstream, place it by
  hand (`instruction-scheduling.md`) or author the pass (`llir-codesign.md`). If you can go direct,
  this whole hazard disappears rather than being scheduled around.
- Slicing: **M+N, same as the other two precisions.** An earlier revision here said N-only was the
  right stopping point for a4w4; the tutorial lineage says otherwise — its MXFP4 champion is the
  M+N variant, ahead of the N-only one on both throughput and matrix efficiency. The reason is the
  one that makes M+N pay anywhere: N-only leaves the global loads bunched into one region, and
  spreading them across four regions relieves the in-flight cap. Scale traffic does not change
  that. Treat M+N as the default for all three precisions and N-only as the earlier step in the
  progression, not as a precision-specific stopping point.

## Budget notes

- LDS now holds A tile + B tile + scale staging; recompute `LDS_bytes` including
  the side-path LDS before deepening the pipeline.
- `R_side` (scale register buffers) is part of `R_total`; count it against 512.

## int4 / int8 weight GEMM (W4A16) — dequant VALU is often the bound

When weights are int4/int8 and activations are bf16/fp16, the hot cost is usually the
**software int4/int8 → bf16 unpack VALU**, not HBM or MFMA-continuity. Run
`scripts/asm_loop_audit.py` and compare **VALU% vs MFMA% first** — the FLOP/peak
roofline has no dequant axis (`../hardware/roofline-models.md ## Low-precision /
dequant-VALU`). The native int4→bf16 offset converter is **binding-gated, not
arch-gated**: `hasattr(rocdl, 'cvt_off_f32_i4')` is **False on gfx942**, yet
`v_cvt_off_f32_i4_e32` assembles on gfx942/gfx950/gfx1100/gfx1201 alike — the gap is the
wrapper, not the silicon, so the fast path stays reachable via inline asm / the raw
intrinsic (probe `hasattr` first, then record a **binding** gap rather than a hardware
ceiling). The offset-binary magic-trick also needs **unsigned-offset** weights; a
two's-complement quant format makes the fix a host-side requantization, not a pure
kernel edit.

## Cheap fp32->bf16 downcast (cut the convert bubble)

When the fp32->bf16/fp16 `cvt` is a top inter-MFMA bubble (ATT convert-class share high), the cost
is LLVM's default RTNE-with-NaN-guard lowered as a ~5-6 VALU/element software sequence
(`v_bfe_u32` / `v_add3_u32` / `v_cmp` / `v_cndmask` / `v_lshrrev` / `v_perm`). **Read the loop's
asm first, per arch.** gfx950 has a native packed convert, `v_cvt_pk_bf16_f32` (CDNA4-only,
`../hardware/isa-mechanisms.md ## Instruction-rate facts that flip a lever's sign (CDNA3/4)`): if
the build already emits it, there is no software sequence to cut and this section does not apply;
if it emits the sequence, the members below apply, and reaching the native cvt directly has no
Gluon source API (inline asm / Tier-B). **gfx942 downgrade:** no native fp32->bf16 convert exists
(`llvm-mc -mcpu=gfx942` rejects it), so the software sequence is the only lowering and the members
below are the whole lever. A **previous revision said gfx950 also lacks the native convert** — that
contradicts the gfx950 ISA tables and is corrected here. Two cheaper members exist; **pick by
tolerance, and default to the safe one.**

**DEFAULT — round-half-up (safe, ~2 VALU/elem, bit-identical to RTNE on non-tie/non-NaN inputs).**
Add the bf16 rounding bias, then take the high half. When the values feeding the cast are finite
(e.g. `exp2` results, post-softmax probabilities), the NaN guard is dead code and this is a pure
free win with **no accuracy cost** — confirm bit-identity on the real tensor once.

```python
# inside @gluon.jit: round-half-up f32 -> bf16, no ties-to-even, no NaN guard
u = x.to(ttgl.int32, bitcast=True)
bf16 = ((u + 0x8000) >> 16).to(ttgl.int16).to(ttgl.bfloat16, bitcast=True)
```

**FALLBACK — round-toward-zero (`rtz`, truncation).** Same ~2 VALU/elem, but **biased**: relative
error ~2⁻⁸ vs RTNE's 2⁻⁹, and the bias **accumulates along a reduction axis**. Use it only when a
measured tolerance margin proves it safe — **never on a kernel already sitting near its oracle
tol**, where it is the likeliest single cause of a correctness failure.

```python
out = acc.to(ttgl.bfloat16, fp_downcast_rounding="rtz")   # cheaper cvt, BUT biased — re-check oracle
```

Ref: upstream `triton/language/core.py` `cast(..., fp_downcast_rounding=)` + `semantic.py`
(`'rtz' -> ROUNDING_MODE.RTZ`, truncating-only guard). Verify **both** members the same way:
convert-class inter-MFMA share down, VALU:MFMA down, and — the part that bites — re-run the oracle;
round-half-up should be unchanged to ~1 ULP, RTZ must be shown to stay under tol.

## The other direction: f32 accuracy on the bf16 matrix core (`bf16x3` / `bf16x6`)

Every section above spends accuracy to buy matrix-core throughput. This one runs the trade the
other way — it spends matrix-core throughput to recover accuracy — and it belongs on this page
because it is the same lever family read backwards, and because the choice between it and a
scaled-MFMA path is made once, at the same moment.

It applies in exactly one situation: **both operands are genuinely f32**, and plain `ieee` is
costing too much. The decomposition pass matches only when `A` and `B` are both f32, so this is
not a way to improve the accuracy of an input that is already bf16.

**That last sentence is a statement about the pass, not about the technique — read it that way.**
The guard is the pass's match condition, and where only one operand is f32 the pass simply does not
fire. What follows is that *the compiler* will not do the split for you, not that splitting is the
wrong move: you can write it out, the arithmetic is the same recurrence, and **the cost lands in a
different place** — rows of a matrix instruction rather than extra instructions. That case has its
own section below (`### One-sided decomposition: when only A is f32`), and it is the one production
reaches for.

**The number in the name is the dot count, so the cost model is exact and needs no measurement.**
Each f32 operand is split into bf16 components — two of them (`hi`, `mid`) for `bf16x3`, three
(`hi`, `mid`, `lo`) for `bf16x6` — and the products are accumulated:

| Precision | Component dots issued | Total |
| --- | --- | --- |
| `bf16x3` | `mid*hi`, `hi*mid`, `hi*hi` | 3 |
| `bf16x6` | `mid*mid`, `lo*hi`, `hi*lo`, then all of `bf16x3`'s | 6 |

So one logical f32 dot becomes three or six bf16 matrix ops. Decide whether the shape can afford
that **before** running anything; if the kernel is already matrix-issue-bound, a 3x multiplier on
matrix work is not recoverable by tuning elsewhere.

**On AMD these two are the entire non-`ieee` menu, and gfx942 has one that gfx950 does not.**
Checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0: the HIP backend's allowed dot input precisions are
`("ieee", "bf16x3", "bf16x6")` on every version, and `tf32` is added back **only when the target is
gfx942** — upstream's own comment calls it "Enable XF32 (TF32) for CDNA3 GPUs". This inverts the
usual direction of the downgrade: a `tf32` input precision carried over from a gfx942 config is not
merely slower on gfx950, it is **not in the allow-list there**. Port the decision, not the string.

Two behaviours that are part of the algorithm rather than side effects:

- **The result is not bit-identical to `ieee`, by construction.** The partial accumulator is
  NaN-scrubbed before the final `hi*hi` dot, and the caller's accumulator is added at the very end
  (the emulation accumulates from zero). An oracle that keys on NaN propagation reads differently
  here than under `ieee`, and that is expected rather than a defect to chase.
- **Upstream already explored the next step up and declined it.** A nine-dot variant is present in
  the pass source but commented out, annotated as not kept for lack of speedup. Treat six as the
  explored ceiling; there is no hidden `bf16x9` to reach for.

**Reachability differs by DSL, and this is the part that decides where to settle the question.**

| Path | How the precision is selected |
| --- | --- |
| plain Triton | `tl.dot(a, b, acc, input_precision="bf16x3")` — **per call site** |
| Gluon on CDNA | **not a call-site parameter.** `gl.amd.cdna4.mfma` / `cdna3.mfma` pass the global f32 default themselves, and `gl.dot_fma` hardcodes it away. The only selector is the process-wide `TRITON_F32_DEFAULT` environment variable (present and identically named on all four versions). |

That asymmetry has a practical consequence: on the Gluon path the setting applies to **every** f32
dot in the process at once, which is the process-level granularity the compiler-contract page warns
about generally. So settle the precision question while still in plain Triton, where it is per-dot
and A/B-able, and carry the answer across — rather than discovering after transcription that the
knob you need has no per-kernel form.

No extra pass flag is required: the bf16 decomposition is registered unconditionally, and only its
`tf32x3` companion sits behind an emulation flag. Selecting the input precision is sufficient.

### One-sided decomposition: when only A is f32

The common shape in a served attention epilogue is asymmetric: the probabilities are f32, the
values are already fp8 or bf16. The pass above will not fire, and the section above says so — but
the split is still available by hand, and the accounting is better than the two-sided case rather
than worse. Surveyed production uses it across three independent kernel groups, and a second,
disjoint archetype (a GDN prefill family) arrives at the same construction with two- and
four-component depths.

**1 — The split itself is the same recurrence the pass uses, written out.** Truncate, widen back,
subtract, repeat on the remainder:

```python
high     = probabilities.to(gl.bfloat16)
residual = (probabilities - high.to(gl.float32)).to(gl.bfloat16)
tail     = (probabilities - high.to(gl.float32) - residual.to(gl.float32)).to(gl.bfloat16)
```

**2 — There are no cross terms, which is why this is not the `bf16x3` cost model.** `B` is already
at or below bf16 precision, so widening it to bf16 is exact and splitting it would recover no
information — the second component is identically zero. `(P_hi + P_res + P_tail) · V` therefore
stays **one matmul** (or two), not three or six dots. The dot-count table above does not apply here.

**3 — Which axis the components go on is decided by what the matrix instruction has left over, and
that is also what decides how many components you can afford.** Two cases:

- **M is already full** — every row is real work, as when the instruction's M is covered by query
  heads or query tokens. Then double along **K**: `gl.join(value, value)` against the joined
  components, because `[lo, hi] · [v, v]^T = lo·v + hi·v` is the same sum.
- **M has spare rows** — the logical M is below the instruction's M, so rows are being issued and
  discarded anyway. Then put the components **on M** and pay **zero extra instructions**: with
  four heads and three components, twelve of a sixteen-row tile carry work that the instruction was
  going to issue regardless.

Read the direction of that carefully: **the component count is computed from the instruction's
leftover capacity, not chosen from an accuracy requirement.** Shipped code gates it exactly that
way — a constexpr that turns the third component on only at the head count where the rows exist.

**4 — Component count and `instr_shape` are one decision, not two.** Two components occupying four
rows of a sixteen-row tile means twelve rows of zeros; the answer is a narrower instruction
(`[4, 64, 64]` with `kWidth=4`) rather than a padded wide one (`[16, 16, 32]` with `kWidth=8`).
Choosing the instruction after fixing the component count re-introduces the waste the whole
construction exists to avoid.

**5 — RTZ is a correctness requirement here, not a style preference.** The split relies on
`residual = p − hi` having the same sign as `hi` so the components add rather than cancel. Truncation
guarantees `hi <= p`; round-to-nearest-even does not, and a `hi` that overshoots makes the residual
negative and the reconstruction worse than the single component it was supposed to improve. So
request truncation explicitly at the downcast (`fp_downcast_rounding="rtz"`), and fold back with
explicit parentheses — `high + (residual + tail)` — so the two small terms are summed before the
large one.

**Accuracy accounting, and the precondition it rests on.** bf16 carries eight significand bits, so
one component is ~8 bits, two ~16 (better than fp16's 11), three ~24, which is f32's significand.
That ladder holds **only because the quantity being split is `exp(s − max)`, which lives in [0, 1]**
— it is short of significand, not of exponent range. Applied to a tensor that needs the exponent
range, the components do not compose this way.

**Scope of the examples.** Only one file of the primary group was read line by line; its siblings
were surveyed by docstring and grep, so the construction above is documented from that one kernel
and should not be assumed uniform across the family. Nothing here is a performance claim — these
are structural observations with no timing attached, and the argument is about what is
expressible and what it costs in issued instructions.

## IR / asm acceptance signals

| Path | Confirm in IR/asm |
| --- | --- |
| scaled MFMA | `v_mfma_scale_f32_16x16x128_f8f6f4` with correct `cbsz/blgp` (1/1 BF8, 4/4 MXFP4) |
| scale side path (direct, default) | scale `buffer_load ... lds` in the same commit group as the tiles; **no** `ds_write` for scales |
| scale side path (round-trip fallback) | `ds_write` then `ds_read_tr` for scales; MFMA scheduled after the `ds_write` |
| dtype gate | operands are native FP8/FP4 (no upcast to FP16 in the hot loop) |

## Reprofile signal

The acceptance bar is beating the plain-Triton target line **for your shapes**, with budget and
IR agreement — not a reference throughput. Two things are worth knowing about what "good" looks
like on this path, and neither is a number:

- **A well-formed scaled-MFMA GEMM can sit very close to matrix-issue saturation**, because the
  scale side path is off the critical chain once it is staged correctly. So a large gap to the
  budget's matrix-issue ceiling is evidence of a *structural* fault (scale staging on the
  critical path, an unhidden `ds_write`, a tile that under-fills the matrix op), not of the dtype
  being intrinsically harder.
- **The narrower dtype does not reach the same fraction as the wider one for free.** The scaled
  path adds traffic the unscaled one does not have, so compare each dtype against **its own**
  calibrated ceiling; comparing an fp4 kernel's efficiency to an fp16 kernel's is not a reading.
