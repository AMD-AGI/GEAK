# Gluon Matrix Reference (CDNA MFMA, gfx950 / gfx942)

Default MFMA shapes (ISA): `../hardware/isa-mechanisms.md` §Default MFMA shapes.
Required companions: `imports-and-launching.md`, `layout-reference.md`, and
`../hardware/capability-matrix.md` (evidence for target/dtype/op and `instr_shape`).
Use this file when lowering a hot matrix subpath to MFMA or CDNA4 scaled MFMA. RDNA WMMA:
`rdna-wmma-reference.md`. gfx1250: separate sub-target.

Keep the whole chain in view: **result layout -> operand layouts -> convert ->
accumulator -> target op -> epilogue/store layout.**

## Matrix Lowering Order

`tl.dot` / `tl.dot_scaled` are not rename targets. There is no generic `gl.dot`;
a Gluon matrix path must use a target-specific MFMA op with explicit result and
operand layouts, or the kernel should remain plain Triton — or, when the tile
cannot fill a matrix instruction at all, take the contraction off the matrix core
entirely (`## When the matrix core is the wrong instrument (gl.dot_fma)`).

1. Confirm the selected matrix subpath is on the benchmark hot path.
2. Check `../hardware/capability-matrix.md` for target, dtype, op path, evidence.
3. Choose the CDNA family: regular MFMA, or CDNA4 scaled MFMA (gfx950 only).
4. Choose the result layout first.
5. Derive `DotOperandLayout` for each operand from the result layout and K width.
6. Convert operands into those layouts with `convert_layout`.
7. Create the accumulator in the result layout and planned accumulator dtype.
8. Call the target-specific matrix op.
9. Convert accumulator/epilogue result to a store-compatible layout (usually a
   `BlockedLayout`) before `gl.store`.
10. Finish epilogue, dtype conversion, store layout, and masks.

## When the matrix core is the wrong instrument (gl.dot_fma)

`gl.dot_fma(a, b, acc)` performs the same contraction without a matrix instruction — it lowers to
an FMA tree on the VALU. It is target-independent (plain `gl`, not `gl.amd.cdna*`) and present on
all four versions (checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0); `acc` is **2D only before 3.8.0**, with
rank-3 batched accumulators accepted from 3.8.0.

Its layout contract is the inverse of the MFMA one, and the inversion is the tell:

- `acc` must carry a **`BlockedLayout`**, not an `AMDMFMALayout` — there is no matrix result layout
  on this path;
- `a` and `b` are still `DotOperandLayout` with `operand_index` 0 and 1, but their **parent must be
  that same `BlockedLayout`**;
- so the operand conversions are built against the blocked accumulator. Handing it layouts lifted
  from a working MFMA path trips an assert rather than lowering quietly.

Reach for it when the tile cannot fill a matrix instruction. MFMA's M is fixed by `instr_shape`, so
a decode-shaped contraction — `M = 1`, the shape a serving kernel spends much of its wall clock in
— occupies one row of an instruction built for 16 or 32 and discards the rest. That is the
half-empty-tile waste of `## Matrix-Family Details` at its limit, and warp placement cannot reach
it: there is no axis left to move warps off. A GEMV-shaped `dot_fma` issues only the work it has.

**Before taking that door, check the other one: you can keep the matrix core and fill the idle
dimension with something else.** `M = 1` leaves 15 of 16 rows empty, but those rows will carry any
*independent* work whose layout lines up with the same instruction — several independent K-groups
turned into one GEMV, several route x quant-group pairs, or several precision components of one
value (`../tile-programming/low-precision.md`). Shipped kernels do each of these. **The test is
whether you have a second piece of independent work whose layout can be aligned to this
instruction.** If you do, fill the rows; `dot_fma` and the VALU are where you go when you do not.
These are not competing recipes — the three cases above reach that conclusion by different routes,
so treat this as a question to ask rather than a pattern to copy.

**Narrow N is the same argument with the axes swapped.** A gate projection, a router logit, an
attention `N`-of-one — any contraction whose *output* is a thin strip fills an `instr_shape[1]` of
16 or 32 exactly as badly as small M fills `instr_shape[0]`, and `warps_per_cta[1] = 1` only stops
you wasting warps, it does not fill the instruction. Apply the small-M reasoning to whichever of M
and N is short; the instruction does not care which one you were thinking about.

### Contracting a reduction onto the FMA tree

A reduction is a contraction, and writing it as one is a **register** decision rather than a matrix
decision. The shape is a batched `1 x 1 x K` dot: for `B` independent rows each reduced over `K`,

```text
a   : (B, 1, K)   DotOperandLayout(0, acc_layout, k_width)
b   : (B, K, 1)   DotOperandLayout(1, acc_layout, k_width)
acc : (B, 1, 1)   BlockedLayout            <- the parent of both operand layouts
```

with `b` the second factor of the reduction — ones for a plain sum, the same tensor for a sum of
squares, a weight vector for a weighted one. Two things are derivable from the shapes alone, and
they are the reason to know this form:

- **The loop-carried accumulator collapses.** Carrying a `(B, K)` partial across the loop holds
  `B * K` values in registers; carrying `(B, 1, 1)` holds `B`. On a state that was sitting near a
  VGPR tier divisor that is the whole difference (`layout-reference.md ## BlockedLayout Constraints
  (wave64)` for the per-lane arithmetic).
- **It costs no extra arithmetic.** The lowering is `batch * M * N * K` FMAs, which here is
  `B * K` — exactly one FMA per input element, the same count the elementwise
  `acc += x * y` form issues. This is the case where the `M * N * K` cost model above is *not* an
  argument against `dot_fma`, because `M` and `N` are both 1.

**Version gate, and it is the textbook case for the 3.8 default.** The rank-3 accumulator this form
needs is accepted **from 3.8.0**; on 3.6.0 / 3.7.0 / 3.7.1 the batched call fails at **trace time**,
because the accumulator shape is unpacked as exactly two values. The downgrade is to loop over the
batch with rank-2 calls, or to keep the partial-sum form and pay the registers.

**Which dtypes get a packed form, and which get the plain tree.** The documented lowering is an FMA
tree on the VALU, and that is all the *contract* promises — but the AMD backend's selection is
readable at the tag, so this does not have to stay an open question.
`v3.8.0:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp:33-47` picks by
`(A element type, accumulator element type)`:

| A / acc | chosen intrinsic | vector size |
| --- | --- | --- |
| BF16 -> F32 | `llvm.amdgcn.fdot2.f32.bf16` (`v_dot2_f32_bf16`, a VALU packed dot) | 2 |
| F16 -> F32 | `llvm.amdgcn.fdot2` | 2 |
| I8 -> I32 | `llvm.amdgcn.sdot4` | 4 |
| anything else | the plain FMA intrinsics, one element at a time | 1 |

So a BF16 reduction accumulating in F32 gets a packed two-element dot per instruction rather than
two FMAs, and shipped kernels name this in their comments ("dot2", "pair-dot", a sixteen-element
dot described as eight native packed BF16 dots — which is exactly `vectorSize = 2`). Read the table
as the selection on this backend at this version, not as a portable guarantee; the contract above
is still only the tree. Likewise, this form does not remove
cross-lane work by fiat: where the combine lands depends on how `K` is distributed by the
accumulator's `BlockedLayout`. If your present form reduces every chunk to keep registers down,
check whether the per-chunk cross-lane tree actually disappeared; if it keeps a wide partial to
avoid those trees, the win you are taking is the registers.

**Verify it the same way as any other conversion change**: in the steady state the two operand
conversions should be gone, not merely cheaper (`## Hot-Loop Conversion Counting`), and the bound
has to still be the one you were attacking — a register win on a bandwidth-bound loop is
`../pitfalls/negative-patterns.md ## Bandwidth-Ceiling Refinement`.

The cost model is the other half of the decision and it is not subtle: the lowering materializes an
FMA per output element per K step, so issue count is `M * N * K` outright. Upstream emits a
compile-time warning past `M * N * K > 2**19`, and that warning is about compile time only — the
runtime crossover against a filled matrix instruction arrives far earlier. Use `dot_fma` for thin
or short contractions, never to avoid deriving a matrix layout.

> **It is not a fallback for an MFMA path that will not compile.** The two failures look alike from
> the error message and are opposite in cause: a verifier rejection is a layout bug
> (`## Matrix Failure Triage`), while `dot_fma` is a decision about shape. Switching to it will
> compile, will be correct, and on a tile that could have filled the matrix core will be slower by
> roughly the matrix-core-to-VALU throughput ratio — a regression that benchmarks cleanly and
> explains nothing.

## CDNA MFMA Shape

```python
mfma_layout = gl.amd.AMDMFMALayout(
    version=cdna_version,           # 4 = gfx950, 3 = gfx942
    instr_shape=instr_shape,        # 3D [M, N, K] on Triton >= 3.6
    transposed=True,
    warps_per_cta=warps_per_cta,
)
a = gl.convert_layout(a, gl.DotOperandLayout(0, mfma_layout, k_width))
b = gl.convert_layout(b, gl.DotOperandLayout(1, mfma_layout, k_width))
acc = gl.zeros((BLOCK_M, BLOCK_N), acc_dtype, layout=mfma_layout)
acc = gl.amd.cdna4.mfma(a, b, acc)    # use gl.amd.cdna3.mfma for gfx942 (version=3)
acc_store = gl.convert_layout(acc, blocked_mn)
```

Fill constants from local source and target evidence; do not copy this tile shape
blindly.

## Matrix-Family Details

- The optional `AMDMFMALayout` field is `element_bitwidth`, not `elem_type`: it is
  an `Optional[int]` taking **32 or 64**, and it describes the result element's
  bit width, not a dtype and not a "layout type". There is no `elem_type` field on
  any version checked. Do not automatically use input operand dtype as result
  layout element type.
- Accumulator dtype follows the matrix op, not source tensor spelling: INT8 MFMA
  accumulates into int32; BF16/FP16 regular MFMA uses fp32; fp8 accumulates fp32
  on both paths, and fp4 only through `cdna4.mfma_scaled` (next bullet).
- **Op selection is per `(version, M, N, K)`, not per dtype (gfx950).** Regular
  `cdna4.mfma` covers BF16/FP16/INT8 **and the fp8 shapes that are registered**:
  `v3.8.0`'s MFMA v4 table registers `mfma_f32_16x16x32_fp8_fp8` and
  `mfma_f32_32x32x16_fp8_fp8` (with the fp8/bf8 mixes), so those are reachable
  from plain `cdna4.mfma`. FP4 (e2m1) has no entry in that table at all, and
  neither do the unregistered fp8 `(M, N, K)` combinations — **those** are what
  `cdna4.mfma_scaled(a, a_scale|None, fmt, b, b_scale|None, fmt, acc)` with an
  e8m0 scale (unit scale materialized when `None`) is required for.
  The `no matching matrix core intrinsic ... f8E4M3FN` error you may hit is that
  registry missing **one shape**, not a dtype-level prohibition: when it fires,
  look up which `(M, N, K)` is registered for your target rather than abandoning
  the dtype. Plain Triton `tl.dot(fp8)` lowers to the scale-less
  `tt.dot_scaled` (the cheaper non-scale instruction) — a structural
  plain-vs-Gluon fp8 gap to expect when transcribing. Per-dtype
  `k_width` / `instr_shape` / scale / acc deltas live in
  `../hardware/capability-matrix.md`.
- On gfx950, direct `gl.amd.cdna4.mfma` for INT8 requires an int32 accumulator.
  Plain Triton `tl.dot` may handle i32->fp32 conversion internally; Gluon
  explicit MFMA does not.
- Derive `k_width` from the instruction shape and the **wave size**, not from the
  element width: `k_width = instr_shape[2] * min(M, N) / 64`. It is
  dtype-independent — the same `[16,16,32]` shape wants `k_width = 8` whether the
  operand is fp16 or bf16. This is the upstream relation inverted
  (`AMDMfmaEncodingAttr::getInstrShapeForOperand` computes
  `kDim = kWidth * (warpSize / min(mDim, nDim))`).
  - **The relation above is the plain-`cdna4.mfma` rule and it holds there.**
    `[16,16,32]` -> `32*16/64 = 8`; `[16,16,16]` (gfx942) -> `4`; `[32,32,16]` ->
    `8`. Surveyed production source agrees at every non-scaled site checked.
  - **On `cdna4.mfma_scaled`, stop deriving and start checking — the halving is an
    authoring convention, not a contract the compiler enforces.** In `v3.8.0`
    `AccelerateAMDMatmul.cpp` the pass computes one internal `kWidth` and then
    splits: the scaled branch builds both operand encodings with `kWidth / 2`
    (`:774`, `:776`), the non-scaled branch with the full `kWidth` (`:788`), with
    `assert(kWidth == 32)` on `:772` as that pass's own precondition on its
    internal value. That is where the halved number comes from — but **that pass
    runs on the plain path only; on the Gluon path you write the operand `k_width`
    yourself and no pass derives it for you** (this is the scope correction that
    applies to the whole paragraph: the pass explains the provenance of a value, it
    does not decide what you may write).
    The consequence is visible in surveyed production source: at the **same**
    `mfma_scaled` instruction shape, one family writes the halved value throughout
    and another writes the unhalved one throughout, one file selects between the
    two on the host from the row count, and **all of them compile and ship** — so
    both are accepted by the verifier and neither is "the" answer. **No formula is
    given here for the scaled case in either direction**, because every candidate
    offered so far is falsified by a production site: the halved value is not
    universal, and a rule derived from the operand's K extent instead disagrees
    with sites that pair a halved `k_width` with a wider `arange`.
    What to do instead: treat `k_width` at a scaled site as **the K packing of the
    operand tile you are building** — pick it, then confirm it against the operand
    layout you actually constructed and against
    `layout_facts.py` below, and **never carry the value across kernels**.
  - **The consequence of picking wrong is a silent numerical error, which is why
    the paragraph above refuses to guess for you.** An unsupported `k_width` at an
    `mfma_scaled` site is **not rejected**: measured on gfx950 / `v3.8.0`, a
    `k_width` of `32` at `instr_shape=[16,16,128]` compiles, issues, and returns a
    plausible-looking wrong matrix — relative error against the fp32 reference
    `1.075e+00`, with no assert, no NaN, and no lowering failure anywhere in the
    chain. Layout assertions do not catch it, because nothing in the layout
    contract is violated. So a **newly chosen `k_width` must be validated against a
    numerical reference on a single tile before it is carried into a kernel**, and
    "it compiled and the kernel ran" is not evidence about it in either direction.
    The `f8f6f4` comment at `:1006-1008` reaching the halved value for fp4 by a
    different route (two fp4 packed per int8) is a cross-check on *one* value, not
    a derivation that generalizes.
  - **The choice is not free even where it is unconstrained, and that is the one
    decidable thing to say about it.** On `v3.8.0`
    `gl.amd.cdna4.compute_efficient_padded_shared_layout` derives a bank-conflict-
    avoiding `PaddedSharedLayout` from the dot-operand layout, and it returns
    **`None`** rather than raising when the inputs fall outside the algorithm's
    covered set. **Check the return value for `None` before you allocate** — that
    one line is what turns a failure somewhere downstream into a failure at the
    call, and it is the single most useful habit on this page.
    Its docstring names **three** parallel causes, not one:

    1. `k_width` outside `{4, 8, 16}`;
    2. element bit-width outside `{4, 8, 16}`;
    3. an MFMA instruction-shape / `k_width` combination the underlying algorithm
       does not handle.

    So a `k_width` of `32`, which the verifier accepts and which production does
    ship, forfeits the helper by cause (1) — but **reading cause (1) as the whole
    contract is the error this bullet used to make**, because cause (3) fires on
    perfectly ordinary `k_width` values and it is **the decode band's main cause**.
    Measured at `AMDMFMALayout(version=4, instr_shape=[16,16,128], transposed=True,
    warps_per_cta=[1,4])` with `dtype=gl.uint8`, sweeping `opIdx x k_width x shape`:
    `k_width=32` is `None` at every shape and both operand indices, as cause (1)
    predicts. For `k_width` **inside** `{4, 8, 16}` the deciding variable is not
    `k_width` at all — it is the **non-K tile length**. At `opIdx=0`, `k_width=16`
    needs a non-K extent of at least 32 and `k_width` of 4 or 8 needs at least 64;
    at `opIdx=1`, `k_width` of 4 or 8 comes back non-`None` across 16..256 while
    `k_width=16` again needs at least 32. **A small tile is the thing that takes the
    helper away from you**, and a decode-band kernel is where small tiles live, so
    expect `None` there for a reason that has nothing to do with the number you
    picked for `k_width`.

    *Scope of that sweep:* one `instr_shape` and one `warps_per_cta`. The three
    causes are the docstring's and are general; the specific non-K thresholds above
    are measured at that one configuration and are not established elsewhere —
    re-measure rather than carrying the numbers. **The axis list matters as much as
    the numbers: the threshold moves with `instr_shape` and with `operand_index`,
    not with the tile alone.** At `[32, 32, 64]` operand 0 is still `None` at a
    non-K extent of 32 while operand 1 already exists at 16 — later and earlier than
    the sweep above respectively, in the same change of instruction shape. A gate
    written against `BLOCK_M` will therefore be wrong for one of the two operands;
    gate on the return value (`layout-reference.md ### Contract points`, the
    `USE_LDS` form). Neither `k_width` value is wrong;
    they have different downstream surfaces, and this is the surface to check
    before picking. On 3.6.0 / 3.7.0 / 3.7.1 the helper does not exist at all, so
    the question does not arise and hand-building is the only road
    (`layout-reference.md ### Contract points — places this goes wrong quietly`
    owns the full contract; `../workloads/moe.md ## Version gates` has the row).
  - An earlier revision divided `instr_shape_k` by the element size in bytes. That
    is dimensionally wrong (elements per byte) and it happens to agree only on
    32x32 shapes: on the 16x16 family it is 2x high, and on fp8 `16x16x128` it is
    8x high. Both of those shapes appear in this pack's own examples with the
    correct values, so the formula contradicted its own worked cases.
  - The authoritative per-shape check is
    `kernel_workflow/scripts/kernel_tools/layout_facts.py gfx950 <opcode>` (the pack's
    `scripts/layout_facts.py` is a shim to it), no GPU needed — but read the right line per routing tier. On the Matrix
    Instruction Calculator path (`cdna1/2/3`, `rdna3/4`) it prints `elems/lane A=`
    directly, which **is** `k_width`. On the gfx950/CDNA4 path it prints
    `vgprs A=` instead, so convert: `k_width = vgprs_A * 4 / dtype_bytes`
    (`v_mfma_f32_16x16x32_f16` reports `A=4` -> `4*4/2 = 8`, matching the formula).
- `instr_shape` is the matrix instruction shape, not a convenient local tile.
  Use the 3D `[M, N, K]` form on Triton >= 3.6. Candidate values per target/dtype
  live in `../hardware/capability-matrix.md`.
- **`gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)`
  is how the scale operand's layout is obtained, and `scale_factor` admits exactly
  one value.** On `v3.8.0` the first statement in its body is
  `assert scale_factor == 32` — the parameter exists, it is not a knob. 32 is the
  hardware group size, and it is the same 32 that OCP MXFP4 quantizes on, so a
  checkpoint whose scales are group-32 E8M0 lines up with this entry point exactly.
  The layout it returns is a `DistributedLinearLayout` derived from the parent
  `AMDMFMALayout`, so it **moves whenever the matrix tile moves**.
  `layout-reference.md ## The Scale Operand's Layout For mfma_scaled` owns the
  contract and the producer/consumer consequence.
- **The hardware block-scale path is reachable, and it is the preferred route when
  your scales already have the right shape.** Driving both scale slots of
  `gl.amd.cdna4.mfma_scaled` with **real `uint8` E8M0 operands** routed through
  `get_mfma_scale_layout` is measured working on gfx950 / `v3.8.0` at
  `a_format="e2m1", b_format="e2m1"`: relative error against the dequantized fp32
  reference `8.8e-08` at a `[16, *, 512]` tile and `1.9e-07` at `[16, 128, 6144]`,
  and it holds at `num_warps=1` / `warps_per_cta=[1,1]` as well. A second
  independent implementation on a production-shaped kernel reached `1.7e-03` and
  passed its numerical oracle. The disassembly shows
  `v_mfma_scale_f32_16x16x128_f8f6f4` with **no** fp32 VALU compensating multiply,
  **no** `scaled_upcast`, and **no** hand-written dequant sequence — that is, the
  scaling really is happening inside the instruction. **If your model's scales are
  group-32 E8M0, drive them here rather than descaling the accumulator afterwards.**
  What decides whether you can is the checkpoint's quantization format, not the
  toolchain (`../workloads/moe.md ### Axis 2 — how the quantization reaches the
  matrix core`).
- `tiles_per_warp` must cover the tile, and the relation is **divisibility, not
  equality**:

```text
(warps_per_cta[d] * tiles_per_warp[d] * instr_shape[d])  divides  tile_size[d]
```

Equality is the special case where the quotient is 1. Gate on the predicate and
report the identity: shipped kernels miss equality on **both**
dimensions of a shape and are still correct, so when a configuration fails the
`==` form, first read off the quotient and only then treat it as a defect.
Recompute whenever stage width, split factor, instruction shape, or
`warps_per_cta` changes.

- **Small-dimension warp placement.** Do not place warps on a tile axis whose
  per-warp share would fall below `instr_shape` on that axis: if
  `tile_size[d] / warps_per_cta[d] < instr_shape[d]`, that warp's MFMA tile is
  **half-empty** and matrix throughput is wasted. Keep a short axis inside one warp
  (`warps_per_cta[d] = 1`) and place the warps on the long axis/axes. This bites
  whenever a tile dim is at or below `instr_shape` (skinny GEMM, attention
  head-block, small-N reductions); the square GEMM examples (`warps_per_cta=[2,2]`)
  do not, because M and N are both large.

## Lowering Ladders

Regular CDNA MFMA:

1. choose `AMDMFMALayout` version from target family (4 gfx950 / 3 gfx942);
2. choose `instr_shape` from supported matrix-op evidence;
3. create `DotOperandLayout` for A and B with planned K width;
4. convert operands exactly at the operand boundary;
5. create the accumulator in result layout and accumulator dtype;
6. call regular `mfma`;
7. finish epilogue, dtype conversion, and store layout.

CDNA4 scaled MFMA (gfx950 only):

1. start from a working regular MFMA or plain Triton anchor;
2. confirm scaled-dot evidence and gfx950 support;
3. plan operand layouts and scale layouts together;
4. verify scale format, scale shape, scale factor, K width, accumulator dtype,
   and store dtype;
5. derive scale packing from the selected instruction shape;
6. stop at regular MFMA or plain Triton if scale-layout support is unclear.

Block-scaled accumulation warning: if the source loop is `acc += dot(a, b) *
scale`, MFMA's built-in accumulator cannot directly express scale-before-add. A
Gluon rewrite usually needs a fresh zero accumulator per K step plus scale
conversion, multiply, and add. Treat this as a strong signal to keep plain Triton
unless the extra mechanism clearly offsets the conversion cost
(`../tile-programming/low-precision.md`, `../pitfalls/negative-patterns.md`).

**gfx942 downgrade.** Both ladders above are written for gfx950. On CDNA3 the regular ladder
holds with `version=3` and `gl.amd.cdna3.mfma`; the scaled ladder does not exist (no
`mfma_scaled`), and FP8 Gluon MFMA is a target-specific blocker — fp8 there is the FNUZ spelling —
so use a plain `tl.dot` comparator (`atoms-reference.md ## CDNA3 gfx942 downgrade`). See the
CDNA4 -> CDNA3 adaptation notes in `../hardware/planning-constants.md`.

## Minimal Dot Recipe

```text
hot dot-like subpath / target family:
result layout / operand layouts / k_width:
convert placement / accumulator dtype:
target op / epilogue/store layout:
correctness oracle:
```

If result layout, operand layouts, instruction shape, accumulator dtype, or store
layout is unknown, stop there instead of guessing.

## Hot-Loop Conversion Counting

For hot loops, count how often operand conversion is paid before reading more
matrix detail. A K-loop typically pays:

```text
per K step:
  load A (and possibly stage to shared)
  load B (and possibly stage to shared)
  convert A to operand layout, if not produced in operand layout
  convert B to operand layout, if not produced in operand layout
  optional scale load/convert
  matrix instruction
total convert cost ~ (K / BLOCK_K) * convert_per_step
```

If the load layout already matches `DotOperandLayout`, both `convert_layout` calls
become reinterpretations and disappear from steady-state cost. If convert cost
grows with `K / BLOCK_K`, reduce the per-K conversion count before broader sweeps
(`../pitfalls/negative-patterns.md ## Hot-Loop Layout Conversion`).

A distinct, **irreducible** convert: when an MFMA **result** is reused as the next
MFMA's **operand** with the free axis becoming the contraction axis (result-N ->
operand-K, e.g. P / dS in attention), CDNA has no `ldmatrix`, so that relayout is a
real cross-lane shuffle (`v_perm` / `permlane*` / `ds_bpermute`), not a
reinterpretation — matching load layouts cannot remove it. Its cost is set by the
structure/data layout, the hardware reason in
`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`.

**Transpose-on-read from swizzled shared (the constructive counterpart to the
`tl.trans`/register-shuffle warning).** When the SAME loaded operand feeds two matmuls
in different orientations — one needs it transposed, the other natural (QK^T then PV in
attention; A and A^T paths in some GEMMs) — do NOT transpose it in registers with
`tl.trans`/a shuffle in the loop. Instead stage it ONCE into swizzled shared memory
(`SwizzledSharedLayout`) and read it transposed for one dot (via a permuted shared
view / `DotOperandLayout` on the transposed access) and natural for the other. This
moves the orientation change to the LDS read addressing (a shared-layout decision)
instead of a hot-loop register conversion, and can unblock a larger MFMA tile that the
register-shuffle form would spill. Guard: it costs shared capacity + an extra `ds_read`
per orientation; verify the swizzle keeps `ds_read` conflict-free
(`../tile-programming/layout-recipes.md`). Prefer loading each operand directly in the
layout each dot needs when only one orientation is used.

## Reduction Accumulator Layout

Use this when the bottleneck is a reduction or accumulator boundary:

1. identify the logical reduction axis and accumulator dtype;
2. choose the parent layout of the expression consuming the reduction state;
3. derive reduction state from that same parent layout;
4. keep the identity value explicit;
5. keep reduction layout changes separate from matrix or wrapper changes;
6. if correct but slower, record whether cost is padding, conversion, mask, or
   launch overhead.

## Accumulator + per-step rescale (online normalization)

When the source rescales the accumulator between matmul steps (online
normalization — e.g. running-max softmax), fold the rescale into the **next**
MFMA's accumulator input; do not add the un-rescaled accumulator a second time:

```text
correct : acc = mfma(p, v, acc * alpha)        # rescale, then accumulate into it
wrong   : acc = acc * alpha + mfma(p, v, acc)   # double-counts acc
```

The MFMA's built-in accumulator adds its third operand, so passing `acc * alpha`
as that operand applies the rescale and the new product in one instruction. The
wrong form is a classic transcription bug: it often passes at a single
K/reduction tile and fails only at >= 2 tiles (when a non-trivial rescale first
occurs).

## Amortize the epilogue (grow M)

When MfmaUtil is low with a filled tile and a serial epilogue (under-amortized), grow
BLOCK_M so each serial epilogue is covered by more MFMA issue. Tile the enlarged
accumulator with `static_range` quadrant loops feeding `cdna.mfma`.

```python
acc = ttgl.zeros([BLOCK_M, BLOCK_N], ttgl.float32, mfma_layout)   # grow BLOCK_M
for qm in ttgl.static_range(2):        # quadrant loop over the enlarged tile
    for qn in ttgl.static_range(2):
        acc = gl.amd.cdna4.mfma(a[qm], b[qn], acc)                # more MMA per epilogue (cdna3 on gfx942)
```

Ref: enlarged-tile quadrant loop (upstream `f16_gemm_streamk_gfx1250.py`; swap the
gfx1250 `wmma` for `cdna4.mfma`, or `cdna3.mfma` on the gfx942 downgrade). Trade-off: larger M costs VGPRs — pair with slicing
(`../tile-programming/slicing.md ## Slice recipe (ttgl.amd.slice)`) if it spills.
Verify: MfmaUtil up.

## Fold scalars off the VALU chain

When the softmax/epilogue folds a per-element scalar multiply into the inner loop, fold
the constant into a fused expression and use base-2 `exp2` with a folded `log2(e)` (CDNA's
`v_exp_f32` is base-2); pre-scale the operand at load instead of scaling every element.

```python
LOG2E = 1.4426950408889634
p = tl.math.exp2(qk * (sm_scale * LOG2E) - m_i[:, None] * LOG2E)   # one fused VALU, base-2
# or fold sm_scale into Q at load so the qk*scale multiply disappears from the hot loop
```

Ref: online-softmax exp2/log2e fold (`../method/profile.md ## Reducing compute-class VALU`;
`../workloads/attention.md ## Online softmax`). Verify: VALU-between-matmul term down,
VALUBusy/MfmaUtil up.

## Shorten the critical path (fast exp2 / hoist rescale)

When the stall is dependency-latency (VALUUtil ~100 but low VALUBusy/MfmaUtil), take the
serial ops off the critical chain: `exp2` instead of `exp`, hoist the reciprocal
normalization out of the K-loop into the epilogue, and raise `num_warps` for more ILP.

```python
# defer the 1/l_i normalization to the epilogue (not per K-block on the acc chain)
acc = acc * (1.0 / l_i)[:, None]        # once, after the loop — off the inner dep chain
```

Ref: `../method/profile.md ## Reducing compute-class VALU`, `## Accumulator + per-step
rescale (online normalization)`. Verify: shorter dep chain -> VALUBusy/MfmaUtil rises.

## Reduce accumulator traffic (keep one fragment, convert once)

When AGPR read-modify / AGPR<->VGPR round-trips are a top inter-MFMA bubble, keep the
accumulator in ONE f32 tile across the whole reduction and convert to the output dtype
exactly once at the epilogue — do not restore/re-cast the accumulator each K-block.

```python
acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)   # one accumulator, K-loop long
# ... mfma accumulates into acc ...
out = acc.to(tl.bfloat16, fp_downcast_rounding="rtz")   # epilogue: convert ONCE
```

Ref: `## Reduction Accumulator Layout`, `## Accumulator + per-step rescale (online
normalization)`. CDNA-only (AGPR). Verify: fewer `v_accvgpr` round-trips; AGPR-shuffle
bubble share down.

## Matrix Failure Triage

| Symptom | Inspect first | Typical fix |
| --- | --- | --- |
| MFMA layout verifier failure | result layout dtype, `instr_shape`, target version | match architecture (v4/v3) and version expectations |
| Store layout mismatch | accumulator/result layout vs pointer/index layout | convert epilogue result to store-compatible layout |
| Lowering failure after conversion | newest operand/result layout rewrite | shrink to one target op and verify each layout |
| CDNA4 INT8 MFMA rejects fp32 acc | accumulator dtype | use int32 acc for direct `cdna4.mfma`, or keep plain `tl.dot` |
| Wrong matrix result | target/dtype/op evidence | check `../hardware/capability-matrix.md`, accumulator dtype, scale format |

Full symptom routing: `../method/triage.md`.
