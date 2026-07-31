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
operand layouts, or the kernel should remain plain Triton.

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

- `AMDMFMALayout.elem_type` describes the result/accumulator layout type accepted
  by the verifier. Do not automatically use input operand dtype as result layout
  element type.
- Accumulator dtype follows the matrix op, not source tensor spelling: INT8 MFMA
  accumulates into int32; BF16/FP16 regular MFMA uses fp32; fp8/fp4 have no regular
  MFMA intrinsic and accumulate fp32 through `cdna4.mfma_scaled` (next bullet).
- **Op selection by dtype (gfx950).** Regular `cdna4.mfma` covers BF16/FP16/INT8.
  **FP8 (e4m3/e5m2) and FP4 (e2m1) have no regular matrix-core intrinsic**
  (`no matching matrix core intrinsic ... f8E4M3FN`); they **must** use
  `cdna4.mfma_scaled(a, a_scale|None, fmt, b, b_scale|None, fmt, acc)` with an e8m0
  scale (unit scale materialized when `None`). Plain Triton `tl.dot(fp8)` instead
  lowers to the scale-less `tt.dot_scaled` (the cheaper non-scale instruction) — a
  structural plain-vs-Gluon fp8 gap to expect when transcribing. Per-dtype
  `k_width` / `instr_shape` / scale / acc deltas live in
  `../hardware/capability-matrix.md`.
- On gfx950, direct `gl.amd.cdna4.mfma` for INT8 requires an int32 accumulator.
  Plain Triton `tl.dot` may handle i32->fp32 conversion internally; Gluon
  explicit MFMA does not.
- Derive `k_width` from local source or selected instruction shape and element
  width. Start with `k_width = instr_shape_k / element_size_bytes`, then verify
  against local examples and compiler errors.
- `instr_shape` is the matrix instruction shape, not a convenient local tile.
  Use the 3D `[M, N, K]` form on Triton >= 3.6. Candidate values per target/dtype
  live in `../hardware/capability-matrix.md`.
- CDNA4 `get_mfma_scale_layout` has a scale-factor contract in local source;
  check it before assuming another factor.
- `tiles_per_warp` must match actual tile coverage:

```text
warps_per_cta[d] * tiles_per_warp[d] * instr_shape[d] == tile_size[d]
```

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
(`../tile-programming/low-precision.md`, `../gluon-negative-patterns.md`).

On gfx942, FP8 Gluon MFMA is a target-specific blocker — use a plain `tl.dot`
comparator. See the CDNA4 -> CDNA3 adaptation notes in
`../hardware/planning-constants.md`.

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
(`../gluon-negative-patterns.md ## Hot-Loop Layout Conversion`).

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
        acc = gl.amd.cdna3.mfma(a[qm], b[qn], acc)                # more MMA per epilogue
```

Ref: enlarged-tile quadrant loop (upstream `f16_gemm_streamk_gfx1250.py`; swap the
gfx1250 `wmma` for `cdna3/4.mfma`). Trade-off: larger M costs VGPRs — pair with slicing
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

Ref: online-softmax exp2/log2e fold (`../phases/profile.md ## Reducing compute-class VALU`;
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

Ref: `../phases/profile.md ## Reducing compute-class VALU`, `## Accumulator + per-step
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

Full symptom routing: `../failure-triage.md`.
