# Gluon Matrix Reference (CDNA MFMA, gfx950 / gfx942)

> **Lookup rather than recall:** MFMA dims/cycles/operand VGPRs — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 facts <V_MFMA_…>`; the lane/VGPR/bit table of an operand — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 layout <V_MFMA_…> -A`; one element — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 locate <V_MFMA_…> A <i> <k>`; scale-operand semantics — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 modifiers`.

Default MFMA shapes (ISA): `../hardware/isa-mechanisms.md` §Default MFMA shapes. Per-target
dtype/op/`instr_shape` evidence: `../hardware/capability-matrix.md`. Layout rules:
`layout-reference.md`. RDNA WMMA and gfx1250 are out of scope.

## Matrix Lowering Order

There is no generic `gl.dot`: a Gluon matrix path uses `gl.amd.cdna4.mfma` / `mfma_scaled` (gfx950)
or `gl.amd.cdna3.mfma` (gfx942) with explicit layouts — result `AMDMFMALayout` first, then
`DotOperandLayout(0|1, mfma_layout, k_width)` operands, accumulator created in the result layout,
`convert_layout` of the result to a store layout before `gl.store`. Otherwise keep plain Triton, or
leave the matrix core (`## When the matrix core is the wrong instrument (gl.dot_fma)`).

## When the matrix core is the wrong instrument (gl.dot_fma)

**`gl.dot_fma(a, b, acc)` is the same contraction as an FMA tree on the VALU.** Plain `gl`, all
versions 3.6.0-3.8.0. Layout contract is the inverse of MFMA:

- `acc` carries a **`BlockedLayout`** (not `AMDMFMALayout`);
- `a` / `b` are `DotOperandLayout(0|1, <that BlockedLayout>, k_width)` — layouts lifted from an
  MFMA path trip an assert.
- **`acc` is rank-2 only before 3.8.0; rank-3 (batched) accumulators from 3.8.0.** On 3.6/3.7 the
  batched call fails at trace time (accumulator shape unpacked as two values).

**Cost is `M * N * K` FMAs outright.** Upstream warns at compile time past `M * N * K > 2**19`
(compile time only — the runtime crossover against a filled MFMA is far earlier). It fits thin or
short contractions (decode `M = 1`, narrow-N gate/router projections). **It is not a fix for an MFMA
path that will not compile**: a verifier rejection is a layout bug (`## Matrix Failure Triage`);
switching to `dot_fma` on a fillable tile compiles, is correct, and is slower by roughly the
matrix-core/VALU throughput ratio.

Before leaving the matrix core, check whether the idle rows of a small-M (or small-N) instruction
can carry other **independent** work whose layout aligns with the same instruction (several
K-groups, route x quant-group pairs, precision components — `../tile-programming/low-precision.md`).

### Contracting a reduction onto the FMA tree

A reduction of `B` rows over `K` is a batched `1 x 1 x K` dot (needs the 3.8.0 rank-3 acc):

```text
a   : (B, 1, K)   DotOperandLayout(0, acc_layout, k_width)
b   : (B, K, 1)   DotOperandLayout(1, acc_layout, k_width)     # ones / same tensor / weights
acc : (B, 1, 1)   BlockedLayout   (parent of both operand layouts)
```

The loop-carried state collapses from `B * K` to `B` registers at the same FMA count (`B * K`).
Downgrade below 3.8.0: loop over the batch with rank-2 calls, or keep the partial-sum form.

**Packed-dot selection on AMD** (`v3.8.0 third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp:33-47`,
by `(A elem, acc elem)`):

| A / acc | intrinsic | vector size |
| --- | --- | --- |
| BF16 -> F32 | `llvm.amdgcn.fdot2.f32.bf16` (`v_dot2_f32_bf16`) | 2 |
| F16 -> F32 | `llvm.amdgcn.fdot2` | 2 |
| I8 -> I32 | `llvm.amdgcn.sdot4` | 4 |
| anything else | plain FMA, one element at a time | 1 |

Backend selection at this version, not a portable contract. Cross-lane combine placement still
depends on how the acc `BlockedLayout` distributes `K`. Verify: the two operand conversions are gone
in steady state (`## Hot-Loop Conversion Counting`) and the loop was not bandwidth-bound
(`../pitfalls/negative-patterns.md ## Bandwidth-Ceiling Refinement`).

## Matrix-Family Details

- **`AMDMFMALayout(..., element_bitwidth=None)` takes 32 or 64** — the result element's bit width.
  There is no `elem_type` field on any version.
- Accumulator dtype follows the op: INT8 -> int32; BF16/FP16 -> fp32; fp8 -> fp32 on both paths; fp4
  only via `mfma_scaled`. **Direct `gl.amd.cdna4.mfma` for INT8 requires an int32 accumulator**
  (plain `tl.dot` converts i32 -> fp32 internally; Gluon does not — fp32 acc fails lowering, 3.8.0 compile-checked).
- **Op selection is per `(version, M, N, K)`, not per dtype.** `v3.8.0`'s MFMA v4 table registers
  regular `mfma_f32_16x16x32_fp8_fp8` and `mfma_f32_32x32x16_fp8_fp8` (and fp8/bf8 mixes), reachable
  from plain `cdna4.mfma`. FP4 (e2m1) and unregistered fp8 shapes need
  `cdna4.mfma_scaled(a, a_scale|None, fmt, b, b_scale|None, fmt, acc)` (unit e8m0 scale when `None`),
  whose shapes are `16x16x128` / `32x32x64`. Error `no matching matrix core intrinsic ... f8E4M3FN`
  = one `(M, N, K)` missing from the registry, not a dtype prohibition.
- **Plain Triton `tl.dot(fp8)` on gfx950 lowers to a scale-less `tt.dot_scaled` and emits the
  scaled-family `v_mfma_f32_32x32x64_f8f6f4`** (twice the K of the regular fp8 MFMA; compile-verified
  3.8.0). Gluon reaches that K only through `mfma_scaled` (scales `None`), which emits the *scaled*
  `v_mfma_scale_f32_32x32x64_f8f6f4` with a unit E8M0 scale (GPU-verified 3.8.0); `cdna4.mfma` stops at
  `16x16x32` / `32x32x16` — a structural plain-vs-Gluon fp8 gap when transcribing. On gfx942
  `tl.dot(fp8e4b8)` stays `tt.dot` (`v_mfma_f32_32x32x16_fp8_fp8`); gfx942 fp8 is FNUZ `float8e4b8`;
  OCP fp8 there fails (`version=3`), is misread as FNUZ (`version=4`), or is upcast to fp16 (`tl.dot`).
- **`k_width = instr_shape[2] * min(M, N) / 64`** for plain `mfma` — wave-size based, dtype
  independent (inverse of `AMDMfmaEncodingAttr::getInstrShapeForOperand`:
  `kDim = kWidth * (warpSize / min(mDim, nDim))`). `[16,16,32]` -> 8; `[32,32,16]` -> 8;
  `[16,16,16]` (gfx942) -> 4. Do **not** divide by element bytes (2x high on 16x16, 8x high on fp8
  `16x16x128`).
- **Scaled `k_width` is NOT derivable — pick it, then validate numerically on one tile.** The
  halved value comes from `AccelerateAMDMatmul.cpp` (`kWidth / 2` for the scaled branch, :774/:776),
  but that pass runs on the plain path only; in Gluon you write `k_width` yourself. Production ships
  both halved and unhalved values at the same `mfma_scaled` shape, and every proposed formula is
  falsified by some site. **A wrong value is silent:** with driven (real E8M0) scales or mixed formats
  only `k_width=16` is correct at `[16,16,128]` (e.g. 32 returns ~100 % relative error, no assert, no
  NaN); with `None` scales and one format 4-32 are correct (GPU-measured gfx950 / 3.8.0;
  `../workloads/moe.md ## Top correctness gate`). A `k_width` validated in one kernel is not validated for another.
- `k_width` also decides whether `compute_efficient_padded_shared_layout` returns a layout (`None`
  outside {4, 8, 16}, and short non-K tiles give `None` at any `k_width`): thresholds and the
  `USE_LDS` gate in `layout-reference.md ### Contract points — places this goes wrong quietly`.
- Cross-check operand packing with `ttgir_bridge.py` (`compiler-layouts.md`) and `amd_isa.py
  facts/layout/locate`; XML operand widths are not Gluon `k_width` (packed FP4). F8F6F4 facts need
  an explicit `--as` format.
- `instr_shape` is the instruction shape `[M, N, K]` (3D on Triton >= 3.6), not a local tile.
- **Scale layout:** `gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)`,
  `assert scale_factor == 32` (the OCP MX group size) — contract in
  `layout-reference.md ## The Scale Operand's Layout For mfma_scaled`.
- **Real e8m0 block scales through `mfma_scaled` work — with fp8/fp4 activations (bf16 operands are
  rejected), drive group-32 E8M0 scales here instead of descaling the accumulator.** GPU-measured gfx950 / 3.8.0, `a_format="e2m1", b_format="e2m1"`,
  `uint8` scales via `get_mfma_scale_layout`: rel err `8.8e-08` at `[16, *, 512]`, `1.9e-07` at
  `[16, 128, 6144]`, also at `num_warps=1`; ISA shows `v_mfma_scale_f32_16x16x128_f8f6f4` with no
  VALU compensating multiply / `scaled_upcast` / dequant. Whether you can depends on the checkpoint's
  format (`../workloads/moe.md ### Axis 2 — how the quantization reaches the matrix core`).
- **`tiles_per_warp`: divisibility, not equality** —
  `warps_per_cta[d] * tiles_per_warp[d] * instr_shape[d]` must **divide** `tile_size[d]`. Shipped
  kernels miss equality on both dims and are correct. Recompute when stage width, split factor,
  `instr_shape` or `warps_per_cta` changes.
- Warps on an axis where `tile_size[d] / warps_per_cta[d] < instr_shape[d]` leave the MFMA half-empty;
  a short axis kept inside one warp (`warps_per_cta[d] = 1`) avoids that.

## Block-scaled accumulation and the gfx942 downgrade

- `acc += dot(a, b) * scale` cannot be expressed with MFMA's built-in accumulate (scale-before-add):
  it needs a fresh zero accumulator per K step + multiply + add. The alternatives are hardware scales
  (`mfma_scaled`) or plain Triton (`../tile-programming/low-precision.md`).
- gfx942: `version=3` + `gl.amd.cdna3.mfma`; no `mfma_scaled`; fp8 Gluon MFMA works on **FNUZ**
  operands only (`float8e4b8` / `float8e5b16`; OCP fails at `version=3` and gives wrong numbers at
  `version=4`) (`../hardware/cdna3-gfx942.md ## MFMA (matrix core)`, `atoms-reference.md ## CDNA3 gfx942
  downgrade`, `../hardware/planning-constants.md`).

## Hot-Loop Conversion Counting

Per-K-step operand `convert_layout` costs `(K / BLOCK_K) * convert_per_step`; when the load layout
already equals the `DotOperandLayout` the converts become reinterpretations
(`../pitfalls/negative-patterns.md ## Hot-Loop Layout Conversion`).

- **An MFMA result reused as the next MFMA's operand A (P·V in attention)** is a free register
  reinterpretation when the producing layout is `AMDMFMALayout(transposed=True)` and the consumer is
  `DotOperandLayout(0, same_layout, k_width=4)`: `convert_layout(..., assert_trivial=True)` compiles and the
  second dot keeps full-rate MFMAs (compile-verified gfx950 / 3.8.0 at `[32,32,16]`, `[16,16,32]`,
  `[32,32,8]`, `[16,16,16]` bf16; the 32x32 bf16 chain also ran in an attention loop on gfx950 / 3.8.0 —
`../workloads/attention.md`). Other `k_width` values (e.g. 8) or a
  non-transposed layout make it a real cross-lane shuffle (`v_permlane*_swap` or an LDS round trip); the
  in-lane `v_perm_b32` from bf16 packing appears either way and is not cross-lane. A result used in
  transposed orientation (dS^T in backward) was not tested and is expected to stay a shuffle. Check with
  `assert_trivial=True`, then count `permlane` / `ds_bpermute` in the loop
  (`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`).
- One operand needed in two orientations: stage once in shared and read the transposed view
  (`smem.permute((1, 0)).load(dot_layout)`) instead of a register transpose — but check for an
  existing `ds_read_*_tr_*` first; the measured loss case is
  `memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`.

## Accumulator + per-step rescale (online normalization)

Fold the rescale into the next MFMA's accumulator input:

```text
correct : acc = mfma(p, v, acc * alpha)
wrong   : acc = acc * alpha + mfma(p, v, acc)   # double-counts acc; passes at 1 tile, fails at >= 2
```

## Reduce accumulator traffic (keep one fragment, convert once)

One fp32 accumulator for the whole K loop, converted once in the epilogue, keeps accumulator traffic out of the loop. Where the
accumulator lands is not fixed (gfx950 / Triton 3.8.0: arch VGPRs on the kernels measured, part in AGPRs once
demand passed 256 VGPRs); the one placement rule and the default split are in
`../tile-programming/llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)` — read the
`v_mfma` destination `v[..]` vs `a[..]`. **Overrides KB**:
`perf_knowledge/languages/triton_amd/deep_codegen.md` ("MFMA accumulator lives in **AGPRs**") and the
"acc stays in AGPR" good signature in `perf_knowledge/languages/triton_amd/isa_verify.md`; basis and scope in
`../workloads/attention.md ### Conditional rescaling (skip the per-block acc *= alpha)`, **Accumulator placement**.

```python
out = acc.to(gl.bfloat16)   # default rounding is rne; fp_downcast_rounding="rtz" truncates and
                            # changes results — use only if the reference also truncates
```

## Matrix Failure Triage

| Symptom | Inspect first | Fix |
| --- | --- | --- |
| MFMA layout verifier failure | `instr_shape`, `version` (4 gfx950 / 3 gfx942), `element_bitwidth` | match the target |
| CDNA4 INT8 MFMA rejects fp32 acc | acc dtype | int32 acc, or plain `tl.dot` |
| Wrong result, no error | scaled `k_width`, scale format, acc dtype | validate one tile vs fp32 reference |

Full routing: `../method/triage.md`.
