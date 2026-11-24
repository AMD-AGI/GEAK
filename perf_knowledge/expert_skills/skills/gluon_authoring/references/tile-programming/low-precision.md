# Low-Precision Side Path: a8w8 (BF8) and a4w4 (MXFP4)

> **MoE / expert GEMM with MXFP4 weights, or any `mfma_scaled` site with driven scales:** use `k_width=16`, the one value correct in every case measured (other values returned wrong numbers silently in some GPU-measured cases — `[32,32,64]` e4m3 x e4m3 with real scales was correct at 4 / 8 / 16; gfx950 / 3.8.0), and bf16 activations cannot enter `mfma_scaled` (3.8.0 accepts only `e2m1` / `e4m3` / `e5m2`) — [`../workloads/moe-lowprec.md`](../workloads/moe-lowprec.md).

> **Lookup over recall:** scaled-MFMA shapes and scale semantics — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 list` (`[scale]` rows), `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 modifiers`; per-dtype MFMA peak of the part — `python3 kernel_workflow/scripts/kernel_tools/extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json` (raw row: `--print <SKU>`).

FP8 / MXFP4 scaled-MFMA paths reuse the backbone (memory path / LDS / pipeline / slicing) plus a
**scale side path**. GEMM-shaped loops only; for a vector stage between two matmuls (attention) read
`../workloads/attention-lowprec.md`. On gfx950 OCP fp8 (`float8_e4m3fn` / `e5m2`) is the native encoding;
`e4m3fnuz` may upcast and invalidate fp8 conclusions.

**gfx942 downgrade (whole page).** Scaled MFMA (`mfma_scaled`, `v_mfma_scale_*`), MXFP4/fp6 and the
e8m0 scale path do not exist on gfx942 (ISA; no `mfma_scaled` in the 3.8.0 `cdna3` module), so the a8w8-scaled and a4w4 sections below are gfx950-only.
gfx942's fp8 is the **FNUZ** encoding (`float8_e4m3fnuz` / `float8_e5m2fnuz`) on the regular
`gl.amd.cdna3.mfma`; an OCP-fp8 tensor carried over from a gfx950 config must be converted, and a
block scale is applied as a separate VALU descale rather than inside the matrix op. Per-arch dtype
cells: `../hardware/capability-matrix.md`.

**Check the oracle type and threshold before choosing a low-precision matmul.** Estimate the fp8/fp4
error under the actual gate on the bf16 anchor: a tight cosine/angular threshold can reject a matmul a
loose abs/rel gate accepts, and a tight elementwise gate can reject one a cosine gate accepts. Measured on
an MXFP8 GEMM against its bf16 anchor (gfx950 / Triton 3.8.0): at common elementwise tolerances
(`1e-2`..`4e-2`) the abs/rel gate is the stricter one, while a cosine gate near `1e-3` accepts; only a
tight cosine threshold (around `1e-4`) rejects what a scale-relative abs/rel gate accepts. The error is
mostly orthogonal to the reference, as it is for any rounding, so "direction vs magnitude" is not a
reason to expect cosine to be stricter. If it
fails, the fallback is a bf16 matmul (dequant -> bf16 MFMA) with reduced precision only on the dequant/scale
side path.

## When scaled MFMA is required here, and when it is not

**The constraint is per-`(version, M, N, K)`, not per-dtype.** The 3.8.0 MFMA v4 table registers fp8
`16x16x32` and `32x32x16` (`mfma_f32_16x16x32_fp8_fp8`, `mfma_f32_32x32x16_fp8_fp8`, plus fp8/bf8 mixes),
reachable from plain `gl.amd.cdna4.mfma`. FP4 (e2m1) and the other fp8 shapes (K=64/128) need
`cdna4.mfma_scaled`. **`no matching matrix core intrinsic ... f8E4M3FN` = that one shape is not
registered**, not a dtype prohibition: pick a registered shape or go scaled. Without a descale pass
`None` scales plus the format string (a unit e8m0 scale is materialized):

```python
acc = gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
```

Plain `tl.dot(fp8)` on gfx950 lowers to a scale-less `tt.dot_scaled` -> `v_mfma_f32_32x32x64_f8f6f4`, so
a faithful transcription shows this instruction gap. Per-dtype `k_width` / `instr_shape` cells:
`../hardware/capability-matrix.md`.

## a8w8 (BF8, e5m2)

Starting points from the upstream gfx950 Gluon tutorials, scale-free build (scales `None`, `e5m2` /
`e2m1`). With real block scales `k_width` becomes 16 and the LDS padding changes; an e4m3 a8w8 task with
block scales may need a different start. Re-check spills at the chosen tile.

- Tile `256 x 256 x 128` (`BLOCK_K=128`; 64 for FP16). `mfma_scaled(..., "e5m2", ...)` with `None`
  scales -> `v_mfma_scale_f32_16x16x128_f8f6f4`, `cbsz/blgp=1/1`. ~32-cycle MFMA: ~2 MFMA per memory op
  (not 4); double `BLOCK_K` if you need 4.
- `k_width = 32` (correct here only because the scales are `None` and the format is uniform; with driven
  scales use 16 — `../workloads/moe.md ## Top correctness gate`); the tutorial's scale-free (`k_width=32`)
  build pads LDS with interval/padding pairs `[[1024, 16], [2048, 32]]` (only the first constructor
  argument: `layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`); its real-scale (`k_width=16`)
  build uses a single `[[1024, 32]]` — derive the padding from the `k_width` in use.
- Slicing M+N (spill-free at this tile in the tutorial).

## a4w4 (MXFP4, e2m1)

- Tile `256 x 256 x 256`. `mfma_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc)`; `cbsz/blgp=4/4`;
  ~16-cycle MFMA (FP16 interleave density). Scale layout: `get_mfma_scale_layout()`.
- **Scales direct to LDS (the upstream default).** `buffer_load_to_shared` of the scales alongside the
  tiles, no `ds_write`: a scale block that is 4 B per thread is the legal 32-bit direct-to-LDS form and lowers
  (check `scale_bytes / threads` against the legal widths:
  `../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`). It puts the
  scales in the same commit group as the tiles (one wait covers both) and removes the `ds_write` trap.
  **Overrides KB** (gfx950 and gfx942, Triton 3.8.0): `perf_knowledge/languages/asm_mfma/mfma_intrinsics.md`
  and `perf_knowledge/operators/quant_fp4_mxfp/backends/gluon.md` prescribe a register round trip
  (global read -> LDS write -> LDS read) as the scale path (the first says it is needed); verified by a compile-only check: a `uint8`
  scale block at 4 B per thread through `buffer_load_to_shared` lowers to one direct-to-LDS load with no
  `ds_write` on both arches.
- *Register round trip (fallback)* for an illegal width or non-contiguous source run: `buffer_load` ->
  LDS store -> `ds_read_tr`; ~4 scale register buffers per operand; and the **`ds_write` trap**
  (ATT-confirmed, gfx950 / Triton 3.8.0 / rocprofiler-sdk 1.3.5): while coalesced LDS-DMA
  (`buffer_load_to_shared`) data is landing, a register-to-LDS scale write completes only after the DMA
  burst — about 300-500 cycles instead of about 60. The `ds_write` itself issues at once; the stall sits at
  the `lgkmcnt` wait that consumes it, so independent MFMAs placed between the `ds_write` and that wait
  hide it (`instruction-scheduling.md`). Separately, a scale whose global load is issued after the DMA
  batch adds a `vmcnt` drain before the `ds_write` (`vmcnt` retires in order, so it also waits for the
  earlier DMA copies) — issue the scale loads before the DMA batch, or keep scales register-resident.
- Slicing: the upstream tutorial lineage uses **M+N** for all three precisions (its MXFP4 champion is M+N,
  ahead of N-only; the tutorial gives register pressure and spills under N-only as the reason).

## Budget notes

Scale staging counts in `LDS_bytes` and scale registers (`R_side`) in `R_total <=` the per-wave budget (512 at 1 wave/SIMD, 256 at 2), which bounds deepening.

## int4 / int8 weight GEMM (W4A16) — dequant VALU is often the bound

The hot cost is usually the software int4/int8 -> bf16 unpack VALU; VALU% vs MFMA% in
`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py` shows whether it is (`../hardware/roofline-models.md ## Low-precision / dequant-VALU`).
**`v_cvt_off_f32_i4` is binding-gated, not arch-gated:** `hasattr(rocdl, 'cvt_off_f32_i4')` is False on
gfx942, yet `v_cvt_off_f32_i4_e32` assembles on gfx942 and gfx950 — it is reachable via inline asm, so its
absence from the bindings is a binding gap, not a hardware ceiling. It reads bits [3:0] as a two's-complement
int4 and returns value/16 (`8 -> -0.5`, `7 -> 0.4375`; ISA `CVT_OFF_TABLE`, LLVM constant folding). Two's-complement
int4 weights feed it directly (fold the x16 into the scale); unsigned-offset weights (zero point 8) need one
`v_xor_b32 ..., 0x88888888` per packed dword first, not host-side requantization. It reads only the low nibble, so
each nibble needs a shift (or SDWA) before the convert.

## Cheap fp32->bf16 downcast (cut the convert bubble)

The default RTNE-with-NaN-guard convert is a ~5-6 VALU/element sequence (`v_bfe_u32` / `v_add3_u32` /
`v_cmp` / `v_cndmask` / `v_lshrrev` / `v_perm`). **gfx950 has native packed `v_cvt_pk_bf16_f32`**
(CDNA4-only; no Gluon source API besides inline asm): read the loop asm first — if it is already
emitted, this section does not apply. On gfx950 / Triton 3.8.0 a plain `.to(bf16)` already lowered to
it (compile-checked), and round-half-up below then costs slightly *more* VALU than the default; the
members below pay where the build emits the software RTNE sequence (gfx942). gfx942 has no native convert (`llvm-mc -mcpu=gfx942` rejects
it), so the members below are the whole lever. Tolerance decides between them; the first is the safe one.

**DEFAULT where no native convert is emitted — round-half-up (~2 VALU/elem; bit-identical to RTNE on
every non-tie, non-NaN input, checked over random and bit-pattern sweeps).**
Add the bf16 rounding bias, then take the high half. When the values feeding the cast are finite
(e.g. `exp2` results, post-softmax probabilities), the NaN guard is unreachable and the only difference
from RTNE is the rounding of exact ties (half-up vs to-even) — a bit-identity check on the real tensor
shows whether any tie occurs.

```python
# inside @gluon.jit: round-half-up f32 -> bf16, no ties-to-even, no NaN guard
u = x.to(ttgl.int32, bitcast=True)
bf16 = ((u + 0x8000) >> 16).to(ttgl.int16).to(ttgl.bfloat16, bitcast=True)
```

**FALLBACK — round-toward-zero (`rtz`, truncation).** Same ~2 VALU/elem, but **biased**: relative
error ~2⁻⁸ vs RTNE's 2⁻⁹, and the bias **accumulates along a reduction axis**. It is safe only where a
measured tolerance margin shows it — **on a kernel already sitting near its oracle tol it is a likely cause
of a correctness failure**.

```python
out = acc.to(ttgl.bfloat16, fp_downcast_rounding="rtz")   # cheaper cvt, BUT biased — re-check oracle
```

Verify both: convert-class share and VALU:MFMA down, and re-run the oracle (round-half-up unchanged to
~1 ULP; RTZ must be shown under tol).

## The other direction: f32 accuracy on the bf16 matrix core (`bf16x3` / `bf16x6`)

Spends matrix throughput to recover f32 accuracy. **The compiler's decomposition pass fires only when
both `A` and `B` are f32**; with one f32 operand, split by hand (`### One-sided decomposition: when only
A is f32`). The number in the name is the dot count:

| Precision | Component dots issued | Total |
| --- | --- | --- |
| `bf16x3` | `mid*hi`, `hi*mid`, `hi*hi` | 3 |
| `bf16x6` | `mid*mid`, `lo*hi`, `hi*lo`, then all of `bf16x3`'s | 6 |

A matrix-issue-bound kernel cannot absorb three or six times the matrix work (it is already at the matrix-issue limit).
**AMD input precisions on 3.8.0 are `("ieee", "bf16x3", "bf16x6")`; `tf32` is allowed only on gfx942**
(installed `compiler.py`) (a `tf32` carried from a gfx942 config is
rejected on gfx950). **Overrides KB**: `perf_knowledge/languages/triton_amd/pitfalls.md` lists only `"ieee"` and
(CDNA3) `"tf32"`; verified by a compile-only check (3.8.0): `bf16x3` / `bf16x6` compile to bf16 MFMAs on gfx950
and gfx942, `tf32` compiles on gfx942 (`v_mfma_f32_32x32x4_xf32`) and is rejected on gfx950. Not bit-identical to `ieee` (partial accumulator NaN-scrubbed before the final
`hi*hi`, caller's acc added last); no `bf16x9` (commented out upstream).

| Path | How the precision is selected |
| --- | --- |
| plain Triton | `tl.dot(a, b, acc, input_precision="bf16x3")` — **per call site** |
| Gluon on CDNA | **no working selector on 3.8.0.** `gl.amd.cdna4.mfma` / `cdna3.mfma` tag the dot with `TRITON_F32_DEFAULT` (`knobs.language.fp32_default`), else the `default_dot_input_precision` compile option, and `gl.dot_fma` hardcodes it away. But `gluon_to_ttgir` does not run the `f32_dot_tc` decomposition pass, so the ISA stays native `v_mfma_f32_*_f32` whatever is set (compile-verified gfx950: the TTGIR carries `inputPrecision = bf16x3`, the `.s` does not change). For bf16x3/bf16x6 in Gluon, split the operands by hand and issue the component bf16 MFMAs (`### One-sided decomposition` below); settle the precision in plain Triton first, where the pass fires. |

So plain Triton is where the precision is A/B-able per dot, before transcribing. No extra pass flag is
needed.

### One-sided decomposition: when only A is f32

Attention epilogue shape: probabilities f32, values fp8/bf16. The pass does not fire; split `P` by hand
(used in production attention and GDN-prefill kernels):

```python
high     = probabilities.to(gl.bfloat16)
residual = (probabilities - high.to(gl.float32)).to(gl.bfloat16)
tail     = (probabilities - high.to(gl.float32) - residual.to(gl.float32)).to(gl.bfloat16)
```

- **No cross terms**: `V` is already <= bf16, so `(P_hi + P_res + P_tail) · V` stays one (or two)
  matmuls, not the 3/6-dot model.
- **Axis from leftover instruction capacity** (structure from one production kernel; MFMA counts and
  accuracy per component count compile- and CPU-checked on Triton 3.8.0 / gfx950). M full (rows = real
  heads/tokens): double along K with `gl.join(value, value)` (`[lo, hi] · [v, v]^T = lo·v + hi·v`) — this
  doubles the MFMA count and adds `v_perm`s. M has spare rows (logical M < instr M): put components on M for
  **no extra MFMA** (4 heads x 3 components = 12 of 16 rows) — but not zero extra instructions: per-row
  selects / converts and a cross-row combine of the component outputs remain; count them before calling it
  free. The component count is computed from leftover rows (a constexpr gates the third component).
- **Component count and `instr_shape` are one decision**: 2 components on 4 rows -> narrower
  `AMDMFMALayout(instr_shape=[4, 64, 64], transposed=False)` with `kWidth=4` (`v_mfma_f32_4x4x4_16b_bf16`;
  `transposed=True` is rejected on 3.8.0), not a padded `[16, 16, 32]` / `kWidth=8`.
- **Rounding: the default RNE downcast per component.** A negative residual (RNE rounding `hi` above `p`) is
  exact in bf16, which is signed, so it only flips the residual's sign. RTZ (`hi <= p`, non-negative residuals)
  keeps fewer bits per component and leaves a one-sided, biased error (CPU numerics check; on gfx950 /
  Triton 3.8.0 a one- or two-component `P @ V` split was more accurate with RNE than RTZ, and the two tie
  at three components); Triton's own
  bf16x3/bf16x6 split (`F32DotTC.cpp`, `arith::TruncFOp`) also rounds RNE. Fold as `high + (residual + tail)`.
- Accuracy ~8/16/24 bits for 1/2/3 components holds because `exp(s - max)` lies in [0, 1] (short of
  significand, not exponent range). Structural observation from one production kernel; no timing.

## IR / asm acceptance signals

| Path | Confirm in IR/asm |
| --- | --- |
| scaled MFMA | `v_mfma_scale_f32_16x16x128_f8f6f4` with correct `cbsz/blgp` (1/1 BF8, 4/4 MXFP4) |
| scale side path (direct, default) | scale `buffer_load ... lds` in the same commit group as the tiles; **no** `ds_write` for scales |
| scale side path (round-trip fallback) | `ds_write` then `ds_read_tr` for scales; MFMA scheduled after the `ds_write` |
| dtype gate | operands are native FP8/FP4 (no upcast to FP16 in the hot loop) |

## Reprofile signal

Compare each dtype to its own ceiling, not fp4 efficiency to fp16's or fp8's: measure the ceiling with a
register-resident `mfma_scaled` loop (gfx950 / Triton 3.8.0: the fp4 ceiling came out below twice the fp8
one). A staged LDS-fed fp8 loop reached most of its ceiling there; a kernel far below that has headroom,
and candidate structural faults are scale staging on the critical path, an unhidden `ds_write`, an
under-filled tile — but the gap alone does not name which.
