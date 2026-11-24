# Gluon — copy/MMA atoms (CDNA MFMA)

Choosing the matrix/copy path by arch. Capability truth: `../hardware/capability-matrix.md`.

## CDNA4 gfx950 — MFMA + scaled

```python
acc = gl.amd.cdna4.mfma(a, b, acc)
# each operand is followed by ITS OWN scale and ITS OWN format string; both formats are required (no defaults, 3.8.0)
acc = gl.amd.cdna4.mfma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
```

**`mfma_scaled` argument order is `(a, a_scale, a_format, b, b_scale, b_format, acc)`** — scales are
not grouped after the operands, and the format strings are not optional.

Shapes: regular `cdna4.mfma` `[16, 16, 32]` default f16. **`mfma_scaled` has two registered shapes —
`[16, 16, 128]` and `[32, 32, 64]`** (the `mfma_scale_f32_*_f8f6f4` entries; fp8 on the scaled path
lands there too). No performance ordering between them is established (one n=1 comparison: 0.9898,
parity) — pick by tile, then measure. Scale layout is **derived**:
`gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)`, `scale_factor`
asserted == 32 (`layout-reference.md ## The Scale Operand's Layout For mfma_scaled`; oracle side
`../tile-programming/low-precision.md`).

## `ds_read_tr` (gfx950)

**No Gluon source API** — the backend emits the transpose LDS reads (`ds_read_b64_tr_b16` and
siblings) itself on gfx950 (`ds_read_tr: true` in `hw_constants.json`; **false on gfx942**). Read the
ISA rather than the capability matrix. **Overrides KB** for Triton 3.8.0: `perf_knowledge/languages/gluon/programming_model.md`
and `perf_knowledge/languages/gluon/overview.md` list `ds_read_tr` among the LDS moves Gluon makes explicit,
and `perf_knowledge/languages/gluon/gemm_cookbook.md` (step 2),
`perf_knowledge/operators/dense_gemm/backends/gluon.md`,
`perf_knowledge/operators/scaled_quant_gemm/backends/gluon.md` and
`perf_knowledge/operators/quant_fp4_mxfp/backends/gluon.md` (**LDS layout** bullets) list it as a step
the kernel author takes; verified by API introspection (no `ds_read_tr` / transpose-load name in `gl`, `gl.amd`, `gl.amd.cdna3`,
`gl.amd.cdna4`) and a compile-only check (a row-major-B fp16 GEMM emits `ds_read_b64_tr_b16` on gfx950 and
plain `ds_read_b64` on gfx942).

- Where it exists the compiler emits it instead of `amdg.in_thread_transpose`, so the "Gluon cannot
  express `in_thread_transpose`" gap has no subject on gfx950; a faithful anchor inherits it free.
- **Moving a transpose onto the read is a saving only when the transpose is not already free.** Check
  the ISA for an existing `ds_read_*_tr_*` before reaching for transpose-on-read; the measured case
  where it lost is `memory-reference.md ### Shared-layout family + transpose-on-read (layout
  dependency)`.

## Memory atoms

| Path | API | Arch |
| --- | --- | --- |
| Buffer load | `gl.amd.cdna4.buffer_load` | gfx950 |
| Async → LDS | `gl.amd.cdna4.async_copy` | gfx950 at **16 B or 4 B per thread (a set, not a floor)** **and gfx942 through this same `cdna4` namespace** at 4 B only (`cdna3` has no async submodule) — `../hardware/capability-matrix.md` row `gfx942`. Destination layout is a gate: swizzled lowers; padded lowers only when the padding interval covers one wave's copy (64 lanes x per-lane bytes: 1 KiB at 128 bit) and the producer layout is derived from its `.offset_bases` (`memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`) |

## CDNA3 gfx942 downgrade

What the gfx950 atoms above lose on CDNA3 (per-arch keys in
`perf_knowledge/hardware/data/hw_constants.json`: `scaled_mfma`, `ds_read_tr`, `fp8_dtype`,
`direct_to_lds_bit_widths`):

- **matrix:** `gl.amd.cdna3.mfma` with `AMDMFMALayout(version=3)`; **no `mfma_scaled`** —
  `mfma_scale_*_f8f6f4` is cannot-select on gfx942 (3.8.0, compile-checked: `LLVM ERROR: Cannot select`
  with a v4 layout, `PassManager::run failed` with v3), a hard compiler failure rather than a fallback
  (`../pitfalls/platform-known-issues.md ## gfx942 / CDNA3 hard failures that affect benchmark validity`),
  so block-scaled formats go through a plain `tl.dot` comparator there. bf16 MFMA on gfx942:
  `16x16x16` and `32x32x8` (16x16x32 / 32x32x16 bf16 are CDNA4-only). The plain-MFMA `k_width`
  relation still holds (`[16, 16, 16]` → 4, `matrix-reference.md ## Matrix-Family Details`).
- **fp8:** **FNUZ** (`float8_e4m3fnuz`, Triton `float8e4b8`) on CDNA3 against OCP `e4m3fn` on CDNA4;
  OCP fp8 MFMA fails on gfx942. The FNUZ spelling is the one dtype CDNA4 lacks, and Triton upcasts
  it there with only a warning (`../pitfalls/platform-known-issues.md`).
- **LDS reads:** **no `ds_read_tr`**.
- **async copy:** the same `cdna4.async_copy` entry points, at 32 bits per thread only.

## Out of scope

gfx1250 (`gl.amd.gfx1250.wmma`, `tdm` module, wave32, `PartitionedSharedLayout`) and RDNA
(`gl.amd.rdna3/rdna4.wmma`) are not gfx950 recipes. NVIDIA TMA / tcgen05 / TMEM have no Gluon analog
on CDNA (TMA is not AMD TDM; TDM is gfx1250-only).
