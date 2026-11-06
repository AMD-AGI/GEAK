# Gluon — copy/MMA atoms (CDNA MFMA + gfx1250 WMMA)

Choosing the matrix/copy path by arch. Capability truth: `../hardware/capability-matrix.md`.

## CDNA4 gfx950 — MFMA + scaled

```python
acc = gl.amd.cdna4.mfma(a, b, acc)
# each operand is followed by ITS OWN scale and ITS OWN format string; both formats are required
acc = gl.amd.cdna4.mfma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
```

The `mfma_scaled` argument order is `(a, a_scale, a_format, b, b_scale, b_format, acc)` — the two
scales are **not** grouped after the two operands, and the format strings are not optional.

Shapes: `[16, 16, 32]` default f16. **`mfma_scaled` has two registered shapes, not one —
`[16, 16, 128]` and `[32, 32, 64]`** (the `mfma_scale_f32_*_f8f6f4` entries; every f8f6f4 type keys
onto them, so fp8 on the scaled path lands there too). An earlier revision named only the second,
which reads as a constraint it is not. **There is no performance ordering between the two in this
pack** — the one comparison on record is a single measurement on a single kernel, 0.9898, i.e.
parity to within anything `n = 1` can resolve. Pick the shape by the tile you need
(`layout-reference.md`), then measure; do not carry a ranking out of this line. Scale layout is **derived, not authored**:
`gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)`, where
`scale_factor` is asserted equal to 32 (`layout-reference.md ## The Scale Operand's Layout For
mfma_scaled`; oracle side `../tile-programming/low-precision.md`).

## `ds_read_tr` (gfx950)

**No Gluon source API** — the backend emits the transpose LDS reads (`ds_read_b64_tr_b16` and
siblings) itself on gfx950 (`ds_read_tr: true` in `hw_constants.json`; **false on gfx942**, the
downgrade has no such instruction). Missing API = scoped ceiling or Scenario B pass; do not assume
from the capability matrix alone — read the ISA. Two consequences:

- **Where it exists the compiler emits it instead of `amdg.in_thread_transpose`**, so the "Gluon
  cannot express `in_thread_transpose`" gap has no subject on gfx950 and a faithful anchor inherits
  the instruction for free.
- **Moving a transpose onto the read is a saving only when the transpose is not already free.** Check
  the ISA for an existing `ds_read_*_tr_*` before reaching for transpose-on-read; the measured case
  where it lost is `memory-reference.md ### Shared-layout family + transpose-on-read (layout
  dependency)`.

## Memory atoms

| Path | API | Arch |
| --- | --- | --- |
| Buffer load | `gl.amd.cdna4.buffer_load` | gfx950 |
| Async → LDS | `gl.amd.cdna4.async_copy` | gfx950 at **16 B or 4 B per thread (a set, not a floor)** **and gfx942 through this same `cdna4` namespace** at 4 B only (`cdna3` has no async submodule) — `../hardware/capability-matrix.md` row `gfx942`. Destination layout is a gate: swizzled lowers, padded has been seen not to (`memory-reference.md`) |
| TDM | `gl.amd.gfx1250.tdm` (module) | gfx1250 only |

## CDNA3 gfx942 downgrade

What the gfx950 atoms above lose on CDNA3 (per-arch keys in
`perf_knowledge/hardware/data/hw_constants.json`: `scaled_mfma`, `ds_read_tr`, `fp8_dtype`,
`direct_to_lds_bit_widths`):

- **matrix:** `gl.amd.cdna3.mfma` with `AMDMFMALayout(version=3)`; **no `mfma_scaled`** —
  `mfma_scale_*_f8f6f4` is cannot-select on gfx942, a hard compiler abort rather than a fallback
  (`../pitfalls/platform-known-issues.md ## gfx942 / CDNA3 hard failures that affect benchmark validity`),
  so block-scaled formats go through a plain `tl.dot` comparator there. The plain-MFMA `k_width`
  relation still holds (`[16, 16, 16]` → 4, `matrix-reference.md ## Matrix-Family Details`).
- **fp8:** **FNUZ** (`float8_e4m3fnuz`) on CDNA3 against OCP `e4m3fn` on CDNA4. FP8 Gluon MFMA on
  gfx942 is a target-specific blocker; the FNUZ spelling is also the one dtype CDNA4 lacks, and
  Triton upcasts it there with only a warning (`../pitfalls/platform-known-issues.md`).
- **LDS reads:** **no `ds_read_tr`** — see the `ds_read_tr` (gfx950) section above.
- **async copy:** the same `cdna4.async_copy` entry points, at 32 bits per thread only (Memory
  atoms table above).

## gfx1250 — WMMA + TDM (not MFMA)

```python
gl.amd.gfx1250.wmma(...)
# tdm is a MODULE, not a callable: async_load / async_store / async_wait / make_tensor_descriptor
gl.amd.gfx1250.tdm.async_load(...)
```

Wave32; `PartitionedSharedLayout` for smem. **Not** gfx950 MFMA recipes.

## RDNA downgrade

`gl.amd.rdna3.wmma` / `gl.amd.rdna4.wmma` — see `rdna-wmma-reference.md`. Do not mix
with `AMDMFMALayout` escalation anchors.

## Blackwell analog (NVIDIA, for cross-read only)

TMA / tcgen05 / TMEM have **no** Gluon analog — scoped ceiling on AMD skills.
Note **TMA is NVIDIA-proprietary** (Hopper/Blackwell async tensor DMA); it is **not**
the same as AMD TDM. The closest AMD mechanism is **gfx1250 (CDNA5 / MI450) TDM**
(`gl.amd.gfx1250.tdm`), a data-center-only async→LDS copy. **RDNA4 (gfx1201) has
neither TMA nor TDM** — no async matrix DMA on RDNA at all.
