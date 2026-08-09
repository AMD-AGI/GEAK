# Gluon — copy/MMA atoms (CDNA MFMA + gfx1250 WMMA)

Choosing the matrix/copy path by arch. Capability truth: `../hardware/capability-matrix.md`.

## CDNA4 gfx950 — MFMA + scaled

```python
acc = gl.amd.cdna4.mfma(a, b, acc, ...)
acc = gl.amd.cdna4.mfma_scaled(a, b, scale_a, scale_b, acc, ...)  # mxfp4/fp8 blockscale
```

Shapes: 16×16×32 default f16; **32×32×64** scaled mxfp4 (`mfma_scaled`). Scale layout
must match oracle (`../tile-programming/low-precision.md`).

## CDNA3 gfx942 downgrade

`gl.amd.cdna3.mfma` — no `mfma_scaled` production path.

## gfx1250 — WMMA + TDM (not MFMA)

```python
gl.amd.gfx1250.wmma(...)
gl.amd.gfx1250.tdm(...)  # async matrix load
```

Wave32; `PartitionedSharedLayout` for smem. **Not** gfx950 MFMA recipes.

## RDNA downgrade

`gl.amd.rdna3.wmma` / `gl.amd.rdna4.wmma` — see `rdna-wmma-reference.md`. Do not mix
with `AMDMFMALayout` escalation anchors.

## `ds_read_tr` (gfx950)

**No Gluon source API** — backend may emit transpose LDS reads. Missing API = scoped
ceiling or Scenario B pass; do not assume from capability matrix alone.

## Memory atoms

| Path | API | Arch |
| --- | --- | --- |
| Buffer load | `gl.amd.cdna4.buffer_load` | gfx950 |
| Async → LDS | `gl.amd.cdna4.async_copy` | gfx950 (16 B/op) |
| TDM | `gl.amd.gfx1250.tdm` | gfx1250 only |

## Blackwell analog (NVIDIA, for cross-read only)

TMA / tcgen05 / TMEM have **no** Gluon analog — scoped ceiling on AMD skills.
Note **TMA is NVIDIA-proprietary** (Hopper/Blackwell async tensor DMA); it is **not**
the same as AMD TDM. The closest AMD mechanism is **gfx1250 (CDNA5 / MI450) TDM**
(`gl.amd.gfx1250.tdm`), a data-center-only async→LDS copy. **RDNA4 (gfx1201) has
neither TMA nor TDM** — no async matrix DMA on RDNA at all.
