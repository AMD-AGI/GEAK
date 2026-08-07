# Gluon — shared memory / LDS layouts

## CDNA (gfx942/gfx950)

- LDS declared via TTGIR shared layout; capacity per CU: 64 KiB (gfx942), 160 KiB
  (gfx950); alloc granularity 512 B vs **1280 B** on gfx950 (`../hardware/planning-constants.md`).
- Swizzle: `PaddedSharedLayout`, blocked layouts recovered from TTGIR transcription.
- Pipeline stages = separate LDS buffers per stage (hand-sized; no `num_stages` knob).

## gfx1250

- `PartitionedSharedLayout` for WMMA operand staging (wave32).
- 320 KiB LDS partition model — probe occupancy per kernel.

## vs CuTeDSL smem/TMEM

| CuTeDSL | Gluon |
| --- | --- |
| TMEM accumulator (sm_100) | **scoped ceiling** — AGPR/VGPR on AMD |
| TMA swizzle modes | LDS swizzle + buffer resource |
| `make_swizzle` XOR | padded/blocked shared layouts |

## Footguns

- gfx950 **1280 B** LDS alignment waste on small tiles.
- gfx942 kernel on gfx950 without relayout → wrong results or spill.

## Anchors

- [triton-lang/triton `gluon/language/amd/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd)
