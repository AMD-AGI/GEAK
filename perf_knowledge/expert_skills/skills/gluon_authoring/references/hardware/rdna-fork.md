# RDNA client GPUs — gfx11* (RDNA3) vs gfx120* (RDNA4)

**Downgrade / optional target** for the three AMD tile skills. **Do not mix** RDNA WMMA
kernels with CDNA MFMA or gfx1250 WMMA+TDM binaries.

**GEAK support scope:** gfx1201 (RDNA4, R9700) is a GEAK-supported client target — its GEAK cards
are `perf_knowledge/hardware/rdna4_gfx1201/{arch,occupancy,pitfalls}.md` and
`kernel_workflow/knowledge/amd_rdna4.md`; gfx1151 (RDNA 3.5 APU) via `kernel_workflow/knowledge/amd_ryzen.md`.
**gfx1100 (RDNA3) and gfx1200 rows are not supported by GEAK** and are kept here for reference
only. SKU peaks: `perf_knowledge/hardware/data/sku.json` (human view `amd-rdna*-skus.md`).

ISA references: [RDNA3 shader ISA (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023.pdf),
[RDNA4 shader ISA (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).

## Product line placement

```text
CDNA Instinct (gfx942/950)  → MFMA wave64, mfma_scaled, ds_read_tr, async→LDS  [primary]
RDNA client     (gfx11/120) → WMMA wave32, no blockscale, hand LDS ping-pong     [downgrade]
MI450           (gfx1250/CDNA5) → WMMA+TDM, named-barrier warpgroups, NOT gfx120* compatible  [separate]
```

## RDNA3 (gfx11*) vs RDNA4 (gfx120*)

| Category | gfx11* (RDNA3) | gfx120* (RDNA4) |
| --- | --- | --- |
| Wave | **wave32** default (client) | wave32; VOPD is wave32-only. **Dynamic VGPR is NOT here** — see §Dynamic VGPR |
| WMMA core | 16×16×16 f16/bf16/i8/**i4** | Same + **16×16×16 FP8/BF8** |
| WMMA ABI | **v16-operand** layout | **v8-operand** layout (different lane map) |
| FP8 GEMM | **No** native FP8 WMMA | `V_WMMA_F32_16X16X16_FP8_*` |
| Block scale / OCP FP4 | **No** `mfma_scaled` | **No** — use dequant+WMMA or IU4 WMMA |
| Sparse | — | **SWMMAC** 4:2 (RDNA4) |
| LDS | **128 KiB/WGP**; **≤64 KiB/work-group** | Same |
| VGPR | ≤256/wave static | ≤256/wave static — **no** dynamic VGPR (see §Dynamic VGPR) |
| WMMA operand ABI | A/B = **8 VGPR**, replicated across lane halves | A/B = **4 VGPR**, no replication |
| WMMA rate (16×16×16 f16) | 32 cyc — 1024 FLOP/WGP/clk | **16 cyc** — 2048 FLOP/WGP/clk |
| GMEM | buffer/global_load + `S_WAITCNT` | **`GLOBAL_LOAD_TR_*`** transpose loads |
| GWS (`DS_GWS`) | present | **removed** |
| Async cp/TDM | **ISA: none** | **ISA: none** (TDM is gfx1250/CDNA5/MI450, not RDNA4) |

## Dynamic VGPR — not an RDNA4 feature

`S_ALLOC_VGPR` is rejected as `invalid instruction` by the ROCm 7.1 / LLVM 20 assembler
for **both gfx1200 and gfx1201**, and no `+dynamic-vgpr` subtarget feature exists for
those targets. Dynamic VGPR belongs to **gfx1250 (CDNA5)**. Plan RDNA4 occupancy on the
static **≤256 VGPR/wave** model only, and do not carry a "dynamic VGPR relieves the
occupancy wall" assumption over from gfx1250 notes.

Provenance: `echo 's_alloc_vgpr 16' | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201`
(also with `-mattr=+dynamic-vgpr`, which the target does not recognize).

## Wave size (gfx11* / gfx120*) — verified

| Source | RDNA3 (gfx11) |
| --- | --- |
| [HIP hardware features](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html) | **Native wavefront = 32**; optional experimental `mwavefrontsize64` → 64 (not HIP-runtime-supported) |
| [GPUOpen WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/) | WMMA intrinsics exist as **`_w32` and `_w64`**; fragment layout differs (C/D: 8 vs 4 VGPRs per lane) |
| [rocWMMA](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/) | gfx1100/1101/1102 listed under **wave32** RDNA targets |

**Planning default for these tile skills: wave32.** Use `wave_size=32` in `calc_perf.py budget/occ`
and WMMA tile math (`R_acc` divides by `num_warps × 32`). Gluon/FlyDSL/TileLang RDNA GEMM paths
and rocWMMA target gfx11 in wave32 mode.

Do **not** assume CDNA wave64 occupancy formulas. Wave64 WMMA is an ISA/compiler option
(`mwavefrontsize64`, `_w64` builtins) — only plan for it if the kernel ISA dump shows
`V_WMMA_*_w64` / wave64 mode; otherwise treat as wave32.

## vs CDNA (do not confuse)

| Dimension | CDNA gfx942/950 | RDNA gfx11/120 |
| --- | --- | --- |
| Matrix op | **MFMA** | **WMMA** / SWMMAC |
| Wave | 64 | **32** |
| **VALU co-execution** | **Yes** — MFMA co-issues with VALU (gfx942 `16x16x16_f16`: 12 of 16 cyc) | **No** — WMMA cannot co-issue with VALU at all |
| Accumulator file | separate **AGPR** (`v_accvgpr_read` to reach from VALU) | none — unified VGPR |
| FP8 | MFMA 16×16×32; gfx950 **mfma_scaled** | RDNA4: 16×16×16 FP8 WMMA only |
| FP4 | gfx950 scaled MFMA | IU4 WMMA (not OCP MXFP4) |
| LDS | 64–160 KiB/**CU** | 128 KiB/**WGP**, ≤64 KiB/**WG** |
| Low-precision | hardware blockscale | **dequant + WMMA** |

## Four-DSL API map (code anchors)

| Capability | FlyDSL | Gluon | TileLang |
| --- | --- | --- | --- |
| gfx11 f16 WMMA | `rdna3_f16_gemm.py` in [ROCm/FlyDSL `kernels/`](https://github.com/ROCm/FlyDSL/tree/main/kernels) | `gl.amd.rdna3.wmma` | [carver/arch/rdna.py](https://github.com/tile-ai/tilelang/blob/main/tilelang/carver/arch/rdna.py) gen=11 |
| gfx120 f16/bf16 | `rdna_f16_gemm.py` in [ROCm/FlyDSL `kernels/`](https://github.com/ROCm/FlyDSL/tree/main/kernels) | `gl.amd.rdna4.wmma` | gen=12 |
| gfx120 FP8 | `rdna_fp8_preshuffle_gemm.py` in [ROCm/FlyDSL `kernels/`](https://github.com/ROCm/FlyDSL/tree/main/kernels) | rdna4.wmma + OCP fp8 | WMMA fp8 lowering |
| Layout | hand ThrVal; v16 vs v8 store map | `AMDWMMALayout` | infer + `T.annotate_layout` |
| Pipeline | hand LDS ping-pong | no `num_stages` | `pipeline_stage` 1–2 |
| Tests | [test_rdna_gemm.py](https://github.com/ROCm/FlyDSL/tree/main/tests/kernels) | [amd.rdna3/4.rst](https://github.com/triton-lang/triton/tree/main/docs/gluon/api) | [test_tilelang_rocm_target.py](https://github.com/tile-ai/tilelang/tree/main/testing/python/amd) |

## Occupancy note (RDNA)

Do **not** apply CDNA's combined arch+accum **512 VGPR** formula on RDNA. Two numbers, two
jobs, and the CDNA habit of reusing one for both is what breaks here:

- **≤256 VGPR / wave** — the addressable cap. Exceed it and the wave does not exist (spills).
- **1536 VGPR / SIMD** (wave32, allocation granule 24, ≤16 waves/SIMD) — the register FILE,
  which is what occupancy divides into. RDNA has **no AGPR file**, so nothing is "combined".

`waves/SIMD = min(16, 1536 // round_up(vgpr, 24))`. Feeding 256 in as the file size (the CDNA
reflex, where cap and file coincide at 512) under-reports occupancy by 2-3x and turns a
register-comfortable kernel into a fake "occupancy-1" story. One implementation for all of
it: `kernel_workflow/scripts/kernel_tools/amd_occupancy.py` (pack shim `scripts/amd_occupancy.py`). When you have the `.s`, skip the model for the register term —
LLVM writes `; Occupancy: N` per kernel and it is the authority **for that term**
(`amd_occupancy.py --asm kernel.s`, also printed by the loop audit's KD block).

**Scope, same as on CDNA.** The emitted value is a register-side bound; on a kernel whose LDS is
allocated dynamically at launch it carries **no LDS term** (the emitter prints `LDSByteSize: 0
bytes/workgroup (compile time only)` in the same block), so where it disagrees with a
hand-derived `min(VGPR-limited, LDS-limited)` the hand derivation is the more correct one. Full
statement, the measured evidence, and the LDS-per-workgroup field to use instead:
`planning-constants.md ## The emitted ; Occupancy: N is a register-term answer`.

## SKU planning peaks

Roofline matrix peaks, BW, `CUs`, `L2_MB`: `amd-rdna3-skus.md` (gfx1100 discrete),
`amd-rdna35-skus.md` (gfx1151 APU), `amd-rdna4-skus.md` (gfx120* / gfx1201).
**RX 9070 XT vs AI PRO R9700** share WMMA ISA — differ in VRAM and slightly in marketed
TFLOPS; use the matching SKU row.

## Skill routing

- plain Triton: the `tile-programming-triton` skill (front end; `tl.*` has no WMMA layout to pin)
- Gluon: the `tile-programming-gluon` skill (`references/gluon/rdna-wmma-reference.md`)
- FlyDSL: the `tile-programming-flydsl` skill (matrix-reference WMMA section)
- TileLang: the `tile-programming-tilelang` skill (gemm-mfma RDNA section)
