---
title: gfx1151 ISA notes — wave32/wave64, register file, reading the disassembly
kind: hardware
gens: [gfx1151]
dtypes: [fp32, bf16, fp16, int8, int4]
regimes: [both]
updated: 2026-09-23
sources:
  - GEAK/kernel_workflow/knowledge/amd_rdna.md §0, §2, §6
  - perf_knowledge/expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py (ROCm 7.2.1 / LLVM 22)
  - on-box rocminfo / rocprofv3, ROCm 7.2.3
---

# gfx1151 ISA notes

> Matrix instructions in [matrix_core_wmma.md](matrix_core_wmma.md); occupancy consequences in
> [arch.md](arch.md).

## TL;DR
> **Route on the `gfx` prefix, never on wave size** — RDNA supports wave32 *and* wave64, so a
> wave64 kernel on `gfx11` is still not a CDNA kernel. Register file is **1536 VGPR/SIMD, granule
> 24, no AGPRs**. When reading disassembly, `v_wmma` present and `v_mfma` absent is the correct
> state, not a fallback.

## Routing rules
1. **`gfx` prefix decides the file.** `gfx9xx` → CDNA (`../cdna*/`). `gfx10/11/12` → RDNA (here).
2. **Capability decides the strategy.** Ask what the matrix path *is*: `mfma`, `wmma`, or **`none`**
   (`gfx10` is RDNA with no matrix instruction at all).
3. **Wave mode is a check, not the decision.** Use the compiled wave size to size tiles and to catch
   a contradiction — if family and wave mode disagree, **stop and re-detect** rather than guessing.

```bash
# rocminfo lists the CPU agent FIRST; scope every field to the gfx agent.
rocminfo | awk '/^ *Name: *gfx/{print $2; exit}'                        # gfx1151
rocminfo | awk '/Name:.*gfx/{f=1} f&&/Wavefront Size:/{print $3; exit}' # 32 (default)
```

## Register file
| | value |
|---|---|
| VGPR per SIMD | **1536** × 4 B |
| allocation granule | **24** |
| AGPR | **none** |
| SIMDs per CU | 2 |
| wave slots | 16/SIMD hard cap |

With no AGPRs there is no accumulator-parking trick: accumulator pressure lands on the VGPR budget
directly and buys occupancy loss at the granule boundaries (≤96 → 16 waves, 120 → 12, 144 → 10,
168 → 9, 192 → 8, 216 → 7). Spills go to **scratch**, which on this part is backed by the same
LPDDR5X the kernel is already streaming from — a spill is far more expensive here than the CDNA
intuition suggests.

## Scalar / dot instructions
- **`sudot4` is available** (mixed-sign dot4). CDNA has only `sdot4`. Relevant for asymmetric int8
  quantisation where the CDNA workaround (bias-correction passes) can simply be dropped.

## Reading the disassembly
```bash
grep -c 'v_wmma' kernel.amdgcn   # > 0 expected on a matrix kernel
grep -c 'v_mfma' kernel.amdgcn   # 0 expected — nonzero means the target is wrong
```
`[measured]` A 2048³ fp16 Triton GEMM compiles to **128 × `v_wmma_f32_16x16x16_f16`, zero
`v_mfma`, zero scalar `v_fma`** — i.e. `tl.dot` really does reach the matrix units on this part.

### Decoding a Tensile kernel name from rocprofv3
```
Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_..._ISA1151_...
                └─ HHS: fp16 in / fp32 accum      │        └─ target ISA
                        MT: macro-tile 128x128x16 └─ matrix instruction 16x16x16
```
`ISA1151` is how you confirm you reached a gfx1151-tuned library kernel rather than a generic
fallback. A peak whose instruction cannot be identified this way should be reported as
**ISA-UNVERIFIED**, not trusted.

## Pitfalls
- **Inferring the architecture from wave size.** Wave64 on RDNA is legal.
- **Assuming a spill is cheap** because there is a large VGPR file. There is no AGPR tier, and
  scratch lands in shared LPDDR5X.
- **Reusing a CDNA `pmc:` counter list.** The raw `TCP_`/`TCC_`/`TD_` names are unavailable here;
  see [tooling_and_profiling.md](tooling_and_profiling.md).

## Verify
- `rocminfo` for target + wave size (scoped to the gfx agent).
- Disassembly greps above.
- `rocprofv3` kernel names for the `ISA1151` tag.

## Sources
- `GEAK/kernel_workflow/knowledge/amd_rdna.md` §0, §2, §6.
- `perf_knowledge/expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py` (occupancy table,
  verified against ROCm 7.2.1 / LLVM 22).
