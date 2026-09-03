# AMD Radeon 8060S / Strix Halo Hardware Reference — gfx1151 / RDNA 3.5

Use this card when `rocminfo` reports `gfx1151` or the target is `radeon8060s`. Do not apply the
Instinct/CDNA card's MFMA, wave64-only, HBM, FNUZ/OCP-FP8, AITER, CK, XCD, or partition-mode
assumptions to this target. Prefer live detection and measured behavior over every static number.

## 0. Detect the target first

```bash
rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+'
cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null
rocm-smi --showproductname 2>/dev/null
```

For a Radeon 8060S, verify that live KFD reports:

- `gfx_target_version 110501` → `gfx1151`;
- `simd_count 80`, two SIMD32 per CU → **40 CUs**;
- four shader arrays, two SIMD arrays per engine, ten CUs per SIMD array;
- native `wave_front_size 32` and `max_waves_per_simd 16`;
- `lds_size_in_kb 64` per CU;
- PCI device `1002:1586`, driven by `amdgpu`.

Record the detected values in the profile. Do not infer them from an MI product table.

## 1. Board and memory model

| Property | Radeon 8060S / Strix Halo |
|---|---|
| Architecture | RDNA 3.5 / `gfx1151` |
| Compute units | 40 (live KFD) |
| Native wave | wave32; wave64 is supported but issues vector work in two halves |
| Memory | unified/shared system memory (UMA), not discrete HBM |
| ROCm device view | small dedicated aperture plus large GTT/shared-memory access |
| Per-CU LDS | 64 KiB live; 128 KiB per WGP in the ISA model |
| Matrix family | RDNA WMMA/VOP3P, not CDNA MFMA |

The 256-bit LPDDR5X-8000 configuration has a 256 GB/s theoretical ceiling. Measure sustained GPU
bandwidth on the target node instead of treating that ceiling as achieved throughput.

UMA consequences:

- weights, KV, host code and CPU activity contend for the same memory channels;
- GTT allocation is not a separate physical pool from system RAM;
- host launch, page residency and copies can matter as much as nominal kernel occupancy;
- memory-capacity checks must use live `MemAvailable`, GTT ownership and co-tenant state;
- never add GTT and process RSS as if they were independent capacities.

## 2. Execution model

- A WGP contains four SIMD32s; a CU is one half of a WGP and contains two SIMD32s.
- A work-group stays on one WGP and may synchronize through LDS.
- Wave32 issues a vector instruction once. Wave64 generally issues VALU and VMEM/LDS instructions
  twice (low then high half), while SALU, scalar memory, branches and messages issue once.
- VOPD dual issue is **wave32-only** and has strict register-bank, source-count and destination
  even/odd rules. Treat VOPD as an ISA-level optimization requiring disassembly and correctness proof.
- EXEC and VCC are 64-bit architectural masks; wave32 uses their low 32 bits.
- VGPR allocation is in blocks of 16 for wave32 and 8 for wave64; a shader may address up to 256 VGPRs.
- Live KFD reports at most 16 waves per SIMD on this device. Actual occupancy is also constrained by
  VGPR use, LDS, work-group size and compiler-generated private/scratch state.

Do not copy an MI300/MI355 `nwarps` or block-size table. A value expressed in waves yields a different
thread count on wave32, and CDNA MFMA kernels have different collective layouts.

## 3. LDS and synchronization

The ISA describes 128 KiB LDS per WGP, split into two 64 KiB CU halves, with 64 banks total. A single
work-group may allocate at most 64 KiB.

- CU mode confines waves to one CU half and permits `LDS_PARAM_LOAD`/`LDS_DIRECT_LOAD`.
- WGP mode exposes one contiguous WGP LDS but does not support those direct/parameter load forms.
- Indexed and atomic LDS operations serialize on bank conflicts.
- Barriers synchronize work-group waves; one-wave groups do not consume a barrier resource.
- Use `s_waitcnt lgkmcnt(...)` for LDS/SMEM/GDS completion. NOP padding is not a memory dependency fix.

Derive swizzles for the 64-bank/two-half layout. MI/CDNA 32-bank padding rules are not portable facts.

## 4. Matrix and low-bit instructions

RDNA3.5 exposes WMMA 16×16×16 instructions through VOP3P for:

- F16 inputs with F32 or F16 accumulation/output forms;
- BF16 inputs with F32 or BF16 forms;
- IU8 inputs with I32 accumulation;
- IU4 inputs with I32 accumulation.

Architectural availability does not prove a particular ROCm framework route is implemented or fast.
WMMA A/B fragments require lane replication; A is column-major in the VGPR view and B/C/D are
row-major. Dependent back-to-back WMMA instructions can require an independent VALU op or `V_NOP`
when the first destination overlaps the next A/B source.

For IU dot/WMMA instructions, NEG bits select signedness rather than ordinary numeric negation. Preserve
exact signed/unsigned algebra and accumulate integer products in I32. `AMD_WMMA_AVAILABLE` is valid on
gfx1151; `AMD_MFMA_AVAILABLE` is a CDNA path and must not be assumed.

## 5. Dtypes and framework claims

Architectural scalar/vector support includes integer, F16, BF16 and F32 operations plus the WMMA forms
listed above. The following are software-route questions, not facts implied by the ISA:

- FP8 checkpoint format and conversion support;
- MXFP4/MXFP6 block scaling;
- AITER/CK kernels written for gfx942/gfx950;
- Triton lowering for a particular op and shape;
- framework quantization loaders and serving dispatch.

Use only a route proven in the selected vLLM/SGLang image. Treat every checkpoint format and fused
kernel as framework-specific until correctness and dispatch evidence proves that exact path on gfx1151.

## 6. Memory instructions, caches and waits

- Prefer GLOBAL when an address is known global. FLAT participates in both VM/VScnt and LGKM domains
  and can tie up LDS machinery unnecessarily.
- `VMcnt` covers vector loads/samples and returning atomics; `VScnt` covers stores/non-return atomics;
  `LGKMcnt` covers LDS, scalar memory, GDS and messages; `EXPcnt` covers exports and LDS direct/param loads.
- GLC/SLC/DLC fields affect first-level behavior, L2 temporal policy and MALL/Infinity Cache policy.
  Cache flags are semantic inputs, not decorative tuning bits.
- Scalar memory may return out of order. Wait before consuming its SGPR destinations.

## 7. Roofline and measurement

For this UMA APU, use an on-box bandwidth probe and the actual served case mix. Do not use MI300/MI355
HBM or peak-FLOP tables. Report:

- exact gfx target and runtime/toolchain;
- wave size and launch geometry;
- VGPR/SGPR/LDS/private/spill metadata from the selected code object;
- host wall, device timing, and dispatch count separately;
- temperature, memory pressure and co-tenant state;
- prefill and decode separately because their routes and bottlenecks differ.

Do not infer a bandwidth-bound or compute-bound classification from architecture alone; measure the
actual prefill/decode shape and preserve the profiler evidence.

## 8. Critical rules

1. Detect `gfx1151` and 40 CUs from the live node; never fall back to `gfx942`/`gfx950`.
2. Compile native code with `--offload-arch=gfx1151`; do not set `HSA_OVERRIDE_GFX_VERSION`.
3. Assume wave32 unless the exact code object proves a deliberate wave64 kernel.
4. Use RDNA WMMA/VOPD rules; do not suggest CDNA MFMA instructions.
5. Keep each work-group at or below 64 KiB LDS and derive 64-bank addressing.
6. Treat UMA/GTT as shared system memory and check live capacity/ownership.
7. Do not suggest AITER/CK/Instinct routes unless the selected Strix image proves that exact path.
8. Require correctness and route evidence before interpreting a kernel benchmark.

## Sources

- AMD, *RDNA 3.5 Instruction Set Architecture Reference Guide*:
  https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture
- AMD/GPUOpen machine-readable ISA:
  https://gpuopen.com/machine-readable-isa/
- AMDResearch IntelliKit PR #123, *metrix: add gfx1151 (Strix Halo) support*:
  https://github.com/AMDResearch/intellikit/pull/123
