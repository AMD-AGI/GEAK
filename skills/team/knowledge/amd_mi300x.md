# AMD MI300X (gfx942) Hardware Reference

## Architecture Overview

| Spec | Value |
|------|-------|
| Architecture | CDNA 3 |
| Compute Units (CUs) | 304 |
| Wavefront Size | 64 threads (some kernels support wave32) |
| SIMDs per CU | 4 |
| Wave Slots per SIMD | 8 → max 32 wavefronts per CU |
| VGPRs per CU | 512 (shared across all waves on a SIMD) |
| SGPRs per CU | 108 |
| LDS per CU | 64 KB (32 banks, 4 bytes/bank) |
| L1 Cache per CU | 32 KB |
| L2 Cache | 256 MB shared |
| HBM3 Stacks | 8 → ~5.3 TB/s peak bandwidth |
| Peak FP32 | ~163 TFLOPS |
| Peak FP16 / BF16 | ~1300 TFLOPS (via MFMA) |

## Memory Hierarchy (fastest → slowest)

1. **Registers (VGPRs/SGPRs)**: Fastest. 512 VGPRs per CU shared across active wavefronts. More VGPRs per wave = fewer concurrent waves (occupancy trade-off).
2. **LDS**: ~2 TB/s effective bandwidth. 64 KB per CU, 32 banks. Use for intra-workgroup data sharing and tiling.
3. **L1 Cache**: 32 KB per CU. Automatic caching of global reads. Cache line = 128 bytes.
4. **L2 Cache**: 256 MB shared across all CUs. ~4 TB/s bandwidth. Good for working sets up to ~200 MB.
5. **HBM3**: ~5.3 TB/s peak. Highest latency. Coalesced 128-byte aligned accesses essential.

## Occupancy and Register Pressure

| VGPRs per wave | Max waves per SIMD | Max waves per CU | Occupancy |
|----------------|--------------------|--------------------|-----------|
| ≤64 | 8 | 32 | 100% |
| ≤96 | 5 | 20 | 62.5% |
| ≤128 | 4 | 16 | 50% |
| ≤256 | 2 | 8 | 25% |
| >256 | 1 | 4 | 12.5% |

High occupancy helps hide memory latency. For memory-bound kernels, target ≥50% occupancy. For compute-bound kernels with high ILP, lower occupancy can be acceptable.

## MFMA Instructions

Matrix Fused Multiply-Add (MFMA) instructions for mixed-precision matrix math:
- `mfma_f32_16x16x16_f16`: 16x16 output, 16-deep accumulate, FP16 input → FP32 output
- `mfma_f32_32x32x8_f16`: 32x32 output, 8-deep
- Also available for BF16, FP8, INT8

## LDS Bank Conflicts

32 banks, each 4 bytes wide. Stride-32 access patterns cause N-way bank conflicts.

**Avoid:** `lds[threadIdx.x * 32]` (all threads hit same bank)
**Good:** `lds[threadIdx.x]` (consecutive, no conflict)
**Padding trick:** Allocate `float lds[N + 1]` to break stride-32 patterns.

## Critical GPU/Profiler Rules

**RULE 1: HIP_VISIBLE_DEVICES is ALREADY SET by the framework.** Do NOT set or export it yourself. Adding `HIP_VISIBLE_DEVICES=X` inline with a command will CRASH rocprofv3.

**RULE 2: Profile commands.** When using rocprof-compute or rocprof, pass ONLY the application command after `--`. Never prefix env vars.
```bash
# CORRECT
rocprof-compute profile --no-roof -- python3 harness.py --profile

# WRONG — will crash
HIP_VISIBLE_DEVICES=0 rocprof-compute profile --no-roof -- python3 harness.py --profile
```

**RULE 3: Use absolute paths.** Do not use `cd /path && command`. Use full paths in all commands.

## Key Optimization Principles

- **Coalesced access**: Consecutive threads should access consecutive memory addresses. Each wavefront reads 64 × element_size bytes per memory instruction.
- **Vector loads**: `float4` loads issue one 128-bit memory instruction → 4x fewer instructions for sequential data. Requires 16-byte alignment.
- **Cache line**: 128 bytes. Access patterns should be aligned to cache lines when possible.
- **Instruction mix**: Balance VALU (vector ALU), VMEM (vector memory), MFMA, and LDS instructions to keep all pipelines busy.
- **Wavefront divergence**: All 64 threads in a wavefront execute in lockstep. Branch divergence serializes both paths.
