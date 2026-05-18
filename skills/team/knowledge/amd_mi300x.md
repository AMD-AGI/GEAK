# AMD MI300X (Instinct) Hardware Reference

## Architecture
- **GPU Architecture**: CDNA 3 (gfx942)
- **Compute Units (CUs)**: 304
- **Stream Processors**: 19,456
- **Wavefront Size**: 64 threads (NOT 32 like NVIDIA warp)
- **Max VGPRs per thread**: 256 (512 for VGPR pairs)
- **Max SGPRs per thread**: 106

## Memory Hierarchy
- **HBM3 Bandwidth**: 5.3 TB/s (8 stacks, 192 GB total)
- **L2 Cache**: 256 MB (shared across all CUs)
- **L1 Cache**: 32 KB per CU (16 KB data + 16 KB instruction)
- **LDS (Local Data Share)**: 64 KB per CU
- **LDS Banks**: 32 banks, 4 bytes per bank per cycle
- **LDS Bandwidth**: ~400 bytes/cycle per CU

## Compute Capabilities
- **Peak FP32**: 163.4 TFLOPS
- **Peak FP16**: 1,307.4 TFLOPS (with MFMA)
- **Peak INT8**: 2,614.9 TOPS (with MFMA)
- **MFMA Instructions**: Matrix Fused Multiply-Add (4x4, 16x16, 32x32)

## Occupancy
- **Max Wavefronts per CU**: 32
- **Max Wavefronts per SIMD**: 8 (4 SIMDs per CU)
- **VGPR budget per CU**: 512 KB (65536 VGPRs * 4 bytes * 2 for pairs)

| VGPRs/thread | Max Waves/SIMD | Occupancy |
|-------------|----------------|-----------|
| 24          | 8              | 100%      |
| 28-32       | 7              | 87.5%     |
| 36          | 6              | 75%       |
| 40-48       | 5              | 62.5%     |
| 56-64       | 4              | 50%       |
| 84          | 3              | 37.5%     |
| 128         | 2              | 25%       |
| 256         | 1              | 12.5%     |

## Launch Configuration
- **Max Threads per Block**: 1024
- **Max Blocks per CU**: 32 (limited by resources)
- **Typical Block Sizes**: 64, 128, 256 (multiples of wavefront size 64)

## Critical Rules
1. **NEVER** set `HIP_VISIBLE_DEVICES` inline with profiler commands. Always use gpu_lock.sh.
2. **NEVER** use `__syncthreads()` across wavefronts of different sizes. Use `__syncthreads()` only for block-level sync.
3. Wavefront-level operations (`__shfl_xor`, `__ballot`, `__any`, `__all`) operate on 64 threads, not 32.
4. Memory coalescing granularity is 64 bytes (one cache line) for global memory.
5. LDS bank conflicts: 32 banks, stride of 32*4=128 bytes causes no conflict. Stride of 4 bytes causes 32-way conflict.
6. Prefer `__launch_bounds__(max_threads, min_waves)` to help compiler optimize register allocation.
