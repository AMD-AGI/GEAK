Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# AMD HIP Native Optimization Examples

This directory contains advanced examples demonstrating low-level HIP/ROCm optimization techniques for writing high-performance GPU kernels on AMD hardware.

## Examples Overview

### 1. Advanced Vector Addition (`advanced_vector_add.cpp`)
**Difficulty:** ⭐ Beginner  
**Key Concepts:** 64-thread waves, vectorized memory, inline assembly

**What you'll learn:**
- Wave-based execution model (64 threads vs NVIDIA's 32)
- Vectorized memory operations with `float4`
- Coalesced memory access patterns
- Inline assembly for explicit instruction control

**Build & Run:**
```bash
hipcc -O3 --offload-arch=gfx942 advanced_vector_add.cpp -o advanced_vector_add
./advanced_vector_add
```

---

### 2. MFMA GEMM BF16 (`mfma_gemm_bf16.cpp`)
**Difficulty:** ⭐⭐⭐ Advanced  
**Key Concepts:** MFMA instructions, AGPR usage, register tiles

**What you'll learn:**
- `v_mfma_f32_16x16x32_bf16` matrix multiply instruction
- VGPR vs AGPR register file management
- Register tile distribution across 64 threads
- BF16 data type usage

**Build & Run:**
```bash
hipcc -O3 --offload-arch=gfx942 mfma_gemm_bf16.cpp -o mfma_gemm
./mfma_gemm
```

---

### 3. Wave Reduction Softmax (`wave_reduction_softmax.cpp`)
**Difficulty:** ⭐⭐ Intermediate  
**Key Concepts:** Wave reductions, numerical stability, __shfl_xor

**What you'll learn:**
- Butterfly reduction across 64-thread waves
- `__shfl_xor` for wave-level communication
- Numerically stable softmax implementation
- Multi-wave cooperation with shared memory

**Build & Run:**
```bash
hipcc -O3 --offload-arch=gfx942 wave_reduction_softmax.cpp -o softmax
./softmax
```

---

### 4. Direct Buffer-to-LDS (`buffer_to_lds_direct.cpp`)
**Difficulty:** ⭐⭐ Intermediate  
**Key Concepts:** Direct transfers, buffer descriptors, readfirstlane hoisting

**What you'll learn:**
- `llvm_amdgcn_raw_buffer_load_lds` intrinsic
- Buffer resource descriptors (SRD)
- Readfirstlane hoisting optimization
- Bypassing VGPRs for memory transfers

**Build & Run:**
```bash
hipcc -O3 --offload-arch=gfx942 buffer_to_lds_direct.cpp -o buffer_lds
./buffer_lds
```

**Expected speedup:** 1.2-1.5x over standard approach

---

## Quick Start

### Prerequisites

1. ROCm 6.0+ installed
2. AMD GPU (CDNA3/CDNA4 architecture)
3. hipcc compiler available

```bash
# Check ROCm
rocm-smi

# Check compiler
hipcc --version

# Identify GPU architecture
rocminfo | grep "Name:"
```

### Build All Examples

```bash
# Replace gfx942 with your GPU architecture
for f in *.cpp; do
    hipcc -O3 --offload-arch=gfx942 "$f" -o "${f%.cpp}"
done
```

### GPU Architecture Flags

- MI350X/MI355X: `--offload-arch=gfx942`
- MI300X/MI325X: `--offload-arch=gfx941`
- MI250X: `--offload-arch=gfx90a`

---

## Learning Path

**Beginner:**
1. Start with `advanced_vector_add.cpp` - understand wave basics
2. Move to `wave_reduction_softmax.cpp` - learn reductions

**Intermediate:**
3. Study `buffer_to_lds_direct.cpp` - master memory optimizations
4. Tackle `mfma_gemm_bf16.cpp` - understand compute primitives

**Advanced:**
- Combine techniques from all examples
- Profile with `rocprofv3`
- Compare with existing kernels

---

## Key AMD GPU Concepts

### Wave Execution (64 threads)
```cpp
constexpr int WAVE_SIZE = 64;  // AMD: 64, NVIDIA: 32
__device__ int laneid() { return threadIdx.x & 0x3F; }
__device__ int waveid() { return threadIdx.x >> 6; }
```

### Register Files
- **VGPRs**: v[0:255] - General purpose vector registers
- **AGPRs**: a[0:255] - Accumulator registers for MFMA

### Memory Hierarchy
- **LDS**: 64 KB shared memory, 32 banks
- **L2 Cache**: Shared across XCDs (chiplets)
- **HBM3/HBM3e**: 5.3-6.9 TB/s bandwidth

---

## Profiling Your Kernels

### Basic Stats
```bash
rocprofv3 --stats ./your_kernel
```

### Detailed Metrics
```bash
rocprofv3 --pmc SQ_INSTS_VALU,SQ_INSTS_MFMA,SQ_LDS_BANK_CONFLICT ./your_kernel
```

### Resource Usage
```bash
hipcc --resource-usage your_kernel.cpp
```

### Key Metrics to Watch
- `SQ_INSTS_MFMA`: MFMA instruction count
- `SQ_LDS_BANK_CONFLICT`: LDS bank conflicts
- `TCC_HIT`/`TCC_MISS`: L2 cache hit/miss
- `VGPR`/`SGPR` usage: Register pressure

---

## Common Optimization Patterns

### 1. Direct Buffer-to-LDS Transfer
```cpp
// Bypass VGPRs for better throughput
llvm_amdgcn_raw_buffer_load_lds(buf_rsrc, lds_ptr, size, offset, 0, 0, 0);
```

### 2. Readfirstlane Hoisting
```cpp
// Compute once before loop
uint32_t lds_base = __builtin_amdgcn_readfirstlane(
    (uint32_t)(uintptr_t)&shared_data[0]
);
// Use lds_base in loop...
```

### 3. Wave Reductions
```cpp
for (int offset = 32; offset > 0; offset >>= 1) {
    float other = __shfl_xor(val, offset, 64);
    val = fmaxf(val, other);
}
```

### 4. Schedule Control
```cpp
__builtin_amdgcn_s_setprio(1);  // High priority
// Compute operations
__builtin_amdgcn_s_setprio(0);  // Normal
```

---

## Performance Expectations

On MI350X:

| Example | Metric | Expected Performance |
|---------|--------|---------------------|
| Vector Add | Bandwidth | ~5-6 TB/s |
| GEMM | Compute | ~100+ GFLOPS (small) |
| Softmax | Throughput | ~3-4 TB/s |
| Buffer-LDS | Speedup | 1.2-1.5x |

---

## Related Knowledge Base

For deeper understanding, see:
- `knowledge-base/amd-knowledge-base/layer-2-compute-stack/hip/hip-advanced-optimization.md`
- `knowledge-base/amd-knowledge-base/layer-2-compute-stack/hip/hip-basics.md`

---

## Troubleshooting

### Compilation Issues

**Problem:** Unknown target architecture
```bash
error: unknown target 'gfx942'
```
**Solution:** Use correct arch flag for your GPU

**Problem:** Undefined MFMA instruction
```bash
error: undefined instruction v_mfma_*
```
**Solution:** MFMA requires CDNA architecture (MI200/MI300/MI350 series)

### Runtime Issues

**Problem:** Low performance
**Solution:**
1. Check GPU clocks: `rocm-smi --showclocks`
2. Profile: `rocprofv3 --stats ./kernel`
3. Verify no register spilling: `hipcc --resource-usage`

**Problem:** Bank conflicts in LDS
**Solution:** Apply swizzling patterns (see examples)

---

## Tips for Best Results

1. **Start simple** - Run examples as-is first
2. **Profile everything** - Use rocprofv3 to understand performance
3. **Experiment** - Change parameters and observe impact
4. **Compare** - Benchmark against rocBLAS, MIOpen
5. **Resource usage** - Monitor VGPR/AGPR/LDS usage

---

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [AMD GPU ISA Documentation](https://www.amd.com/en/support/graphics/amd-radeon-pro/amd-radeon-pro-5000-series)

---

**Last Updated:** 2024  
**Target ROCm:** 6.0+  
**Hardware:** CDNA3/CDNA4 (MI300X, MI350X series)

