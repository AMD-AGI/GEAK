---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# NVIDIA CUDA Native Optimization Examples

This directory contains advanced examples demonstrating low-level CUDA optimization techniques for writing high-performance GPU kernels on NVIDIA hardware.

## Examples Overview

### 1. Tensor Cores MMA (`tensor_cores_mma.cu`)
**Difficulty:** ⭐⭐⭐ Advanced  
**Architecture:** Ampere/Ada/Hopper (sm_80+)  
**Key Concepts:** mma.sync, tensor core fragments, register layout

**What you'll learn:**
- PTX inline assembly for `mma.sync` instructions
- Register fragment layout (4× `__nv_bfloat162` per thread)
- Loading 16×16 tiles to match tensor core layout
- Accumulator management (float vs half precision)

**Expected speedup:** 10-20x over CUDA cores

---

### 2. Shared Memory Swizzling (`shared_memory_swizzling.cu`)
**Difficulty:** ⭐⭐ Intermediate  
**Architecture:** All (sm_70+)  
**Key Concepts:** Bank conflicts, XOR swizzling, padding trick

**What you'll learn:**
- XOR-based address swizzling formula
- Padding trick (+1 column) for simple swizzle
- Bank conflict measurement and profiling
- Transpose with/without conflicts

**Expected speedup:** 2-4x for conflict-heavy patterns

---

### 3. Async Copy Pipeline (`async_copy_pipeline.cu`)
**Difficulty:** ⭐⭐⭐ Advanced  
**Architecture:** Ampere+ (sm_80+)  
**Key Concepts:** cp.async, pipelining, commit/wait groups

**What you'll learn:**
- `cp.async` PTX instructions for async memory copy
- Multi-stage ping-pong buffers (2-stage vs 3-stage)
- Commit and wait group management
- Overlapping memory transfer with computation

**Expected speedup:** 1.5-3x for memory-bound kernels

---

### 4. Warp Shuffle Ops (`warp_shuffle_ops.cu`)
**Difficulty:** ⭐⭐ Intermediate  
**Architecture:** All (sm_30+)  
**Key Concepts:** __shfl operations, reductions, broadcast

**What you'll learn:**
- `__shfl_down_sync` for reductions
- `__shfl_sync` for arbitrary exchanges
- `__shfl_up_sync` for prefix sum
- Row-wise softmax without shared memory

**Benefits:** No shared memory, lower latency, no bank conflicts

---

### 5. Flash Attention V2 (`flash_attention_v2.cu`)
**Difficulty:** ⭐⭐⭐ Expert  
**Architecture:** Ampere+ (sm_80+)  
**Key Concepts:** Online softmax, tiling, all techniques combined

**What you'll learn:**
- Complete Flash Attention 2 implementation
- Online softmax algorithm (O(N) memory)
- Tile-based computation
- Combining tensor cores + async copy + shuffles
- Causal masking support

**Expected speedup:** 2-4x over standard attention

---

### 6. Type Conversions (`type_conversions.cu`)
**Difficulty:** ⭐ Beginner  
**Architecture:** All (FP8 requires Hopper sm_90+)  
**Key Concepts:** BF16, FP16, FP8, vectorized conversions

**What you'll learn:**
- Vectorized BF16 ↔ Float conversions
- FP16 ↔ Float conversions
- FP8 support (Hopper)
- Precision analysis and performance comparison

**Performance:** ~4x throughput with vectorized operations

---

### 7. Vector and Broadcast Ops (`vector_broadcast_ops.cu`)
**Difficulty:** ⭐⭐ Intermediate  
**Architecture:** All (sm_70+)  
**Key Concepts:** Broadcasting, layer norm, RMS norm

**What you'll learn:**
- Row-wise and column-wise broadcast
- Layer normalization implementation
- RMS normalization
- Fused operations for efficiency

**Speedup:** 1.5-2x with fusion

---

### 8. Efficient Transpose (`transpose_optimized.cu`)
**Difficulty:** ⭐⭐ Intermediate  
**Architecture:** All (sm_70+)  
**Key Concepts:** Shared memory, bank conflicts, padding

**What you'll learn:**
- Efficient transpose with shared memory
- Bank conflict avoidance with +1 padding
- In-place transpose techniques
- Row-major ↔ Column-major conversion

**Expected speedup:** 3-10x over naive transpose

---

## Quick Start

### Prerequisites

1. CUDA Toolkit 12.0+
2. NVIDIA GPU (Compute Capability 7.0+, 8.0+ for most)
3. nvcc compiler available

```bash
# Check CUDA
nvcc --version
nvidia-smi

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Build All Examples

```bash
# Auto-detect architecture
make all

# Or specify architecture
make all CUDA_ARCH=sm_89    # RTX 4090 (Ada)
make all CUDA_ARCH=sm_80    # A100 (Ampere)
make all CUDA_ARCH=sm_90    # H100 (Hopper)
```

### Compute Capability Guide

- **sm_70**: Volta (V100)
- **sm_75**: Turing (RTX 20xx, T4)
- **sm_80**: Ampere (A100, RTX 30xx)
- **sm_86**: Ampere (RTX 30xx mobile)
- **sm_89**: Ada (RTX 40xx)
- **sm_90**: Hopper (H100)

---

## Learning Path

**Beginner:**
1. Start with `06_type_conversions` - understand data types
2. Move to `04_warp_shuffle` - learn warp primitives
3. Study `08_transpose` - master shared memory

**Intermediate:**
4. `02_shared_memory_swizzling` - eliminate bank conflicts
5. `07_vector_ops` - understand broadcast patterns

**Advanced:**
6. `01_tensor_cores` - maximize compute throughput
7. `03_async_copy_pipeline` - hide memory latency
8. `05_flash_attention` - combine all techniques

---

## Key NVIDIA GPU Concepts

### Warp Execution (32 threads)
```cuda
constexpr int WARP_SIZE = 32;
__device__ int laneid() { return threadIdx.x & 31; }
__device__ int warpid() { return threadIdx.x >> 5; }
```

### Memory Hierarchy
- **Shared Memory**: 48-164 KB per SM, 32 banks
- **L1 Cache**: Combined with shared memory
- **L2 Cache**: Shared across SMs
- **HBM**: Up to 3.35 TB/s (H100)

### Tensor Cores
- **Ampere**: m16n8k16 (BF16, TF32)
- **Ada**: m16n8k16 (BF16, FP8)
- **Hopper**: m16n8k16 + wgmma (FP8, FP16, BF16)

---

## Profiling Your Kernels

### Nsight Compute

**Basic profile:**
```bash
ncu --set full ./your_kernel
```

**Specific metrics:**
```bash
# Tensor core utilization
ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed ./kernel

# Bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./kernel

# Memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./kernel

# Warp execution efficiency
ncu --metrics smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio ./kernel
```

### Key Metrics

- **Tensor Core Utilization**: Target >80%
- **Shared Memory Bank Conflicts**: Target 0
- **Memory Bandwidth**: Target >70% of peak
- **Occupancy**: Balance with resource usage

---

## Common Optimization Patterns

### 1. Tensor Core Fragment Loading
```cuda
// Load 16x16 tile for tensor cores
__nv_bfloat162 frag_a[2];
__nv_bfloat162 frag_b[2];
// ... load data into fragments ...
hmma16816_bf16(frag_d, frag_a, frag_b, frag_c);
```

### 2. Async Copy with Pipeline
```cuda
cp_async_commit_group();
// Issue multiple loads...
cp_async_wait_group(N-1);  // Wait for all but last N
__syncthreads();
```

### 3. Warp Shuffle Reduction
```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
```

### 4. Swizzled Shared Memory
```cuda
int swizzled_idx = row ^ ((col >> 2) & 0x7);
tile[swizzled_idx * COLS + col] = value;
```

---

## Performance Expectations

### RTX 4090 (Ada Lovelace)
- Peak FP32: 83 TFLOPS
- Peak BF16 Tensor: 166 TFLOPS (with sparsity: 330 TFLOPS)
- Memory Bandwidth: 1008 GB/s

| Example | Expected Speedup |
|---------|-----------------|
| Tensor Cores | 10-20x |
| Swizzling | 2-4x |
| Async Copy | 1.5-3x |
| Combined | 50-100x |

### A100 (Ampere)
- Peak FP32: 19.5 TFLOPS
- Peak BF16 Tensor: 312 TFLOPS
- Memory Bandwidth: 1555 GB/s (40GB), 2039 GB/s (80GB)

### H100 (Hopper)
- Peak FP32: 67 TFLOPS
- Peak FP8 Tensor: 989 TFLOPS
- Memory Bandwidth: 3.35 TB/s

---

## Troubleshooting

### Compilation Issues

**Problem:** PTX instruction not supported
```
error: PTX .version directive must be at least 7.0 for sm_80
```
**Solution:** Update CUDA Toolkit or specify correct architecture

**Problem:** Undefined tensor core instruction
```
error: undefined instruction 'mma.sync'
```
**Solution:** Requires sm_70+ (Volta or newer)

### Runtime Issues

**Problem:** Low tensor core utilization
**Solution:**
1. Check tile sizes (must be multiples of 16)
2. Ensure proper fragment loading
3. Profile with Nsight Compute

**Problem:** Bank conflicts
**Solution:**
1. Apply swizzling or padding
2. Profile conflicts with ncu
3. Verify access patterns

---

## Tips for Best Results

1. **Profile first** - Use Nsight Compute to identify bottlenecks
2. **Start simple** - Run examples as-is before modifying
3. **Experiment** - Change tile sizes, data types, pipeline depth
4. **Combine techniques** - Maximum speedup from using all optimizations
5. **Validate** - Always check correctness after optimization

---

## Related Knowledge Base

For deeper understanding, see:
- `knowledge-base/nvidia-knowledge-base/layer-2-compute-stack/cuda/cuda-advanced-optimization.md`
- `knowledge-base/nvidia-knowledge-base/layer-2-compute-stack/cuda/cuda-basics.md`

---

## Additional Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Tensor Core Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

---

**Last Updated:** 2024  
**Target CUDA:** 12.0+  
**Hardware:** Ampere/Ada/Hopper (sm_80+)

