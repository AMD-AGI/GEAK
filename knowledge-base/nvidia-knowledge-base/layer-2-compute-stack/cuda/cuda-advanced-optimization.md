---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Native CUDA Advanced Optimization Techniques

This guide covers advanced GPU optimization techniques using native CUDA for achieving peak performance on NVIDIA GPUs.

## Table of Contents
1. [Tensor Core Operations](#1-tensor-core-operations)
2. [Shared Memory Swizzling](#2-shared-memory-swizzling)
3. [Asynchronous Copy and Pipelining](#3-asynchronous-copy-and-pipelining)
4. [Warp Shuffle Operations](#4-warp-shuffle-operations)
5. [Flash Attention 2](#5-flash-attention-2)
6. [Type Conversions](#6-type-conversions)
7. [Vector and Broadcast Operations](#7-vector-and-broadcast-operations)
8. [Efficient Transpose](#8-efficient-transpose)

---

## 1. Tensor Core Operations

### Warp-Level MMA (Ampere, Ada, Hopper)

Tensor cores perform matrix multiply-accumulate at warp scope using the `mma.sync` PTX instruction.

**Native CUDA Implementation (BF16):**

```cuda
__device__ void hmma16816_bf16(
    __nv_bfloat162* d,  // 4 registers output
    __nv_bfloat162* a,  // 2 registers input A  
    __nv_bfloat162* b,  // 2 registers input B
    float* c            // 4 registers accumulator
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};\n"
        : "=f"(d[0].x), "=f"(d[0].y), "=f"(d[1].x), "=f"(d[1].y)
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}
```

### Register Fragment Layout

For m16n8k16 operation:
- Each thread in warp holds 2 BF16 pairs for A (4 elements)
- Each thread holds 2 BF16 pairs for B (4 elements)
- Each thread accumulates 4 FP32 values for C

### TF32 Tensor Cores

```cuda
__device__ void hmma16816_tf32(
    float* d,    // 4 registers output
    float* a,    // 4 registers input A
    float* b,    // 2 registers input B  
    float* c     // 4 registers accumulator
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}
```

---

## 2. Shared Memory Swizzling

### Bank Conflict Basics

NVIDIA GPUs have 32 shared memory banks (4 bytes each). Conflict occurs when multiple threads in a warp access different addresses in the same bank.

### XOR Swizzling Pattern

```cuda
__device__ int swizzle_address(int row, int col, int stride) {
    // XOR bits to distribute across banks
    int swizzled_row = row ^ ((col >> 2) & 0x7);
    return swizzled_row * stride + col;
}
```

### Padding Trick

Simple alternative to swizzling:

```cuda
// Add +1 to column dimension
__shared__ float tile[ROWS][COLS + 1];  // Padding avoids conflicts
```

### Measuring Bank Conflicts

Use Nsight Compute:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./kernel
```

---

## 3. Asynchronous Copy and Pipelining

### cp.async Instructions (Ampere+)

```cuda
__device__ void async_copy_gmem_to_smem(
    void* smem_ptr,
    const void* gmem_ptr,
    size_t size
) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"((unsigned)__cvta_generic_to_shared(smem_ptr)),
           "l"(gmem_ptr),
           "n"(size)
    );
}
```

### Commit and Wait

```cuda
__device__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ void cp_async_wait_group(int n) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}
```

### Multi-Stage Pipeline

```cuda
__global__ void pipelined_kernel() {
    __shared__ float smem[STAGES][TILE_SIZE];
    
    // Stage 0: Initiate first load
    async_copy_gmem_to_smem(&smem[0], &gmem[0], size);
    cp_async_commit_group();
    
    for (int i = 1; i < num_tiles; ++i) {
        int stage = i % STAGES;
        int prev_stage = (i - 1) % STAGES;
        
        // Start next load
        async_copy_gmem_to_smem(&smem[stage], &gmem[i * size], size);
        cp_async_commit_group();
        
        // Wait for previous stage
        cp_async_wait_group(STAGES - 2);
        __syncthreads();
        
        // Compute on previous stage
        compute(smem[prev_stage]);
        __syncthreads();
    }
}
```

---

## 4. Warp Shuffle Operations

### Basic Shuffle Operations

```cuda
// Broadcast from lane 0
__device__ float warp_broadcast(float val) {
    return __shfl_sync(0xffffffff, val, 0);
}

// Exchange with neighbor
__device__ float warp_exchange(float val, int offset) {
    return __shfl_xor_sync(0xffffffff, val, offset);
}

// Get from specific lane
__device__ float warp_get(float val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}
```

### Warp Reductions

```cuda
// Sum reduction across warp
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Max reduction across warp
__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
```

### Prefix Sum with Shuffles

```cuda
__device__ float warp_prefix_sum(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float temp = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x >= offset) val += temp;
    }
    return val;
}
```

---

## 5. Flash Attention 2

### Online Softmax Pattern

```cuda
__device__ void online_softmax_attention(
    float* output,       // [Q_SIZE, D]
    const float* query,  // [Q_SIZE, D]
    const float* key,    // [KV_SIZE, D]
    const float* value,  // [KV_SIZE, D]
    int qsize, int kvsize, int d
) {
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    for (int q = 0; q < qsize; ++q) {
        output[q] = 0.0f;
    }
    
    // Process KV in chunks
    for (int kv_chunk = 0; kv_chunk < kvsize; kv_chunk += CHUNK_SIZE) {
        // Compute scores for this chunk
        float scores[CHUNK_SIZE];
        // ... compute Q @ K^T ...
        
        // Update max
        float old_max = max_score;
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            max_score = fmaxf(max_score, scores[i]);
        }
        
        // Rescale previous accumulation
        float scale = expf(old_max - max_score);
        for (int q = 0; q < qsize; ++q) {
            output[q] *= scale;
        }
        sum_exp *= scale;
        
        // Add new contribution
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            float exp_score = expf(scores[i] - max_score);
            sum_exp += exp_score;
            // output += exp_score * value[...]
        }
    }
    
    // Final normalization
    for (int q = 0; q < qsize; ++q) {
        output[q] /= sum_exp;
    }
}
```

---

## 6. Type Conversions

### BF16 ↔ FP32

```cuda
__device__ float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

// Vectorized conversion
__device__ float2 bf162_to_float2(__nv_bfloat162 val) {
    return __bfloat1622float2(val);
}
```

### FP16 ↔ FP32

```cuda
__device__ float half_to_float(__half val) {
    return __half2float(val);
}

__device__ __half float_to_half(float val) {
    return __float2half(val);
}
```

### FP8 Support (Hopper)

```cuda
__device__ float fp8_to_float(__nv_fp8_e4m3 val) {
    return float(val);  // Implicit conversion
}

__device__ __nv_fp8_e4m3 float_to_fp8(float val) {
    return __nv_fp8_e4m3(val);
}
```

---

## 7. Vector and Broadcast Operations

### Row/Column Broadcast

```cuda
// Broadcast row to all columns
__device__ void broadcast_row(
    float* output,       // [ROWS, COLS]
    const float* row,    // [COLS]
    int rows, int cols
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int r = tid / cols;
    int c = tid % cols;
    
    if (r < rows && c < cols) {
        output[r * cols + c] = row[c];
    }
}
```

### Layer Normalization

```cuda
__device__ void layer_norm(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    int size
) {
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    float mean = sum / size;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / size;
    float inv_std = rsqrtf(variance + 1e-5f);
    
    // Normalize
    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
```

---

## 8. Efficient Transpose

### Shared Memory Transpose with Padding

```cuda
__global__ void transpose_smem(
    float* output,
    const float* input,
    int rows, int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load to shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    
    // Write transposed
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

---

## Performance Tips

### Achieving Peak Tensor Core Utilization

1. **Keep tensor cores busy**: Use m16n8k16 or larger tiles
2. **Pipeline memory and compute**: Use async copy
3. **Minimize register pressure**: Balance tile sizes
4. **Avoid bank conflicts**: Use swizzling or padding

### Expected Performance

On RTX 4090 (Ada):
- Tensor cores: ~83 TFLOPS FP32, ~166 TFLOPS BF16
- Memory bandwidth: ~1 TB/s
- Achievable: 70-90% of peak with proper optimization

### Profiling Metrics

```bash
# Tensor core utilization
ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed

# Bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum

# Memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed
```

---

## Summary

Key techniques for NVIDIA GPU optimization:
- **Tensor cores** for 10-20x compute speedup
- **Shared memory swizzling** to eliminate bank conflicts
- **Async copy** for 1.5-3x latency hiding
- **Warp shuffles** for fast reductions
- **Combined** for 50-100x total speedup

By applying these native CUDA optimization techniques, you can achieve performance comparable to highly-optimized libraries like cuBLAS and cuDNN.

