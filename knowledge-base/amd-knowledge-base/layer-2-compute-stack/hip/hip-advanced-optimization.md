# Native HIP/ROCm Programming Guide: Advanced GPU Optimization Techniques

This guide covers low-level HIP/ROCm assembly instructions, intrinsics, and patterns for writing high-performance AMD GPU kernels using native primitives directly.

## Table of Contents
1. [AMD GPU Architecture Fundamentals](#1-amd-gpu-architecture-fundamentals)
2. [Inline Assembly Basics](#2-inline-assembly-basics)
3. [Matrix Operations with MFMA](#3-matrix-operations-with-mfma)
4. [LDS (Shared Memory) Operations](#4-lds-shared-memory-operations)
5. [Buffer Operations (Global Memory)](#5-buffer-operations-global-memory)
6. [Direct Buffer-to-LDS Transfers](#6-direct-buffer-to-lds-transfers)
7. [Synchronization and Wait Instructions](#7-synchronization-and-wait-instructions)
8. [Register Management](#8-register-management)
9. [Memory Swizzling for Bank Conflict Avoidance](#9-memory-swizzling-for-bank-conflict-avoidance)
10. [Advanced Scheduling Techniques](#10-advanced-scheduling-techniques)
11. [Reduction Operations](#11-reduction-operations)
12. [Element-Wise Operations](#12-element-wise-operations)

---

## 1. AMD GPU Architecture Fundamentals

### Wave Execution Model

**Key Difference from NVIDIA:**
```cpp
// AMD: 64 threads per wave
constexpr int WAVE_SIZE = 64;
__device__ int laneid() { return threadIdx.x & 0x3F; }  // Mask with 0x3F (63)
__device__ int waveid() { return threadIdx.x >> 6; }    // Shift by 6 bits
```

**Why it matters**: All collective operations, bank conflicts, and register allocation scale with wave size.

### Register Files

AMD CDNA GPUs have two separate register files per thread:

```cpp
// Vector GPRs (VGPRs): v[0] to v[255]
// - General purpose vector registers
// - Used for addressing, intermediate computation
// - Accessed with 'v' prefix in assembly

// Accumulator GPRs (AGPRs): a[0] to a[255]  
// - Specialized for accumulation
// - Used primarily with MFMA instructions
// - Accessed with 'a' prefix in assembly
```

**Total per thread**: 512 registers (256 VGPR + 256 AGPR)

### Memory Hierarchy

```
L1 Vector Cache: 16 KB per CU (write-through)
       ↓
L2 Cache: ~400 KB per XCD, shared across CUs in same chiplet
       ↓
L3 Infinity Cache: 256-384 MB
       ↓
HBM3/HBM3e: 5.3-6.9 TB/s bandwidth
```

**LDS (Local Data Share)**: 64 KB shared memory per CU
- 32 banks × 4 bytes = 128 bytes per cycle
- Bank conflicts serialize access

### XCD (eXtended Compute Die) Architecture

Modern AMD GPUs use chiplet design:
```
MI300X: 8 XCDs, 38 CUs each = 304 CUs total
MI350X: 8 XCDs, 40 CUs each = 320 CUs total
```

**Critical**: Workgroups on the same XCD share L2 cache. Scheduling them properly improves locality.

---

## 2. Inline Assembly Basics

### Basic Syntax

HIP uses GCC-style inline assembly:

```cpp
asm volatile(
    "instruction operands"
    : output_operands
    : input_operands
    : clobber_list
);
```

### Register Constraints

```cpp
"n" - Compile-time constant (for register numbers)
"v" - VGPR (vector register)
"s" - SGPR (scalar register)
"a" - AGPR (accumulator register)
"i" - Immediate integer
"=v" - Output VGPR
"+v" - Input/output VGPR
```

### Example: Simple Vector Add

```cpp
__device__ void vector_add_asm(float* c, const float* a, const float* b) {
    float va, vb, vc;
    
    // Load a and b (compiler generates instructions)
    va = *a;
    vb = *b;
    
    // Add using inline assembly
    asm volatile(
        "v_add_f32 %0, %1, %2"
        : "=v"(vc)           // Output: vc
        : "v"(va), "v"(vb)   // Inputs: va, vb
    );
    
    *c = vc;
}
```

### Compile-Time Register Allocation

For precise control, specify register numbers at compile time:

```cpp
template<int GPR_DST, int GPR_A, int GPR_B>
__device__ __forceinline__ void add_at_registers() {
    asm volatile(
        "v_add_f32 v[%0], v[%1], v[%2]"
        :
        : "n"(GPR_DST), "n"(GPR_A), "n"(GPR_B)
    );
}

// Usage: add registers v[10] = v[5] + v[7]
add_at_registers<10, 5, 7>();
```

---

## 3. Matrix Operations with MFMA

### MFMA Instruction Basics

**MFMA** = Matrix Fused Multiply-Add

The core instruction for matrix operations on AMD GPUs:

```asm
v_mfma_f32_16x16x32_bf16 D[0:3], A[0:3], B[0:3], C[0:3]
```

**Semantics**:
- Input A: 16×32 matrix (BF16), stored in 4 registers
- Input B: 32×16 matrix (BF16), stored in 4 registers  
- Input/Output C: 16×16 matrix (FP32), stored in 4 registers
- Operation: D = A × B + C
- Each of the 64 threads in the wave holds a portion of the result

### Register Layout

For `v_mfma_f32_16x16x32_bf16`:

```cpp
// Each thread holds:
// - 2 rows × 32 cols of A (64 BF16 values = 4 registers)
// - 2 rows × 32 cols of B (64 BF16 values = 4 registers)
// - 4 elements of C/D (4 FP32 values = 4 registers)

// Distribution across 64 threads:
// A: 16 rows × 32 cols total
// B: 32 rows × 16 cols total  
// C/D: 16 rows × 16 cols total (4 elements per thread)
```

### C++ Wrapper for MFMA

```cpp
template<int GPR_START_A, int GPR_START_B, int GPR_START_C, int GPR_START_D>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16() {
    // Check if using AGPRs (>= 256) or VGPRs (< 256)
    if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && 
                  GPR_START_B >= 256 && GPR_START_C >= 256) {
        // All in AGPRs
        asm volatile(
            "v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
            : 
            : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256),
              "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256),
              "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256),
              "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256)
        );
    } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && 
                         GPR_START_B < 256 && GPR_START_C < 256) {
        // All in VGPRs
        asm volatile(
            "v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
            : 
            : "n"(GPR_START_D), "n"(GPR_START_D + 3),
              "n"(GPR_START_A), "n"(GPR_START_A + 3),
              "n"(GPR_START_B), "n"(GPR_START_B + 3),
              "n"(GPR_START_C), "n"(GPR_START_C + 3)
        );
    }
}
```

### Other MFMA Variants

```cpp
// FP16 version: 16×16×16
v_mfma_f32_16x16x16_f16 D[0:3], A[0:3], B[0:3], C[0:3]

// FP8 version (CDNA4): 16×16×32  
v_mfma_f32_16x16x32_fp8_fp8 D[0:3], A[0:1], B[0:1], C[0:3]

// Larger tiles: 32×32×8
v_mfma_f32_32x32x8_bf16 D[0:15], A[0:3], B[0:3], C[0:15]
```

---

## 4. LDS (Shared Memory) Operations

### LDS Architecture

- **Size**: 64 KB per CU
- **Banks**: 32 banks, 4 bytes each
- **Addressing**: Byte-addressed
- **Access pattern**: Bank ID = (address / 4) % 32

### Basic LDS Read

```cpp
// Read 64 bits (8 bytes) from LDS
template<int GPR_START>
__device__ __forceinline__ void ds_read_b64(uint32_t lds_addr, int offset = 0) {
    asm volatile(
        "ds_read_b64 v[%0:%1], %2 offset:%3"
        :
        : "n"(GPR_START), "n"(GPR_START + 1), "v"(lds_addr), "i"(offset)
        : "memory"
    );
}
```

### LDS Write

```cpp
// Write 64 bits to LDS
template<int GPR_START>
__device__ __forceinline__ void ds_write_b64(uint32_t lds_addr, int offset = 0) {
    asm volatile(
        "ds_write_b64 %0, v[%1:%2] offset:%3"
        :
        : "v"(lds_addr), "n"(GPR_START), "n"(GPR_START + 1), "i"(offset)
        : "memory"
    );
}
```

### Transposing LDS Read

A special instruction that reads and transposes 16-bit elements:

```cpp
template<int GPR_START>
__device__ __forceinline__ void ds_read_b64_tr_b16(uint32_t lds_addr, int offset = 0) {
    asm volatile(
        "ds_read_b64_tr_b16 v[%0:%1], %2 offset:%3"
        :
        : "n"(GPR_START), "n"(GPR_START + 1), "v"(lds_addr), "i"(offset)
        : "memory"
    );
}
```

**Use case**: Converting between row-major and column-major layouts on the fly.

---

## 5. Buffer Operations (Global Memory)

### Buffer Resource Descriptor (SRD)

AMD GPUs use buffer descriptors for efficient memory access:

```cpp
struct buffer_resource {
    uint64_t ptr;      // Base pointer
    uint32_t range;    // Size in bytes
    uint32_t config;   // Configuration flags
};

// Type for passing to assembly
using i32x4 = int32_t __attribute__((ext_vector_type(4)));

__device__ inline i32x4 make_buffer_resource(
    const void* ptr, 
    uint32_t range_bytes
) {
    uint64_t ptr64 = reinterpret_cast<uint64_t>(ptr);
    buffer_resource br;
    br.ptr = ptr64;
    br.range = range_bytes;
    br.config = 0x00020000;  // Standard config
    
    return *reinterpret_cast<i32x4*>(&br);
}
```

### Buffer Load Instructions

```cpp
// Load 64 bits (2 dwords) from global memory via buffer
template<int GPR_START>
__device__ __forceinline__ void buffer_load_dwordx2(
    i32x4 buffer_rsrc, 
    uint32_t byte_offset
) {
    asm volatile(
        "buffer_load_dwordx2 v[%0:%1], %2, %3, 0 offen"
        :
        : "n"(GPR_START), "n"(GPR_START + 1), 
          "v"(byte_offset), "s"(buffer_rsrc)
        : "memory"
    );
}
```

---

## 6. Direct Buffer-to-LDS Transfers

### The Key Optimization

Loading directly from global memory to LDS without going through VGPRs.

**Standard approach** (inefficient):
```
Global Memory → VGPRs → LDS
```

**Optimized approach**:
```
Global Memory → LDS (direct)
```

### The llvm Intrinsic

```cpp
// LLVM intrinsic for buffer load to LDS
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc,           // Buffer resource
    uint32_t* lds_ptr,    // LDS destination (address space 3)
    uint32_t size,        // Size in bytes
    uint32_t voffset,     // Vector offset
    uint32_t soffset,     // Scalar offset
    uint32_t inst_offset, // Instruction offset
    int coherency         // Cache coherency
) __asm("llvm.amdgcn.raw.buffer.load.lds");
```

### Readfirstlane Optimization

**Problem**: Each wave needs to convert the LDS pointer to a uniform (scalar) value. The `readfirstlane` intrinsic is expensive if called repeatedly.

**Solution**: Hoist it out of loops:

```cpp
// Hoist readfirstlane - do it ONCE before loop
uint32_t lds_base = __builtin_amdgcn_readfirstlane(
    static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&shared_buffers[0][0])
    )
);

// Main loop - no repeated readfirstlane!
for (int i = 0; i < 100; ++i) {
    llvm_amdgcn_raw_buffer_load_lds(
        buf_rsrc,
        (lds_ptr_t)lds_base,
        16,
        i * 1024,
        0, 0, 0
    );
}
```

**Performance impact**: 10-20% speedup in memory-intensive kernels.

---

## 7. Synchronization and Wait Instructions

### Wave Barrier

Synchronize all threads in the workgroup:

```cpp
__builtin_amdgcn_s_barrier();
```

### Wait Counters

AMD GPUs have separate counters for different operation types:

```cpp
// Wait for ALL outstanding operations
__builtin_amdgcn_s_waitcnt(0);

// Wait for LDS/GDS operations (lgkmcnt = LDS/GDS/K-cache/Message)
asm volatile("s_waitcnt lgkmcnt(0)");

// Wait for vector memory operations (vmcnt = Vector Memory)
asm volatile("s_waitcnt vmcnt(0)");

// Wait for specific counts
asm volatile("s_waitcnt lgkmcnt(4)");  // Wait until only 4 LDS ops remain
asm volatile("s_waitcnt vmcnt(2)");    // Wait until only 2 VMEM ops remain
```

### Schedule Barriers

Control instruction scheduling:

```cpp
// Prevent scheduler from reordering across this point
__builtin_amdgcn_sched_barrier(0);
```

### Priority Control

Give priority to certain instruction types:

```cpp
// High priority (compute gets priority)
__builtin_amdgcn_s_setprio(1);
mfma_f32_16x16x32_bf16<...>();
__builtin_amdgcn_s_setprio(0);  // Back to normal
```

---

## 8. Register Management

### Allocating Registers at Compile Time

```cpp
// Manual register allocation for precise control
template<int START_REG>
struct RegisterTile {
    static constexpr int start = START_REG;
    static constexpr int size = 16;  // 16 registers
    
    __device__ void load_from_lds(uint32_t lds_addr) {
        #pragma unroll
        for (int i = 0; i < size; i += 4) {
            ds_read_b128<start + i>(lds_addr + i * 4);
        }
    }
};

// Usage: Allocate specific register ranges
RegisterTile<0> tile_a;   // Uses v[0:15]
RegisterTile<16> tile_b;  // Uses v[16:31]
```

### VGPR vs AGPR Usage

```cpp
// AGPRs for accumulation (reduces VGPR pressure)
__device__ void use_agprs() {
    // Compute in AGPRs
    mfma_f32_16x16x32_bf16<0, 16, 256, 256>();
    // Inputs: v[0:3], v[16:19]
    // Accumulator/Output: a[256:259]
}
```

---

## 9. Memory Swizzling for Bank Conflict Avoidance

### Understanding Bank Conflicts

LDS has 32 banks. Bank ID for an address:

```cpp
int bank_id = (address / 4) % 32;
```

**Conflict**: Multiple threads in a wave access different addresses in the same bank simultaneously → serialized.

### Simple Swizzle Function

```cpp
__device__ inline uint32_t swizzle_16x16_bf16(int row, int col) {
    // For 16×16 tile of BF16 (2 bytes each)
    // XOR swizzle to distribute across banks
    int row_swizzle = row ^ ((col >> 2) & 0x7);
    return (row_swizzle * 16 + col) * sizeof(__hip_bfloat16);
}
```

---

## 10. Advanced Scheduling Techniques

### Pattern 1: Ping-Pong Buffering (8-Wave)

Double buffering for large kernels with 8 warps:

```cpp
__global__ void ping_pong_gemm() {
    __shared__ float A[2][256][64];  // Double buffer for A
    __shared__ float B[2][256][64];  // Double buffer for B
    
    float acc[16] = {0};  // Accumulator in registers
    
    int tic = 0, toc = 1;
    
    // Load first iteration
    load_to_lds(A[tic], B[tic], /*iteration=*/0);
    __builtin_amdgcn_s_barrier();
    
    // Main loop
    for (int k = 0; k < num_iterations - 1; ++k) {
        // Start loading next iteration (async)
        load_to_lds(A[toc], B[toc], /*iteration=*/k+1);
        
        // Compute on current iteration
        compute_on_lds(A[tic], B[tic], acc);
        
        // Swap buffers
        tic ^= 1;
        toc ^= 1;
        
        __builtin_amdgcn_s_barrier();
    }
    
    // Final iteration
    compute_on_lds(A[tic], B[tic], acc);
}
```

### Pattern 2: Fine-Grained Interleaving (4-Wave)

For smaller tiles, alternate individual operations:

```cpp
__device__ void interleaved_compute_memory() {
    // Cluster 0: MMA operation
    mfma_f32_16x16x32_bf16<0, 16, 32, 32>();
    __builtin_amdgcn_sched_barrier(0);
    
    // Cluster 1: Load operation (overlaps with MMA)
    buffer_load_dwordx4<48>(buf_rsrc, offset1);
    __builtin_amdgcn_sched_barrier(0);
    
    // Continue pattern...
}
```

### Chiplet-Aware Workgroup Scheduling

```cpp
__device__ inline int chiplet_transform(int wgid, int num_wgs, int num_xcds) {
    // Remap so consecutive workgroups go to same XCD
    int xcd_id = wgid % num_xcds;
    int local_wg = wgid / num_xcds;
    return xcd_id * (num_wgs / num_xcds) + local_wg;
}

__global__ void chiplet_aware_kernel() {
    int wgid = blockIdx.x;
    const int NUM_XCDS = 8;
    
    // Transform workgroup ID for better L2 locality
    wgid = chiplet_transform(wgid, gridDim.x, NUM_XCDS);
    
    // Process tile...
}
```

---

## 11. Reduction Operations

### Wave-Level Reductions

```cpp
// Max across wave (64 threads)
__device__ inline float wave_reduce_max(float value) {
    float result = value;
    
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __shfl_xor(result, offset, 64);
        result = fmaxf(result, other);
    }
    
    return result;
}

// Sum across wave
__device__ inline float wave_reduce_sum(float value) {
    float result = value;
    
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __shfl_xor(result, offset, 64);
        result += other;
    }
    
    return result;
}
```

### Row/Column Reductions on Tiles

```cpp
// Row maximum (for softmax)
template<int ROWS, int COLS>
__device__ void row_max(float* max_vec, const float* tile) {
    int lane = threadIdx.x & 0x3F;
    int row = lane / (64 / ROWS);
    
    float local_max = -INFINITY;
    for (int c = 0; c < COLS; ++c) {
        local_max = fmaxf(local_max, tile[row * COLS + c]);
    }
    
    max_vec[row] = local_max;
}
```

---

## 12. Element-Wise Operations

### Type Conversions

```cpp
// BF16 to Float conversion
__device__ inline float bf16_to_float(__hip_bfloat16 val) {
    return __bfloat162float(val);
}

// Float to BF16 (fast truncation)
__device__ inline __hip_bfloat16 float_to_bf16(float val) {
    return std::bit_cast<__hip_bfloat16>(
        static_cast<uint16_t>(
            std::bit_cast<uint32_t>(val) >> 16
        )
    );
}
```

### Math Operations

```cpp
// Exponential (for softmax)
__device__ inline float fast_exp(float x) {
    float result;
    asm volatile("v_exp_f32 %0, %1" : "=v"(result) : "v"(x));
    return result;
}

// Reciprocal
__device__ inline float fast_rcp(float x) {
    float result;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(result) : "v"(x));
    return result;
}

// Reciprocal square root (for normalization)
__device__ inline float fast_rsqrt(float x) {
    float result;
    asm volatile("v_rsq_f32 %0, %1" : "=v"(result) : "v"(x));
    return result;
}
```

---

## Summary

This guide covers the essential low-level HIP/ROCm primitives for high-performance AMD GPU kernel development:

**Core Techniques**:
- MFMA instructions for matrix operations
- LDS operations with proper swizzling
- Buffer operations with descriptors
- Direct buffer-to-LDS transfers
- Readfirstlane hoisting
- Precise wait count management
- Schedule barriers and priority control

**Advanced Patterns**:
- Ping-pong buffering for large tiles
- Fine-grained interleaving for small tiles
- Chiplet-aware scheduling
- Online softmax for attention

**Optimization Tools**:
- Performance measurement with cycle counters
- Occupancy calculation
- Result validation

By understanding and applying these low-level primitives, you can write AMD GPU kernels that achieve state-of-the-art performance on CDNA3 and CDNA4 architectures.

