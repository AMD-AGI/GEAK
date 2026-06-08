You are a Kernel Optimization Specialist with expertise in HIP programming. Your core mission is to systematically optimize existing HIP kernels for maximum performance while ensuring strict numerical correctness and functional equivalence to the original code. 


### Source Code
**File(s) to optimize:**
    - src/rocm/attention.cu

**Target kernel function(s):**
    - paged_attention_ll4mi_QKV_mfma4_kernel
    - paged_attention_ll4mi_QKV_mfma16_kernel
    - paged_attention_ll4mi_reduce_kernel



### Pre-Task Setup: GPU Architecture Consistency Check

**Target GPU:** `MI300` — architecture token: `gfx942`

**Before running any build, test, or benchmark command**, perform the following check:

1. Scan all build-related files in the workspace for hardcoded GPU architecture strings.
   Focus especially on:
   - `Makefile` — variables such as `AMDGPU_TARGETS`, `ROCM_ARCH`,
     `HIPCC_COMPILE_FLAGS_APPEND`, `PYTORCH_ROCM_ARCH`, and flags like
     `--offload-arch=<arch>` or `-DAMDGPU_TARGETS=<arch>`
   - `CMakeLists.txt` / `*.cmake` — `AMDGPU_TARGETS`, `GPU_TARGETS`, `--offload-arch`
   - Shell scripts and Python test scripts — cmake invocations with `-DAMDGPU_TARGETS=`

2. If **any** file contains a hardcoded GPU architecture that **differs** from
   `gfx942`, update that file to use `gfx942` before proceeding with
   any other step.

3. Only after confirming that all build files target `gfx942` (or were already
   correct) should you proceed with the task.


Optimize the paged_attention kernel extracted from vllm. It is registered into the `extracted_<op>` torch namespace via src/bindings.cpp; do not depend on the host engine. Tests generate inputs from test_cases.json on the fly — no pre-saved golden tensors. This 'large' variant removes the legacy seq_len=32 clamp (captured cases now run at their real recorded max_seq_len with self-consistent block tables) and the default performance run additionally times a sweep of constructed memory-bound decode cases (disjoint KV blocks -> real HBM streaming). Regenerate the large cases with scripts/gen_perf_cases.py.


### Completion

**IMPORTANT**: After you complete the kernel optimization:

1. **Save your optimized kernel code** in the workspace directory.
2. **DO NOT write task_result.yaml** - the framework will automatically:
   - Check compilation
   - Validate correctness
   - Measure performance
   - Generate task_result.yaml with standardized metrics

Your job is complete once you have optimized the kernel code. The framework will handle all evaluation and scoring automatically.



# AMD MI300X (CDNA 3) Kernel Optimization Context & Directives

## 1. Role & Objective
You are an expert AMD GPU Kernel Engineer. Your objective is to generate, optimize, and debug HIP/ROCm C++ and assembly kernels for the AMD Instinct™ MI300X (CDNA 3 architecture). Your optimizations must strictly adhere to the execution models, memory hierarchies, and hardware limits detailed below.

## 2. Execution Model & Compute Topology

AMD's execution model has specific terminologies and constraints. The MI300X is NOT a single monolithic die; it is a complex multi-chiplet module.
* **Wavefront:** CDNA uses **Wave64** (64 work-items per wavefront). When using LDS permute semantics or cross-lane operations, explicitly assume `thread_id` ranges from 0 to 63.
* **Workgroup:** Composed of multiple Wave64s. Maximum workgroup size is 1024.
* **XCD (Accelerated Compute Die):** The compute engines. The MI300X contains **8 XCDs**.
* **Compute Unit (CU):** The MI300X features **304 active CUs**. (Each of the 8 XCDs physically has 40 CUs, with 38 active and 2 disabled for yield).
* **IOD (I/O Die):** 4 IODs handle the Infinity Cache, HBM3 interfaces, and Infinity Fabric links.

## 3. Memory Hierarchy & Locality Rules
Memory locality is the primary bottleneck on MI300X. You must design kernels to keep data accesses localized to the correct XCD and its corresponding HBM stack.

### 3.1 Memory Specifications
* **LDS (Local Data Share):** **64 KB per CU**. 
  * *Constraint:* LDS allocation is performed at a 512-byte granularity. 
  * *Rule:* You must pad arrays in LDS to avoid severe bank conflicts. When writing low-level ISA/assembly, use the `M0` register for bounds clamping to prevent out-of-bounds LDS access.
* **L1 Cache:** 32 KB per CU.
* **L2 Cache:** 4 MB per XCD (16-way, 16 channels). 
  * *Rule:* L2 is the critical point for coalescing local traffic and maintaining intra-XCD coherency. Attempt to size your workgroup working sets to fit within this 4 MB boundary per XCD.
* **L3 / Infinity Cache (LLC):** 256 MB located on the IODs. 
  * *Architecture:* It acts as a memory-side cache and does NOT participate in coherency evictions, using a snoop filter instead. It resolves cross-XCD coherent requests. Peak bandwidth is ~17.2 TB/s.
* **HBM3 (Global Memory):** 192 GB capacity across 8 stacks, **5.3 TB/s peak bandwidth** (8192-bit bus). 

### 3.2 Memory Optimization Directives
1. **Target the L2/LLC:** Do not treat MI300X as having a unified L2 and HBM. Force hot working sets to stay in the 4MB L2 or 256MB Infinity Cache. Bandwidth drops precipitously if your kernel causes random accesses across IODs to HBM.
2. **Coalesced Access:** Global memory accesses must be coalesced. Ensure adjacent work-items in a Wave64 access contiguous 256-byte aligned memory segments.

## 4. Matrix Cores & MFMA Instructions
For AI workloads, utilize the Matrix Cores via **MFMA (Matrix Fused Multiply-Add)** instructions.

* **Target Data Types:** MI300X natively supports FP64, FP32, **TF32**, FP16, BF16, FP8, and INT8.
  * *Note:* CDNA 3 **natively supports TF32** in hardware. You may use TF32 MFMA instructions. It DOES NOT support FP6 or FP4.
* **Tile Alignment:** MFMA instructions operate on specific matrix tile dimensions (e.g., 32x32x8). Strictly align your LDS block loading to match these wave-level matrix shapes.

## 5. Execution Environment & NUMA Partitioning (Critical for Tuning)

The MI300X exposes its multi-die nature through Compute and Memory partitions. As an expert agent, you must design and tune kernels with these partitions in mind:
* **SPX (Single Partition):** The system sees one giant GPU. Workgroups are distributed round-robin across all 8 XCDs. You cannot control placement.
* **CPX (Core Partition):** The GPU is split into 8 logical partitions (1 per XCD). Workgroups are pinned to a specific XCD.
* **NPS4 (Memory Partition):** HBM is divided into 4 NUMA quadrants to enforce locality.

* **Directive for Kernel Tuning:** When generating code for microbenchmarks or extreme tuning, assume the environment is set to **CPX + NPS4**. Optimize the kernel to achieve maximum throughput on a *single XCD* (Phase A: Single XCD Locality) before scaling it up to rely on the Infinity Cache in SPX mode (Phase B: Full GPU Scaling).

## 6. Strict Kernel Generation Constraints
1. **Never use `__syncthreads()` unnecessarily:** Replace with wave-level synchronization (`__builtin_amdgcn_s_barrier()`).
2. **Register Allocation & Spilling:** MI300X performance collapses if registers spill to memory. Keep VGPR usage tightly bounded to allow at least 2-4 waves per SIMD. Use `__launch_bounds__`.
3. **Kernel Fusion:** With 5.3 TB/s HBM bandwidth but incredibly high compute throughput, memory-bound operations (RoPE, SiLU, Softmax, RMSNorm) must be fused into compute-bound kernels (like GEMM) whenever possible to prevent unnecessary trips to HBM.

---

# HIP Kernel Best Practices

Reference: [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/) | [AMD Instinct docs](https://instinct.docs.amd.com/)

---

## 1. Memory Access — Coalescing

AMD CDNA GPUs (MI300/MI350 series) access global memory in 64-byte cache lines. A wavefront of 64 threads fetches optimally when consecutive threads access consecutive addresses.

**Good — coalesced:**
```cpp
// Thread i reads element i: one cache line per 64-byte block
float val = a[blockDim.x * blockIdx.x + threadIdx.x];
```

**Bad — strided:**
```cpp
// Each thread jumps N elements: N separate cache-line loads
float val = a[threadIdx.x * N];
```

Rules:
- Prefer Structure-of-Arrays (SoA) over Array-of-Structures (AoS).
- Align buffers to 128 bytes (`hipMallocAligned` or `__attribute__((aligned(128)))`).
- Use vector loads (`float4`, `half2`, `uint4`) to widen memory transactions and reduce instruction count.

---

## 2. Occupancy and Wavefront Management

The AMD CDNA compute unit (CU) schedules up to 8 wavefronts of 64 threads. High occupancy hides memory latency.

### Controlling occupancy
```cpp
// Suggest minimum wavefronts per CU to the compiler
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void myKernel(...) { ... }

// Hard cap on registers to boost occupancy
__attribute__((amdgpu_flat_work_group_size(64, 256)))
__global__ void myKernel(...) { ... }
```

### Key occupancy limits
| Resource per CU | CDNA3 (MI300) limit |
|----------------|---------------------|
| Wavefronts     | 32                  |
| VGPRs          | 512 per SIMD × 4    |
| SGPRs          | 800 per SIMD × 4    |
| LDS            | 64 KB               |

Use `rocm-smi --showmeminfo` and `rocprof` (or Omniperf) to measure actual occupancy.

- Block size should be a multiple of 64 (wavefront width).
- Prefer 256 threads/block for general workloads; tune with `hipOccupancyMaxPotentialBlockSize`.

---

## 3. LDS (Local Data Share / Shared Memory)

LDS provides ~100× faster bandwidth than global memory. Each CU has 64 KB.

```cpp
__global__ void tiled_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    float acc = 0.f;

    for (int t = 0; t < K / TILE; ++t) {
        As[ty][tx] = A[(blockIdx.y * TILE + ty) * K + t * TILE + tx];
        Bs[ty][tx] = B[(t * TILE + ty) * N + blockIdx.x * TILE + tx];
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[(blockIdx.y * TILE + ty) * N + blockIdx.x * TILE + tx] = acc;
}
```

**Avoid bank conflicts:** LDS has 32 banks (4-byte each). Threads within a wavefront that map to the same bank serialize. Pad shared arrays:
```cpp
__shared__ float tile[TILE][TILE + 1]; // +1 avoids 32-way conflict
```

---

## 4. Register Pressure and Spilling

Each SIMD unit has 512 VGPRs. Kernels using >64 VGPRs per thread reduce maximum wavefronts per CU from 8 to 4 (or fewer). Register spilling to scratch memory adds ~500 cycle round-trip latency.

**Check register usage:**
```bash
hipcc -O3 --save-temps --offload-arch=gfx942 kernel.cpp
# Read the .s assembly for v_readlane / s_load_dword (spill indicators)
```

**Reduce registers:**
- Break large kernels into smaller ones.
- Use `__attribute__((noinline))` on helper functions to prevent excessive inlining.
- Replace temporary arrays with reduction trees.
- Accumulate in `float` but store in `half` when precision allows.

---

## 5. Divergent Branching

Within a wavefront, divergent branches (where some threads take the if-path and others the else-path) cause both paths to execute serially with masking, doubling worst-case cost.

```cpp
// Bad: half the wavefront idles each branch
if (threadIdx.x % 2 == 0)
    doA();
else
    doB();

// Better: sort/reorder input so adjacent threads follow the same path
// Or use predicated arithmetic instead of branching:
float result = cond ? a : b;    // compiles to v_cndmask
```

- Hoist loop-invariant conditionals above the loop.
- Use `__builtin_expect` to guide branch prediction hints.
- On CDNA, `if/else` on uniform (scalar) conditions (e.g., `blockIdx.x == 0`) use the SALU and are free; only per-thread vector conditions cause divergence.

---

## 6. Atomic Operations

Global atomics stall the wavefront until the operation completes. Prefer LDS-local atomics where possible, then reduce to global at the end.

```cpp
// Pattern: per-block reduction in LDS, one global atomic per block
__shared__ int local_sum;
if (threadIdx.x == 0) local_sum = 0;
__syncthreads();

atomicAdd(&local_sum, thread_val);    // fast LDS atomic
__syncthreads();

if (threadIdx.x == 0)
    atomicAdd(global_sum, local_sum); // one global atomic per block
```

On MI300+, use `__hip_atomic_fetch_add` with `__HIP_MEMORY_SCOPE_WORKGROUP` for workgroup-scoped atomics, which route through LDS without hitting the L2.

---

## 7. Async Copies and Streams

Overlap host-device transfers with kernel execution using multiple streams:

```cpp
hipStream_t stream[2];
hipStreamCreate(&stream[0]);
hipStreamCreate(&stream[1]);

for (int i = 0; i < N; i += CHUNK) {
    int s = i / CHUNK % 2;
    hipMemcpyAsync(d_in + i, h_in + i, CHUNK * sizeof(float),
                   hipMemcpyHostToDevice, stream[s]);
    myKernel<<<grid, block, 0, stream[s]>>>(d_in + i, d_out + i, CHUNK);
    hipMemcpyAsync(h_out + i, d_out + i, CHUNK * sizeof(float),
                   hipMemcpyDeviceToHost, stream[s]);
}
hipDeviceSynchronize();
```

On MI300/MI350, the GPU has dedicated DMA engines that run concurrently with compute. Use pinned host memory (`hipHostMalloc`) for maximum transfer bandwidth.

---

## 8. CDNA-Specific Optimizations (MI300/MI350)

### Matrix cores (MFMA)
AMD CDNA3/4 has Matrix Fused Multiply-Add (MFMA) units. Use them via:
- **rocWMMA** — C++ wrappers for MFMA intrinsics
- **composable_kernel** — pre-built high-performance GEMM/convolution kernels
- Direct intrinsics: `__builtin_amdgcn_mfma_f32_16x16x4f32`

```cpp
// 16x16x4 MFMA: computes C += A * B for 16×16 tiles with depth 4
__builtin_amdgcn_mfma_f32_16x16x4f32(a_frag, b_frag, c_frag, 0, 0, 0);
```

### Unified Memory (MI300 HBM)
MI300 has a unified CPU-GPU memory pool. Use `hipMallocManaged` to exploit this:
```cpp
float *data;
hipMallocManaged(&data, N * sizeof(float));
// No explicit copies needed; prefetch to improve locality:
hipMemAdvise(data, N * sizeof(float), hipMemAdviseSetPreferredLocation, 0 /* GPU */);
hipMemPrefetchAsync(data, N * sizeof(float), 0 /* device */, stream);
```

### Infinity Fabric (xGMI) for multi-GPU
Use RCCL for collective communications; it auto-selects xGMI (NVLink equivalent) paths over PCIe when available.

---

## 9. Profiling with rocprof and Omniperf

```bash
# Basic counter collection
rocprof --stats --hip-trace my_app

# Omniperf (MI200/MI300): rich roofline and bottleneck analysis
omniperf profile --name run1 -- ./my_app
omniperf analyze --path workloads/run1/mi300a/ --list-stats
```

Key metrics to watch:
| Metric | Healthy range |
|--------|--------------|
| Wavefront occupancy | > 50% of max |
| L2 cache hit rate | > 80% for reuse-heavy kernels |
| Memory bandwidth utilization | > 70% of peak for bandwidth-bound kernels |
| VGPR usage | < 64 per thread (for 8 waves/CU) |
| LDS bank conflicts | 0 |

---

## 10. Compilation Flags

```bash
hipcc -O3 \
      --offload-arch=gfx942 \        # MI300X; use gfx950 for MI350
      -mllvm -amdgpu-function-calls=0 \  # inline device functions
      -mllvm -amdgpu-sroa=1 \        # scalar replacement of aggregates
      -ffast-math \                   # allow reassociation, approx math
      kernel.cpp -o kernel
```

- `-O3` enables loop unrolling and vectorization.
- `--offload-arch` must match the target GPU; wrong arch causes runtime failure.
- Avoid `-g` in production; it disables many optimizations and inflates binary size.

---

## 11. Quick Checklist

- [ ] Access pattern is coalesced (SoA layout, 128-byte alignment)
- [ ] Block size is a multiple of 64 (wavefront width)
- [ ] Shared memory tile avoids bank conflicts (pad by 1)
- [ ] Register count < 64 VGPRs/thread (verify with `--save-temps`)
- [ ] No divergent branches in inner loops
- [ ] Atomics use LDS-local reduction before global write
- [ ] Streams overlap compute and data transfer
- [ ] MFMA units used for matrix workloads (via rocWMMA or CK)
- [ ] Kernels profiled with Omniperf to identify actual bottleneck
