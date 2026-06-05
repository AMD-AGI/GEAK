# HIP / C++ Kernel Programming for CDNA3 — MI300X / gfx942

> Scope: writing high-performance HIP/C++ compute kernels for AMD Instinct MI300X (CDNA3, `gfx942`):
> toolchain, launch configuration, the wave64 programming model, `__launch_bounds__`, LDS shared
> memory, wave/cross-lane primitives with **64-bit masks**, streams/async copy, cooperative groups,
> and a complete tiled LDS GEMM. For raw AMDGCN intrinsics & MFMA microkernels see
> `hip_intrinsics_async.md`; for Triton see `triton_amd.md`.

---

## 0. MI300X hardware constants (memorize)

| Resource | Value (per unit) |
|---|---|
| Compute Units (CUs) | **304** active (8 XCDs × 38) |
| SIMDs per CU | 4 |
| Wavefront (warp) size | **64 lanes** |
| VGPRs | **512 / SIMD lane-slot**, granularity 16 |
| SGPRs | 102 usable / wave (104 alloc) |
| LDS (`__shared__`) | **64 KB / CU**, 32 banks × 4 B (128 B/cycle) |
| AGPRs (accum, MFMA) | 256 / lane (shared budget with VGPR on CDNA3) |
| L1 vector cache | 32 KB / CU |
| L2 | 4 MB / XCD |
| Infinity Cache (LLC) | 256 MB |
| Peak clock | ~2.1 GHz |
| HBM3 | 192 GB, ~5.3 TB/s |

> **The two facts that break CUDA habits:** wavefront is **64** (not 32), and LDS is only **64 KB**
> (vs 228 KB on H100). Every porting bug traces back to one of these.

---

## 1. Toolchain: hipcc / amdclang

```bash
# Compile for MI300X (CDNA3). gfx942 covers MI300X/MI300A/MI325X.
hipcc --offload-arch=gfx942 -O3 kernel.hip -o kernel
# Multiple targets (fat binary):
hipcc --offload-arch=gfx942 --offload-arch=gfx950 -O3 kernel.hip -o kernel
# amdclang++ directly (hipcc is a wrapper around it):
amdclang++ -x hip --offload-arch=gfx942 -O3 -munsafe-fp-atomics kernel.hip -o kernel
```

Useful flags:
| Flag | Purpose |
|---|---|
| `--offload-arch=gfx942` | target CDNA3 (required) |
| `-munsafe-fp-atomics` | enable HW fp atomics (`global_atomic_add_f32`) — big for split-K/reductions |
| `-ffast-math` / `-fgpu-flush-denormals-to-zero` | relax FP (check accuracy) |
| `--save-temps` | keep `.s` AMDGCN ISA for inspection |
| `-Rpass-analysis=kernel-resource-usage` | print VGPR/SGPR/LDS/scratch per kernel |
| `-mllvm -amdgpu-waves-per-eu=N` | global occupancy hint |

Inspect a built binary:
```bash
rocminfo | grep -E "Compute Unit|SIMD|Wavefront"   # confirm gfx942, wavefront 64
roc-obj-ls kernel ; llvm-objdump -d --arch=amdgcn kernel | less   # disassemble ISA
```

---

## 2. The wave64 programming model

A workgroup (HIP "block") is partitioned into wavefronts of **64 lanes** that execute in lockstep on
one SIMD. `warpSize == 64` on all current AMD datacenter GPUs.

```cpp
__global__ void k() {
    int lane   = threadIdx.x % warpSize;     // 0..63  — NOT 0..31
    int wave   = threadIdx.x / warpSize;     // wave index within block
    // warpSize is a runtime int; do NOT hardcode 32 or 64 in portable code.
}
```

**Block-size rule:** make `blockDim` a **multiple of 64** (e.g. 64, 128, 256). 256 threads = 4
wavefronts is a common sweet spot. A block sized 128 on a wave64 machine = 2 waves; a block of 32
wastes half a wave.

**Grid-size rule:** target **≥ 1024 workgroups** so the 304 CUs (×~8 resident blocks) stay fed and
work spreads across all 8 XCDs.

Porting trap: "wave-aware" CUDA code assuming 32-lane warps *runs* on AMD but uses **half** the
machine — every `__shfl`/`__ballot`/manual warp reduction must be revisited.

---

## 3. Launch configuration & `__launch_bounds__`

```cpp
// __launch_bounds__(maxThreadsPerBlock, minWavesPerEU)
__global__ void __launch_bounds__(256, 2)
my_kernel(const float* __restrict__ a, float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * 2.0f;
}

dim3 block(256);                              // 4 wavefronts
dim3 grid((n + block.x - 1) / block.x);
my_kernel<<<grid, block, /*dynLDS=*/0, stream>>>(a, b, n);
HIP_CHECK(hipGetLastError());
```

- `__launch_bounds__(maxTPB, minWavesPerEU)` caps registers so `minWavesPerEU` wavefronts fit per
  SIMD. It is the C++ analogue of Triton's `waves_per_eu`. With 512 VGPRs/EU, `minWavesPerEU=2`
  forces VGPR ≤ 256; `=4` forces ≤ 128. Too aggressive → **scratch spills** (HBM) → 3–5× slower.
- Always check actual usage: `-Rpass-analysis=kernel-resource-usage` or `--save-temps` →
  `.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size` (LDS), `.private_segment_fixed_size`
  (scratch — want **0**).
- `__restrict__` on pointers is important on AMD: enables wider `global_load_dwordx4` and reordering.

---

## 4. LDS / `__shared__` memory (64 KB, bank-conflict aware)

```cpp
// Static LDS:
__global__ void k() {
    __shared__ float tile[64][64];           // 16 KB — fits well within 64 KB
    // ...
    __syncthreads();                         // workgroup barrier (s_barrier)
}

// Dynamic LDS (size chosen at launch, 3rd <<<>>> arg):
extern __shared__ char smem[];
__global__ void k2() {
    float* a = reinterpret_cast<float*>(smem);
    float* b = a + BLOCK_M * BLOCK_K;
    // ...
}
k2<<<grid, block, (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N)*sizeof(float), stream>>>();
```

**Capacity:** 64 KB/CU. Two blocks per CU → ≤ 32 KB each. Budget LDS so target occupancy survives.

**Bank conflicts (critical with wave64):** 32 banks × 4 B. A wavefront issues memory for 64 lanes
but there are only 32 banks → addresses are serviced in two phases; same-bank/different-row accesses
across the phase **conflict** and serialize. Mitigations:
- **Pad** the inner dimension: `__shared__ float tile[64][64+1];` breaks the stride that maps
  columns onto the same bank.
- **Swizzle** (XOR the column index with a function of the row) for transpose-heavy patterns.
- Prefer **128-bit** LDS access (`float4` / `ds_read_b128`) — fewer instructions, better bandwidth
  (16-byte values reach ~80% peak; 4-byte ~50%).

```cpp
// vectorized 128-bit LDS load
float4 v = *reinterpret_cast<float4*>(&tile[r][c]);   // -> ds_read_b128
```

LDS throughput: a single wave can move up to 64 B/cycle (16 lanes/cycle); 4-B accesses take 8 cycles
for 64 lanes (50% peak), 16-B accesses 20 cycles (80% peak) — so vectorize.

---

## 5. Wave / cross-lane primitives — **64-bit masks**

This is where wave64 changes the API surface most vs CUDA.

```cpp
// __ballot returns a 64-bit mask on AMD (uint64), not 32-bit.
unsigned long long active = __ballot(pred);          // bit i = pred of lane i, i in 0..63
int count = __popcll(active);                         // popcount over 64 bits

// __shfl family — width defaults to warpSize (64). Do not assume 32.
float up   = __shfl_up(val, 1);
float down = __shfl_down(val, 1);
float xed  = __shfl_xor(val, 16);
float bcst = __shfl(val, srcLane);                    // half-float shuffle NOT supported

// Masked-sync variants: mask is 64-bit; every participating lane must pass the SAME mask.
unsigned long long m = 0xFFFFFFFFFFFFFFFFull;         // all 64 lanes
float r = __shfl_down_sync(m, val, 1);

// any/all/match
bool a = __any(pred);  bool b = __all(pred);
unsigned long long same = __match_any(key);          // 64-bit result
```

Rules & perf notes:
- Mask type **must be 64-bit** (`unsigned long long`). A 32-bit mask triggers a static-assert in
  `amd_warp_sync_functions.h` on CDNA.
- **Contiguous, hole-free masks are faster.** `0xFF` (no holes) beats `0xFB` (a hole) — the backend
  uses faster cross-lane ops for prefix masks. Reduce over `0..N-1` lanes, not scattered lanes.
- These intrinsics carry **no memory barrier** on any platform — add `__syncthreads()`/fences if you
  need ordering of memory side effects.
- Feature-detect with `HIP_ARCH_HAS_WARP_BALLOT` (device) / `hasWarpBallot` device prop (host).

### Wave64-correct block reduction
```cpp
__device__ float warp_reduce_sum(float v) {           // 64-lane reduce
    for (int off = warpSize / 2; off > 0; off >>= 1)  // 32,16,8,4,2,1
        v += __shfl_down(v, off);
    return v;                                          // lane 0 holds the sum
}

__global__ void block_reduce(const float* in, float* out, int n) {
    __shared__ float partial[64];                      // up to 64 waves/block (we use ≤ blockDim/64)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (tid < n) ? in[tid] : 0.0f;
    v = warp_reduce_sum(v);                             // intra-wave (64 lanes)
    int lane = threadIdx.x % warpSize;
    int wave = threadIdx.x / warpSize;
    if (lane == 0) partial[wave] = v;
    __syncthreads();
    if (wave == 0) {                                    // reduce the per-wave partials
        int nwaves = blockDim.x / warpSize;
        v = (lane < nwaves) ? partial[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) atomicAdd(out, v);              // HW fp atomic (-munsafe-fp-atomics)
    }
}
```

---

## 6. Grid-stride loops (portable, full-occupancy)

```cpp
__global__ void saxpy(int n, float a, const float* x, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {                // stride = total threads
        y[i] = a * x[i] + y[i];                        // 128-bit coalesced if aligned
    }
}
// launch a grid that saturates the device, not one-thread-per-element:
int blocks = 304 * 8;                                  // ~CUs × resident blocks
saxpy<<<blocks, 256, 0, stream>>>(n, 2.0f, x, y);
```
Coalescing: consecutive lanes should touch consecutive addresses so the wave issues
`global_load_dwordx4`. Use `float4`/`int4` for 128-bit transactions.

---

## 7. Streams & async memory

```cpp
hipStream_t s;  HIP_CHECK(hipStreamCreate(&s));
HIP_CHECK(hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s));  // needs pinned host mem
kernel<<<grid, block, 0, s>>>(d, n);
HIP_CHECK(hipMemcpyAsync(h, d, bytes, hipMemcpyDeviceToHost, s));
HIP_CHECK(hipStreamSynchronize(s));

// Pinned host memory enables true async DMA overlap:
float* h;  HIP_CHECK(hipHostMalloc(&h, bytes));        // or hipHostMalloc with flags
```

- Overlap copy and compute by issuing on **separate streams**; use `hipEventRecord`/`hipStreamWaitEvent`
  for cross-stream dependencies.
- On MI300X, prefer **one process per GPU** for multi-GPU (avoids launch serialization); set
  `GPU_MAX_HW_QUEUES=2` and disable NUMA balancing (`kernel.numa_balancing=0`) for training-scale runs.
- For HIP graphs (kill per-launch overhead in decode loops): capture with
  `hipStreamBeginCapture`/`hipStreamEndCapture` → `hipGraphInstantiate` → `hipGraphLaunch`.

---

## 8. Cooperative groups (higher-level wave primitives)

```cpp
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cg_reduce(const float* in, float* out, int n) {
    cg::thread_block blk = cg::this_thread_block();
    // tile size must be a power of 2 and ≤ warpSize (64 on CDNA3)
    cg::thread_block_tile<64> wave = cg::tiled_partition<64>(blk);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (tid < n) ? in[tid] : 0.0f;
    for (int off = wave.size() / 2; off > 0; off >>= 1)
        v += wave.shfl_down(v, off);                   // tile-scoped shuffle
    if (wave.thread_rank() == 0) atomicAdd(out, v);
}
```
- `thread_block_tile<N>`: `N` must be a power of two and ≤ wavefront size (so ≤ 64 on CDNA3 —
  `tiled_partition<64>` is the full wave; `<32>` is a half-wave sub-tile).
- Members: `thread_rank()`, `size()`, `sync()`, `shfl/shfl_up/shfl_down/shfl_xor()`, `ballot()`,
  `any()`, `all()`, `match_any()`, `match_all()`.
- Grid-wide sync (`cg::grid_group::sync()`) requires a **cooperative launch**
  (`hipLaunchCooperativeKernel`) and a grid that fits resident — use sparingly.

---

## 9. Worked example — tiled LDS GEMM (C = A·B, fp32)

Classic shared-memory blocked GEMM, tuned for CDNA3: 256-thread blocks (4 waves), 64×64 output tile,
padded LDS to avoid bank conflicts, 128-bit loads. (For real perf use MFMA — see
`hip_intrinsics_async.md`; this shows the LDS/tiling skeleton.)

```cpp
#define TM 64        // output tile rows
#define TN 64        // output tile cols
#define TK 16        // K-block
#define BX 16        // threads.x  (16*16 = 256 = 4 waves)
#define BY 16

__global__ void __launch_bounds__(BX*BY, 2)
gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
           float* __restrict__ C, int M, int N, int K) {
    // +1 padding column kills 32-bank conflicts for the wave64 access pattern
    __shared__ float As[TK][TM + 1];     // store A transposed: [k][m]
    __shared__ float Bs[TK][TN + 1];     // [k][n]

    int tx = threadIdx.x, ty = threadIdx.y;     // 0..15
    int row0 = blockIdx.y * TM;                 // tile origin
    int col0 = blockIdx.x * TN;

    // each thread computes a 4x4 micro-tile of C -> 64x64 / (16x16) = 4x4
    float acc[4][4] = {{0}};

    for (int k0 = 0; k0 < K; k0 += TK) {
        // cooperative load of A (TM x TK) and B (TK x TN) into LDS
        for (int i = ty; i < TM; i += BY)
            for (int kk = tx; kk < TK; kk += BX)
                As[kk][i] = A[(row0 + i) * K + (k0 + kk)];
        for (int kk = ty; kk < TK; kk += BY)
            for (int j = tx; j < TN; j += BX)
                Bs[kk][j] = B[(k0 + kk) * N + (col0 + j)];
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += a[i] * b[j];          // FMA-mapped
        }
        __syncthreads();                                // before overwriting LDS next iter
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int r = row0 + ty * 4 + i, c = col0 + tx * 4 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K, hipStream_t s) {
    dim3 block(BX, BY);                                 // 256 threads = 4 waves
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);    // aim ≥1024 blocks for big M,N
    gemm_tiled<<<grid, block, 0, s>>>(A, B, C, M, N, K);
}
```

AMD-specific points in this kernel:
- `__launch_bounds__(256, 2)` → 2 waves/EU target; keeps VGPRs ≤ 256, avoids spill.
- LDS arrays padded `+1` → conflict-free across the 32 banks with 64-lane waves.
- 256-thread block = exactly 4 wavefronts; each thread does a 4×4 register micro-tile (register
  blocking) so the inner loop is FMA-bound, not LDS-bound.
- Double-buffering the LDS tiles + replacing the FMA inner loop with `__builtin_amdgcn_mfma_*` is the
  upgrade path to peak — see `hip_intrinsics_async.md`.

---

## 10. CUDA → HIP porting checklist (AMD pitfalls)

| Pitfall | Symptom | Fix |
|---|---|---|
| `warpSize`/mask assumed 32 | wrong reductions, static-assert on mask | use 64; `unsigned long long` masks |
| Block size not multiple of 64 | wasted lanes | use 64/128/256 |
| `__ballot` stored in `unsigned` | truncated mask | `unsigned long long` + `__popcll` |
| LDS > 64 KB (ported from H100) | launch fail / occupancy 1 | shrink tile; budget to 64 KB |
| No LDS padding/swizzle | bank conflicts, slow | pad inner dim `+1` or swizzle |
| `__launch_bounds__` too tight | scratch spill (HBM) | check `.private_segment_fixed_size==0` |
| Scalar global loads | low BW | `float4`/`__restrict__` → `dwordx4` |
| fp atomics slow/unused | reduction bottleneck | `-munsafe-fp-atomics` |
| Mask with holes | slower cross-lane | use contiguous prefix masks |
| Half-float `__shfl` | unsupported | shuffle as int/float, repack |

---

## Sources

1. HIP C++ language extensions (warpSize, `__launch_bounds__`, `__shfl`/`__ballot` 64-bit masks, hole-free mask perf, half-float shuffle) — ROCm HIP docs: <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html>
2. Introduction to the HIP programming model (wavefront 64 on CDNA, SIMD, block sizing) — ROCm HIP docs: <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html>
3. HIP hardware implementation (LDS banks, 64 KB/CU, occupancy) — ROCm HIP docs: <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html>
4. Cooperative groups (thread_block_tile ≤ warpSize, member ops) — ROCm HIP docs: <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html>
5. AMD Instinct MI300X workload optimization (304 CUs, VGPR/LDS limits, ≥1024 grid, multi-GPU, GPU_MAX_HW_QUEUES) — ROCm Documentation: <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
6. `__ballot` 64-bit return & mask requirements (warpSize 32 vs 64) — ROCm/HIP issue #3667: <https://github.com/ROCm/HIP/issues/3667>
7. AMD matrix cores / Lab Notes (CDNA tiling, MFMA upgrade path from LDS GEMM) — AMD GPUOpen: <https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/>
