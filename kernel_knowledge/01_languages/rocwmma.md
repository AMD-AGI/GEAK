# rocWMMA — C++ Matrix-Core Fragment API (MI300X / CDNA3 / gfx942)

> Scope: **rocWMMA**, AMD's header-only C++ library for mixed-precision matrix-multiply-accumulate on the
> matrix cores. The fragment API (`rocwmma::fragment`, `fill_fragment`, `load_matrix_sync`, `mma_sync`,
> `store_matrix_sync`), supported tile shapes / dtypes on gfx942, how fragments map to MFMA, the
> cooperative (multi-wave) API, and how it compares to raw MFMA intrinsics and to CK / ck_tile. Target:
> **gfx942 (MI300X/MI300A/MI325X)**, gfx950 (CDNA4) noted. For the underlying MFMA hardware see
> `asm_mfma_intrinsics.md`; for library GEMM/attention see `composable_kernel.md` / `ck_tile.md`.
>
> rocWMMA deliberately **mirrors NVIDIA's `nvcuda::wmma`** API so WMMA code ports across vendors. Min
> ROCm 6.4, header at `/opt/rocm/include/rocwmma/rocwmma.hpp`. Repo moved into `ROCm/rocm-libraries`
> (the standalone `ROCm/rocWMMA` is deprecated).

---

## 1. Mental model: fragments are a typed view of MFMA lane registers

A `fragment` is a small object, **stored in packed VGPRs**, that holds *this lane's share* of a
BlockM×BlockN×BlockK tile. rocWMMA hides the scattered MFMA lane layout (see `asm_mfma_intrinsics.md`
§2.2): you call `load_matrix_sync` and it distributes a tile from memory into the right lanes; you call
`mma_sync` and it issues the right `v_mfma_*`; you call `store_matrix_sync` and it gathers back. **Vector
elements inside a fragment have no guaranteed order or locality** — never index `frag.x[i]` assuming a
particular row/col; only do *elementwise* math (alpha/beta scaling, activation) on them.

`mma_sync` / `load` / `store` are **wave-cooperative**: all 64 lanes of the wavefront must execute them
together (like a warp-synchronous op). With LDS source/dest, you may need an explicit
`synchronize_workgroup()` between produce and consume.

---

## 2. The `fragment` template

```cpp
rocwmma::fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT /*, Scheduler*/>
```

| param | values | meaning |
|---|---|---|
| **MatrixT** | `matrix_a`, `matrix_b`, `accumulator` | A, B, or C/D fragment context |
| **BlockM/N/K** | e.g. 16,16,16 / 32,32,8 | the MFMA tile this fragment maps to |
| **DataT** | `float16_t`, `bfloat16_t`, `float8_t`, `bfloat8_t`, `int8_t`, `float`, `double`, `int32_t`, `xfloat32_t` | element type |
| **DataLayoutT** | `row_major`, `col_major` | in-memory layout (A/B); accumulator usually defaulted |
| **Scheduler** | `default_schedule` (or coop schedulers, §5) | wave cooperation pattern |

Type triple notation `<Ti / To / Tc>` = Input(A/B) / Output(C/D) / Compute(accum). **On CDNA the matrix
unit always accumulates in 32-bit and converts to the output type** — so for bf16/fp8 inputs your
`accumulator` fragment is `float` even if you store bf16.

### 2.1 Core functions

| function | effect |
|---|---|
| `fill_fragment(frag, value)` | broadcast-set every element (zero the accumulator before the K-loop) |
| `load_matrix_sync(frag, ptr, ldm)` | load a tile from global/LDS into the fragment per its layout |
| `load_matrix_sync(frag, ptr, ldm, layout)` | overload with explicit runtime layout (`mem_row_major`/`mem_col_major`) |
| `store_matrix_sync(ptr, frag, ldm[, layout])` | gather a fragment back to memory |
| `mma_sync(d, a, b, c)` | `D = A·B + C`; `c == d` aliasing is valid (in-place accumulate) |
| `synchronize_workgroup()` | barrier across all wavefronts in the block (needed around LDS staging) |

Layout combos: all 8 of N(col)/T(row) across A,B,C/D are supported (C and D layouts match). Up to **4
wavefronts per thread block**; valid `(TBlock_X, TBlock_Y)` include `WaveSize×{1,2,4}`,
`2·WaveSize×{1,2}`, `4·WaveSize×1` (WaveSize = **64** on CDNA).

---

## 3. Supported tile shapes & dtypes on gfx942

`BlockM/N` are *minimum recommended* (below → padding → perf hit; above → powers of 2 OK). `BlockK` is a
minimum; **in practice BlockK ≤ 32**. F8/BF8 and F64 **require gfx942/gfx950**.

| Types `Ti/To/Tc` | BlockM | BlockN | BlockK (gfx942) | notes |
|---|---|---|---|---|
| f16 / f32 / f32 | 16 | 16 | 16 | |
| f16 / f16 / f32 | 16 | 16 | 16 | accum f32, store f16 |
| f16 / f16 / f32 | 32 | 32 | 8 | larger tile |
| f16 / f16 / f16 | 16 | 16 | 16 | full-fp16 (lossy accum) |
| **bf16** / f32 / f32 | 16 | 16 | **16** | (gfx908 = 8) |
| **bf16** / bf16 / f32 | 16 | 16 | **16** | the LLM default |
| bf16 / bf16 / f32 | 32 | 32 | 8 | |
| **f8** / f32 / f32 | 16 | 16 | **32** | gfx942/gfx950 only; 2× K density |
| f8 / f32 / f32 | 32 | 32 | 16 | |
| **i8** / i32 / i32 | 16 | 16 | **32** | (gfx908/90a = 16) |
| f32 / f32 / f32 | 16 | 16 | 4+ | |
| f32 / f32 / f32 | 32 | 32 | 2+ | |
| f64 / f64 / f64 | 16 | 16 | 4+ | gfx90a/942/950 only |

gfx942 fp8: rocWMMA assumes the gfx942 "NANOO" (`fnuz`) f8 format unless OCP f8 is selected; match your
quantization. rocWMMA also supports **partial fragments** (FragMNK < BlockMNK) — internally padded to the
nearest supported BlockMNK (padding costs perf).

---

## 4. Complete rocWMMA GEMM tile example

Single-wave-per-16×16-output HGEMM (`D = α·A·B + β·C`), A row-major M×K, B col-major K×N, C/D row-major
M×N. Pattern from rocWMMA's `samples/simple_hgemm.cpp` (MIT, AMD):

```cpp
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

const int ROCWMMA_M = 16, ROCWMMA_N = 16, ROCWMMA_K = 16;          // tile (bf16/fp16 on gfx942)
const uint32_t WAVE_SIZE = getWarpSize();                          // 64 on CDNA
const int T_BLOCK_X = 4 * WAVE_SIZE;   // 4 waves along X
const int T_BLOCK_Y = 4;               // 4 along Y  -> 16 waves/block (16x16 output blocks)

__global__ void hgemm(uint32_t m, uint32_t n, uint32_t k,
                      const float16_t* a, const float16_t* b,
                      const float16_t* c, float16_t* d,
                      uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
                      float alpha, float beta)
{
    // declare fragments
    fragment<matrix_a,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, row_major> fragA;
    fragment<matrix_b,    ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, col_major> fragB;
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>            fragAcc;
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t>            fragC;

    fill_fragment(fragAcc, 0.0f);                                  // zero accumulator

    // which 16x16 output block does THIS wave own?
    uint32_t majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    uint32_t minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);
    uint32_t cRow = majorWarp * ROCWMMA_M;
    uint32_t cCol = minorWarp * ROCWMMA_N;

    if (cRow < m && cCol < n) {
        // ---- K reduction loop ----
        for (uint32_t i = 0; i < k; i += ROCWMMA_K) {
            load_matrix_sync(fragA, a + (cRow * lda + i), lda);    // A tile (row-major)
            load_matrix_sync(fragB, b + (i + cCol * ldb), ldb);   // B tile (col-major)
            mma_sync(fragAcc, fragA, fragB, fragAcc);             // acc += A*B  (issues v_mfma)
        }
        // ---- epilogue: D = alpha*acc + beta*C ----
        load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc, mem_row_major);
        for (int i = 0; i < fragC.num_elements; ++i)              // elementwise only!
            fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
        store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd, mem_row_major);
    }
}

// host launch
auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
auto gridDim  = dim3(ceil_div(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                     ceil_div(n, ROCWMMA_N * T_BLOCK_Y));
hgemm<<<gridDim, blockDim>>>(m, n, k, dA, dB, dC, dD, lda, ldb, ldc, ldd, alpha, beta);
```

Compile: `hipcc --offload-arch=gfx942 -O3 -I/opt/rocm/include hgemm.cpp -o hgemm`.

> This naïve version loads A/B straight from global each K-step — fine for learning, **not** competitive.
> For real perf you stage tiles through **LDS**, reuse across the block, and software-pipeline (rocWMMA's
> `perf_hgemm` sample does this). At that point you are re-implementing what CK/ck_tile already do.

---

## 5. The cooperative API (multi-wave)

There is **no `load_matrix_coop_sync` function** in current rocWMMA — cooperation is expressed via the
fragment's **`Scheduler` template parameter**, which controls how multiple waves in a block share the
load/compute of one logical tile (better global-load coalescing across waves):

| Scheduler | behavior |
|---|---|
| `default_schedule` | each wave operates independently (the §4 case) |
| `coop_row_major_2d<TBX,TBY>` | waves contribute in row-major grid order |
| `coop_col_major_2d<TBX,TBY>` | waves contribute in col-major grid order |
| `coop_row_slice_2d<TBX,TBY>` | partition into rows; only same-row waves cooperate |
| `coop_col_slice_2d<TBX,TBY>` | partition into cols; only same-col waves cooperate |
| `single<TBX,TBY,WaveIdx>` | only one designated wave participates |

Use a coop scheduler when several waves jointly load a large shared tile into LDS — it spreads the global
load across waves for full coalescing. Pair with `synchronize_workgroup()` before the consuming
`mma_sync`. Related: the **Transforms API** (`rocwmma_transforms.hpp`) — `apply_transpose`,
`apply_data_layout`, `to_register_file` — for in-register tile reshaping (e.g. transpose without LDS
round-trip). Include only `rocwmma.hpp`, `rocwmma_coop.hpp`, `rocwmma_transforms.hpp` in user code.

---

## 6. How rocWMMA maps to MFMA (and where the abstraction leaks)

- A `fragment<...,16,16,16,bf16>` + `mma_sync` lowers to `v_mfma_f32_16x16x16_bf16`; the `<32,32,8>`
  variant lowers to `v_mfma_f32_32x32x8_*`. `load_matrix_sync` generates the lane-distributed
  `ds_read`/`global_load` placement that matches the MFMA layout.
- Same MI300X power/clock fact applies: **16×16×16 generally yields higher achievable FLOPs than
  32×32×8** — prefer the 16-tile fragment.
- The accumulator lives in AGPRs/VGPRs; the same large-tile `v_accvgpr_read/write` and spill traps from
  `asm_mfma_intrinsics.md` §3 apply when you build big multi-fragment tiles. Check the disassembly.

---

## 7. rocWMMA vs raw MFMA intrinsics vs CK/ck_tile

| dimension | raw MFMA intrinsics | **rocWMMA** | CK / ck_tile |
|---|---|---|---|
| abstraction | per-lane register placement by hand | typed fragments, WMMA-style API | full device GEMM/attention ops |
| portability | gfx-specific intrinsic names | **portable** (CDNA + RDNA + maps to nvcuda::wmma) | AMD-only, but turnkey |
| lane-layout burden | you own it (or use the calculator) | hidden | hidden |
| LDS staging / pipelining | manual | manual (you write the LDS + pipeline) | **built-in** (v3/v4, CShuffle, async) |
| peak perf for dense GEMM | max (if you out-schedule LLVM) | good with effort; rarely beats CK | **best turnkey** (CK v3 ≈ 615 TFLOP/s @4096³) |
| best use | a tiny hot custom op, research | **portable custom op**, fusing MFMA into bespoke kernels, WMMA ports | production GEMM/FMHA/MoE |

**Decision guide:**
- Need a **production** dense GEMM / attention / MoE on MI300X → use **CK / ck_tile** (don't hand-roll).
- Need a **custom fused kernel** with matrix-multiply inside (e.g. a fused norm+matmul, or a research op)
  and want it **portable** across AMD (and CUDA via nvcuda::wmma) → **rocWMMA**.
- Need to **out-schedule** the compiler on a tiny hot loop, or place fragments in an exotic layout →
  **raw `__builtin_amdgcn_mfma_*` intrinsics** (see `asm_mfma_intrinsics.md`).
- rocWMMA's strength is the **ergonomic, portable** middle ground; its weakness is that you must still
  write LDS staging + pipelining yourself to approach CK's numbers.

---

## 8. Quick tuning knobs

| knob | values | effect |
|---|---|---|
| fragment BlockM/N/K | 16×16×16 (default), 32×32×8 | MFMA size — prefer 16-tile on MI300X |
| dtype triple | bf16/bf16/**f32**, f8/f32/f32, i8/i32/i32 | precision; accum is always 32-bit |
| waves/block (TBlock_X/Y) | up to 4 waves | larger output tile per block, occupancy trade |
| Scheduler | default vs coop_* | multi-wave shared-tile coalescing |
| LDS staging | manual | the real perf lever — stage + reuse + pipeline |
| layout | row/col per A,B,C | match memory to avoid transposing loads |

---

## Sources
- rocWMMA API Reference Guide (fragment template, load/mma/store_matrix_sync, tile-shape & dtype tables, schedulers): https://rocm.docs.amd.com/projects/rocWMMA/en/develop/api-reference/api-reference-guide.html
- rocWMMA GitHub repo (samples: simple_hgemm.cpp, perf_hgemm.cpp; now under ROCm/rocm-libraries): https://github.com/ROCm/rocWMMA
- rocWMMA `samples/simple_hgemm.cpp` (the fragment GEMM walked through in §4): https://github.com/ROCm/rocWMMA/blob/develop/samples/simple_hgemm.cpp
- AMD GPUOpen — How to accelerate AI with WMMA (fragment model, load/store/mma_sync semantics): https://gpuopen.com/learn/wmma_on_rdna3/
- Matrix Core Programming on CDNA3/CDNA4 (MFMA shapes/dtypes that rocWMMA fragments map to): https://salykova.github.io/matrix-cores-cdna
- ROCm/amd_matrix_instruction_calculator (verify the MFMA that a fragment lowers to): https://github.com/ROCm/amd_matrix_instruction_calculator
- AMD CDNA3 ISA Reference (Ch.7 Matrix Arithmetic — the instructions behind mma_sync): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
