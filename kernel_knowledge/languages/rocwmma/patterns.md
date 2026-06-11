---
title: rocWMMA — fragment GEMM, cooperative & transform patterns
kind: language
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp32]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/rocWMMA@develop:samples/simple_hgemm.cpp
  - ROCm/rocWMMA@develop:samples/perf_hgemm.cpp
  - https://rocm.docs.amd.com/projects/rocWMMA/en/develop/api-reference/api-reference-guide.html
---

# rocWMMA patterns

## TL;DR
The three patterns that matter: (1) the **fragment GEMM tile** (declare → fill → K-loop load/mma →
elementwise epilogue → store); (2) the **cooperative API** (`Scheduler` template param + LDS staging) for
multi-wave shared tiles; (3) the **Transforms API** for in-register reshape/transpose without an LDS
round-trip. The naïve fragment GEMM teaches the API but is not fast — perf comes from LDS staging +
pipelining, which rocWMMA leaves to you.

## Pattern 1 — fragment GEMM tile (the canonical shape)
Single-wave-per-16×16-output HGEMM `D = α·A·B + β·C`, A row-major M×K, B col-major K×N, C/D row-major.
Pattern from `samples/simple_hgemm.cpp`:

```cpp
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
const int M=16, N=16, K=16;                       // 16x16x16 tile (bf16/fp16 on gfx942)
const uint32_t WS = getWarpSize();                // 64 on CDNA
// blockDim = (4*WS, 4) -> 16 waves/block

__global__ void hgemm(/* m,n,k, a,b,c,d, lda..ldd, alpha,beta */) {
  fragment<matrix_a,    M,N,K, float16_t, row_major> fragA;
  fragment<matrix_b,    M,N,K, float16_t, col_major> fragB;
  fragment<accumulator, M,N,K, float32_t>            fragAcc;
  fragment<accumulator, M,N,K, float16_t>            fragC;
  fill_fragment(fragAcc, 0.0f);                    // zero accumulator before K-loop

  uint32_t cRow = ((blockIdx.x*blockDim.x + threadIdx.x)/WS) * M;
  uint32_t cCol =  (blockIdx.y*blockDim.y + threadIdx.y)     * N;
  if (cRow < m && cCol < n) {
    for (uint32_t i = 0; i < k; i += K) {
      load_matrix_sync(fragA, a + (cRow*lda + i), lda);    // A tile (row-major)
      load_matrix_sync(fragB, b + (i + cCol*ldb), ldb);    // B tile (col-major)
      mma_sync(fragAcc, fragA, fragB, fragAcc);            // acc += A*B (issues v_mfma)
    }
    load_matrix_sync(fragC, c + (cRow*ldc + cCol), ldc, mem_row_major);
    for (int i = 0; i < fragC.num_elements; ++i)           // ELEMENTWISE ONLY
      fragC.x[i] = alpha*fragAcc.x[i] + beta*fragC.x[i];
    store_matrix_sync(d + (cRow*ldd + cCol), fragC, ldd, mem_row_major);
  }
}
```
Compile: `hipcc --offload-arch=gfx942 -O3 -I/opt/rocm/include hgemm.cpp -o hgemm`.
**This loads A/B from global each K-step — fine for learning, not competitive.** The epilogue loop is the
only place you touch `frag.x[i]`, and only for elementwise scale.

### Core functions used above
| function | effect |
|---|---|
| `fill_fragment(frag, v)` | broadcast-set every element (zero accumulator before K-loop) |
| `load_matrix_sync(frag, ptr, ldm[, layout])` | load a tile into the fragment per its layout (overload takes runtime `mem_row_major`/`mem_col_major`) |
| `mma_sync(d, a, b, c)` | `D = A·B + C`; `c == d` aliasing valid (in-place accumulate) |
| `store_matrix_sync(ptr, frag, ldm[, layout])` | gather a fragment back to memory |
| `synchronize_workgroup()` | barrier across all wavefronts in the block (around LDS staging) |

All 8 layout combos of N(col)/T(row) across A/B/C are supported; C and D layouts match.

## Pattern 2 — cooperative (multi-wave) API
There is **no `load_matrix_coop_sync` function**. Cooperation is expressed via the fragment's
**`Scheduler` template parameter**, controlling how multiple waves share the load/compute of one logical
tile (spreads global loads across waves for full coalescing):

| Scheduler | behavior |
|---|---|
| `default_schedule` | each wave operates independently (Pattern 1) |
| `coop_row_major_2d<TBX,TBY>` | waves contribute in row-major grid order |
| `coop_col_major_2d<TBX,TBY>` | waves contribute in col-major grid order |
| `coop_row_slice_2d<TBX,TBY>` | partition into rows; only same-row waves cooperate |
| `coop_col_slice_2d<TBX,TBY>` | partition into cols; only same-col waves cooperate |
| `single<TBX,TBY,WaveIdx>` | only one designated wave participates |

Use a coop scheduler when several waves jointly load a large shared tile into LDS, then pair with
`synchronize_workgroup()` before the consuming `mma_sync`. Include `rocwmma_coop.hpp`.

## Pattern 3 — Transforms API (in-register reshape)
`rocwmma_transforms.hpp` provides `apply_transpose`, `apply_data_layout`, `to_register_file` for
in-register tile reshaping — e.g. transpose a fragment without an LDS round-trip. Include only
`rocwmma.hpp`, `rocwmma_coop.hpp`, `rocwmma_transforms.hpp` in user code (the rest are internal).

## Pattern 4 — making it fast (what `perf_hgemm.cpp` adds)
The naïve GEMM → competitive GEMM gap is closed by, in order:
1. **LDS staging** — cooperatively load A/B block tiles into LDS once, reuse across all output fragments
   in the block (cuts redundant global loads).
2. **Multiple accumulator fragments per wave** — a 2×2 or 4×4 grid of `16×16` accumulators per wave to
   raise arithmetic intensity (watch AGPR/VGPR pressure and `v_accvgpr_read/write` spills — check the
   disassembly).
3. **Software pipelining** — double-buffer the LDS stage so the next K-block loads while the current one
   MMAs.
4. **Prefer `16×16×16`** accumulators over `32×32×8` on MI300X.

At step 3 you have re-built CK's inner loop. If the goal is production GEMM, use CK instead; rocWMMA wins
only when the matmul is *embedded* in a larger custom (and portable) kernel.

## Pitfalls
- Mixing a `Scheduler` other than `default_schedule` without `synchronize_workgroup()` around the LDS
  stage → races / wrong results.
- Building big multi-fragment accumulator tiles silently spills AGPRs → `v_accvgpr_read/write` traffic
  tanks perf. Always inspect ISA.
- `load_matrix_sync` with a wrong `ldm` or layout silently transposes the load — parity bug, not a crash.

## Verify
- `samples/simple_hgemm.cpp` for correctness baseline; `samples/perf_hgemm.cpp` for the staged version.
- ISA dump (`-S` or `AMDGCN_ENABLE_DUMP=1`): want `v_mfma_f32_16x16x16_bf16`, LDS `ds_read_b128`, minimal
  `v_accvgpr_read`.
- Bench against hipBLASLt/CK on the exact shape.

## Sources
- rocWMMA samples (`simple_hgemm.cpp`, `perf_hgemm.cpp`): `ROCm/rocWMMA@develop:samples/` —
  https://github.com/ROCm/rocWMMA/blob/develop/samples/simple_hgemm.cpp
- rocWMMA API Reference (schedulers, transforms, function semantics):
  https://rocm.docs.amd.com/projects/rocWMMA/en/develop/api-reference/api-reference-guide.html
- AMD GPUOpen WMMA fragment model: https://gpuopen.com/learn/wmma_on_rdna3/
- overview: [overview.md](overview.md)
