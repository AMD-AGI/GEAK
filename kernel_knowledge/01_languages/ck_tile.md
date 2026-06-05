# ck_tile — Tile-Based Composable Kernel Programming Model (MI300X / CDNA3 / gfx942)

> Scope: `include/ck_tile`, the newer **tile-programming** front-end of Composable Kernel — tile
> windows, tile distributions / distributed tensors, the pipeline+policy+epilogue decomposition, and how
> to write a GEMM and an FMHA. For the classic `DeviceGemm*` device-op model see `composable_kernel.md`.
> Target: **gfx942 (MI300X/MI300A/MI325X)**; gfx950 (CDNA4) noted where relevant.
>
> Why ck_tile exists: classic CK encodes data movement as deeply nested `constexpr` tensor-descriptor
> transforms — extremely fast but very hard to read/author. ck_tile keeps the same compile-time
> coordinate-transform engine underneath but exposes a **CUTLASS-CuTe-like** surface: you reason about
> *tiles*, *windows*, and *distributions* instead of raw descriptors. It is the path AMD now uses for
> new LLM kernels (FMHA paged-KV, fused-MoE, fp8 GEMM).

---

## 1. The five abstractions

ck_tile is layered. Every header is in `include/ck_tile/`; you include one header per component
(`#include "ck_tile/core.hpp"`, `#include "ck_tile/ops/gemm.hpp"`, `#include "ck_tile/ops/fmha.hpp"`).

| Abstraction | header | role |
|---|---|---|
| **TensorView** | `core/tensor/tensor_view.hpp` | a strided, possibly-padded N-D view over a raw pointer (global / LDS / VGPR) |
| **TileDistribution** | `core/tensor/tile_distribution.hpp` | the **thread↔element map**: which lane/wave owns which tile coords |
| **TileWindow** | `core/tensor/tile_window.hpp` | a *moving* sub-view + distribution → the load/store gateway (coalescing, vectorization, OOB guard) |
| **DistributedTensor** | `core/tensor/...` | the in-register result of `tile_window.load()`: storage + the cooperation pattern |
| **Pipeline / Policy / Epilogue** | `ops/gemm/`, `ops/fmha/` | the mainloop schedule, its layout policy, and the writeback |

The golden rule (verbatim from the docs): *the tile APIs (`make_naive_tensor_view`, `make_tile_window`)
only **declare** memory addresses; the real loading/writing happens inside the **pipeline** and
**epilogue**.* A window is a cursor, not a copy.

### 1.1 TensorView — raw memory → padded N-D view

```cpp
// Wrap a raw A pointer (M x K, row-major) as a 2D view, then pad K up to a tile multiple.
auto a_view = make_naive_tensor_view<address_space_enum::global>(
    a_ptr, make_tuple(M, K), make_tuple(K, 1),          // lengths, strides
    number<AK1>{}, number<1>{});                        // last-dim vector access hints

auto a_view_pad = pad_tensor_view(
    a_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    sequence<false, true>{});                           // pad K (guard OOB), not M
```

### 1.2 TileDistribution — the thread↔element encoding

This is the single most important (and most cryptic) ck_tile object. It is a `tile_distribution_encoding`
that declares, in compile-time `sequence`/`tuple` form, how the wavefront's 64 lanes and the block's waves
tile a 2D region and how many elements each lane holds (the per-lane "Y" space). Example for a GEMM tile,
M and N each tiled as a `<4,2,8,4>` hierarchy (Repeat × Warp × Lane × Vector):

```cpp
using Encoding = tile_distribution_encoding<
    sequence<>,                               // R: no replication
    tuple<sequence<4, 2, 8, 4>,               // H for M: {MRepeat, MWarp, MLane, MVec}
          sequence<4, 2, 8, 4>>,              // H for N
    tuple<sequence<1, 2>, sequence<1, 2>>,    // Ps -> RH major (which dim each P/thread idx drives)
    tuple<sequence<1, 1>, sequence<2, 2>>,    // Ps -> RH minor
    sequence<1, 1, 2, 2>,                     // Ys -> RH major (the per-lane register dims)
    sequence<0, 3, 0, 3>>;                    // Ys -> RH minor
constexpr auto dist = make_static_tile_distribution(Encoding{});
```

You almost never write this by hand — a **Policy** (§2.2) generates it from the tile sizes and the
chosen WarpGemm (MFMA). But you must recognize it: the `<4,2,8,4>` pattern reads as
`Repeat(4) × Warp(2) × Lane(8) × Vector(4)` and is what maps the tile onto the 32×32×8 MFMA lanes.

### 1.3 TileWindow + DistributedTensor — the data gateway

```cpp
auto a_win = make_tile_window(
    a_view_pad,                                       // the (padded) tensor view
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),  // window/tile shape
    {iM, 0},                                          // origin: this block's M offset, K=0
    dist);                                            // the tile distribution

auto a_tile = load_tile(a_win);                       // -> distributed_tensor in VGPR
a_win = move_tile_window(a_win, {0, KPerBlock});      // slide the cursor along K for next iter
```

Tile-level verbs operating on distributed tensors: `load_tile`, `store_tile`, `update_tile`,
`shuffle_tile` (re-distribute across lanes, e.g. for transpose), `slice_tile`, and `sweep_tile`
(iterate the per-lane Y elements with a lambda). Reductions across the distribution use
`block_tile_reduce` (the FMHA row-max/row-sum primitive).

---

## 2. Anatomy of a ck_tile GEMM

A GEMM kernel is assembled from **four** template pieces (this composition is the heart of ck_tile):

```
GemmKernel< TilePartitioner, GemmPipeline, EpiloguePipeline >
              │                  │              │
              │                  │              └─ writeback (+ CShuffle, + fused elementwise)
              │                  └─ the K-loop mainloop schedule (vN)
              └─ maps (M,N,K) → grid/blocks: gridDim = ceil(M/kM) × ceil(N/kN)
```

### 2.1 TilePartitioner

Maps the problem onto the GPU grid. Defines the block tile `kM × kN × kK`, computes
`gridSize.x = M/kM`, `gridSize.y = N/kN`, and the K iteration count. On MI300X (304 CUs) you size
`kM,kN` so `ceil(M/kM)·ceil(N/kN) ≈ k·304` to fully occupy CUs (e.g. for M=4864,N=4096 a 256×256 block
gives 19×16 = **304** blocks = one block per CU).

### 2.2 Pipeline + Policy (the swappable mainloop)

The pipeline name encodes its dataflow: **`GemmPipelineAgBgCrCompV3`** =
**A** from **g**lobal, **B** from **g**lobal, **C** in **r**egisters, **Comp**ute-optimized, **V3**.

| Pipeline | dataflow | character | pick when |
|---|---|---|---|
| `GemmPipelineAGmemBGmemCRegV1` | A,B global→LDS→reg, C in reg, single buffer | simplest, low VGPR | teaching / small / memory-bound |
| `GemmPipelineAgBgCrCompV3` | + double-buffer LDS, 2-stage prefetch, compute-opt schedule | the GEMM workhorse | **compute-bound large GEMM** |
| `GemmPipelineAgBgCrMemV3` / `...CompV4` | memory-opt / deeper prefetch variants | extra overlap | very large K / fp8 K-dense |
| async-input persistent (`*_async`) | `buffer_load` direct-to-LDS (DGL), persistent block | hides global latency w/o reg staging | latest LLM GEMM/MoE |

Each pipeline takes a **`GemmPipelineProblem`** (dtypes, layouts, tile sizes, flags) and a
**`GemmPipelinePolicy`** (the layout brains). The default policy (e.g.
`UniversalGemmPipelineAgBgCrPolicy`) provides:

- `MakeADramTileDistribution()` / `MakeBDramTileDistribution()` — the global-load distributions
  (warp layout, lanes per dim, **vector load width per lane**, lane repeats).
- `MakeALdsBlockDescriptor()` / `MakeBLdsBlockDescriptor()` — the LDS staging layout, including an
  **XOR swizzle** to eliminate bank conflicts.
- `GetWarpGemm()` — selects the MFMA WarpGemm instance (e.g. 32×32×8 bf16, 16×16×16, 16×16×32 fp8).

The **WarpGemm** struct is where ck_tile meets the matrix core. Real instance for fp16 32×32×8:

```cpp
struct WarpGemmAttributeMfmaImplF16F16F32M32N32K8 {
    using AVecType = fp16x4_t;       // 4 fp16 per lane along K
    static constexpr index_t kM = 32, kN = 32, kK = 8;
    static constexpr index_t kAMLane = 32, kBNLane = 32;   // A/B lane layout
    static constexpr index_t kABKLane = 2, kABKPerLane = 4;// 2 K-lanes, 4 K elems/lane
    static constexpr index_t kCMLane = 2, kCNLane = 32;    // C lane layout
    static constexpr index_t kCM0PerLane = 4, kCM1PerLane = 4; // 4 loops × 4 contiguous
    __device__ void operator()(CVecType& c, const AVecType& a, const BVecType& b) const {
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0); // <-- the matrix core
    }
};
```

### 2.3 Epilogue (+ CShuffle)

The MFMA C accumulator sits in a scattered per-lane layout that is **not** coalescable for the global
store. The **CShuffle epilogue** re-tiles C through LDS into a vectorized layout (`shuffle_tile` /
`store_tile`), optionally applies a fused elementwise op (bias, activation, residual add), then writes.
Knobs: `CShuffleDataType` (LDS staging precision), the C store vector width (8 for bf16), and the
shuffle granularity (`MXdlPerWavePerShuffle`).

### 2.4 Full GEMM mainloop skeleton

```cpp
template <typename ADataType, typename BDataType, typename CDataType,
          index_t BlockSize, index_t MPerBlock, index_t NPerBlock, index_t KPerBlock>
__global__ void ck_tile_gemm_kernel(const ADataType* a_g, const BDataType* b_g,
                                    CDataType* c_g, index_t M, index_t N, index_t K)
{
    // 1. views (+ pad), 2. distributions (from policy), 3. tile windows on A,B (LDS) and C
    auto a_lds = /* make_tile_window on A LDS scratch */;
    auto b_lds = /* make_tile_window on B LDS scratch */;
    decltype(make_c_block_tile()) c_reg{};  // C accumulator in VGPR
    clear_tile(c_reg);

    const index_t num_k = K / KPerBlock;

    // ---- prologue: preload first K-tile ----
    load_tile(a_lds, a_dram_win);
    load_tile(b_lds, b_dram_win);
    block_sync_lds();                                  // __syncthreads on LDS

    // ---- software-pipelined hot loop (double buffer + prefetch) ----
    for (index_t k = 0; k < num_k - 1; ++k) {
        a_dram_win = move_tile_window(a_dram_win, {0, KPerBlock});
        b_dram_win = move_tile_window(b_dram_win, {0, KPerBlock});
        async_load_tile(a_lds_next, a_dram_win);       // buffer_load -> LDS for next iter
        async_load_tile(b_lds_next, b_dram_win);
        block_gemm(c_reg, a_lds, b_lds);               // MFMA on current tile (overlaps the load)
        block_sync_lds();
        swap(a_lds, a_lds_next);  swap(b_lds, b_lds_next);
    }
    block_gemm(c_reg, a_lds, b_lds);                   // drain last tile

    // ---- epilogue: CShuffle re-tile + fused op + coalesced store ----
    CShuffleEpilogue{}(c_g, c_reg, M, N);
}
```

Build & run (gfx942):

```bash
cd composable_kernel && mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ gfx942
make tile_example_gemm_basic -j
./tile_example_gemm_basic -m 4096 -n 4096 -k 4096 -v 1
# universal GEMM example:
make tile_example_universal_gemm -j
./bin/tile_example_universal_gemm -m=4096 -n=4096 -k=4096 -v=0
```

> **Perf reality check (Issue #1727):** at 4096³ bf16, ck_tile `universal_gemm` measured ~0.382 ms /
> 359 TFLOP/s vs classic-CK `DeviceGemmXdlUniversal` v3 at ~0.223 ms / 615 TFLOP/s — same 256×256×64 tile.
> ck_tile's strength today is **fusion & attention/MoE**, not raw square dense GEMM. Always benchmark
> ck_tile GEMM against the classic v3 instance before shipping it as your dense path.

---

## 3. FMHA (FlashAttention-2) with ck_tile

ck_tile is the **production** FMHA on MI300X (`example/ck_tile/01_fmha`, kernels `fmha_fwd` / `fmha_bwd`,
incl. **paged-KV** for decode). The forward kernel maps FlashAttention-2 one-to-one onto tiles.

| FA-2 step | ck_tile mechanism |
|---|---|
| S = Q·Kᵀ | `gemm0` (a BlockGemm pipeline) producing S tile in registers |
| m = rowmax(S) | `block_tile_reduce` (max) across the distribution |
| P = exp(S − m) | `sweep_tile` lambda over the per-lane Y elements |
| ℓ = rowsum(P) + correction | `block_tile_reduce` (sum) + running-stat rescale |
| O = P·V (+ rescale prev O) | `gemm1` BlockGemm; O accumulator rescaled by `exp(m_prev−m)` |

Pipeline variants (swappable into `fmha_fwd_kernel`):

| Pipeline | dataflow | best for |
|---|---|---|
| `qr_ks_vs` | Q in **r**egisters, K/V streamed via **s**mem | general prefill |
| `qr_ks_vs_async` | + `buffer_load` async K/V direct-to-LDS | latency-hidden prefill (MI300X default) |
| paged-KV variants | KV gathered through a block/page table | **decode** with paged KV-cache (sglang/vLLM) |

Kernel construction mirrors GEMM — make Q/K/V/O DRAM tile windows, run gemm0 → softmax → gemm1,
epilogue stores O:

```cpp
auto q_win = make_tile_window(q_view, make_tuple(kM0, kK0), {i_m, 0}, q_dist);
auto k_win = make_tile_window(k_view, make_tuple(kN0, kK0), {0, 0},  k_dist);
auto v_win = make_tile_window(v_view, make_tuple(kN1, kK1), {0, 0},  v_dist);
// loop over KV tiles: S=gemm0(Q,K); m=reduce_max(S); P=exp(S-m);
//                     l=reduce_sum(P); O = O*scale + gemm1(P,V);
```

Tuning knobs that matter on MI300X for FMHA: head-dim tile (`kK0`/`kK1` = 64/128), the
`qr_ks_vs_async` vs sync pipeline, `kM0` (Q rows per block, 64/128), causal masking specialization,
and the page-size for paged-KV. fp8 KV-cache uses the fp8 WarpGemm + per-tile scale.

---

## 4. ck_tile vs classic CK — when to use which

| dimension | classic CK (`composable_kernel.md`) | ck_tile (this file) |
|---|---|---|
| surface | nested tensor-descriptor transforms | tile windows + distributions |
| authoring | hard to read/extend | CuTe-like, much more legible |
| dense square GEMM (bf16) | **currently faster** (v3 Intrawave, 615 TFLOP/s @4096³) | ~1.7× slower in #1727 — improving |
| FMHA / attention | legacy softmax-GEMM | **the production path** (paged-KV decode/prefill) |
| fused-MoE / fp8 / mxfp4 | grouped-GEMM instances | active new development, easier fusion |
| epilogue fusion (bias/act/residual) | possible but verbose | natural via the Epilogue template |
| pipeline swap | recompile a new instance | drop in a different `GemmPipeline*` type |

**Decision:** dense bf16 GEMM → benchmark classic-CK v3 first. Attention, paged-KV decode, fp8/MoE
fusion, or any custom fused op → ck_tile.

---

## 5. Scheduler / policy tuning cheat-sheet (gfx942)

| knob | where | values | effect |
|---|---|---|---|
| pipeline version | `GemmPipeline*V3/V4`, fmha `*_async` | V1/V3/V4 | prefetch depth & overlap |
| block tile | `GemmPipelineProblem` | 256×256×64 (prefill), 128×128, 64×256 (skinny) | reuse vs occupancy |
| WarpGemm (MFMA) | policy `GetWarpGemm()` | 32×32×8 (bf16), 16×16×16, 16×16×32 / 32×32×16 (fp8) | matrix-core tile |
| AK1/BK1 vector | DRAM distribution | 8 (bf16), 16 (fp8) | global-load width; must match alignment |
| LDS swizzle | LDS block descriptor (`make_xor_transform`) | on | kills LDS bank conflicts |
| CShuffle vec | epilogue | 8 (bf16) | coalesced C store width |
| async load | `async_load_tile` / `*_async` pipeline | on | direct global→LDS (DGL), no VGPR stage |
| occupancy target | TilePartitioner tile sizes | blocks ≈ k·304 | fill all MI300X CUs, avoid tail |

---

## Sources
- Hands-On with CK-Tile: build & run optimized GEMM on AMD GPUs (ROCm Blog, Apr 2025 — WarpGemm struct, pipeline/policy, gfx942 build): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- From Theory to Kernel: FlashAttention-v2 with CK-Tile (ROCm Blog — fmha pipeline mapping, qr_ks_vs): https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ck_tile Tile Window (data access gateway) concept doc: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_window.html
- ck_tile Tensor Views concept doc: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tensor_views.html
- ck_tile Sweep Tile (per-lane Y iteration): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/sweep_tile.html
- Block GEMM optimization on MI300 with CK Tile (LDS sizing, pipeline stages, 256×256 / 304 CU): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- `include/ck_tile/README.md` (component layout: core/host/ops/gemm/ops/fmha): https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md
- Issue #1727 — ck_tile universal_gemm vs classic CK v3 perf gap: https://github.com/ROCm/composable_kernel/issues/1727
- ck_tile space-filling curve / CShuffle traversal doc: https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/space_filling_curve.html
