# Composable Kernel (CK) — Classic Programming Model (MI300X / CDNA3 / gfx942)

> Scope: the **classic CK** programming model (`include/ck`, the `DeviceGemm*` / `DeviceBatchedGemm*` /
> fMHA device-op family, the instance/template system, and the XDL blockwise-GEMM pipelines v1–v5).
> ck_tile (the newer tile-based front-end in `include/ck_tile`) is covered in `ck_tile.md`. Everything
> here is for **CDNA3 / gfx942 (MI300X, MI300A, MI325X)**, with gfx950 (CDNA4) notes where they differ.
>
> Repo note (2025+): `ROCm/composable_kernel` is now a **read-only mirror**; active development moved into
> the monorepo `ROCm/rocm-libraries` (`projects/composablekernel/`). Paths below are given relative to the
> CK source root (`include/ck/...`, `example/...`) which is identical in both layouts.

CK is a **tensor-coordinate-transform + tile** library. The entire library is built on one idea:
describe data movement as a **composition of coordinate transforms** on a tensor descriptor, and let the
compiler fuse the index math into the load/store address calculation at compile time. There is *no*
runtime index arithmetic in a well-written CK kernel — every transform is `constexpr`.

---

## 1. The descriptor hierarchy (mental model)

CK structures a kernel as a strict hierarchy. Each level owns a *tile* of the level above and a
*descriptor* that maps its local coordinates into the parent's address space.

| Level | CK object | owns | typical MI300X size |
|---|---|---|---|
| **Grid** | `GridwiseGemm_xdl_cshuffle_v3` | whole C tensor | M×N |
| **Block / workgroup** | `BlockwiseGemmXdlops_pipeline_vX` | `MPerBlock × NPerBlock` C tile | 256×256 |
| **Wave / warp** | `WaveGemm` (XDL warp tile) | `MPerXDL × NPerXDL` repeated | 32×32 ×(MRepeat×NRepeat)=4×4 |
| **Thread / lane** | MFMA lane fragment | per-lane VGPR fragment | e.g. 4 acc regs |

The block tile is laid out across waves by **WaveMap** (`MRepeat × NRepeat`, e.g. `4×4` → 16 waves =
4 SIMDs × 4 — but CK uses **256 threads = 4 waves**, and each wave issues `MRepeat×NRepeat` MFMAs).
Watch the terminology: `MXdlPerWave`/`NXdlPerWave` (a.k.a. `MRepeat`/`NRepeat`) is how many MFMA tiles
**one wave** computes, not how many waves exist.

### 1.1 Tensor descriptors and coordinate transforms

A `TensorDescriptor` is a `constexpr` object built by composing transforms. The primitives:

| Transform | meaning |
|---|---|
| `make_naive_tensor_descriptor(lengths, strides)` | raw strided view of global memory |
| `make_naive_tensor_descriptor_packed(lengths)` | contiguous (strides implied) |
| `transform_tensor_descriptor(desc, transforms, lower_dims, upper_dims)` | the workhorse: re-index |
| `make_merge_transform({d0,d1})` | fuse two dims into one (e.g. flatten K0,K1 → K) |
| `make_unmerge_transform({d0,d1})` | split one dim into two (tile a dimension) |
| `make_pass_through_transform(len)` | identity on a dim |
| `make_pad_transform(len, left, right)` | pad for alignment / guard OOB |
| `make_xor_transform(...)` | LDS swizzle to kill bank conflicts |

Example — tile the K dimension of a row-major A (M×K) into `K0 × KPerBlock` for the block loop:

```cpp
// A is M x K, row-major. Split K = K0 * K1 (K1 = KPerBlock).
constexpr auto a_grid_desc_m_k =
    make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(K, I1)); // strides {K,1}

constexpr auto a_grid_desc_k0_m_k1 = transform_tensor_descriptor(
    a_grid_desc_m_k,
    make_tuple(make_unmerge_transform(make_tuple(K0, K1)),  // K -> (K0, K1)
               make_pass_through_transform(M)),
    make_tuple(Sequence<1>{}, Sequence<0>{}),               // lower dims of input
    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));           // upper dims (K0,M,K1)
```

The crucial property: `a_grid_desc_k0_m_k1.CalculateOffset({k0,m,k1})` compiles to a handful of
integer ops with the K/K0/K1 constants folded — the transform chain is *erased*. This is why CK GEMMs
hit hipBLASLt-class throughput without a tuned assembly kernel per shape.

### 1.2 The `*_xdl_cshuffle` epilogue

`CShuffle` is CK's epilogue strategy: the MFMA accumulator lane layout (scattered 4×4 sub-blocks) is
**not** coalescable for the global store. CShuffle stages C through **LDS** to re-tile it into a
vectorized, coalesced layout before the epilogue elementwise op and the global write. Knobs:
`CShuffleMXdlPerWavePerShuffle`, `CShuffleNXdlPerWavePerShuffle`,
`CShuffleBlockTransferScalarPerVector_NPerBlock` (the store vector width, usually 8 for bf16).

---

## 2. Device ops — the public API surface

A *device op* is the user-facing object. You pick a templated instance, query whether it supports your
problem, make an argument, and `Run` it on a stream. The contract is uniform across op families.

| Device op | header | use |
|---|---|---|
| `DeviceGemm<...>` (base) | `device/device_gemm.hpp` | generic GEMM interface |
| `DeviceGemm_Xdl_CShuffle<...>` | `device/impl/device_gemm_xdl_cshuffle.hpp` | older XDL GEMM (v1/v2 pipeline) |
| `DeviceGemmXdlUniversal<...>` | `device/impl/device_gemm_xdl_universal.hpp` | **the current** unified XDL GEMM (v3+) |
| `DeviceBatchedGemmXdl<...>` | `device/impl/device_batched_gemm_xdl.hpp` | batched GEMM |
| `DeviceGroupedGemm*` | `device/impl/device_grouped_gemm_*.hpp` | variable-shape / MoE grouped GEMM |
| `DeviceGemmMultipleD*` | `device/impl/device_gemm_multiple_d_*.hpp` | GEMM + multiple D (bias/residual fuse) |
| `DeviceBatchedGemmSoftmaxGemm*` / fmha | `tensor_operation/.../device_*` (+ `example/ck_tile/01_fmha`) | attention |

### 2.1 The five-call lifecycle

Every device op follows the same sequence:

```cpp
using DeviceOp = ck::tensor_operation::device::DeviceGemmXdlUniversal<
    Row, Col, Row,                         // A=row, B=col, C=row  (this is "RCR")
    BF16, BF16, BF16,                      // ADataType, BDataType, CDataType
    F32, BF16,                             // AccDataType (=fp32), CShuffleDataType
    PassThrough, PassThrough, PassThrough, // A/B/C elementwise ops
    GemmDefault,                           // GemmSpecialization (padding policy)
    /* BlockSize */ 256,
    /* MPerBlock,NPerBlock,KPerBlock */ 256, 256, 64,
    /* AK1,BK1 */ 8, 8,                    // global-load vector width along K
    /* MPerXDL,NPerXDL */ 32, 32,          // the MFMA tile (32x32x8 bf16)
    /* MXdlPerWave,NXdlPerWave */ 4, 4,    // WaveMap 4x4 -> 256x256 per block
    /* ABlockTransfer... */ S<8,32,1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, 0,
    /* BBlockTransfer... */ S<8,32,1>, S<1,0,2>, S<1,0,2>, 2, 8, 8, 0,
    /* CShuffle MXdl,NXdl */ 1, 1,
    /* CShuffleBlockTransfer */ S<1,32,1,8>, 8,
    BlockGemmPipelineScheduler::Intrawave, // scheduler
    BlockGemmPipelineVersion::v3,          // pipeline version
    BF16, BF16>;                           // compute types

auto op   = DeviceOp{};
auto arg  = op.MakeArgument(a_ptr, b_ptr, c_ptr, M, N, K,
                            /*StrideA*/ K, /*StrideB*/ K, /*StrideC*/ N,
                            /*KBatch*/ 1, PassThrough{}, PassThrough{}, PassThrough{});
if(!op.IsSupportedArgument(arg))          // (1) capability check — ALWAYS gate on this
    throw std::runtime_error("instance does not support this shape/alignment");
auto invoker = op.MakeInvoker();
float ms = invoker.Run(arg, StreamConfig{stream, /*time_kernel*/ true}); // timed launch
```

`IsSupportedArgument` is the heart of instance selection: it checks M/N/K divisibility against the
tile, K against `KPerBlock × KBatch`, pointer alignment against the vector widths (`AK1/BK1`), and
layout/spec compatibility. **An instance that returns `false` will produce garbage if forced** — never
skip the check.

### 2.2 Layout shorthand (R/C)

CK names instances by the A/B/C layouts: `R`=row-major, `C`=col-major.

| code | A | B | C | typical use |
|---|---|---|---|---|
| **RCR** | row | col | row | the standard `x · Wᵀ` linear layer (W stored KxN col = NxK row) |
| **RRR** | row | row | row | A·B both row-major |
| **CRR** | col | row | row | rare |

For an `nn.Linear` (`Y = X·Wᵀ`), X is row-major (M×K) and W is row-major (N×K), which is **RCR** — the
most-tuned layout in CK's instance DB.

---

## 3. The blockwise-GEMM pipelines (v1–v5)

The "pipeline" is the **hot K-loop scheduler**: how global→LDS prefetch, LDS→VGPR reads, and MFMA
issue are interleaved to hide latency. Selected via two template params on the universal GEMM:
`BlockGemmPipelineScheduler` and `BlockGemmPipelineVersion`. Implemented in
`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_vX.hpp`.

### 3.1 Scheduler: Intrawave vs Interwave

| Scheduler | mechanism | best when |
|---|---|---|
| **Intrawave** | one wave's own loads and MFMAs are software-pipelined / interleaved (`HotLoopScheduler()` issues explicit `s_setprio` + scheduling barriers to overlap `ds_read`/`buffer_load` with `v_mfma`) | **compute-bound** large GEMM, high VGPR pressure tolerated; the MI300X default for prefill |
| **Interwave** | latency hidden by *switching to another wave* on the SIMD (classic GPU occupancy hiding); fewer prefetch buffers, lower VGPR | **memory-bound** / skinny / low-occupancy or when Intrawave spills |

### 3.2 Pipeline version semantics

| Version | prefetch stages | LDS buffering | character | when to pick |
|---|---|---|---|---|
| **v1** | 1 | single | simplest, lowest VGPR; global→LDS→compute serialized per stage | small/odd shapes, debugging, low occupancy |
| **v2** | 1–2 | single, better SW pipeline | improved overlap over v1 | mid shapes |
| **v3** | **2** | double-buffer LDS, async-ish prefetch | **the workhorse** for large compute-bound bf16/fp16 GEMM | default for prefill matmuls on MI300X |
| **v4** | ≥2 | + ping-pong / extra prefetch | very large K, max overlap, needs VGPR headroom | huge-K compute-bound |
| **v5** | tuned | newest, persistent / async-input variants | latest LLM GEMM/MoE paths | bleeding edge; benchmark vs v3 |

A real winning instance string emitted by CK's profiler for **bf16 4096×4096×4096 RCR on MI300X**:

```
DeviceGemmXdlUniversal<Default, RCR>
  BlkSize: 256   BlkTile: 256x256x64   WaveTile(MPerXDL,NPerXDL): 32x32   WaveMap: 4x4
  VmemReadVec(AK1,BK1): 8x8
  BlkGemmPipelineScheduler: Intrawave   BlkGemmPipelineVersion: v3   PrefetchStages: 2
  => 0.223 ms, 615 TFLOP/s, 451 GB/s
```

> **Perf gotcha (real, Issue #1727):** the *ck_tile* `universal_gemm` for the same 4096³ case ran ~0.382 ms
> / 359 TFLOP/s — ~1.7× slower than this classic-CK v3 instance — at identical tile size. **Lesson:** for
> compute-bound dense GEMM today, classic-CK `DeviceGemmXdlUniversal` v3/Intrawave is often still the
> stronger baseline; benchmark before assuming ck_tile wins. (See `ck_tile.md` for where ck_tile pulls ahead.)

---

## 4. The instance / template system

CK ships **pre-instantiated** device ops so you don't compile a fresh template per call. Two layers:

1. **Instance factory headers** — `library/src/tensor_operation_instance/gpu/gemm*/...` define lists of
   concrete `DeviceGemmXdlUniversal<...>` specializations (different tiles/pipelines), registered via
   `add_device_gemm_xdl_universal_*_instances(...)`.
2. **`DeviceOperationInstanceFactory`** — at runtime you query all registered instances for a layout/dtype:

```cpp
using namespace ck::tensor_operation::device;
using DeviceOpPtr = DeviceGemmPtr<Row,Col,Row, BF16,BF16,BF16, PassThrough,PassThrough,PassThrough>;

std::vector<DeviceOpPtr> ops;
ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceGemm<Row,Col,Row, BF16,BF16,BF16, PassThrough,PassThrough,PassThrough>>::GetInstances(ops);

// Sweep: keep the fastest instance that supports the argument (this is "Tier-B" CK autotune).
float best = 1e30f; DeviceOpPtr* winner = nullptr;
for(auto& op : ops) {
    auto arg = op->MakeArgumentPointer(a,b,c,M,N,K,K,K,N,1,PassThrough{},PassThrough{},PassThrough{});
    if(!op->IsSupportedArgument(arg.get())) continue;        // skip incompatible
    float ms = op->MakeInvokerPointer()->Run(arg.get(), StreamConfig{nullptr, true});
    if(ms < best) { best = ms; winner = &op; }
}
```

This loop *is* how `ckProfiler` and the CK fallback in frameworks pick a config. To tune a fixed LLM
shape: run the sweep once offline, record the winning instance index, and pin it.

### 4.1 Tuning knobs (ranked by impact on MI300X)

| Knob | values | effect | priority |
|---|---|---|---|
| `MPerBlock × NPerBlock` | 256×256, 256×128, 128×128, 128×64, 64×64 | block tile; bigger = more reuse, fewer blocks (occupancy/tail trade-off) | ★★★★★ |
| `KPerBlock` | 32, 64, 128 | K-loop tile; bigger = better MFMA/load overlap, more LDS+VGPR | ★★★★ |
| `BlockGemmPipelineVersion` | v1/v2/**v3**/v4/v5 | hot-loop schedule | ★★★★ |
| `BlockGemmPipelineScheduler` | **Intrawave** / Interwave | overlap strategy | ★★★★ |
| `MPerXDL × NPerXDL` | **32×32**, 16×16 | MFMA tile; 32×32 = best for large M,N; 16×16 for skinny | ★★★ |
| `MXdlPerWave × NXdlPerWave` | 4×4, 4×2, 2×2, … | waves-per-tile mapping; drives VGPR & occupancy | ★★★ |
| `AK1 / BK1` | 8 (bf16), 16 (fp8), 4 | global-load vector width; must match alignment | ★★★ |
| `GemmSpecialization` | Default / MNKPadding / MNPadding | pad guards for non-divisible shapes (small perf cost) | ★★ |
| `KBatch` (split-K) | 1, 2, 4, 8 | atomic split of K across blocks → fills CUs for **small-M decode** | ★★★ (decode) |

**Heuristics for LLM shapes on MI300X (304 CUs):**
- **Prefill (large M):** 256×256×64, MPerXDL=NPerXDL=32, WaveMap 4×4, v3, Intrawave, AK1=BK1=8.
- **Decode (M = batch ≪ N,K):** small M tile (e.g. 16/32×256), **split-K (`KBatch`≥2)** to occupy CUs,
  Interwave often wins, 16×16 MFMA. The 256×256 tile leaves most CUs idle when M is tiny.
- Aim for `ceil(M/MPerBlock)·ceil(N/NPerBlock) ≈ k·304` to avoid a wave-quantization tail.

---

## 5. Batched GEMM, grouped GEMM (MoE), and fMHA

- **`DeviceBatchedGemmXdl`** — same template + a batch stride; one grid dim for batch. Used for
  multi-head projection stacks.
- **`DeviceGroupedGemm` / MoE** — handles a *vector of* (M,N,K) problems with different M (expert token
  counts). `MakeArgument` takes arrays of pointers/strides; one kernel sweeps all groups. This is the CK
  path behind fused-MoE on MI300X (`device_grouped_gemm_xdl_*`, plus `b_scale`/`mx` variants for
  fp8/mxfp4 weights, see §6).
- **fMHA (attention)** — classic CK had `DeviceBatchedGemmSoftmaxGemm*`, but the *production* FMHA is
  now ck_tile (`example/ck_tile/01_fmha`, `fmha_fwd`/`fmha_bwd`, paged-KV). Use the ck_tile FMHA for
  MI300X prefill/decode attention; see `ck_tile.md`. Classic-CK softmax-GEMM is effectively legacy.

---

## 6. Low-precision: fp8, mxfp4, and `*_b_scale` instances

MI300X adds native fp8 (OCP **and** the gfx942-only "NANOO"/fnuz format) MFMA. CK exposes:

| Variant | what | header hint |
|---|---|---|
| `..._fp8` instances | fp8 (e4m3/e5m2) A and/or B, fp32 acc | `gemm_xdl_universal_f8_*` |
| `..._b_scale` | per-tile dequant scale on B (weight-only fp8/int) | `blockwise_gemm_pipeline_xdlops_v3_b_scale.hpp` |
| `..._ab_scale` | scales on both A and B (full fp8 GEMM w/ scaling) | `..._v1_ab_scale`, `..._v3_ab_scale` |
| `..._mx_gemm` / `mx_gemm_bpreshuffle` | **microscaled** (mxfp8/mxfp4 block-scaled) GEMM | `device_moe_mx_gemm_bpreshuffle.hpp` |

For fp8 use `AK1=BK1=16` (fp8 packs 16 along K for the 16×16×32 / 32×32×16 fp8 MFMA), and prefer the
fp8-specific tiles (`KPerBlock` doubles vs bf16 since K-density doubles). `bpreshuffle` pre-permutes the
weight into the MFMA register layout at load time → faster decode GEMM for static weights. gfx950
(CDNA4) extends this with native **fp4/mxfp4** MFMA and larger K MFMA tiles; CK gates these behind
`DTYPES` build flags (e.g. `tf32`, `fp4`).

---

## 7. Building & profiling CK on gfx942

```bash
# build the dev tree + the profiler, targeting MI300X only (fast build)
cd composable_kernel && mkdir build && cd build
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -D GPU_TARGETS="gfx942" -D CMAKE_BUILD_TYPE=Release ..
make -j ckProfiler

# autotune a fixed bf16 RCR shape: layout=1(RCR)? see --help; args: prec a b c M N K strideA strideB strideC
./bin/ckProfiler gemm 1 1 1 0 1 1 4096 4096 4096 4096 4096 4096
#   -> prints every instance's TFLOP/s; the top line is your pinned config
```

Compile-time tips: restrict `GPU_TARGETS=gfx942` (CK otherwise builds for every gfx — huge), build only
the instance group you need (`make device_gemm_xdl_universal_f16_instance`), and use the
**CK-Tile dispatcher** (new unified codegen/arch-filter front-end, C++ & Python) to emit only the
instances your shapes require.

---

## 8. Quick decision guide

| situation | pick |
|---|---|
| dense bf16/fp16 prefill GEMM, large M·N·K | `DeviceGemmXdlUniversal` RCR, 256×256×64, 32×32 MFMA, **v3 Intrawave** |
| decode GEMM, M = batch (tiny) | small-M tile + **split-K (KBatch≥2)**, Interwave, 16×16 MFMA |
| fp8 weight-only (LLM linear) | `..._b_scale` fp8 instance, AK1/BK1=16, bpreshuffle weights |
| MoE | `DeviceGroupedGemm*` (+ `mx`/`b_scale` for low-precision experts) |
| attention | **ck_tile FMHA** (`fmha_fwd`, paged-KV) — not classic softmax-GEMM |
| non-divisible M/N/K | add `GemmSpecialization::MNKPadding` |
| instance returns `IsSupportedArgument()==false` | wrong alignment/divisibility — change tile or pad; never force |

---

## Sources
- CK GEMM optimization concept (MI300, CK Tile): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- CK User Guide / docs root (1.2.0): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/
- `BlockwiseGemmXdlops_pipeline_v1_ab_scale` (Intrawave) doxygen — pipeline template params: https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1_blockwise_gemm_xdlops__pipeline__v1__ab__scale_3_01_block_gemm_pipeline_scheduler_1f98d5cb27163c1a3364a8c8f61866821.html
- `DeviceGemm` device-op base struct (lifecycle: MakeArgument/IsSupportedArgument/MakeInvoker): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
- `device_moe_mx_gemm_bpreshuffle.hpp` source (mxfp4/fp8 MoE GEMM): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.1.1/doxygen/html/device__moe__mx__gemm__bpreshuffle_8hpp_source.html
- CK CHANGELOG (TF32 gfx942/gfx950, persistent async input scheduler, CK-Tile dispatcher): https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- Issue #1727 — ck_tile universal_gemm vs classic CK v3 perf (615 vs 359 TFLOP/s, instance string): https://github.com/ROCm/composable_kernel/issues/1727
- ROCm "Optimizing with Composable Kernel" how-to: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
