Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

::: {.meta description="C++ library for accelerating mixed precision matrix multiply-accumulate operations
leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
:keywords: rocWMMA, ROCm, library, API, tool"}
:::

# API reference guide

This document provides information about rocWMMA functions, data types, and other programming constructs.

## Synchronous API

In general, rocWMMA API functions ( `load_matrix_sync`, `store_matrix_sync`, `mma_sync` ) are assumed to be synchronous when
used in the context of global memory.

When using these functions in the context of shared memory (e.g. LDS memory), additional explicit workgroup synchronization (`synchronize_workgroup`)
may be required due to the nature of this memory usage.

## Supported GPU architectures

List of supported CDNA architectures (wave64):

- gfx908
- gfx90a
- gfx940
- gfx941
- gfx942

:::: note
::: title
Note
:::

gfx9 = gfx908, gfx90a, gfx940, gfx941, gfx942

gfx940+ = gfx940, gfx941, gfx942
::::

List of supported RDNA architectures (wave32):

- gfx1100
- gfx1101
- gfx1102

:::: note
::: title
Note
:::

gfx11 = gfx1100, gfx1101, gfx1102
::::

## Supported data types

rocWMMA mixed precision multiply-accumulate operations support the following data type combinations.

Data Types **\<Ti / To / Tc\>** = \<Input type / Output Type / Compute Type\>, where:

- Input Type = Matrix A / B
- Output Type = Matrix C / D
- Compute Type = Math / accumulation type
- i8 = 8-bit precision integer
- f8 = 8-bit precision floating point
- bf8 = 8-bit precision brain floating point
- f16 = half-precision floating point
- bf16 = half-precision brain floating point
- f32 = single-precision floating point
- i32 = 32-bit precision integer
- xf32 = single-precision tensor floating point
- f64 = double-precision floating point

:::: note
::: title
Note
:::

f16 represents equivalent support for both [Float16]{#float16} and \_\_half types.

Current f8 support is NANOO (optimized) format.
::::

| Ti / To / Tc | BlockM | BlockN | BlockK Range* | CDNA Support | RDNA Support |
|--------------|--------|--------|---------------|--------------|--------------|
| bf8 / f32 / f32 | 16 | 16 | 32+ | gfx940+ | - |
| bf8 / f32 / f32 | 32 | 32 | 16+ | gfx940+ | - |
| f8 / f32 / f32 | 16 | 16 | 32+ | gfx940+ | - |
| f8 / f32 / f32 | 32 | 32 | 16+ | gfx940+ | - |
| i8 / i32 / i32 | 16 | 16 | 16+ | gfx908, gfx90a | gfx11 |
| i8 / i32 / i32 | 16 | 16 | 32+ | gfx940+ | - |
| i8 / i32 / i32 | 32 | 32 | 8+ | gfx908, gfx90a | - |
| i8 / i32 / i32 | 32 | 32 | 16+ | gfx940+ | - |
| i8 / i8 / i32 | 16 | 16 | 16+ | gfx908, gfx90a | gfx11 |
| i8 / i8 / i32 | 16 | 16 | 32+ | gfx940+ | - |
| i8 / i8 / i32 | 32 | 32 | 8+ | gfx908, gfx90a | - |
| i8 / i8 / i32 | 32 | 32 | 16+ | gfx940+ | - |
| f16 / f32 / f32 | 16 | 16 | 16+ | gfx9 | gfx11 |
| f16 / f32 / f32 | 32 | 32 | 8+ | gfx9 | - |
| f16 / f16 / f32 | 16 | 16 | 16+ | gfx9 | gfx11 |
| f16 / f16 / f32 | 32 | 32 | 8+ | gfx9 | - |
| f16 / f16 / f16** | 16 | 16 | 16+ | gfx9 | gfx11 |
| f16 / f16 / f16** | 32 | 32 | 8+ | gfx9 | - |
| bf16 / f32 / f32 | 16 | 16 | 8+ | gfx908 | - |
| bf16 / f32 / f32 | 16 | 16 | 16+ | gfx90a, gfx940+ | gfx11 |
| bf16 / f32 / f32 | 32 | 32 | 4+ | gfx908 | - |
| bf16 / f32 / f32 | 32 | 32 | 8+ | gfx90a, gfx940+ | - |
| bf16 / bf16 / f32 | 16 | 16 | 8+ | gfx908 | - |
| bf16 / bf16 / f32 | 16 | 16 | 16+ | gfx90a, gfx940+ | gfx11 |
| bf16 / bf16 / f32 | 32 | 32 | 4+ | gfx908 | - |
| bf16 / bf16 / f32 | 32 | 32 | 8+ | gfx90a, gfx940+ | - |
| bf16 / bf16 / bf16** | 16 | 16 | 8+ | gfx908 | - |
| bf16 / bf16 / bf16** | 16 | 16 | 16+ | gfx90a, gfx940+ | gfx11 |
| bf16 / bf16 / bf16** | 32 | 32 | 4+ | gfx908 | - |
| bf16 / bf16 / bf16** | 32 | 32 | 8+ | gfx90a, gfx940+ | - |
| f32 / f32 / f32 | 16 | 16 | 4+ | gfx9 | - |
| f32 / f32 / f32 | 32 | 32 | 2+ | gfx9 | - |
| xf32 / xf32 / xf32 | 16 | 16 | 8+ | gfx940+ | - |
| xf32 / xf32 / xf32 | 32 | 32 | 4+ | gfx940+ | - |
| f64 / f64 / f64 | 16 | 16 | 4+ | gfx90a, gfx940+ | - |

### Data Type Support Summary (Natural Language Description)

The following describes rocWMMA supported data type combinations in natural language for better searchability:

**8-bit Floating Point Types (bf8, f8):**
- **bf8/f32/f32**: Brain float 8-bit input with float32 output and compute. Supports BlockM×BlockN of 16×16 (BlockK≥32) or 32×32 (BlockK≥16). Only available on CDNA gfx940+ architectures (MI300 series). Not supported on RDNA.
- **f8/f32/f32**: Float 8-bit input with float32 output and compute. Same block size support as bf8. Only available on CDNA gfx940+ architectures. Not supported on RDNA.

**8-bit Integer Types (i8):**
- **i8/i32/i32**: Int8 input with int32 output and compute. On gfx908/gfx90a: supports 16×16 (BlockK≥16) and 32×32 (BlockK≥8). On gfx940+: supports 16×16 (BlockK≥32) and 32×32 (BlockK≥16). RDNA gfx11 supports 16×16 only.
- **i8/i8/i32**: Int8 input and output with int32 compute. Same architecture support as i8/i32/i32.

**Half Precision Types (f16):**
- **f16/f32/f32**: Half precision input with float32 output and compute. Supports 16×16 (BlockK≥16) on all gfx9 CDNA and gfx11 RDNA. Supports 32×32 (BlockK≥8) on gfx9 CDNA only.
- **f16/f16/f32**: Half precision input and output with float32 compute. Same support as f16/f32/f32.
- **f16/f16/f16**: Full half precision pipeline (note: CDNA internally uses 32-bit accumulation, then converts). Same support as f16/f32/f32.

**Brain Float 16 Types (bf16):**
- **bf16/f32/f32**: BFloat16 input with float32 output and compute. On gfx908: 16×16 (BlockK≥8), 32×32 (BlockK≥4). On gfx90a/gfx940+: 16×16 (BlockK≥16), 32×32 (BlockK≥8). RDNA gfx11 supports 16×16 (BlockK≥16).
- **bf16/bf16/f32**: BFloat16 input and output with float32 compute. Same architecture support as bf16/f32/f32.
- **bf16/bf16/bf16**: Full BFloat16 pipeline (note: CDNA internally uses 32-bit accumulation). Same support as bf16/f32/f32.

**Single Precision Types (f32, xf32):**
- **f32/f32/f32**: Full single precision pipeline. Supports 16×16 (BlockK≥4) and 32×32 (BlockK≥2) on all gfx9 CDNA. Not supported on RDNA.
- **xf32/xf32/xf32**: Tensor float32 format. Supports 16×16 (BlockK≥8) and 32×32 (BlockK≥4). Only available on gfx940+ (MI300 series). Not supported on RDNA.

**Double Precision Types (f64):**
- **f64/f64/f64**: Full double precision pipeline. Supports 16×16 (BlockK≥4) only. Available on gfx90a (MI200 series) and gfx940+ (MI300 series). Not supported on RDNA.

:::: note
::: title
Note
:::

\* = BlockK range lists the minimum possible value. Other values in the range are powers of 2 larger than the minimum. Practical BlockK values are usually 32 and smaller.

\*\* = CDNA architectures matrix unit accumulation is natively 32-bit precision and is converted to the desired type.
::::

## Supported matrix layouts

(N = col major, T = row major)

| LayoutA | LayoutB | LayoutC | LayoutD |
|---------|---------|---------|---------|
| N | N | N | N |
| N | N | T | T |
| N | T | N | N |
| N | T | T | T |
| T | N | N | N |
| T | N | T | T |
| T | T | N | N |
| T | T | T | T |

### Matrix Layout Summary (Natural Language Description)

rocWMMA supports all combinations of row-major (T) and column-major (N) layouts for the four matrices involved in matrix multiply-accumulate operations:

- **Matrix A (LayoutA)**: The first input matrix, can be either column-major (N) or row-major (T).
- **Matrix B (LayoutB)**: The second input matrix, can be either column-major (N) or row-major (T).
- **Matrix C (LayoutC)**: The input accumulator matrix, can be either column-major (N) or row-major (T).
- **Matrix D (LayoutD)**: The output result matrix, always matches the layout of Matrix C.

**Key observations:**
- LayoutC and LayoutD are always the same (both N or both T).
- All 8 combinations of LayoutA × LayoutB × LayoutC/D are supported.
- For GEMM operations computing D = A × B + C, you can use any combination of input layouts.
- Column-major (N) is typically more efficient for certain access patterns on AMD GPUs.

## Supported thread block sizes

rocWMMA generally supports and tests up to 4 wavefronts per threadblock. The X dimension is expected to be a multiple of the wave size and will be scaled as such.

| TBlock_X | TBlock_Y | Total Threads (CDNA) | Total Threads (RDNA) |
|----------|----------|----------------------|----------------------|
| WaveSize | 1 | 64 | 32 |
| WaveSize | 2 | 128 | 64 |
| WaveSize | 4 | 256 | 128 |
| WaveSize×2 | 1 | 128 | 64 |
| WaveSize×2 | 2 | 256 | 128 |
| WaveSize×4 | 1 | 256 | 128 |

**Note:** WaveSize (RDNA) = 32, WaveSize (CDNA) = 64

### Thread Block Size Summary (Natural Language Description)

rocWMMA supports up to 4 wavefronts per thread block. Here's how to configure thread block dimensions:

**For CDNA architectures (MI100, MI200, MI300 series):**
- WaveSize = 64 threads per wavefront
- Supported configurations:
  - 64×1 = 64 threads (1 wavefront)
  - 64×2 = 128 threads (2 wavefronts)
  - 64×4 = 256 threads (4 wavefronts)
  - 128×1 = 128 threads (2 wavefronts)
  - 128×2 = 256 threads (4 wavefronts)
  - 256×1 = 256 threads (4 wavefronts)

**For RDNA architectures (RX 7000 series):**
- WaveSize = 32 threads per wavefront
- Supported configurations:
  - 32×1 = 32 threads (1 wavefront)
  - 32×2 = 64 threads (2 wavefronts)
  - 32×4 = 128 threads (4 wavefronts)
  - 64×1 = 64 threads (2 wavefronts)
  - 64×2 = 128 threads (4 wavefronts)
  - 128×1 = 128 threads (4 wavefronts)

**Best practices:**
- The X dimension should always be a multiple of WaveSize for optimal performance.
- Using 4 wavefronts (maximum supported) typically provides better occupancy and latency hiding.
- For memory-bound kernels, consider using larger thread blocks to improve cache utilization.

## Using rocWMMA API

This section describes how to use the rocWMMA library API.

## rocWMMA datatypes

### matrix_a

::: doxygenstruct
rocwmma::matrix_a
:::

### matrix_b

::: doxygenstruct
rocwmma::matrix_b
:::

### accumulator

::: doxygenstruct
rocwmma::accumulator
:::

### row_major

::: doxygenstruct
rocwmma::row_major
:::

### col_major

::: doxygenstruct
rocwmma::col_major
:::

### fragment

::: {.doxygenclass members=""}
rocwmma::fragment
:::

## rocWMMA enumeration

### layout_t

::: doxygenenum
rocwmma::layout_t
:::

## rocWMMA API functions

::: doxygenfunction
rocwmma::fill_fragment
:::

::: doxygenfunction
rocwmma::load_matrix_sync(fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\>& frag, const DataT\* data, uint32_t ldm)
:::

::: doxygenfunction
rocwmma::load_matrix_sync(fragment\<MatrixT, BlockM, BlockN, BlockK, DataT\>& frag, const DataT\* data, uint32_t ldm, layout_t layout)
:::

::: doxygenfunction
rocwmma::store_matrix_sync(DataT\* data, fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\> const& frag, uint32_t ldm)
:::

::: doxygenfunction
rocwmma::store_matrix_sync(DataT\* data, fragment\<MatrixT, BlockM, BlockN, BlockK, DataT\> const& frag, uint32_t ldm, layout_t layout)
:::

::: doxygenfunction
rocwmma::mma_sync
:::

::: doxygenfunction
rocwmma::synchronize_workgroup
:::

## rocWMMA cooperative API functions

::: doxygenfunction
rocwmma::load_matrix_coop_sync(fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\>& frag, const DataT\* data, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)
:::

::: doxygenfunction
rocwmma::load_matrix_coop_sync(fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\>& frag, const DataT\* data, uint32_t ldm)
:::

::: doxygenfunction
rocwmma::load_matrix_coop_sync(fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\>& frag, const DataT\* data, uint32_t ldm, uint32_t waveIndex)
:::

::: doxygenfunction
rocwmma::store_matrix_coop_sync(DataT\* data, fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\> const& frag, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)
:::

::: doxygenfunction
rocwmma::store_matrix_coop_sync(DataT\* data, fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\> const& frag, uint32_t ldm)
:::

::: doxygenfunction
rocwmma::store_matrix_coop_sync(DataT\* data, fragment\<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT\> const& frag, uint32_t ldm, uint32_t waveIndex)
:::

### rocWMMA transforms API functions

::: doxygenfunction
rocwmma::applyTranspose(FragT &&frag)
:::

::: doxygenfunction
rocwmma::applyDataLayout(FragT &&frag)
:::

## Sample programs

See a sample code for calling rocWMMA functions `load_matrix_sync`, `store_matrix_sync`, `fill_fragment`, and `mma_sync` [here](https://github.com/ROCm/rocWMMA/blob/develop/samples/simple_hgemm.cpp).
For more such sample programs, refer to the [Samples directory](https://github.com/ROCm/rocWMMA/tree/develop/samples).

## Emulation tests

The emulation test is a smaller test suite specifically designed for emulators. It comprises a selection of test cases from the full ROCWMM test set, allowing for significantly faster execution on emulated platforms. Despite its concise nature, the emulation test supports `smoke`, `regression`, and `extended` modes.

For example, run a smoke test.

``` bash
rtest.py --install_dir <build_dir> --emulation smoke
```
