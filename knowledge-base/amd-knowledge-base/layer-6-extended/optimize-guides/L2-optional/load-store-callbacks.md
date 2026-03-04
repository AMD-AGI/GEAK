---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocFFT/en/latest/how-to/load-store-callbacks.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Load and Store Callbacks

rocFFT provides experimental functionality to invoke user-defined device functions during data transfer operations. These callbacks execute when reading input from global memory at the transform's start or writing output at its conclusion.

## Setting Up Callbacks

Users can register custom callback functions through:
- `rocfft_execution_info_set_load_callback()`
- `rocfft_execution_info_set_store_callback()`

**Important requirement:** Callback functions must be compiled as relocatable device code using the `-fgpu-rdc` compiler and linker flag.

## Data Type Requirements

The element types for load and store operations depend on the transform configuration:

| Transform Type | Load Type | Store Type |
|---|---|---|
| C2C half-precision | `_Float16_2` | `_Float16_2` |
| C2C single-precision | `float2` | `float2` |
| C2C double-precision | `double2` | `double2` |
| R2C single-precision | `float` | `float2` |
| R2C half-precision | `_Float16` | `_Float16_2` |
| R2C double-precision | `double` | `double2` |
| C2R half-precision | `_Float16_2` | `_Float16` |
| C2R single-precision | `float2` | `float` |
| C2R double-precision | `double2` | `double` |

## Function Signatures

```c
Tdata load_callback(Tdata* buffer, size_t offset,
                    void* callback_data, void* shared_memory);

void store_callback(Tdata* buffer, size_t offset, Tdata element,
                    void* callback_data, void* shared_memory);
```

### Parameters

- **Tdata**: Element data type for load or store operations
- **buffer**: Device memory pointer from `rocfft_execute()`
- **offset**: Element-based position within the buffer
- **element**: Data to store (store callbacks only)
- **callback_data**: User-supplied context pointer
- **shared_memory**: Currently unused (always null)

## Constraints

The callbacks execute once per processed element. Multiple kernels may decompose a single transform, potentially calling load and store callbacks across separate kernel invocations. Planar format input/output is not compatible with callbacks.
