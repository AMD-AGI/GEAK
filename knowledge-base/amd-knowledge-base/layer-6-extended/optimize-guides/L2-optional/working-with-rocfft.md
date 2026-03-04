---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocFFT/en/latest/how-to/working-with-rocfft.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Working with rocFFT

## Workflow

To compute an FFT with rocFFT, create a plan (a handle to internal data structures holding transform details), execute it with specified data buffers, and destroy it when complete.

### Basic Steps

1. Initialize the library with `rocfft_setup()`
2. Create a plan using `rocfft_plan_create()` or build a detailed plan with `rocfft_plan_description_create()` and related configuration functions
3. Optionally allocate a work buffer by:
   - Calling `rocfft_plan_get_work_buffer_size()`
   - Creating an execution info object with `rocfft_execution_info_create()`
   - Allocating buffer space via `hipMalloc` and passing it to `rocfft_execution_info_set_work_buffer()`
4. Execute the plan with `rocfft_execute()` on your data buffers
5. Free work buffers and execution info if allocated
6. Destroy the plan with `rocfft_plan_destroy()`
7. Terminate the library with `rocfft_cleanup()`

### Example Code

```cpp
#include <iostream>
#include <vector>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft/rocfft.h"

int main()
{
        rocfft_setup();

        size_t N = 16;
        size_t Nbytes = N * sizeof(float2);

        float2 *x;
        hipMalloc(&x, Nbytes);

        std::vector<float2> cx(N);
        for (size_t i = 0; i < N; i++)
        {
                cx[i].x = 1;
                cx[i].y = -1;
        }

        hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);

        rocfft_plan plan = nullptr;
        size_t length = N;
        rocfft_plan_create(&plan, rocfft_placement_inplace,
             rocfft_transform_type_complex_forward, rocfft_precision_single,
             1, &length, 1, nullptr);

        size_t work_buf_size = 0;
        rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
        void* work_buf = nullptr;
        rocfft_execution_info info = nullptr;
        if(work_buf_size)
        {
                rocfft_execution_info_create(&info);
                hipMalloc(&work_buf, work_buf_size);
                rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
        }

        rocfft_execute(plan, (void**) &x, nullptr, info);
        hipDeviceSynchronize();

        if(work_buf_size)
        {
                hipFree(work_buf);
                rocfft_execution_info_destroy(info);
        }

        rocfft_plan_destroy(plan);

        std::vector<float2> y(N);
        hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);

        for (size_t i = 0; i < N; i++)
        {
                std::cout << y[i].x << ", " << y[i].y << std::endl;
        }

        hipFree(x);
        rocfft_cleanup();

        return 0;
}
```

## Library Setup and Cleanup

Call `rocfft_setup()` before any library APIs and `rocfft_cleanup()` at program termination to properly allocate and free resources.

## Plans

A plan collects most parameters needed for FFT computation, including:

- Transform type (complex or real)
- Dimensions (1D, 2D, or 3D)
- Data length/extent per dimension
- Number of batched datasets
- Floating-point precision
- In-place or out-of-place execution
- Input/output buffer format (array type)
- Data layout in buffers
- Output scaling factor

Plans do **not** include input/output buffer handles, work buffer handles, or device execution controls—these are specified during execution.

## Data

Allocate, initialize, and manage input/output buffers. Query work buffer requirements using `rocfft_plan_get_work_buffer_size()` and pass allocated buffers to the library via `rocfft_execution_info_set_work_buffer()`. rocFFT minimizes its own device memory allocations; you manage the work buffers.

## Transform and Array Types

**Complex FFT**: Transforms complex data (forward or backward) with two storage formats:

- **Planar**: Real and imaginary components in separate arrays (`RRRRR...` and `IIIII...`)
- **Interleaved**: Components stored as contiguous pairs (`RIRIRIRI...`)

**Real FFT**: Transforms real data to/from Hermitian complex data. Real backward FFTs require Hermitian-symmetric input; rocFFT produces undefined results otherwise.

Use `rocfft_transform_type` and `rocfft_array_type` enumerations to specify these.

## Batches

Efficiency improves by batching transforms. Use the `number_of_transforms` parameter in `rocfft_plan_create()` to specify batch size. The GPU benefits from receiving as much data as possible in fewer API calls, reducing control transfer overhead.

## Result Placement

The library supports both in-place and out-of-place transforms via the `rocfft_result_placement` enumeration:

- **In-place**: Only input buffers provided; results overwrite input data
- **Out-of-place**: Distinct output buffers receive results

"rocFFT can often overwrite input buffers on real inverse transforms, even if requested as out-of-place," allowing better optimization.

## Strides and Distances

Configure custom data layouts using `rocfft_plan_description_set_data_layout()`:

- **Strides**: Gaps in memory between successive elements in a dimension
  - `stride == 1`: contiguous elements
  - `stride > 1`: gaps between elements
- **Distance**: Stride between corresponding elements of successive FFT instances in a batch (measured in complex or real units)

For tightly packed 1D data, `distance == length`. Column-major access patterns use small distances with large strides.

## Overwriting Non-Contiguous Buffers

rocFFT respects specified strides for input reading and output writing but may write temporary results contiguously. Temporary data might overwrite non-strided locations, though the library respects total buffer size. For a 1D transform with stride 2 on a 2N-element buffer, at most 2N elements receive temporary writes.

## Input and Output Fields

rocFFT allows inputs and outputs as **fields** composed of multiple **bricks**, each on different devices with distinct layouts. This is experimental and subject to change.

**Workflow**:

1. Create a field with `rocfft_field_create()`
2. Add bricks:
   - Allocate brick with `rocfft_brick_create()`, defining lower (inclusive) and upper (exclusive) coordinates
   - Specify device and memory strides
   - Add to field with `rocfft_field_add_brick()`
   - Destroy brick with `rocfft_brick_destroy()`
3. Assign field as input/output via `rocfft_plan_description_add_infield()` or `rocfft_plan_description_add_outfield()`
4. Destroy field with `rocfft_field_destroy()`
5. Create and execute plan, passing arrays of pointers in brick-addition order

For in-place transforms, provide non-empty input pointer arrays and empty output arrays.

## Transforms of Real Data

See the Real data documentation for details.

## Reproducibility of Results

"The results of an FFT computation generated by rocFFT are bitwise reproducible" across runs when keeping constant:

- FFT parameters
- rocFFT library version
- GPU model

Valid FFT plans following data overlap rules are required.

## Result Scaling

Use `rocfft_plan_description_set_scale_factor()` to efficiently combine scaling multiplication with FFT computation, set before plan creation.

## Loading and Storing Callbacks

See the Loading and storing callbacks documentation.

## Runtime Compilation

See the Runtime compilation documentation.
