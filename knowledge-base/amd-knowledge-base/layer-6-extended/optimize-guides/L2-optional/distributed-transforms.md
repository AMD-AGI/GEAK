Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocFFT/en/latest/how-to/distributed-transforms.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Distributed Transforms

rocFFT enables FFT distribution across multiple devices within a single process or across multiple Message Passing Interface (MPI) ranks. Input and output data layouts are described as fields to accomplish this.

## Multiple Devices in a Single Process

A transform can span multiple devices in a single process by passing distinct device IDs to `rocfft_brick_create()` when creating bricks in the input and output fields.

This capability was introduced in ROCm 6.0 with rocFFT 1.0.25.

## Message Passing Interface

MPI allows distributing transforms across multiple processes organized into MPI ranks.

### Enabling MPI Support

To activate rocFFT's MPI functionality, enable the `ROCFFT_MPI_ENABLE` CMake option during library compilation (disabled by default). For Cray MPI environments, enable `ROCFFT_CRAY_MPI_ENABLE` instead.

rocFFT's MPI support requires a GPU-aware MPI library capable of transferring data to and from HIP devices.

MPI transform support debuted in ROCm 6.3 with rocFFT 1.0.29.

### Implementation Steps

To distribute a transform across MPI ranks:

1. **Add communicator**: Each rank invokes `rocfft_plan_description_set_comm()` to attach an MPI communicator to the plan description, enabling rocFFT to coordinate computation across all communicator ranks.

2. **Define fields and bricks**: All ranks allocate identical fields and call `rocfft_plan_description_add_infield()` and `rocfft_plan_description_add_outfield()`. Each rank creates bricks only for data residing on that rank using `rocfft_brick_create()` and `rocfft_field_add_brick()`. Each brick exists on precisely one rank.

3. **Create plan**: Each rank calls `rocfft_plan_create()`, allowing rocFFT to distribute brick information across all ranks.

4. **Execute transform**: Each rank invokes `rocfft_execute()` with arrays of pointers to its bricks, ordered identically to their addition sequence. For in-place operations, supply only input pointers with an empty output array.

**Important**: Different ranks may return different API values. Developers must confirm all ranks successfully created their plans before executing distributed transforms.
