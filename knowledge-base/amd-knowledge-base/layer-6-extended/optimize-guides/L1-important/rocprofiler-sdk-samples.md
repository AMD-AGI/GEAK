Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprofiler-sdk", "samples", "examples"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/samples.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCprofiler-SDK Samples

## Overview

The ROCprofiler-SDK provides sample programs designed to demonstrate the profiler's functionality in practical scenarios.

## Finding Samples

Sample programs and tools are distributed through the ROCm installation:

- **Sample programs location:** `/opt/rocm/share/rocprofiler-sdk/samples`
- **rocprofv3 tool location:** `/opt/rocm/bin`

## Building Samples

To compile the samples from any directory, execute these commands:

```bash
cmake -B build-rocprofiler-sdk-samples /opt/rocm/share/rocprofiler-sdk/samples -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-rocprofiler-sdk-samples --target all --parallel 8
```

## Running Samples

After building, navigate to the build directory and run the test suite:

```bash
cd build-rocprofiler-sdk-samples
ctest -V
```

The `-V` flag produces verbose output, showing detailed information throughout test execution.
