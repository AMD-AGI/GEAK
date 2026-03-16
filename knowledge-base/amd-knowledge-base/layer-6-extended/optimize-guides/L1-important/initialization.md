---
tags: ["optimization", "performance", "hip", "initialization", "setup"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/initialization.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Initialization

## Overview

The initialization process sets up the environment and resources needed for GPU operations, covering three main areas:

1. **HIP Runtime Setup** - Reads environment variables, configures active/visible devices, loads libraries, initializes internal buffers, sets up the compiler and HSA runtime, and checks for resource availability.

2. **GPU Querying and Selection** - Identifies and queries available GPU devices on the system.

3. **Context Creation** - Establishes contexts for each GPU device to manage resources and execute kernels.

## Initialize the HIP Runtime

The HIP runtime initializes automatically upon the first HIP API call. However, you can explicitly initialize it using `hipInit()` to control timing and ensure the GPU is ready before other operations begin.

> "You can use `hipDeviceReset()` to delete all streams created, memory allocated, kernels running and events created by the current process."

After resetting, any new HIP API call will reinitialize the runtime.

## Querying and Setting GPUs

When multiple GPUs are available, you can query and select specific devices based on properties like global memory size, shared memory per block, cooperative launch support, and managed memory capability.

### Querying GPUs

Use `hipGetDeviceProperties()` to retrieve a `hipDeviceProp_t` structure containing device characteristics. The `hipGetDeviceCount()` function returns the total number of available GPUs, enabling iteration over them.

**Example Code:**

```c
#include <hip/hip_runtime.h>
#include <iostream>

int main()
{
    int deviceCount;
    if (hipGetDeviceCount(&deviceCount) == hipSuccess)
    {
        for (int i = 0; i < deviceCount; ++i)
        {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, i) == hipSuccess)
                std::cout << "Device" << i << prop.name << std::endl;
        }
    }

    return 0;
}
```

### Setting the GPU

The `hipSetDevice()` function designates which GPU handles subsequent operations by:

- **Binding the current thread** to the specified device's context
- **Preparing resource allocation** for memory and stream creation
- **Verifying device availability** and capability for HIP operations
