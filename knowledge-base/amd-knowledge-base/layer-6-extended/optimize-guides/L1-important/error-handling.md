---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "error-handling", "debugging"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Error Handling in HIP

## Overview

HIP provides error detection and management capabilities for runtime functions and kernel launches. According to the documentation, "Every HIP runtime function, apart from launching kernels, has `hipError_t` as return type."

For kernel launches specifically, developers should use `hipGetLastError()` and `hipPeekAtLastError()` to catch errors, since kernels don't return error codes directly.

## Key Functions

The main error handling utilities include:

- **hipGetLastError()** - Returns the last error and resets it to `hipSuccess`
- **hipPeekAtLastError()** - Returns the error without clearing it
- **hipGetErrorString()** - Provides human-readable error descriptions
- **hipGetErrorName()** - Returns error names

## Best Practices

The documentation recommends three key approaches:

1. Check errors after each API call to prevent error propagation
2. Use macros for error checking to reduce code duplication
3. Handle errors gracefully by freeing resources and providing meaningful messages

## HIP_CHECK Macro Pattern

A typical error-checking macro follows this structure:

```cpp
#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}
```

## Complete Example

The documentation provides a full working example demonstrating error handling in a vector addition kernel, showing proper allocation, data transfer, kernel execution, and cleanup with error checking at each step.
