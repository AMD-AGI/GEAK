---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/howto/memory.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# rocSOLVER Memory Model

rocSOLVER manages workspace memory through a configurable device memory scheme integrated with the rocBLAS memory model. Unlike traditional LAPACK, workspace pointers aren't explicitly passed as function arguments.

## Automatic Workspace

By default, rocSOLVER automatically allocates and manages device memory for internal workspace. This scheme:

- Automatically increases allocated memory when functions require more space
- Persists memory between function calls
- Can be verified via `rocblas_is_managing_device_memory` returning `true`
- Can be re-enabled by calling `rocblas_set_workspace` with `nullptr` or size `0`

**Drawback:** Automatic reallocation triggers synchronization events the user cannot control.

## User-Owned Workspace

Manual memory management provides greater control. The workflow involves three steps:

### 1. Determine Minimum Required Size

Query rocSOLVER for memory requirements:

```c
size_t memory_size;
rocblas_start_device_memory_size_query(handle);
rocsolver_dgetrf(handle, 1024, 1024, nullptr, lda, nullptr, nullptr);
rocsolver_dgetrs(handle, rocblas_operation_none, 1024, 1, nullptr, lda, nullptr, nullptr, ldb);
rocblas_stop_device_memory_size_query(handle, &memory_size);
```

Pass `nullptr` to device pointer arguments during the query phase.

### 2. Allocate Memory

```c
void* device_memory;
hipMalloc(&device_memory, memory_size);
```

### 3. Set Workspace

```c
rocblas_set_workspace(handle, device_memory, memory_size);
// perform computations
rocblas_set_workspace(handle, nullptr, 0);
hipFree(device_memory);
```

This approach eliminates unexpected synchronization points during computation.
