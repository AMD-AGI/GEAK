---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/how-to/using-rocsparse.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# rocSPARSE User Guide

## Overview

rocSPARSE is a sparse linear algebra library optimized for AMD GPUs. This guide covers device management, stream handling, storage formats, and API capabilities.

## Device and Stream Management

### HIP Device Selection

Before calling rocSPARSE functions, set your target device using `hipSetDevice()`. rocSPARSE queries but doesn't set the device—developers must ensure device validity. Create a handle via `rocsparse_create_handle()` after device selection and destroy it with `rocsparse_destroy_handle()` when finished.

**Important:** You cannot switch devices between handle creation and destruction. Create a new handle for each device.

### Stream Binding

rocSPARSE functions execute on HIP streams. Bind custom streams using `rocsparse_set_stream()`. All rocSPARSE operations within a handle use the associated stream. Manage stream lifecycle independently.

## Execution Model

Most rocSPARSE functions are **non-blocking and asynchronous** relative to the host. Use `hipDeviceSynchronize()` or `hipStreamSynchronize()` to force synchronization when needed.

Multiple handles can run concurrently across devices, but each handle operates on a single device only.

## Graph Capture Support

Many rocSPARSE functions support HIP graph capture via standard HIP Graph Management APIs. Notably, functions supporting graph capture include:

- **Level 1:** axpyi, doti, dotci, gthr, roti, sctr
- **Level 2:** csrmv, csrsv, coomv, bsrmv, ellmv
- **Level 3:** csrmm, csrsm, bsrmm, bsrsm
- **Generic:** spmv (compute stage only), spmm, spgemm, spsv, spsm

**Limitations:** Some operations like `spmv` preprocess stages and `sddmm` with non-default algorithms have restricted graph support.

## Storage Formats

rocSPARSE supports multiple sparse matrix formats:

### COO (Coordinate)

Three arrays store matrix data:
- `coo_val`: non-zero values
- `coo_row_ind`: row indices
- `coo_col_ind`: column indices

Matrix must be sorted by row then column indices.

### CSR (Compressed Sparse Row)

Efficient row-based storage:
- `csr_val`: non-zero values
- `csr_row_ptr`: row pointers (length m+1)
- `csr_col_ind`: column indices

### CSC (Compressed Sparse Column)

Column-oriented variant of CSR with `csc_col_ptr` and `csc_row_ind`.

### BSR/GEBSR (Block Sparse)

Block-structured formats for matrices with regular block patterns. BSR handles uniform blocks; GEBSR supports rectangular blocks.

### ELL (Ellpack-Itpack)

Stores fixed maximum non-zeros per row. Rows with fewer elements are padded with zeros.

### HYB (Hybrid)

Combines ELL for regular portions and COO for irregular portions to optimize memory and computation.

## Indexing and Pointer Modes

### Index Base

rocSPARSE supports both **0-based and 1-based indexing**, selectable via `rocsparse_index_base` parameter.

### Pointer Mode

Control scalar parameter location:
- **Host mode:** Scalars (alpha, beta) allocated on host heap/stack
- **Device mode:** Scalars allocated on GPU memory

Set via `rocsparse_set_pointer_mode()`. Device mode allows asynchronous returns; host mode blocks until GPU results transfer back.

## Logging and Profiling

### Activity Logging (Deprecated)

Set `ROCSPARSE_LAYER` environment variable (bit mask: 1=trace, 2=bench, 4=debug). Optionally redirect output with `ROCSPARSE_LOG_*_PATH` variables.

**Note:** This feature is deprecated and impacts performance.

### ROC-TX Integration

Enable profiling with:
```bash
ROCSPARSE_ROCTX=1 rocprofv3 --kernel-trace --marker-trace -- ./program
```

Generates trace files viewable in Perfetto UI. Unavailable on Windows and static library builds.

## Related Libraries

**hipSPARSE** provides a portable abstraction layer supporting both rocSPARSE and cuSPARSE backends, prioritizing convenience over raw performance. Use rocSPARSE directly when performance is critical.
