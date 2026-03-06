Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprofiler-sdk", "intercept"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/intercept_table.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Runtime Intercept Tables

## Overview

The ROCprofiler-SDK documentation explains how tools can access raw API dispatch tables beyond standard callback or buffer tracing services. This approach enables "interception of HIP, HSA, and ROCTx APIs" through dispatch table manipulation.

## Key Architectural Components

### Forward Declaration
APIs begin with a public C function declaration with default visibility:

```c
extern "C"
{
int foo(int) __attribute__((visibility("default")));
}
```

### Internal Implementation
The actual logic resides in a namespace implementation:

```c
namespace impl
{
int foo(int val)
{
    return (2 * val);
}
}
```

### Dispatch Table Structure
A struct holds function pointers, initialized once:

```c
namespace impl
{
struct dispatch_table
{
    int (*foo_fn)(int) = nullptr;
};

dispatch_table*& construct_dispatch_table()
{
    static dispatch_table* tbl = new dispatch_table{};
    tbl->foo_fn = impl::foo;
    return tbl;
}

dispatch_table* get_dispatch_table()
{
    static dispatch_table*& tbl = construct_dispatch_table();
    return tbl;
}
}
```

### Public API Implementation
The exported function delegates through the dispatch table:

```c
extern "C"
{
int foo(int val)
{
    return impl::get_dispatch_table()->foo_fn(val);
}
}
```

## Dispatch Table Chaining

ROCprofiler-SDK preserves original function pointers and replaces them with its own wrappers. This creates a chain where the public API calls the SDK wrapper, which then invokes the original implementation. Tools receive access via `rocprofiler_at_intercept_table_registration`.

**Reference:** Sample implementations available at the project's intercept_table directory.
