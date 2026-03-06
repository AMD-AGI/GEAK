Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprofiler-sdk", "callbacks"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/callback_services.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCprofiler-SDK Callback Tracing Services

## Overview

The callback tracing services offer synchronous callbacks triggered on the current CPU thread when events occur. For instance, when tracing an API function like `hipSetDevice`, the system invokes a user-defined callback before and after the function executes on the invoking thread.

## Subscribing to Callback Tracing Services

Tools configure callback tracing during initialization using:

```c
rocprofiler_status_t
rocprofiler_configure_callback_tracing_service(
    rocprofiler_context_id_t context_id,
    rocprofiler_callback_tracing_kind_t kind,
    rocprofiler_tracing_operation_t* operations,
    size_t operations_count,
    rocprofiler_callback_tracing_cb_t callback,
    void* callback_args);
```

### Key Parameters

- **kind**: High-level specification of services (domain), such as HIP API, HSA API, or kernel dispatches
- **operations**: Array restricting callbacks to specific domain operations. Pass `nullptr` and `0` to trace all operations
- **callback**: User-specified function invoked on events
- **callback_args**: User data passed to the callback

The function returns an error if the callback service for a given context and domain is configured multiple times.

## Callback Tracing Callback Function

```c
typedef void (*rocprofiler_callback_tracing_cb_t)(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t* user_data,
    void* callback_data);
```

### Record Structure

```c
typedef struct rocprofiler_callback_tracing_record_t {
    rocprofiler_context_id_t context_id;
    rocprofiler_thread_id_t thread_id;
    rocprofiler_correlation_id_t correlation_id;
    rocprofiler_callback_tracing_kind_t kind;
    uint32_t operation;
    rocprofiler_callback_phase_t phase;
    void* payload;
} rocprofiler_callback_tracing_record_t;
```

The `payload` field's underlying type varies by domain. For HIP APIs, cast to `rocprofiler_callback_tracing_hip_api_data_t*`.

### User Data Persistence

The `user_data` parameter stores information between callback phases, unique per operation instance. This enables tracking values like timestamps across enter/exit phases.

## Callback Tracing Record

Query functions support introspection:

- `rocprofiler_query_callback_tracing_kind_name()`: Retrieves domain name
- `rocprofiler_query_callback_tracing_kind_operation_name()`: Retrieves operation name
- `rocprofiler_iterate_callback_tracing_kinds()`: Iterates all tracing kinds
- `rocprofiler_iterate_callback_tracing_kind_operations()`: Iterates operations per kind

## Code Object Tracing

Code object tracing provides critical GPU asynchronous activity information. Two payload types exist:

1. **Code Object Load** (`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`, `ROCPROFILER_CODE_OBJECT_LOAD`): Unique identifier for kernel symbol bundles loaded on GPU agents
2. **Kernel Symbol Register** (`ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`): Globally unique kernel symbol identifiers with static properties

### Important Note

Kernel identifiers remain unique across identical symbols on different agents and after reload cycles. The SDK doesn't provide external query interfaces—tools must copy relevant information during load callbacks.

### Event Sequence

1. Load code object (LOAD phase)
2. Load kernel symbols (LOAD phase, repeats per symbol)
3. Application execution
4. Unload kernel symbols (UNLOAD phase, repeats per symbol)
5. Unload code object (UNLOAD phase)

String fields like `kernel_name` remain valid until SDK finalization, eliminating copy requirements for constants.
