---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "counters", "rocprofiler"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/rocprofiler-sdk/api-reference/counter_collection_services.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCprofiler-SDK Counter Collection Services

## Overview

The ROCprofiler-SDK offers two counter collection modes:

- **Dispatch counting**: Collects per-kernel metrics in isolation, enabling detailed performance analysis while enforcing serialized kernel execution
- **Device counting**: Gathers device-level counters across time ranges without targeting specific kernels

## Key Definitions

**Profile Config** specifies which counters to collect on an agent and must be supplied to collection APIs.

**Counter ID** uniquely identifies a counter per architecture, enabling retrieval of metadata like names.

**Instance ID** encodes both counter identity and dimension information for collected values.

**Dimensions** provide context through hardware register sources (XCC, AID, shader engine, agent, shader array, WGP, instance).

## Setup Process

### Initialization

Create context and buffer infrastructure:

```c
rocprofiler_context_id_t ctx{0};
rocprofiler_buffer_id_t buff;
ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");
ROCPROFILER_CALL(rocprofiler_create_buffer(ctx, 4096, 2048,
                 ROCPROFILER_BUFFER_POLICY_LOSSLESS, buffered_callback,
                 user_data, &buff), "buffer creation failed");
```

### Service Configuration

For dispatch counting:
```c
ROCPROFILER_CALL(rocprofiler_configure_buffer_dispatch_counting_service(
    ctx, buff, dispatch_callback, nullptr),
    "Could not setup buffered service");
```

For device counting:
```c
ROCPROFILER_CALL(rocprofiler_configure_device_counting_service(
    ctx, buff, agent_id, set_profile, nullptr),
    "Could not setup buffered service");
```

## Profile Creation

1. **Query available agents** using `rocprofiler_query_available_agents`, filtering for GPU types
2. **Enumerate supported counters** via `rocprofiler_iterate_agent_supported_counters`
3. **Retrieve counter metadata** using `rocprofiler_query_counter_info`
4. **Construct profile** by passing counter arrays to `rocprofiler_create_counter_config`

Important: "Profiles are immutable. To collect a new counter set, construct a new profile."

## Callbacks

**Dispatch Callback** receives kernel launch information and returns selected profile for counter collection.

**Set Profile Callback** invoked after context start, allowing profile specification per agent.

**Buffered Callback** processes collected counter data through record iteration.

## Derived Metrics

Expressions perform computations on hardware metrics, supporting standard operators (+, -, *, /) and special functions.

### Reduce Function

Aggregates counter values across dimensions:
- `sum`: Total across all dimensions
- `avr`: Average value
- `min`/`max`: Extreme values

Dimension-specific reduction: `reduce(GL2C_HIT,sum,[DIMENSION_XCC])`

### Select Function

Filters results by dimension indices: `select(Y, [DIMENSION_XCC=[0],DIMENSION_SHADER_ENGINE=[2]])`

### Accumulate Function

Sums basic counters over specified cycles:
- `HIGH_RES`: Every clock cycle
- `LOW_RES`: Every four cycles
- `NONE`: No summing

## Kernel Serialization

Dispatch counting requires serialized kernel execution. Applications with co-dependent kernels risk deadlock. Solutions include:
- Restructuring to avoid simultaneous kernels
- Filtering co-dependent kernels from collection
- Switching to device-wide counting mode
