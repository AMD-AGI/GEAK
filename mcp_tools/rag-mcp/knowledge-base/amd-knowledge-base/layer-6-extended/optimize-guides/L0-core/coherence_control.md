---
tags: ["optimization", "performance", "hip", "memory", "coherence"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Coherence Control

## Overview

Memory coherence determines how different system components (CPU and GPU) access and see updates to shared memory. HIP supports two coherence types:

**Coarse-grained coherence:** Memory visibility requires explicit synchronization via `hipDeviceSynchronize()`, `hipStreamSynchronize()`, or blocking operations like `hipMemcpy()`. Caches flush before data becomes visible to prevent conflicts.

**Fine-grained coherence:** Memory remains coherent during concurrent modifications across system components, ensuring current data visibility regardless of kernel boundaries. This approach may use limited GPU cache policies or read-only access on some AMD hardware.

## Memory Coherence Control Methods

| API | Flag | Coherence |
|-----|------|-----------|
| `hipHostMalloc` | `hipHostMallocDefault` | Fine-grained |
| `hipHostMalloc` | `hipHostMallocNonCoherent` | Coarse-grained |
| `hipExtMallocWithFlags` | `hipDeviceMallocDefault` | Coarse-grained |
| `hipExtMallocWithFlags` | `hipDeviceMallocFinegrained` | Fine-grained |
| `hipMallocManaged` | (default) | Fine-grained |
| `hipMallocManaged` | `hipMemAdviseSetCoarseGrain` | Coarse-grained |
| `malloc` | (default) | Fine-grained |
| `malloc` | `hipMemAdviseSetCoarseGrain` | Coarse-grained |

**Note:** The `HIP_HOST_COHERENT` environment variable affects `hipHostMalloc()` behavior when coherence flags remain unset.

## Synchronization Functions and Memory Visibility

Different synchronization methods provide varying visibility guarantees:

| Function | Fine-grained Host Memory | Coarse-grained Host Memory |
|----------|--------------------------|---------------------------|
| `hipStreamSynchronize()` | Yes | Yes |
| `hipDeviceSynchronize()` | Yes | Yes |
| `hipEventSynchronize()` | Yes | Event-dependent |
| `hipStreamWaitEvent()` | Yes | No |

### Event Configuration

`hipEventCreateWithFlags()` allows scope control:

- **`hipEventReleaseToSystem`:** Executes system-scope release operations, making both coherence types visible across agents (may involve cache flushing)
- **`hipEventDisableTiming`:** Improves synchronization performance by disabling profiling
