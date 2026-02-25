---
tags: ["optimization", "performance", "hip", "memory", "unified"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Unified Memory Management in HIP

## Overview

Unified memory management in HIP provides "a single address space accessible from both CPU and GPU," with managed memory offering automatic page migration between devices. This architecture differs from conventional setups where CPUs and GPUs maintain separate memory spaces.

## Core Concepts

### Unified Memory
A unified memory system allows processors to access host and other GPU memory without explicit copying. The system handles memory accessibility through various methods depending on hardware support.

### Managed Memory
Managed memory extends unified memory by monitoring access patterns and intelligently migrating data between device and system memories. When a GPU kernel accesses managed memory not in local device memory, a page-fault triggers, and the GPU requests the page from the host or another device.

## System Requirements

Managed memory support varies by GPU architecture:

| Architecture | `hipMallocManaged()` / `__managed__` | System Allocators |
|---|---|---|
| CDNA4 | ✅ | ✅ |
| CDNA3 | ✅ | ✅ |
| CDNA2 | ✅ | ✅ |
| CDNA1 | ✅ | ❌ |
| RDNA1 | ✅ | ❌ |
| GCN5 | ✅ | ❌ |

**Critical requirement:** Set `HSA_XNACK=1` and use a GPU kernel mode driver supporting Heterogeneous Memory Management (HMM) for proper functionality.

## Memory Allocation Approaches

### 1. HIP Allocated Managed Memory
`hipMallocManaged()` provides dynamic allocation on all GPUs with unified memory support. The `__managed__` attribute handles static allocation.

### 2. System Allocated Unified Memory
Starting with CDNA2, standard allocators (`new`, `malloc()`, `allocate()`) reserve unified memory. Memory allocated here registers with `hipHostRegister()` for device accessibility.

### 3. HIP Allocated Non-Managed Memory
`hipMalloc()` and `hipHostMalloc()` provide unified memory allocation without automatic migration between device and host.

## Device Attributes for Checking Support

| Attribute | Purpose |
|---|---|
| `hipDeviceAttributeManagedMemory` | Indicates managed memory allocation support |
| `hipDeviceAttributePageableMemoryAccess` | Coherent pageable memory access capability |
| `hipDeviceAttributeConcurrentManagedAccess` | Full unified memory with concurrent CPU access |

## Basic Example

Here's a simple example using `hipMallocManaged()`:

```cpp
int *a, *b, *c;
HIP_CHECK(hipMallocManaged(&a, sizeof(*a)));
HIP_CHECK(hipMallocManaged(&b, sizeof(*b)));
HIP_CHECK(hipMallocManaged(&c, sizeof(*c)));

*a = 1;
*b = 2;

add<<<1, 1>>>(a, b, c);
HIP_CHECK(hipDeviceSynchronize());

std::cout << *a << " + " << *b << " = " << *c << std::endl;

HIP_CHECK(hipFree(a));
HIP_CHECK(hipFree(b));
HIP_CHECK(hipFree(c));
```

## Performance Optimization Techniques

### Data Prefetching
Move data to desired devices before use with `hipMemPrefetchAsync()`. Use `hipCpuDeviceId` to specify CPU as target. Warning: profiling recommended as prefetching isn't always beneficial.

### Memory Advice
`hipMemAdvise()` provides hints about memory usage patterns including:
- `hipMemAdviseSetPreferredLocation` — ideal memory location
- `hipMemAdviseSetReadMostly` — read-heavy access patterns
- `hipMemAdviseSetAccessedBy` — which devices access the memory

### Memory Range Attributes
Query memory range characteristics using `hipMemRangeGetAttribute()` with attributes from `hipMemRangeAttribute` enum.

### Stream Memory Attachment
`hipStreamAttachMemAsync()` attaches memory to streams, reducing transfer overhead by only migrating memory when kernels on that stream require access.

## Key Advantages

Unified memory simplifies GPU computing by eliminating explicit host-device copies, particularly beneficial for sparse memory access patterns. Using pinned memory improves bandwidth for transfers between host and GPUs.

## Important Considerations

Unified memory can introduce performance overhead. Developers should thoroughly test and profile code to ensure it suits their specific use case before committing to this approach.
