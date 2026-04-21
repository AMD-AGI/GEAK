---
tags: ["optimization", "performance", "hip", "memory", "virtual"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Virtual Memory Management

## Overview

Virtual memory management optimizes GPU memory handling by separating physical memory allocation from virtual address mapping. This approach reduces memory usage and unnecessary `memcpy` operations compared to traditional allocation methods.

## Memory Allocation Process

### Check Virtual Memory Support

Verify device capability using `hipDeviceGetAttribute()`:

```c
int vmm = 0, currentDev = 0;
hipDeviceGetAttribute(
    &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
);
```

### Allocate Physical Memory

Use `hipMemCreate()` to allocate physical memory with proper granularity alignment:

```c
size_t granularity = 0;
hipMemGenericAllocationHandle_t allocHandle;
hipMemAllocationProp prop = {};
prop.type = hipMemAllocationTypePinned;
prop.location.type = hipMemLocationTypeDevice;
prop.location.id = currentDev;
hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum);
padded_size = ROUND_UP(size, granularity);
hipMemCreate(&allocHandle, padded_size, &prop, 0);
```

### Reserve Virtual Address Range

Reserve contiguous virtual addresses using `hipMemAddressReserve()`:

```c
hipMemAddressReserve(&ptr, padded_size, 0, 0, 0);
hipMemMap(ptr, padded_size, 0, allocHandle, 0);
```

### Set Memory Access Permissions

Enable read/write access via `hipMemSetAccess()`:

```c
hipMemAccessDesc accessDesc = {};
accessDesc.location.type = hipMemLocationTypeDevice;
accessDesc.location.id = currentDev;
accessDesc.flags = hipMemAccessFlagsProtReadwrite;
hipMemSetAccess(ptr, padded_size, &accessDesc, 1);
```

## Dynamic Memory Expansion

Extend allocations while maintaining virtual address continuity:

```c
hipMemAddressReserve(&new_ptr, (new_size - padded_size), 0, ptr + padded_size, 0);
hipMemMap(new_ptr, (new_size - padded_size), 0, newAllocHandle, 0);
hipMemSetAccess(new_ptr, (new_size - padded_size), &accessDesc, 1);
```

## Memory Cleanup

Release resources in order:

```c
hipMemUnmap(ptr, size);
hipMemRelease(allocHandle);
hipMemAddressFree(ptr, size);
```

## Virtual Aliases

Multiple virtual addresses can map to the same physical memory. However, "RDNA cards may not produce correct results if users access two different virtual addresses that map to the same physical address" due to L1 cache incoherence. Use volatile pointers to ensure correctness.
