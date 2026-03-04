---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "multi-gpu", "peer-to-peer"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Multi-device Management

## Device Enumeration

Device enumeration identifies all available GPUs connected to the host system. A single machine can have multiple GPUs, each with its own unique identifier. By listing these devices, applications can decide which GPU to use for computation. The system queries available GPUs that support the chosen `HIP_PLATFORM`. If no specific GPU is defined, "device 0 is selected."

The provided code example demonstrates how to:
- Query the total device count using `hipGetDeviceCount()`
- Iterate through each device to retrieve its properties
- Display information such as device name, global memory, shared memory per block, warp size, thread dimensions, and grid size

## Device Selection

After enumerating available GPUs, the next step involves selecting a specific device for computation. This is crucial in multi-GPU systems where different GPUs may have varying capabilities or workloads. By selecting the appropriate device, computational tasks are directed to the correct GPU, optimizing performance and resource utilization.

The example shows how to:
- Set the active device using `hipSetDevice()`
- Allocate memory on the selected device with `hipMalloc()`
- Launch kernels on the chosen device
- Copy results between different devices

## Stream and Event Behavior

In multi-device systems, streams enable asynchronous task execution, allowing multiple devices to process data concurrently. Events provide synchronization mechanisms across streams and devices, ensuring tasks on one device complete before dependent tasks on another device begin. This coordination prevents race conditions and optimizes data flow.

The example demonstrates:
- Creating separate streams and events for each device
- Recording events to measure kernel execution time
- Asynchronous kernel launches on different devices
- Proper cleanup and synchronization

## Peer-to-Peer Memory Access

Peer-to-peer memory access allows one GPU to directly read or write to another GPU's memory, reducing data transfer times by eliminating the need for host involvement. This significantly improves performance for applications requiring frequent inter-GPU data exchange.

The documentation provides two implementations:
1. **With peer-to-peer enabled**: Uses `hipDeviceEnablePeerAccess()` to establish direct GPU-to-GPU communication, improving transfer performance
2. **Without peer-to-peer**: Still works but internally uses a host staging buffer, incurring performance penalties
