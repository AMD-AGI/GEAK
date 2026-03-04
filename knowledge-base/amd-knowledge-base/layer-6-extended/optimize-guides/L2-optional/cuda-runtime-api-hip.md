---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# CUDA Runtime API Supported by HIP

## Overview

This documentation provides a comprehensive mapping of NVIDIA CUDA Runtime API functions to their HIP equivalents. The reference tables organize APIs across 40 categories, enabling developers to port CUDA code to HIP for AMD GPU compatibility.

## Table Status Legend

The documentation uses column markers to indicate function status:

- **A** (Added) - Version when HIP support was introduced
- **D** (Deprecated) - Version marking deprecation
- **C** (Changed) - Version when modifications occurred
- **R** (Removed) - Version marking removal
- **E** (Experimental) - Experimental feature designation

## Major API Categories

### Core Functionality Areas

1. **Device Management** - Device selection, properties, and configuration
2. **Memory Management** - Allocation, transfers, and deallocation operations
3. **Stream & Event Management** - Asynchronous execution control
4. **Execution Control** - Kernel launching and configuration
5. **Graph Management** - Computation graph creation and execution
6. **Error Handling** - Error detection and reporting

### Interoperability Sections

The API includes support for graphics framework integration:
- OpenGL interoperability
- Direct3D 9, 10, and 11 support
- VDPAU and EGL interoperability
- Graphics resource mapping

### Specialized Features

- Texture and surface object management
- Unified addressing
- Peer device memory access
- External resource interoperability
- Stream-ordered memory allocation
- Occupancy calculation utilities

## Implementation Status

Most fundamental CUDA Runtime functions have corresponding HIP implementations. Coverage spans from version 1.5.0 onward, with ongoing additions in recent releases. Some specialized NVIDIA features lack direct HIP equivalents, particularly proprietary technologies like NvSciSync and certain Direct3D variants.
