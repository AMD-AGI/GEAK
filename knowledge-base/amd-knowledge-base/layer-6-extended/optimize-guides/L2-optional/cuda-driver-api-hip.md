---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Driver_API_functions_supported_by_HIP.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# CUDA Driver API Supported by HIP

## Overview

This documentation page provides comprehensive mapping tables showing which CUDA Driver API functions and data types are supported by HIP (Heterogeneous-Interface for Portability). The page serves as a reference for developers migrating CUDA code to HIP or writing portable GPU code.

## Structure

The documentation is organized into 45 sections covering different functional areas:

**Core Categories:**
- Data types and error handling
- Device and context management
- Memory management (standard and virtual)
- Stream and event management
- Kernel execution and graphs
- Texture and surface objects
- Interoperability with graphics APIs

## Key Information

### Table Format

Each section uses standardized columns:
- **CUDA column**: Original CUDA API name and version introduced
- **Status flags**: Added (A), Deprecated (D), Changed (C), Removed (R), Experimental (E)
- **HIP equivalent**: Corresponding HIP function/type name
- **HIP version**: When support was introduced in HIP

### Coverage Highlights

The reference includes mappings for:
- **Error codes**: Comprehensive error type translations (e.g., `CUDA_ERROR_OUT_OF_MEMORY` → `hipErrorOutOfMemory`)
- **Device attributes**: 200+ device capability queries
- **Memory operations**: Including virtual memory and memory pools
- **Graph operations**: Extensive support for CUDA graph primitives
- **JIT compilation**: Compilation options and settings
- **Modern features**: Tensor maps, conditional graphs, and device-side updates

## Notable Coverage Gaps

Several advanced features lack HIP equivalents:
- Direct3D interoperability (D3D9, D3D10, D3D11)
- VDPAU interoperability
- Some profiling-only features
- Certain memory synchronization primitives
- Advanced coredump configuration options

## Usage

This table supports HIPIFY tool functionality, which automatically converts CUDA code to HIP. Developers can reference specific APIs to understand compatibility and any required adjustments during code migration.
