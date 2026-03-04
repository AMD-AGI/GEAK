---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "compiler", "device-libs", "llvm", "reference"]
priority: "L0-core"
source_url: "https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# AMD Device Libraries Reference

## Overview

The AMD Device Libraries are a collection of LLVM bitcode libraries that provide essential runtime support for GPU kernels. These libraries implement standard math functions, intrinsics, and other runtime support needed by GPU code.

## Repository Information

**Location:** https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs

**Purpose:** Device libraries provide:
- Mathematical functions (sin, cos, exp, log, etc.)
- GPU-specific intrinsics
- Runtime support functions
- Optimized implementations for AMD GPU architectures

## Key Components

- **Math Libraries:** Optimized implementations of standard mathematical functions
- **Built-in Functions:** GPU intrinsics and built-in operations
- **Runtime Support:** Essential runtime functionality for kernel execution
- **Architecture-Specific Optimizations:** Tuned implementations for different GPU generations

## Compilation and Linking

Device libraries are:
- Compiled to LLVM bitcode
- Linked with user kernels during compilation
- Optimized based on target GPU architecture
- Automatically included by the ROCm compiler toolchain

## Integration with ROCm

These libraries are fundamental to:
- HIP application compilation
- OpenCL kernel execution
- Mathematical operation performance
- Portable GPU code development

## Documentation

For implementation details, supported functions, and usage examples, refer to the source code and documentation in the GitHub repository.
