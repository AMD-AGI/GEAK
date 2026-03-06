Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "compiler", "comgr", "llvm", "reference"]
priority: "L0-core"
source_url: "https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# AMD Code Object Manager (COMGR) Reference

## Overview

The Code Object Manager (COMGR) is a library that provides APIs for compiling and inspecting AMD GPU code objects. It is part of the ROCm LLVM project and serves as a key component in the ROCm compiler toolchain.

## Repository Information

**Location:** https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr

**Purpose:** COMGR provides functionality for:
- Compiling device code to code objects
- Inspecting code object metadata
- Managing code object symbols
- Linking code objects

## Key Features

- **Code Object Compilation:** Transform high-level GPU code into executable code objects
- **Metadata Inspection:** Query code object properties and metadata
- **Symbol Management:** Handle symbols within code objects
- **Code Object Linking:** Link multiple code objects together

## Integration with ROCm

COMGR is used by:
- HIP runtime for JIT compilation
- ROCm language runtimes
- Performance tools requiring code introspection
- Development tools needing code object manipulation

## Documentation

For detailed API documentation, implementation details, and examples, refer to the source code and headers in the GitHub repository.
