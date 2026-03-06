Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Using hipify-clang

## Overview

`hipify-clang` serves as a Clang-based translator for converting NVIDIA CUDA source code into HIP sources. The tool parses CUDA code into an Abstract Syntax Tree (AST) and applies transformation matchers to generate HIP output.

### Key Strengths

- Successfully parses complex code constructs or reports errors clearly
- Supports Clang options including `-I`, `-D`, and `–cuda-path`
- Seamlessly accommodates new CUDA versions through statically linked Clang
- Well-supported as a compiler extension

### Limitations

- Input CUDA code must be syntactically correct for successful translation
- Requires CUDA installation (version 7.0 minimum; latest supported is 12.8.1)
- Demands all necessary includes and defines for successful code translation

## Release Dependencies

**CUDA Requirements:** Minimum version 7.0; latest supported is 12.8.1

**LLVM+Clang:** Compatibility depends on your CUDA version. The recommended stable release is 20.1.8, with minimum version 4.0.0 required.

A detailed compatibility matrix is provided showing supported LLVM versions for each CUDA release across Windows and Linux platforms.

## Basic Usage

Process a single file with required headers:

```bash
./hipify-clang square.cu \
  --cuda-path=/usr/local/cuda-12.8 \
  -I /usr/local/cuda-12.8/samples/common/inc
```

Supply `hipify-clang` arguments first, then use `--` separator for Clang compilation arguments:

```bash
./hipify-clang cpp17.cu \
  --cuda-path=/usr/local/cuda-12.8 \
  -- -std=c++17
```

Process multiple files in a single command with absolute or relative paths.

## JSON Compilation Database

Use a `compile_commands.json` file for automation (Clang 8.0.0+):

```bash
-p <folder containing compile_commands.json>
```

This allows Clang options to be provided via the JSON file while `hipify-clang` options remain on the command line.

## Hipification Statistics

### Text Output

```bash
hipify-clang intro.cu \
  -cuda-path="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" \
  --print-stats
```

Returns conversion metrics including reference counts, conversion percentages, byte replacements, and line-of-code changes.

### CSV Export

```bash
hipify-clang intro.cu \
  -cuda-path="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" \
  --print-stats-csv
```

Generates a `.csv` file with detailed statistics for spreadsheet analysis. When processing multiple files, statistics appear per file and in aggregate.
