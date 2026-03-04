---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "profiling", "architecture", "model"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/projects/rocprofiler-compute/conceptual/performance-model.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Performance Model

ROCm Compute Profiler provides extensive metrics for understanding application performance on AMD Instinct MI-series accelerators, including GCN GPUs (MI50), CDNA (MI100), CDNA2 (MI250X/MI250/MI210), CDNA3 (MI300A/MI300X/MI325X), and CDNA4 (MI350X/MI355X) architectures.

## Architecture Overview

| Architecture | Chip Packaging | Supported Series | Partition Modes |
|---|---|---|---|
| CDNA | Single Die | MI100 | ❌ |
| CDNA 2 | Two GCDs | MI200, MI210, MI250 | ❌ |
| CDNA 3 | Dozen chiplets | MI300A, MI300X, MI325X | Compute & Memory |
| CDNA 4 | Multi-Die with IODs | MI350X, MI355X | Compute & Memory |

## Data Type Support

All architectures support FP32, FP64, FP16, and INT32 operations. CDNA2 and CDNA3 add FP64 GEMM, BF16 GEMM, and INT8 GEMM support. CDNA3 additionally supports TF32 GEMM and FP8/BF8, while CDNA4 supports FP8/BF8 but not FP32 Packed or TF32 GEMM.

## Key Hardware Blocks

This documentation covers five primary hardware components:

- **Compute Unit (CU)** — Processing core with pipeline descriptions, metrics, local data share, and vector L1 cache
- **L2 Cache (TCC)** — Secondary cache layer
- **Shader Engine (SE)** — Execution engine coordination
- **Command Processor (CP)** — Command queue management
- **System Speed-of-Light** — Theoretical performance ceiling metrics

Understanding these blocks helps developers interpret profiling data and optimize workload performance on AMD accelerators.
