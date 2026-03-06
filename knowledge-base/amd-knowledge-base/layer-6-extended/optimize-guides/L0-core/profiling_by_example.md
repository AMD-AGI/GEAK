Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "profiling", "rocprofiler", "examples"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/projects/rocprofiler-compute/tutorial/profiling-by-example.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Profiling by Example — ROCm Compute Profiler 3.3.1

## Overview

This documentation provides practical examples of using ROCm Compute Profiler for performance analysis on AMD CDNA accelerators. Examples reference HIP code in the ROCm/rocm-systems repository.

## VALU Arithmetic Instruction Mix

The instruction mix example demonstrates counting different types of VALU operations using inline assembly. The sample kernel includes:

- 32-bit floating point addition
- 32-bit floating point multiplication
- 32-bit floating point square-root
- 32-bit floating point fused multiply-add

**Compilation:**
```bash
hipcc -O3 instmix.hip -o instmix
rocprof-compute profile -n instmix --no-roof -- ./instmix
```

Results show exactly one instruction of each arithmetic type per wave, including INT32, INT64, F16/F32/F64 operations (ADD, MUL, FMA, transcendental, and conversion).

## Infinity Fabric Transactions

Seven experiments explore Infinity Fabric behavior across different memory types and locations:

### Experiment 1: Coarse-grained, Accelerator-Local HBM Reads

Local device memory shows:
- 64-byte read requests dominate (>99%)
- ~40 GiB/kernel read bandwidth
- 100% HBM traffic from local accelerator
- Minimal uncached reads for kernel code/arguments

### Experiment 2: Fine-grained, Accelerator-Local HBM Reads

Fine-grained local memory produces nearly identical results to coarse-grained, with slightly increased HBM stalls from additional Infinity Fabric stress.

### Experiment 3: Fine-grained, Remote-Accelerator HBM Reads

Remote accelerator access shows:
- 100% remote read traffic
- Uncached reads at ~200% (each 64B request counted twice per request flow)
- 17.85% Infinity Fabric stalls
- Higher latency than local memory

### Experiment 4: Fine-grained, CPU-DRAM Reads

CPU DRAM access exhibits:
- 100% remote read traffic
- 200% uncached read traffic
- 91.29% PCIe stalls
- Significantly higher stalls than Infinity Fabric connections

### Experiment 5: Coarse-grained, CPU-DRAM Reads

Coarse-grained CPU memory differs by:
- Removal of uncached read traffic
- Conversion to standard 64-byte reads
- Maintained PCIe stalls (~91%)

### Experiment 6: Fine-grained, CPU-DRAM Writes

Write operations demonstrate:
- 64-byte write requests
- 100% remote write and atomic traffic
- 100% uncached write traffic
- No write stalls (non-posted over PCIe)

### Experiment 7: Fine-grained, CPU-DRAM atomicAdd

Atomic operations show:
- 32-byte atomic requests (vs. 64-byte for reads/writes)
- 100% atomic traffic classification
- ~0.4 GiB bandwidth (reduced problem size)
- System-scope atomics to fine-grained memory

## Vector Memory Operation Counting

### Global/Generic (FLAT) Operations

**Global Write:**
```c
__global__ void global_write(int* ptr, int zero) {
  ptr[threadIdx.x] = zero;
}
```
Generates one Global/Generic Write instruction via `global_store_dword`.

**Generic Write to LDS:**
Uses address space casting to force generic FLAT instructions targeting local memory. Results show one LDS instruction when targeting local data share.

**Global Read:**
Single global/generic read instruction accessing global memory via vector L1 cache.

**Generic Read from Global Memory:**
Generic FLAT instruction dynamically targeting global memory shows one L1-L2 read request.

**Global Atomic:**
Single atomic instruction produces one L1-L2 atomic request.

**Generic Mixed Atomic:**
Generic atomic targeting both LDS and global memory shows one LDS instruction plus one L1-L2 atomic request, demonstrating dynamic address space resolution.

### Spill/Scratch (BUFFER)

Stack memory example using private address space accessed via buffer instructions:

```c
__global__ void knl(int* out, int filter) {
  int x[1024];
  x[filter] = 0;
  if (threadIdx.x < filter)
    out[threadIdx.x] = x[threadIdx.x];
}
```

Results show stack writes backed by global memory, traveling through the same memory hierarchy with one L1-L2 write request.

## Instructions-Per-Cycle and Utilization

### VALU Operations

Simple `v_mov_b32` instruction achieves:
- IPC: 1.0 instructions/cycle
- VALU Utilization: ~99.98%
- 64 active threads (full wavefront)

### MFMA Operations

Matrix instruction `v_mfma_f32_32x32x8bf16_1k` (64-cycle execution):
- IPC: 0.0626 instructions/cycle
- Issued IPC: 1.0 instructions/cycle
- MFMA Utilization: ~99.99%
- 64 quad-cycles per instruction

The difference between IPC and Issued IPC reflects scheduler cycles versus total active CU cycles.

### Internal Instructions

No-op instruction `s_nop 0x0` demonstrates:
- IPC: 6.79 instructions/cycle
- Issued IPC: 1.0
- Internal instructions don't consume functional units

## LDS Examples

### LDS Bandwidth

Local data share performance metrics measure throughput and utilization for shared memory operations.

### Bank Conflicts

Analysis identifies conflicts in LDS bank access patterns affecting performance.

## Occupancy Limiters

### VGPR Limited

Maximum occupancy constrained by vector register availability.

### LDS Limited

Occupancy constrained by local data share allocation.

### SGPR Limited

Occupancy constrained by scalar register availability.

---

**Note:** All examples target CDNA accelerators (MI250/MI250X). Results vary by ROCm version, hardware generation, and system configuration.
