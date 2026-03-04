---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# MI300 and MI200 Series Performance Counters and Metrics

*Comprehensive guide to hardware performance counters and derived metrics for AMD Instinct MI300 and MI200 GPUs*

**Source:** [AMD Instinct Documentation - Performance Counters](https://instinct.docs.amd.com/latest/gpu-arch/mi300-mi200-performance-counters.html)

---

## Overview

This document describes the hardware performance counters and derived metrics available for AMD Instinct MI300 and MI200 GPUs. These counters are essential for:
- Performance profiling and optimization
- Identifying bottlenecks in GPU applications
- Understanding hardware utilization
- Optimizing kernel performance

**Note:** These counters can be accessed and analyzed using the ROCProfiler tool.

---

## Performance Counter Categories

Hardware performance counters are organized into the following categories:

1. **Command Processor Counters** - Command dispatch and execution tracking
2. **Graphics Register Bus Manager Counters** - GPU-wide activity monitoring
3. **Shader Processor Input Counters** - Wavefront scheduling and launch
4. **Compute Unit Counters** - Core execution metrics
5. **L1 Cache Counters** - Instruction and scalar data cache
6. **Vector L1 Cache Subsystem Counters** - Vector memory hierarchy
7. **L2 Cache Access Counters** - Unified L2 cache metrics

---

## 1. Command Processor Counters

The command processor manages kernel dispatch and execution. It's divided into two subsystems:

### Command Processor-Fetcher (CPF) Counters

Handles fetching commands from memory:

| Counter | Unit | Description |
|---------|------|-------------|
| `CPF_CMP_UTCL1_STALL_ON_TRANSLATION` | Cycles | Cycles stalled waiting on L1 address translation |
| `CPF_CPF_STAT_BUSY` | Cycles | Cycles the fetcher is busy |
| `CPF_CPF_STAT_IDLE` | Cycles | Cycles the fetcher is idle |
| `CPF_CPF_STAT_STALL` | Cycles | Cycles the fetcher is stalled |
| `CPF_CPF_TCIU_BUSY` | Cycles | Cycles texture cache interface is busy |
| `CPF_CPF_TCIU_IDLE` | Cycles | Cycles texture cache interface is idle |
| `CPF_CPF_TCIU_STALL` | Cycles | Cycles stalled waiting on free tags |

### Command Processor-Compute (CPC) Counters

Handles compute kernel dispatch:

| Counter | Unit | Description |
|---------|------|-------------|
| `CPC_ME1_BUSY_FOR_PACKET_DECODE` | Cycles | Cycles microengine is busy decoding packets |
| `CPC_UTCL1_STALL_ON_TRANSLATION` | Cycles | Cycles stalled waiting on L1 address translation |
| `CPC_CPC_STAT_BUSY` | Cycles | Cycles command processor-compute is busy |
| `CPC_CPC_STAT_IDLE` | Cycles | Cycles command processor-compute is idle |
| `CPC_CPC_STAT_STALL` | Cycles | Cycles command processor-compute is stalled |
| `CPC_CPC_TCIU_BUSY` | Cycles | Cycles texture cache interface is busy |
| `CPC_CPC_TCIU_IDLE` | Cycles | Cycles texture cache interface is idle |
| `CPC_CPC_UTCL2IU_BUSY` | Cycles | Cycles L2 translation cache interface is busy |
| `CPC_CPC_UTCL2IU_IDLE` | Cycles | Cycles L2 translation cache interface is idle |
| `CPC_CPC_UTCL2IU_STALL` | Cycles | Cycles L2 translation cache interface is stalled |
| `CPC_ME1_DC0_SPI_BUSY` | Cycles | Cycles microengine processor is busy |

**Key Insight:** High stall counts indicate memory translation bottlenecks or command queue issues.

---

## 2. Graphics Register Bus Manager (GRBM) Counters

GPU-wide activity monitoring:

| Counter | Unit | Description |
|---------|------|-------------|
| `GRBM_COUNT` | Cycles | Free-running GPU cycles (reference clock) |
| `GRBM_GUI_ACTIVE` | Cycles | GPU active cycles |
| `GRBM_CP_BUSY` | Cycles | Any command processor block busy |
| `GRBM_SPI_BUSY` | Cycles | Any shader processor input busy |
| `GRBM_TA_BUSY` | Cycles | Any texture addressing unit busy |
| `GRBM_TC_BUSY` | Cycles | Any texture cache block busy |
| `GRBM_CPC_BUSY` | Cycles | Command processor-compute busy |
| `GRBM_CPF_BUSY` | Cycles | Command processor-fetcher busy |
| `GRBM_UTCL2_BUSY` | Cycles | Unified L2 translation cache busy |
| `GRBM_EA_BUSY` | Cycles | Efficiency arbiter block busy |

**Usage:** Calculate GPU utilization as `GRBM_GUI_ACTIVE / GRBM_COUNT * 100%`

---

## 3. Shader Processor Input (SPI) Counters

Manages wavefront scheduling and dispatch to compute units:

| Counter | Unit | Description |
|---------|------|-------------|
| `SPI_CSN_BUSY` | Cycles | Cycles with outstanding waves |
| `SPI_CSN_WINDOW_VALID` | Cycles | Cycles enabled by performance counter start event |
| `SPI_CSN_NUM_THREADGROUPS` | Workgroups | Total threadgroups dispatched |
| `SPI_CSN_WAVE` | Wavefronts | Total wavefronts dispatched |

**Key Metric:** `SPI_CSN_BUSY / GRBM_COUNT` indicates wavefront scheduling efficiency.

---

## 4. Compute Unit (CU) Counters

Core execution metrics for compute units:

### Instruction Mix Counters

| Counter | Unit | Description |
|---------|------|-------------|
| `SQ_INSTS_VALU` | Instructions | Vector ALU instructions |
| `SQ_INSTS_VMEM` | Instructions | Vector memory instructions |
| `SQ_INSTS_SALU` | Instructions | Scalar ALU instructions |
| `SQ_INSTS_SMEM` | Instructions | Scalar memory instructions |
| `SQ_INSTS_FLAT` | Instructions | Flat memory instructions |
| `SQ_INSTS_FLAT_LDS_ONLY` | Instructions | LDS-only flat instructions |
| `SQ_INSTS_LDS` | Instructions | LDS instructions |
| `SQ_INSTS_GDS` | Instructions | Global data share instructions |
| `SQ_INSTS_BRANCH` | Instructions | Branch instructions |

### Matrix Operations Counters (MFMA)

Critical for AI/ML workloads:

| Counter | Unit | Description |
|---------|------|-------------|
| `SQ_INSTS_MFMA_MOPS_FP64` | Ops | FP64 matrix ops (per wave) |
| `SQ_INSTS_MFMA_MOPS_FP32` | Ops | FP32 matrix ops (per wave) |
| `SQ_INSTS_MFMA_MOPS_FP16` | Ops | FP16 matrix ops (per wave) |
| `SQ_INSTS_MFMA_MOPS_BF16` | Ops | BF16 matrix ops (per wave) |
| `SQ_INSTS_MFMA_MOPS_I8` | Ops | INT8 matrix ops (per wave) |

**AI/ML Performance:** These counters directly measure the utilization of Matrix Core units, critical for deep learning performance.

### Wavefront Management

| Counter | Unit | Description |
|---------|------|-------------|
| `SQ_WAVES` | Wavefronts | Number of waves sent to SQs |
| `SQ_WAVES_RESTORED` | Wavefronts | Waves restored from VMEM save area |
| `SQ_WAVES_SAVED` | Wavefronts | Waves saved to VMEM save area |
| `SQ_WAVE_CYCLES` | Cycles | Total wave cycles across all CUs |
| `SQ_BUSY_CYCLES` | Cycles | Number of busy cycles |
| `SQ_ACTIVE_INST_ANY` | Cycles | Cycles with any instruction active |
| `SQ_ACTIVE_INST_VALU` | Cycles | Cycles with VALU instruction active |
| `SQ_ACTIVE_INST_MFMA` | Cycles | Cycles with MFMA instruction active |

**Occupancy Calculation:**
```
Occupancy = SQ_WAVE_CYCLES / (Number_of_CUs × Max_Waves_per_CU × Total_Cycles)
```

### Local Data Share (LDS) Counters

On-chip shared memory metrics:

| Counter | Unit | Description |
|---------|------|-------------|
| `SQ_LDS_BANK_CONFLICT` | Cycles | LDS bank conflict stalls |
| `SQ_LDS_ADDR_CONFLICT` | Cycles | LDS address conflict stalls |
| `SQ_LDS_UNALIGNED_STALL` | Cycles | LDS unaligned access stalls |
| `SQ_LDS_MEM_VIOLATIONS` | Events | LDS memory access violations |

**Optimization Tip:** High bank conflicts indicate poor LDS access patterns requiring stride adjustments.

---

## 5. L1 Cache Counters

### L1 Instruction Cache (L1i)

| Counter | Unit | Description |
|---------|------|-------------|
| `SQC_INST_REQ` | Requests | Instruction fetch requests |
| `SQC_INST_LEVEL_CACHE_HIT` | Requests | L1 instruction cache hits |
| `SQC_DATA_LEVEL_CACHE_HIT` | Requests | L1 constant cache hits |
| `SQC_ICACHE_MISSES` | Misses | L1 instruction cache misses |
| `SQC_DCACHE_MISSES` | Misses | L1 constant cache misses |

### Scalar L1 Data Cache (L1d)

| Counter | Unit | Description |
|---------|------|-------------|
| `SQC_TC_REQ` | Requests | Requests to L2 cache |
| `SQC_TC_DATA_READ_REQ` | Requests | Data read requests to L2 |
| `SQC_TC_DATA_WRITE_REQ` | Requests | Data write requests to L2 |

**Cache Hit Rate:**
```
L1i Hit Rate = SQC_INST_LEVEL_CACHE_HIT / SQC_INST_REQ × 100%
```

---

## 6. Vector L1 Cache Subsystem Counters

### Texture Addressing Unit (TA) Counters

| Counter | Unit | Description |
|---------|------|-------------|
| `TA_FLAT_READ_WAVEFRONTS` | Wavefronts | Flat read ops |
| `TA_FLAT_WRITE_WAVEFRONTS` | Wavefronts | Flat write ops |
| `TA_FLAT_ATOMIC_WAVEFRONTS` | Wavefronts | Flat atomic ops |
| `TA_BUFFER_READ_WAVEFRONTS` | Wavefronts | Buffer read ops |
| `TA_BUFFER_WRITE_WAVEFRONTS` | Wavefronts | Buffer write ops |
| `TA_BUFFER_ATOMIC_WAVEFRONTS` | Wavefronts | Buffer atomic ops |
| `TA_ADDR_STALLED_BY_TC_CYCLES` | Cycles | Cycles stalled by texture cache |
| `TA_BUSY` | Cycles | Cycles TA is busy |

### Texture Cache Per Pipe (TCP) Counters

Vector L1 data cache:

| Counter | Unit | Description |
|---------|------|-------------|
| `TCP_TOTAL_CACHE_ACCESSES` | Accesses | Total L1 accesses (hits + misses) |
| `TCP_TCP_LATENCY` | Cycles | Wave access latency to vector L1d |
| `TCP_TCC_READ_REQ_LATENCY` | Cycles | Vector L1d to L2 read latency |
| `TCP_TCC_WRITE_REQ_LATENCY` | Cycles | Vector L1d to L2 write latency |
| `TCP_TCC_READ_REQ` | Requests | Read requests to L2 |
| `TCP_TCC_WRITE_REQ` | Requests | Write requests to L2 |
| `TCP_TCC_ATOMIC_WITH_RET_REQ` | Requests | Atomic with return to L2 |
| `TCP_TCC_ATOMIC_WITHOUT_RET_REQ` | Requests | Atomic without return to L2 |
| `TCP_PENDING_STALL_CYCLES` | Cycles | Stalled waiting on L2 data |
| `TCP_UTCL1_REQUEST` | Requests | Address translation requests |
| `TCP_UTCL1_TRANSLATION_HIT` | Hits | Address translation hits |
| `TCP_UTCL1_TRANSLATION_MISS` | Misses | Address translation misses |
| `TCP_UTCL1_PERMISSION_MISS` | Misses | Address translation permission misses |

**Cache Efficiency:**
```
TCP Hit Rate = (TCP_TOTAL_CACHE_ACCESSES - TCP_TCC_READ_REQ - TCP_TCC_WRITE_REQ) / TCP_TOTAL_CACHE_ACCESSES × 100%
```

---

## 7. L2 Cache Access Counters

Texture Cache per Channel (TCC) - unified L2 cache:

| Counter | Unit | Description |
|---------|------|-------------|
| `TCC_HIT` | Requests | L2 cache hits |
| `TCC_MISS` | Requests | L2 cache misses |
| `TCC_READ` | Requests | L2 read requests |
| `TCC_WRITE` | Requests | L2 write requests |
| `TCC_ATOMIC` | Requests | L2 atomic requests |
| `TCC_REQ` | Requests | Total L2 requests |
| `TCC_WRITEBACK` | Writebacks | Lines written back to memory |
| `TCC_EA_WRREQ` | Requests | Write requests to memory |
| `TCC_EA_WRREQ_64B` | Requests | 64-byte write requests |
| `TCC_EA_WRREQ_STALL` | Cycles | Cycles write requests stalled |
| `TCC_EA_RDREQ` | Requests | Read requests from memory |
| `TCC_EA_RDREQ_32B` | Requests | 32-byte read requests |
| `TCC_TOO_MANY_EA_WRREQS_STALL_CYCLES` | Cycles | Stalled due to too many writes |

**L2 Cache Metrics:**
```
L2 Hit Rate = TCC_HIT / TCC_REQ × 100%
Memory Bandwidth Utilization = (TCC_EA_RDREQ + TCC_EA_WRREQ) / Peak_Memory_Bandwidth
```

---

## Common Derived Metrics

### GPU Utilization
```
GPU_Active_Percentage = (GRBM_GUI_ACTIVE / GRBM_COUNT) × 100%
```

### Wavefront Occupancy
```
Occupancy = (SQ_WAVE_CYCLES / (Num_CUs × Max_Waves × Total_Cycles)) × 100%
```

### Memory Hierarchy Efficiency
```
L1_Hit_Rate = (TCP_TOTAL_CACHE_ACCESSES - TCP_TCC_READ_REQ) / TCP_TOTAL_CACHE_ACCESSES × 100%
L2_Hit_Rate = TCC_HIT / TCC_REQ × 100%
```

### Compute Efficiency
```
VALU_Efficiency = SQ_ACTIVE_INST_VALU / SQ_BUSY_CYCLES × 100%
MFMA_Efficiency = SQ_ACTIVE_INST_MFMA / SQ_BUSY_CYCLES × 100%
```

### Memory Bandwidth Utilization
```
Memory_BW_Used = (TCC_EA_RDREQ × 32B + TCC_EA_WRREQ_64B × 64B) / Time
Memory_BW_Percentage = Memory_BW_Used / Peak_Memory_BW × 100%
```

---

## Using ROCProfiler

### Basic Profiling

```bash
# Profile with specific counters
rocprof --stats --timestamp on \
  -i input.txt \
  -o output.csv \
  ./your_application

# Example input.txt with counters
pmc : SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM
pmc : TCC_HIT TCC_MISS TCC_REQ
pmc : GRBM_GUI_ACTIVE GRBM_COUNT
```

### Advanced Profiling with Metrics

```bash
# Profile derived metrics
rocprof --stats \
  -i metrics.txt \
  -o metrics_output.csv \
  ./your_application

# Example metrics.txt
pmc : GRBM_GUI_ACTIVE GRBM_COUNT
pmc : SQ_WAVE_CYCLES SQ_BUSY_CYCLES
pmc : TCC_HIT TCC_MISS TCC_REQ
```

### Python Integration

```python
import pandas as pd

# Read ROCProfiler output
df = pd.read_csv('output.csv')

# Calculate derived metrics
df['GPU_Utilization'] = (df['GRBM_GUI_ACTIVE'] / df['GRBM_COUNT']) * 100
df['L2_Hit_Rate'] = (df['TCC_HIT'] / df['TCC_REQ']) * 100

print(f"Average GPU Utilization: {df['GPU_Utilization'].mean():.2f}%")
print(f"Average L2 Hit Rate: {df['L2_Hit_Rate'].mean():.2f}%")
```

---

## Performance Analysis Workflow

### 1. Initial Assessment
```bash
# Get overall GPU utilization
rocprof -i basic_counters.txt ./app
# basic_counters.txt:
# pmc : GRBM_GUI_ACTIVE GRBM_COUNT GRBM_SPI_BUSY
```

### 2. Identify Bottlenecks

**Memory-bound indicators:**
- High `TCP_PENDING_STALL_CYCLES`
- Low L1/L2 hit rates
- High `TCC_EA_RDREQ_STALL`

**Compute-bound indicators:**
- High `SQ_ACTIVE_INST_VALU` or `SQ_ACTIVE_INST_MFMA`
- Low memory stalls
- High wavefront occupancy

**Launch-bound indicators:**
- Low `SPI_CSN_BUSY`
- Low wavefront occupancy
- Short kernel runtime

### 3. Deep Dive Analysis

For memory issues:
```bash
# Memory hierarchy analysis
rocprof -i memory_counters.txt ./app
# memory_counters.txt:
# pmc : TCP_TOTAL_CACHE_ACCESSES TCP_TCC_READ_REQ TCP_TCC_WRITE_REQ
# pmc : TCC_HIT TCC_MISS TCC_REQ TCC_WRITEBACK
# pmc : TCC_EA_RDREQ TCC_EA_WRREQ
```

For compute analysis:
```bash
# Instruction mix and MFMA utilization
rocprof -i compute_counters.txt ./app
# compute_counters.txt:
# pmc : SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_SALU
# pmc : SQ_INSTS_MFMA_MOPS_FP16 SQ_INSTS_MFMA_MOPS_BF16
# pmc : SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_MFMA
# pmc : SQ_WAVE_CYCLES SQ_BUSY_CYCLES
```

---

## Optimization Guidelines

### For High Memory Stalls
1. **Improve cache locality** - Restructure data access patterns
2. **Use LDS effectively** - Cache frequently accessed data in shared memory
3. **Increase arithmetic intensity** - Add compute operations between memory accesses
4. **Optimize memory coalescing** - Ensure aligned, contiguous memory accesses

### For Low Compute Utilization
1. **Increase occupancy** - Reduce register usage or shared memory per thread
2. **Minimize divergence** - Reduce branching within wavefronts
3. **Use MFMA instructions** - Leverage matrix cores for AI/ML workloads
4. **Optimize work distribution** - Balance workload across CUs

### For LDS Bank Conflicts
1. **Adjust access stride** - Use power-of-2 strides with offsets
2. **Pad shared memory** - Add padding to break conflict patterns
3. **Reorder data layout** - Reorganize shared memory structure

---

## MI300 vs MI200 Differences

### MI300 Specific Features
- **Enhanced matrix cores** - Higher MFMA throughput
- **Larger cache hierarchy** - More L2 cache per compute unit
- **Improved memory bandwidth** - HBM3 support
- **Unified memory architecture** - APU-style unified memory

### Counter Availability
Most counters are consistent between MI300 and MI200, but MI300 may have:
- Additional MFMA precision modes
- Enhanced memory compression counters
- Improved power monitoring counters

**Note:** Always verify counter availability for your specific GPU model using `rocprof --list-basic`.

---

## Quick Reference Commands

```bash
# List all available counters
rocprof --list-basic

# List derived metrics
rocprof --list-derived

# Basic profiling
rocprof --stats ./your_app

# Profile specific kernel
rocprof --stats --hip-trace ./your_app

# Generate timeline
rocprof --stats --timestamp on --hip-trace -o timeline.json ./your_app
```

---

## Troubleshooting

### Counter Read Failures
```bash
# Check if counters are available
rocprof --list-basic | grep COUNTER_NAME

# Verify GPU type
rocminfo | grep "Name:"

# Check permissions
ls -la /dev/kfd
```

### High Overhead
- Use fewer counters per run (max 4-6 per group)
- Profile in release mode, not debug
- Use sampling mode for long-running applications

### Inconsistent Results
- Run multiple iterations and average
- Ensure GPU is not being used by other processes
- Check thermal throttling with `rocm-smi`

---

## Additional Resources

- **ROCProfiler Documentation:** [ROCm Profiler Tools](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- **AMD Instinct Architecture:** [MI300 Architecture Guide](https://instinct.docs.amd.com/latest/gpu-arch/mi300-arch.html)
- **Performance Optimization:** [ROCm Performance Guide](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-arch.html)
- **CDNA ISA Documentation:** [CDNA3 ISA Reference](https://instinct.docs.amd.com/latest/gpu-arch/mi300-isa.html)

---

## Summary

MI300 and MI200 performance counters provide comprehensive visibility into:
- ✅ Command dispatch and scheduling
- ✅ Compute unit execution and instruction mix
- ✅ Memory hierarchy (L1, L2, HBM) performance
- ✅ Cache hit rates and memory bandwidth
- ✅ Wavefront occupancy and scheduling
- ✅ Matrix core (MFMA) utilization
- ✅ LDS usage and bank conflicts

**Key Takeaways:**
1. Start with high-level metrics (GPU utilization, occupancy)
2. Drill down into specific subsystems based on bottlenecks
3. Use ROCProfiler for automated counter collection
4. Combine multiple counter groups for comprehensive analysis
5. Validate optimizations with before/after profiling

For AI/ML workloads, focus on:
- MFMA instruction counters
- Memory bandwidth utilization
- Cache hierarchy efficiency
- Wavefront occupancy

---

*Last updated: 2025-11-04*
*AMD Instinct MI300/MI200 Series | CDNA Architecture*

