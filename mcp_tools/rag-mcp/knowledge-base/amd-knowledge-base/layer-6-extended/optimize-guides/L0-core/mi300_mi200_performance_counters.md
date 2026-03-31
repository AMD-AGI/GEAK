---
tags: ["optimization", "performance", "profiling", "counters", "mi300", "mi200"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# MI300 and MI200 Series Performance Counters and Metrics

## Overview

This documentation details hardware performance counters and derived metrics available for AMD Instinct MI300 and MI200 GPUs. You can access this information using ROCprofiler-SDK.

**Note:** Preliminary validation of MI300 and MI200 Series performance counters is ongoing. Counters marked with an asterisk (*) require further evaluation.

## Performance Counter Categories

### Command Processor Counters

#### Command Processor-Fetcher

| Counter | Unit | Definition |
|---------|------|-----------|
| `CPF_CMP_UTCL1_STALL_ON_TRANSLATION` | Cycles | Compute unified translation cache (L1) stall cycles waiting on translation |
| `CPF_CPF_STAT_BUSY` | Cycles | Command processor-fetcher busy cycles |
| `CPF_CPF_STAT_IDLE` | Cycles | Command processor-fetcher idle cycles |
| `CPF_CPF_STAT_STALL` | Cycles | Command processor-fetcher stall cycles |
| `CPF_CPF_TCIU_BUSY` | Cycles | Texture cache interface unit interface busy cycles |
| `CPF_CPF_TCIU_IDLE` | Cycles | Texture cache interface unit interface idle cycles |
| `CPF_CPF_TCIU_STALL` | Cycles | Texture cache interface unit interface stall cycles |

#### Command Processor-Compute

| Counter | Unit | Definition |
|---------|------|-----------|
| `CPC_ME1_BUSY_FOR_PACKET_DECODE` | Cycles | Micro engine busy cycles decoding packets |
| `CPC_UTCL1_STALL_ON_TRANSLATION` | Cycles | Unified translation cache (L1) stall cycles |
| `CPC_CPC_STAT_BUSY` | Cycles | Command processor-compute busy cycles |
| `CPC_CPC_STAT_IDLE` | Cycles | Command processor-compute idle cycles |
| `CPC_CPC_STAT_STALL` | Cycles | Command processor-compute stall cycles |
| `CPC_CPC_TCIU_BUSY` | Cycles | Texture cache interface unit interface busy cycles |
| `CPC_CPC_TCIU_IDLE` | Cycles | Texture cache interface unit interface idle cycles |
| `CPC_CPC_UTCL2IU_BUSY` | Cycles | Unified translation cache (L2) interface busy cycles |
| `CPC_CPC_UTCL2IU_IDLE` | Cycles | Unified translation cache (L2) interface idle cycles |
| `CPC_CPC_UTCL2IU_STALL` | Cycles | Unified translation cache (L2) interface stall cycles |
| `CPC_ME1_DC0_SPI_BUSY` | Cycles | Micro engine processor busy cycles |

### Graphics Register Bus Manager (GRBM) Counters

| Counter | Unit | Definition |
|---------|------|-----------|
| `GRBM_COUNT` | Cycles | Free-running GPU cycles |
| `GRBM_GUI_ACTIVE` | Cycles | GPU active cycles |
| `GRBM_CP_BUSY` | Cycles | Command processor block busy cycles |
| `GRBM_SPI_BUSY` | Cycles | Shader processor input busy cycles |
| `GRBM_TA_BUSY` | Cycles | Texture addressing unit busy cycles |
| `GRBM_TC_BUSY` | Cycles | Texture cache block busy cycles |
| `GRBM_CPC_BUSY` | Cycles | Command processor-compute busy cycles |
| `GRBM_CPF_BUSY` | Cycles | Command processor-fetcher busy cycles |
| `GRBM_UTCL2_BUSY` | Cycles | Unified translation cache (L2) busy cycles |
| `GRBM_EA_BUSY` | Cycles | Efficiency arbiter block busy cycles |

### Shader Processor Input (SPI) Counters

| Counter | Unit | Definition |
|---------|------|-----------|
| `SPI_CSN_BUSY` | Cycles | Cycles with outstanding waves |
| `SPI_CSN_WINDOW_VALID` | Cycles | Enabled cycles by perfcounter_start event |
| `SPI_CSN_NUM_THREADGROUPS` | Workgroups | Dispatched workgroups count |
| `SPI_CSN_WAVE` | Wavefronts | Dispatched wavefronts count |
| `SPI_RA_REQ_NO_ALLOC` | Cycles | Arbiter cycles with requests but no allocation |
| `SPI_RA_RES_STALL_CSN` | Cycles | Stall cycles due to pipeline slot shortage |
| `SPI_RA_TMP_STALL_CSN` | Cycles | Stall cycles due to temp space shortage |
| `SPI_RA_WAVE_SIMD_FULL_CSN` | SIMD-cycles | Wave slot shortage impact |
| `SPI_RA_VGPR_SIMD_FULL_CSN` | SIMD-cycles | Vector GPR slot shortage impact |
| `SPI_RA_SGPR_SIMD_FULL_CSN` | SIMD-cycles | Scalar GPR slot shortage impact |
| `SPI_RA_LDS_CU_FULL_CSN` | CU | LDS space shortage impact |
| `SPI_RA_BAR_CU_FULL_CSN` | CU | Barrier wait compute units |
| `SPI_RA_BULKY_CU_FULL_CSN` | CU | BULKY resource wait compute units |
| `SPI_RA_TGLIM_CU_FULL_CSN` | Cycles | Thread group limit stall cycles |
| `SPI_RA_WVLIM_STALL_CSN` | Cycles | Wave limit stall cycles |
| `SPI_VWC_CSC_WR` | Qcycles | VGPR initialization quad-cycles |
| `SPI_SWC_CSC_WR` | Qcycles | SGPR initialization quad-cycles |

### Compute Unit Counters

#### Instruction Mix

The document lists extensive instruction counters tracking VALU operations across various data types (F16, F32, F64, INT32, INT64), memory operations (vector/scalar), and specialized matrix FMA instructions.

Key counters include:
- `SQ_INSTS` - Total instructions issued
- `SQ_INSTS_VALU` - Vector ALU instructions
- `SQ_INSTS_MFMA` - Matrix FMA instructions
- `SQ_INSTS_VMEM_RD` / `SQ_INSTS_VMEM_WR` - Vector memory operations
- `SQ_INSTS_SALU` - Scalar ALU instructions
- `SQ_INSTS_SMEM` - Scalar memory instructions
- `SQ_INSTS_LDS` - LDS instructions
- `SQ_INSTS_FLAT` - Flat instructions
- `SQ_INSTS_BRANCH` - Branch instructions

#### Matrix FMA Operations

| Counter | Unit | Definition |
|---------|------|-----------|
| `SQ_INSTS_VALU_MFMA_MOPS_I8` | IOP | 8-bit integer matrix FMA ops (units of 512) |
| `SQ_INSTS_VALU_MFMA_MOPS_F16` | FLOP | F16 floating matrix FMA ops (units of 512) |
| `SQ_INSTS_VALU_MFMA_MOPS_BF16` | FLOP | BF16 floating matrix FMA ops (units of 512) |
| `SQ_INSTS_VALU_MFMA_MOPS_F32` | FLOP | F32 floating matrix FMA ops (units of 512) |
| `SQ_INSTS_VALU_MFMA_MOPS_F64` | FLOP | F64 floating matrix FMA ops (units of 512) |

#### Level Counters

These measure inflight instruction counts. Use formula: Latency = `SQ_ACCUM_PREV_HIRES` ÷ instruction count

- Vector memory latency
- Wave latency
- LDS latency
- Scalar memory latency
- Instruction fetch latency

#### Wavefront Counters

Track wavefront dispatch states including full-width (64 threads), partial-width, and context save/restore operations.

#### Wavefront Cycle Counters

Measure time spent on specific instruction types:
- `SQ_ACTIVE_INST_VMEM` - Vector memory instruction cycles
- `SQ_ACTIVE_INST_LDS` - LDS instruction cycles
- `SQ_ACTIVE_INST_VALU` - VALU instruction cycles
- `SQ_ACTIVE_INST_SCA` - Scalar instruction cycles

#### LDS Counters

| Counter | Unit | Definition |
|---------|------|-----------|
| `SQ_LDS_ATOMIC_RETURN` | Cycles | Atomic return cycles |
| `SQ_LDS_BANK_CONFLICT` | Cycles | Bank conflict stall cycles |
| `SQ_LDS_ADDR_CONFLICT` | Cycles | Address conflict stall cycles |
| `SQ_LDS_UNALIGNED_STALL` | Cycles | Unaligned load/store stall cycles |
| `SQ_LDS_MEM_VIOLATIONS` | Count | Memory violation threads |
| `SQ_LDS_IDX_ACTIVE` | Cycles | Indexed operation cycles |

### L1 Instruction and Scalar L1 Data Cache Counters

| Counter | Unit | Definition |
|---------|------|-----------|
| `SQC_ICACHE_REQ` | Req | L1i cache requests |
| `SQC_ICACHE_HITS` | Count | L1i cache hits |
| `SQC_ICACHE_MISSES` | Count | L1i cache non-duplicate misses |
| `SQC_DCACHE_REQ` | Req | Scalar L1d requests |
| `SQC_DCACHE_HITS` | Count | Scalar L1d hits |
| `SQC_DCACHE_MISSES` | Count | Scalar L1d non-duplicate misses |
| `SQC_TC_REQ` | Req | Texture cache requests from instruction/constant caches |
| `SQC_TC_STALL` | Cycles | L2 cache request stall cycles |

### Vector L1 Cache Subsystem Counters

#### Texture Addressing Unit (TA)

16 instances (n=0-15) tracking:
- Buffer/flat wavefront processing
- Read/write/atomic operations
- Stall cycles by texture cache or data unit
- Coalesced cycle counts

#### Texture Data Unit (TD)

16 instances (n=0-15) tracking:
- Busy cycles and stall sources
- Wavefront load/store/atomic instructions
- Coalescable wavefront counts

#### Texture Cache Per Pipe (TCP)

16 instances (n=0-15) tracking:
- Clock gate cycles
- Tag conflict stalls (read/write/atomic)
- Pending data stalls
- Cache accesses (read/write/atomic)
- Translation cache requests and hits
- L2 cache request classifications by coherency mode

#### Texture Cache Arbiter (TCA)

32 instances (n=0-31) tracking busy and total cycles.

### L2 Cache Access Counters

The L2 cache, also known as texture cache per channel, maintains separate counter sets for MI300 and MI200 hardware.

#### MI300 Counters

Comprehensive tracking includes:
- `TCC_CYCLE[n]` - Free-running clocks
- `TCC_BUSY[n]` - Busy cycles
- `TCC_REQ[n]` - All request types
- `TCC_READ[n]`, `TCC_WRITE[n]`, `TCC_ATOMIC[n]` - Operation-specific counts
- `TCC_HIT[n]`, `TCC_MISS[n]` - Cache performance
- `TCC_WRITEBACK[n]` - Memory writebacks
- `TCC_EA0_WRREQ[n]` - Write requests to efficiency arbiter
- `TCC_EA0_RDREQ[n]` - Read requests to efficiency arbiter
- Various credit stall counters (IO, GMI, DRAM)
- `TCC_TAG_STALL[n]` - Tag pipeline stalls

#### MI200 Counters

Similar structure using `TCC_EA_*` naming convention instead of `TCC_EA0_*`.

**Latency Calculations:**
- Average write latency = `TCC_EA0_WRREQ_LEVEL` ÷ `TCC_EA0_WRREQ`
- Average atomic latency = `TCC_EA0_ATOMIC_LEVEL` ÷ `TCC_EA0_ATOMIC`
- Average read latency = `TCC_EA0_RDREQ_LEVEL` ÷ `TCC_EA0_RDREQ`

## Derived Metrics

The documentation provides a comprehensive list of 30+ derived metrics including:

- **`GPUBusy`** - GPU activity percentage
- **`L2CacheHit`** - L2 hit rate percentage (0-100%)
- **`LDSBankConflict`** - LDS bank conflict stall percentage
- **`MemUnitBusy`** - Memory unit active time percentage
- **`MemUnitStalled`** - Memory unit stall time percentage
- **`VALUUtilization`** - Active vector ALU thread percentage
- **`WriteUnitStalled`** - Write unit stall percentage
- Instruction counts (scalar/vector fetch/write operations)
- Data transfer metrics (FetchSize, WriteSize in kilobytes)

**Optimization guidance:** Reduce "ALUStalledByLDS" by minimizing LDS bank conflicts and access count. Lower "MemUnitStalled" by reducing fetch/write volume.

## Aggregated Metrics

Summed/aggregated metrics over all instances:
- `TA_*_sum` - Texture addressing unit totals
- `TCC_*_sum` - L2 cache aggregates
- `TCP_*_sum` - Vector L1 cache totals
- `TD_*_sum` - Texture data unit totals

Averaged metrics:
- `TCC_BUSY_avr` - Average L2 busy cycles
- `TA_BUSY_avr` - Average texture addressing busy cycles
