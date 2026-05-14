# AMD MI300X Architecture & Optimization Guide

## Hardware Specifications

- **Architecture**: CDNA 3 (gfx942)
- **Compute Units (CUs)**: 304
- **Wavefront Size**: 64 (some kernels can use 32)
- **Peak HBM Bandwidth**: ~5.3 TB/s (8 stacks HBM3)
- **LDS per CU**: 64 KB (32 banks, 4 bytes per bank)
- **VGPRs per CU**: 512 (Vector General Purpose Registers)
- **SGPRs per CU**: 108 (Scalar General Purpose Registers)
- **MFMA (Matrix Fused Multiply-Add)**: Available for dense linear algebra
- **L1 Cache per CU**: 32 KB
- **L2 Cache**: 256 MB (shared across all CUs)
- **Wave Slots per SIMD**: 8 (maximum occupancy)

## Key Architecture Concepts

### Wavefront Execution
- A wavefront is 64 work-items executing in lockstep (SIMD)
- Each CU has 4 SIMD units, each can run 1 wavefront at a time
- Maximum theoretical occupancy: 4 SIMDs x 8 wave slots = 32 wavefronts per CU
- Occupancy is limited by: VGPR usage, LDS usage, wavefront count

### Memory Hierarchy (fastest to slowest)
1. **Registers (VGPRs/SGPRs)**: Fastest, per-wavefront, limited count
2. **LDS (Local Data Share)**: 64 KB per CU, shared within workgroup, ~2 TB/s effective BW
3. **L1 Cache**: 32 KB per CU, hardware-managed
4. **L2 Cache**: 256 MB shared, ~4 TB/s effective BW
5. **HBM (Global Memory)**: ~5.3 TB/s peak, highest latency

### MFMA Instructions
- Matrix Fused Multiply-Add for dense math (GEMM, convolutions)
- Available sizes: 16x16, 32x32 (fp16, bf16, fp32, int8)
- Much higher throughput than scalar VALU for matrix operations
- Triton's `tl.dot` maps to MFMA on AMD GPUs

## GPU and Profiler Rules (CRITICAL)

1. **HIP_VISIBLE_DEVICES is ALREADY SET** in the environment by the scheduler.
   Do NOT prefix commands with `HIP_VISIBLE_DEVICES=X`. Do NOT set or export it.
   Adding it inline will CRASH rocprofv3.

2. **Profile commands**: Pass ONLY the python command, e.g.:
   `python3 /path/to/harness.py --profile`
   Do NOT prefix with env vars -- rocprofv3 uses `os.execvpe()`, not a shell.

3. **Use absolute paths** in all commands. Do not use `cd /path && ...`.

4. **rocprof-compute** (formerly omniperf) is the profiling tool. It generates
   detailed per-kernel metrics including cache hit rates, wavefront stats,
   instruction mix, and bottleneck analysis.

## Optimization Principles for MI300X

### Memory Access Patterns
- **Coalesced access is critical**: Adjacent threads must access adjacent memory addresses
- **Vector loads (float4/float2)**: Use to maximize HBM bandwidth utilization
- **LDS bank conflicts**: 32 banks, 4 bytes each. Pad arrays to avoid stride-32 patterns
- **Cache line size**: 128 bytes. Align data structures accordingly

### Occupancy Tuning
- Higher occupancy helps hide memory latency (more wavefronts to schedule)
- But higher occupancy means fewer registers per wavefront
- Trade-off: register pressure vs. latency hiding
- Use `waves_per_eu` to control occupancy in Triton

### Instruction Mix
- VALU (Vector ALU): General arithmetic
- VMEM (Vector Memory): Global/L2 memory access
- MFMA: Matrix operations
- LDS: Local data share operations
- Balance between these determines kernel performance profile
