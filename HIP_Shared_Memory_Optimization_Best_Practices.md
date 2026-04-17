# HIP Shared Memory (LDS) Optimization Best Practices

## Overview

Shared memory on AMD GPUs is called **Local Data Share (LDS)**. It provides low-latency, high-bandwidth on-chip memory that can be used as a staging area to transform strided global memory access into coalesced patterns.

## AMD GPU Memory Hierarchy

1. **Registers**: ~256KB per CU, fastest
2. **LDS (Local Data Share)**: 64-128KB per CU, shared memory
3. **L1 Cache**: 16-32KB per CU
4. **L2 Cache**: 4-8MB, shared across GPU
5. **HBM (High Bandwidth Memory)**: 32-128GB, 1.6-3.2 TB/s

## LDS Performance Characteristics

- **Internal bandwidth**: ~20-40 TB/s (50-100x faster than global memory)
- **Latency**: 4-8 cycles vs 200-400 cycles for global memory
- **Capacity**: 64 KB per CU on MI300/MI250

## Key Optimization Techniques

### 1. Tiled Matrix Multiplication with Shared Memory