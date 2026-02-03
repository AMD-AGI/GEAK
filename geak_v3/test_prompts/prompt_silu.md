# HIP Kernel Optimization Task

Please optimize the HIP code implementation for better performance on the ROCm platform.

## Target Hardware: AMD MI308 GPU
- 64KB LDS (Local Data Share) per Compute Unit (CU)
- 80 CUs total

## Optimization Guidelines

### Memory & Data Access
1. **Chunked processing**: Divide large data into fixed-size chunks (e.g., threads × items/elements) to fit in registers/shared memory, enable streaming computation, and minimize global memory accesses. Process each chunk independently while carrying over state.

2. **Shared memory for state propagation**: Use shared memory as a buffer to handle inter-chunk dependencies, avoiding redundant global memory reads. Store and shift data for efficient access by threads.

3. **Delayed operations**: Postpone writes to shared memory until after dependent reads to prevent data races and overwrites, ensuring correct sequential dependencies.

4. **Vectorized I/O**: Perform loads/stores in vector types (e.g., 4 or 8 elements for float/half) for coalesced memory access. Use direct mode for aligned data or warp-transpose for flexibility, reducing instruction count and boosting bandwidth.

5. **Bounded accesses**: Implement conditional checks in loads/stores (e.g., if index < length) to safely handle variable data sizes and prevent out-of-bounds errors.

### Parallelism & Efficiency
6. **CUB primitives**: Employ CUB library for parallel operations: BlockLoad/BlockStore for efficient, coalesced input/output with temporary shared memory; BlockScan for prefix computations where needed.

7. **Loop unrolling**: Apply `#pragma unroll` to inner loops (e.g., over dimensions or elements) to reduce branching overhead and enable compiler optimizations like instruction scheduling.

8. **Branch divergence minimization**: Structure code to minimize divergent branches within warps, ensuring threads execute the same path where possible.

9. **Instruction-level parallelism**: Maximize ILP by interleaving independent instructions to hide latencies.

### Resource Management
10. **Resource limiting for occupancy**: Reduce shared memory (LDS) and register usage per workgroup to boost occupancy, allowing more concurrent workgroups per CU for improved parallelism and latency hiding.

11. **Type and feature handling**: Use templates for data types (e.g., float/half/bf16, optional complex); boolean switches for optional features like activations.

### AMD-Specific Optimizations
12. **Performance-enhancing techniques specific to AMD GPUs**: Apply AMD-specific optimizations like wavefront management or ROCm-tuned configurations.

13. **Kernel fusion or splitting opportunities**: Fuse multiple kernels to reduce launches and global memory traffic, or split for better resource utilization.

14. **Stream and asynchronous execution**: Use ROCm streams for overlapping computation and data transfer asynchronously.

You can apply other optimization techniques that fit the kernel.

## Critical Requirements

### Code Integrity (MUST follow)
1. MUST keep the exact same kernel function name
2. MUST maintain the same kernel function signature and parameter types
3. MUST keep the same kernel launch configuration structure
4. MUST ensure the code is directly compilable and runnable
5. MUST preserve the same algorithm logic and correctness
6. MUST maintain the same comments and code formatting style
7. If a kernel parameter is not used, you should remove it and not return it in the code
8. MUST define `shared_memory_size` before kernel launch if using shared memory

## Expected Deliverables

Return the optimized implementation including:
1. The optimized kernel function with the exact same name and signature
2. Any modified kernel launch parameters (if needed)
3. Any additional helper functions or kernels (if needed)
4. Any changes to the launch configuration (if needed)

**The code must be directly compilable and runnable with the same interface as the original implementation. Do not modify the input types and values used when calling the kernel in the main function.**
