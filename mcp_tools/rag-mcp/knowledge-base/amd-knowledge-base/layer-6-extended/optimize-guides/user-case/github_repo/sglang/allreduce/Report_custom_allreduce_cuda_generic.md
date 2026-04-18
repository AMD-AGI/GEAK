# Kernel: Custom AllReduce Kernel

## Variant Context
- Input semantic type: Collective communication for tensor parallelism
- Datatype(s): FP16, BF16, FP32
- Data representation: Contiguous tensors across GPUs
- Target architecture: CUDA (NVLink), ROCm (xGMI/Infinity Fabric)

## Functionality
This kernel implements custom all-reduce operations optimized for LLM inference workloads. It provides lower latency than NCCL for small to medium tensor sizes commonly encountered in tensor-parallel inference, using direct GPU-to-GPU memory access via NVLink or PCIe.

## Optimization 1: Two-Phase AllReduce with Barrier Synchronization
- Commit ID: (custom_all_reduce.cu)
- Optimization type: Communication
- Summary: Implements a two-phase all-reduce using direct memory access and lightweight barrier synchronization.

- Detailed explanation:
  The custom all-reduce uses a two-phase approach:
  1. Phase 1: Each GPU writes its data to a shared buffer accessible by all GPUs
  2. Barrier: Lightweight GPU-side synchronization using atomic operations
  3. Phase 2: Each GPU reads and reduces data from all other GPUs
  
  This avoids the overhead of NCCL's more general-purpose implementation for small tensors.

- Code excerpt:
    ```cpp
    // Phase 1: Write local data to shared buffer
    __global__ void allreduce_phase1(
        float* local_data,
        float** remote_buffers,
        int rank,
        int world_size,
        int num_elements
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements) {
            // Write to position visible to all ranks
            remote_buffers[rank][idx] = local_data[idx];
        }
    }

    // Barrier synchronization
    __device__ void barrier_sync(
        volatile int* barrier_flags,
        int rank,
        int world_size
    ) {
        // Signal completion
        if (threadIdx.x == 0) {
            atomicAdd((int*)&barrier_flags[rank], 1);
        }
        __threadfence_system();
        
        // Wait for all ranks
        if (threadIdx.x == 0) {
            for (int i = 0; i < world_size; i++) {
                while (barrier_flags[i] < expected_count) {}
            }
        }
        __syncthreads();
    }

    // Phase 2: Read and reduce
    __global__ void allreduce_phase2(
        float* output,
        float** remote_buffers,
        int world_size,
        int num_elements
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements) {
            float sum = 0.0f;
            for (int r = 0; r < world_size; r++) {
                sum += remote_buffers[r][idx];
            }
            output[idx] = sum;
        }
    }
    ```

- Evidence mapping:
  - "Two-phase approach" → separate `phase1` and `phase2` kernels
  - "Direct memory access" → `remote_buffers[rank][idx]` accesses other GPU memory
  - "Lightweight barrier" → atomic operations instead of NCCL synchronization

## Optimization 2: One-Shot AllReduce for Small Tensors
- Commit ID: (custom_all_reduce.cu)
- Optimization type: Latency
- Summary: Fuses both phases into a single kernel for very small tensors to minimize kernel launch overhead.

- Detailed explanation:
  For tensors small enough to fit in shared memory or registers, the kernel fuses both phases:
  1. Load local data
  2. Exchange via shared memory or direct access
  3. Reduce and store
  
  This eliminates the barrier kernel launch between phases.

- Code excerpt:
    ```cpp
    __global__ void allreduce_oneshot(
        float* data,
        float** peer_ptrs,
        int rank,
        int world_size,
        int num_elements
    ) {
        extern __shared__ float smem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Load local value
        float local_val = (idx < num_elements) ? data[idx] : 0.0f;
        
        // Store to shared memory for exchange
        smem[threadIdx.x] = local_val;
        __syncthreads();
        
        // Signal ready and wait
        // ... barrier logic ...
        
        // Read from all peers and reduce
        float sum = local_val;
        for (int r = 0; r < world_size; r++) {
            if (r != rank) {
                sum += peer_ptrs[r][idx];
            }
        }
        
        if (idx < num_elements) {
            data[idx] = sum;
        }
    }
    ```

- Evidence mapping:
  - "Single kernel" → all operations in `allreduce_oneshot`
  - "Shared memory" → `extern __shared__ float smem[]`
  - "Fused reduce" → sum computed in same kernel as exchange

## Optimization 3: ROCm-Specific Memory Ordering
- Commit ID: (custom_all_reduce_hip.cuh)
- Optimization type: Memory / Synchronization
- Summary: Uses ROCm-specific memory ordering primitives for correct cross-GPU synchronization on AMD hardware.

- Detailed explanation:
  AMD GPUs have different memory consistency models than NVIDIA GPUs. This optimization uses HIP-specific memory fences and atomic operations to ensure correct ordering of memory operations across GPUs connected via xGMI or Infinity Fabric.

- Code excerpt:
    ```cpp
    // ROCm-specific memory fence
    #ifdef __HIP_PLATFORM_AMD__
    __device__ void cross_gpu_fence() {
        __threadfence_system();
        // Additional HIP-specific synchronization if needed
    }
    #endif

    // Atomic operations with proper memory ordering
    __device__ void signal_completion(volatile int* flag, int value) {
        #ifdef __HIP_PLATFORM_AMD__
        __atomic_store_n(flag, value, __ATOMIC_RELEASE);
        #else
        atomicExch((int*)flag, value);
        #endif
    }
    ```

- Evidence mapping:
  - "ROCm-specific" → `#ifdef __HIP_PLATFORM_AMD__`
  - "Memory ordering" → `__ATOMIC_RELEASE` for proper visibility
  - "System fence" → `__threadfence_system()` for cross-GPU coherence
