# Kernel: Custom All-Reduce

## Variant Context
- Input semantic type: Collective communication (all-reduce)
- Datatype(s): fp16, bf16, fp32
- Data representation: Dense tensors for tensor parallelism
- Target architecture: Generic CUDA/HIP with NVLink/xGMI

## Functionality
The custom all-reduce kernel provides optimized collective communication for tensor parallelism in multi-GPU inference. It bypasses NCCL for small tensors where kernel launch overhead dominates, using direct GPU-to-GPU memory access via NVLink or PCIe.

## Optimization 1: One-Shot All-Reduce for Small Tensors
- Commit ID: (custom_all_reduce.cu initial implementation)
- Optimization type: Communication / Latency
- Summary: Implement low-latency all-reduce using direct P2P memory access
- Detailed explanation:
  For small tensors, NCCL's ring-based algorithm has high latency due to multiple communication rounds. This optimization uses a one-shot algorithm where each GPU writes to all other GPUs' buffers directly, then each GPU reduces locally.
- Code excerpt:
    ```cpp
    // One-shot all-reduce: each GPU writes to all peers
    template <typename T, int RANKS>
    __global__ void one_shot_allreduce_kernel(
        T* __restrict__ result,
        T* __restrict__ self_buffer,
        T** __restrict__ peer_buffers,  // Pointers to peer GPU buffers
        int* __restrict__ barrier,
        int rank,
        int num_elements) {
      
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      const int stride = gridDim.x * blockDim.x;
      
      // Step 1: Copy local data to all peer buffers
      for (int i = tid; i < num_elements; i += stride) {
        T val = self_buffer[i];
        // Write to each peer's receive buffer
        for (int r = 0; r < RANKS; r++) {
          if (r != rank) {
            peer_buffers[r][rank * num_elements + i] = val;
          }
        }
      }
      
      // Barrier synchronization
      __syncthreads();
      if (threadIdx.x == 0) {
        atomicAdd(barrier, 1);
        while (atomicLoad(barrier) < RANKS * gridDim.x) {}
      }
      __syncthreads();
      
      // Step 2: Reduce all received data
      for (int i = tid; i < num_elements; i += stride) {
        T sum = self_buffer[i];
        for (int r = 0; r < RANKS; r++) {
          if (r != rank) {
            sum += peer_buffers[rank][r * num_elements + i];
          }
        }
        result[i] = sum;
      }
    }
    ```
- Evidence mapping:
  - "Direct P2P access" → `peer_buffers[r][...]` writes to remote GPU memory
  - "One-shot algorithm" → Single write phase + single reduce phase
  - "Barrier sync" → Atomic counter for cross-GPU synchronization

## Optimization 2: Two-Shot All-Reduce for Medium Tensors
- Commit ID: (custom_all_reduce.cuh)
- Optimization type: Communication / Bandwidth
- Summary: Implement two-shot algorithm for better bandwidth utilization on medium tensors
- Detailed explanation:
  For medium-sized tensors, a two-shot algorithm provides better bandwidth utilization. In the first shot, each GPU reduces a portion of the data. In the second shot, the partial results are gathered.
- Code excerpt:
    ```cpp
    // Two-shot all-reduce: reduce-scatter + all-gather
    template <typename T, int RANKS>
    __global__ void two_shot_allreduce_kernel(
        T* __restrict__ result,
        T* __restrict__ self_buffer,
        T** __restrict__ peer_buffers,
        int* __restrict__ barriers,
        int rank,
        int num_elements) {
      
      const int chunk_size = num_elements / RANKS;
      const int my_chunk_start = rank * chunk_size;
      
      // Shot 1: Reduce-scatter
      // Each GPU is responsible for reducing one chunk
      for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
        T sum = self_buffer[my_chunk_start + i];
        for (int r = 0; r < RANKS; r++) {
          if (r != rank) {
            // Read peer's contribution to my chunk
            sum += peer_buffers[r][my_chunk_start + i];
          }
        }
        // Store reduced chunk
        self_buffer[my_chunk_start + i] = sum;
      }
      
      // Barrier
      barrier_sync(barriers, rank, RANKS);
      
      // Shot 2: All-gather
      // Copy reduced chunks from all peers
      for (int r = 0; r < RANKS; r++) {
        int chunk_start = r * chunk_size;
        for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
          if (r == rank) {
            result[chunk_start + i] = self_buffer[chunk_start + i];
          } else {
            result[chunk_start + i] = peer_buffers[r][chunk_start + i];
          }
        }
      }
    }
    ```
- Evidence mapping:
  - "Reduce-scatter" → Each GPU reduces its assigned chunk
  - "All-gather" → Reduced chunks collected from all GPUs
  - "Better bandwidth" → Each element transferred once per phase

## Optimization 3: Fused Residual Add with All-Reduce
- Commit ID: (custom_all_reduce.cu)
- Optimization type: Fusion
- Summary: Fuse residual connection addition with all-reduce operation
- Detailed explanation:
  In transformer models, all-reduce is often followed by a residual addition. This optimization fuses both operations, reducing memory traffic by avoiding a separate kernel for the addition.
- Code excerpt:
    ```cpp
    // Fused all-reduce + residual add
    template <typename T, int RANKS>
    __global__ void allreduce_residual_kernel(
        T* __restrict__ result,
        const T* __restrict__ input,
        const T* __restrict__ residual,
        T** __restrict__ peer_buffers,
        int* __restrict__ barrier,
        int rank,
        int num_elements) {
      
      // ... all-reduce computation ...
      
      // Final reduction with residual addition
      for (int i = tid; i < num_elements; i += stride) {
        T sum = input[i];
        for (int r = 0; r < RANKS; r++) {
          if (r != rank) {
            sum += peer_buffers[rank][r * num_elements + i];
          }
        }
        // Fused residual add
        result[i] = sum + residual[i];
      }
    }
    ```
- Evidence mapping:
  - "Fused addition" → `sum + residual[i]` in same kernel
  - "Reduced memory traffic" → Residual read once during reduction
  - "Single kernel" → No separate residual add kernel needed

## Optimization 4: Quick Reduce for Ultra-Low Latency
- Commit ID: (custom_quickreduce.cu)
- Optimization type: Latency
- Summary: Implement quick reduce algorithm for minimum latency on very small tensors
- Detailed explanation:
  For very small tensors (e.g., single tokens during decode), even the one-shot algorithm has too much overhead. Quick reduce uses a simpler synchronization mechanism and minimal memory operations.
- Code excerpt:
    ```cpp
    // Quick reduce for minimal latency
    template <typename T, int RANKS, int MAX_ELEMENTS>
    __global__ void quick_reduce_kernel(
        T* __restrict__ result,
        volatile T* __restrict__ buffers,  // Shared buffer for all ranks
        volatile int* __restrict__ flags,   // Synchronization flags
        int rank,
        int num_elements) {
      
      // Use registers for small data
      T local_data[MAX_ELEMENTS / blockDim.x];
      
      // Load local data
      for (int i = 0; i < MAX_ELEMENTS / blockDim.x; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < num_elements) {
          local_data[i] = result[idx];
          buffers[rank * MAX_ELEMENTS + idx] = local_data[i];
        }
      }
      
      // Memory fence and signal ready
      __threadfence_system();
      if (threadIdx.x == 0) {
        flags[rank] = 1;
      }
      
      // Wait for all ranks
      if (threadIdx.x < RANKS) {
        while (flags[threadIdx.x] == 0) {}
      }
      __syncthreads();
      
      // Reduce from all buffers
      for (int i = 0; i < MAX_ELEMENTS / blockDim.x; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < num_elements) {
          T sum = local_data[i];
          for (int r = 0; r < RANKS; r++) {
            if (r != rank) {
              sum += buffers[r * MAX_ELEMENTS + idx];
            }
          }
          result[idx] = sum;
        }
      }
      
      // Reset flag for next iteration
      if (threadIdx.x == 0) {
        flags[rank] = 0;
      }
    }
    ```
- Evidence mapping:
  - "Register storage" → `T local_data[]` for fast access
  - "Volatile buffers" → Ensures visibility across GPUs
  - "Flag-based sync" → Simpler than atomic barriers

## Optimization 5: Multi-Stage Pipeline
- Commit ID: (custom_all_reduce.cuh)
- Optimization type: Scheduling
- Summary: Implement multi-stage pipelining for overlapping communication and computation
- Detailed explanation:
  For larger tensors, pipelining allows overlapping data transfer with reduction computation. While one chunk is being transferred, the previous chunk is being reduced.
- Code excerpt:
    ```cpp
    // Pipelined all-reduce
    template <typename T, int RANKS, int NUM_STAGES>
    __global__ void pipelined_allreduce_kernel(
        T* __restrict__ result,
        T* __restrict__ self_buffer,
        T** __restrict__ peer_buffers,
        int rank,
        int num_elements) {
      
      const int chunk_size = num_elements / NUM_STAGES;
      
      // Double buffering for pipeline
      __shared__ T stage_buffer[2][CHUNK_SIZE];
      
      // Prologue: start first transfer
      async_copy(stage_buffer[0], peer_buffers, 0, chunk_size);
      
      for (int stage = 0; stage < NUM_STAGES; stage++) {
        int curr_buf = stage % 2;
        int next_buf = (stage + 1) % 2;
        
        // Start next transfer (if not last stage)
        if (stage < NUM_STAGES - 1) {
          async_copy(stage_buffer[next_buf], peer_buffers, 
                     (stage + 1) * chunk_size, chunk_size);
        }
        
        // Wait for current transfer
        wait_copy(curr_buf);
        
        // Reduce current chunk
        reduce_chunk(result + stage * chunk_size, 
                     stage_buffer[curr_buf], chunk_size);
      }
    }
    ```
- Evidence mapping:
  - "Double buffering" → `stage_buffer[2]` for overlap
  - "Async copy" → Transfer starts before previous reduce completes
  - "Pipeline stages" → `NUM_STAGES` chunks processed in sequence
