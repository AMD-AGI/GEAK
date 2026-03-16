# Kernel: Custom All-Reduce Kernels

## Variant Context
- Input semantic type: Distributed communication (all-reduce for tensor parallelism)
- Datatype(s): bf16, fp16, fp32
- Data representation: Distributed tensors across GPUs
- Target architecture: CUDA (SM70+), ROCm (gfx942)

## Functionality
These kernels implement custom all-reduce operations for tensor-parallel LLM inference. They bypass NCCL/RCCL for lower latency in small-message scenarios:
- Intra-node all-reduce using shared memory or NVLink/xGMI
- Support for various reduction operations (sum, max)
- Optimized for LLM hidden dimensions

The custom implementation provides:
- Lower latency than NCCL for small messages
- Better integration with CUDA/HIP graphs
- Reduced synchronization overhead

## Optimization 1: Quick All-Reduce for ROCm
- Commit ID: c012ca5e3
- Optimization type: Communication / Latency
- Summary: Implemented quick all-reduce using shared memory for low-latency intra-node communication
- Detailed explanation:
  The quick all-reduce implementation uses:
  - Shared memory buffers accessible by all GPUs in a node
  - Two-phase algorithm: scatter-reduce then all-gather
  - Barrier synchronization using atomic operations
  
  This provides significantly lower latency than RCCL for small tensors.

- Code excerpt:
    ```cpp
    // From quick_ar_comm.cc
    class QuickARComm {
    public:
        void allReduce(void* input, void* output, size_t count, 
                       hipDataType dtype, hipStream_t stream) {
            // Phase 1: Each GPU reduces its portion
            int chunk_size = count / world_size_;
            int my_offset = rank_ * chunk_size;
            
            // Copy local data to shared buffer
            hipMemcpyAsync(shared_buffer_ + my_offset, 
                          input + my_offset,
                          chunk_size * sizeof(T), 
                          hipMemcpyDeviceToDevice, stream);
            
            // Barrier synchronization
            barrier_.wait(stream);
            
            // Phase 2: Reduce from all GPUs
            reduceKernel<<<grid, block, 0, stream>>>(
                output, shared_buffer_, count, world_size_);
            
            barrier_.wait(stream);
        }
        
    private:
        void* shared_buffer_;  // IPC shared memory
        int rank_;
        int world_size_;
        Barrier barrier_;
    };
    ```
- Evidence mapping:
  - Shared memory IPC → `shared_buffer_` accessible across GPUs
  - Two-phase algorithm → Scatter-reduce + all-gather pattern
  - Barrier sync → `barrier_.wait(stream)` for coordination

## Optimization 2: Custom All-Gather for ROCm
- Commit ID: 45f2bd7c3
- Optimization type: Communication
- Summary: Implemented custom all-gather using peer-to-peer memory access
- Detailed explanation:
  The custom all-gather uses direct GPU-to-GPU memory access:
  - Each GPU writes its data to a shared buffer
  - Other GPUs read directly from the buffer
  - Avoids intermediate copies through host memory

- Code excerpt:
    ```cpp
    // Custom all-gather implementation
    void customAllGather(void* output, const void* input, 
                         size_t count, hipStream_t stream) {
        int chunk_size = count / world_size_;
        
        // Each GPU writes to its portion of output
        for (int i = 0; i < world_size_; i++) {
            if (i == rank_) {
                // Local copy
                hipMemcpyAsync(output + rank_ * chunk_size,
                              input, chunk_size * sizeof(T),
                              hipMemcpyDeviceToDevice, stream);
            } else {
                // Remote read from peer GPU
                hipMemcpyAsync(output + i * chunk_size,
                              peer_buffers_[i] + i * chunk_size,
                              chunk_size * sizeof(T),
                              hipMemcpyDeviceToDevice, stream);
            }
        }
    }
    ```
- Evidence mapping:
  - Peer buffers → `peer_buffers_[i]` for direct GPU access
  - Chunk-based distribution → `chunk_size = count / world_size_`
  - Async operations → `hipMemcpyAsync` for overlap

## Optimization 3: CUDA Custom All-Reduce with NVLink
- Commit ID: (core implementation)
- Optimization type: Communication / Bandwidth
- Summary: Optimized all-reduce using NVLink for high-bandwidth GPU communication
- Detailed explanation:
  The CUDA implementation leverages NVLink:
  - Direct peer-to-peer memory access
  - Ring-based or tree-based reduction algorithms
  - Kernel-based reduction for compute overlap

- Code excerpt:
    ```cpp
    // From custom_ar_kernels.cu
    template<typename T>
    __global__ void customAllReduceKernel(
        T* output, T** inputs, int count, int world_size) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;
        
        // Reduce from all GPUs
        T sum = 0;
        #pragma unroll
        for (int i = 0; i < world_size; i++) {
            sum += inputs[i][idx];
        }
        
        output[idx] = sum;
    }
    
    // Ring all-reduce for larger messages
    template<typename T>
    void ringAllReduce(T* data, size_t count, int rank, int world_size,
                       cudaStream_t stream) {
        int chunk_size = count / world_size;
        
        // Reduce-scatter phase
        for (int step = 0; step < world_size - 1; step++) {
            int send_chunk = (rank - step + world_size) % world_size;
            int recv_chunk = (rank - step - 1 + world_size) % world_size;
            
            // Send and receive with reduction
            sendRecvReduce(data + send_chunk * chunk_size,
                          data + recv_chunk * chunk_size,
                          chunk_size, stream);
        }
        
        // All-gather phase
        for (int step = 0; step < world_size - 1; step++) {
            int send_chunk = (rank - step + 1 + world_size) % world_size;
            int recv_chunk = (rank - step + world_size) % world_size;
            
            sendRecv(data + send_chunk * chunk_size,
                    data + recv_chunk * chunk_size,
                    chunk_size, stream);
        }
    }
    ```
- Evidence mapping:
  - Direct reduction → `inputs[i][idx]` for peer memory access
  - Ring algorithm → Reduce-scatter + all-gather phases
  - Chunk-based → `chunk_size = count / world_size`

## Optimization 4: Tensor Parallel Size Optimization
- Commit ID: f15f52a53
- Optimization type: Configuration
- Summary: Enable custom all-reduce when TP size is less than local world size
- Detailed explanation:
  The optimization enables custom all-reduce for specific configurations:
  - When tensor parallel size < number of GPUs per node
  - Falls back to NCCL/RCCL for cross-node communication
  - Provides best of both worlds for hybrid parallelism

- Code excerpt:
    ```cpp
    // From ROCmDistributedOp.cc
    void ROCmDevice::allReduce(Buffer& buffer, ReduceOp op) {
        if (tp_size_ < local_world_size_ && use_custom_ar_) {
            // Use custom all-reduce for intra-node
            quick_ar_comm_->allReduce(
                buffer.data(), buffer.data(), 
                buffer.size(), buffer.dtype(), stream_);
        } else {
            // Fall back to RCCL for cross-node or large TP
            rcclAllReduce(buffer.data(), buffer.data(),
                         buffer.size(), dtype, rcclSum,
                         comm_, stream_);
        }
    }
    ```
- Evidence mapping:
  - Condition check → `tp_size_ < local_world_size_`
  - Custom path → `quick_ar_comm_->allReduce`
  - Fallback → `rcclAllReduce` for other cases
