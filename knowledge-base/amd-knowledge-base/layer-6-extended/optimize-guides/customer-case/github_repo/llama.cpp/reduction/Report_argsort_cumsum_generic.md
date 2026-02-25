# Kernel: Argsort and Cumsum (Reduction Operations)

## Variant Context
- Input semantic type: Sorting and prefix operations
- Datatype(s): FP32, INT32
- Data representation: Dense tensors
- Target architecture: Generic (NVIDIA, AMD)

## Functionality
These kernels implement sorting and prefix sum operations used in various parts of LLM inference:
- **Argsort**: Returns indices that would sort an array, used in top-k/top-p sampling
- **Cumsum**: Cumulative sum (prefix sum), used in top-p sampling and attention

Key features:
- CUB library integration for optimized primitives
- Warp-level and block-level algorithms
- Support for descending order sorting

---

## Optimization 1: CUB-based Argsort with Strided Iterators
- Commit ID: d1e355648
- Optimization type: Compute (library optimization)
- Summary: Replace custom init_offsets kernel with CUB strided iterators for argsort
- Detailed explanation: The argsort operation needs to track original indices while sorting values. This optimization uses CUB's strided iterator to generate indices on-the-fly, eliminating the need for a separate kernel to initialize the index array.

- Code excerpt:
    ```cpp
    // CUDA: Replace init_offsets kernel with iterators in cub-based argsort
    #include <cub/cub.cuh>
    #include <cub/iterator/counting_input_iterator.cuh>
    #include <cub/iterator/strided_input_iterator.cuh>
    
    template<typename T, bool DESCENDING>
    void argsort_cub(
        const T * __restrict__ input,
        int32_t * __restrict__ indices,
        const int n,
        cudaStream_t stream) {
        
        // Use counting iterator instead of explicit index array
        cub::CountingInputIterator<int32_t> counting_iter(0);
        
        // Temporary storage
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes,
            input, sorted_values,
            counting_iter, indices,
            n, 0, sizeof(T) * 8, stream);
        
        void * temp = allocate(temp_bytes);
        
        if constexpr (DESCENDING) {
            cub::DeviceRadixSort::SortPairsDescending(
                temp, temp_bytes,
                input, sorted_values,
                counting_iter, indices,
                n, 0, sizeof(T) * 8, stream);
        } else {
            cub::DeviceRadixSort::SortPairs(
                temp, temp_bytes,
                input, sorted_values,
                counting_iter, indices,
                n, 0, sizeof(T) * 8, stream);
        }
    }
    ```

- Evidence mapping:
  - "CUB integration" → `cub::DeviceRadixSort::SortPairs`
  - "Strided iterator" → `cub::CountingInputIterator` for indices
  - "No init kernel" → indices generated on-the-fly

---

## Optimization 2: Cumsum Race Condition Fix
- Commit ID: 5fa66c6e6
- Optimization type: Correctness (race condition)
- Summary: Fix race condition in cumsum kernel for correct parallel execution
- Detailed explanation: The parallel cumsum algorithm requires careful synchronization to avoid race conditions when combining partial sums across thread blocks. This fix ensures correct results for all input sizes.

- Code excerpt:
    ```cpp
    // cuda: fix race condition in cumsum
    template<typename T, int BLOCK_SIZE>
    __global__ void cumsum_block(
        const T * __restrict__ input,
        T * __restrict__ output,
        T * __restrict__ block_sums,
        const int n) {
        
        __shared__ T sdata[BLOCK_SIZE];
        
        const int tid = threadIdx.x;
        const int gid = blockIdx.x * BLOCK_SIZE + tid;
        
        // Load to shared memory
        sdata[tid] = gid < n ? input[gid] : 0;
        __syncthreads();
        
        // Inclusive scan within block
        for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
            T val = 0;
            if (tid >= offset) {
                val = sdata[tid - offset];
            }
            __syncthreads();  // Critical: sync before write
            sdata[tid] += val;
            __syncthreads();  // Critical: sync after write
        }
        
        // Write result
        if (gid < n) {
            output[gid] = sdata[tid];
        }
        
        // Store block sum for second pass
        if (tid == BLOCK_SIZE - 1) {
            block_sums[blockIdx.x] = sdata[tid];
        }
    }
    
    // Second pass: add block sums
    template<typename T>
    __global__ void cumsum_add_block_sums(
        T * __restrict__ output,
        const T * __restrict__ block_sums,
        const int n,
        const int block_size) {
        
        const int gid = blockIdx.x * block_size + threadIdx.x;
        if (gid < n && blockIdx.x > 0) {
            output[gid] += block_sums[blockIdx.x - 1];
        }
    }
    ```

- Evidence mapping:
  - "Race condition fix" → proper `__syncthreads()` placement
  - "Two-pass algorithm" → block scan + add block sums
  - "Correct synchronization" → sync before and after shared memory write

---

## Optimization 3: Additional Reduction Operations
- Commit ID: 389ac78b2
- Optimization type: Algorithm (new operations)
- Summary: Add TRI (triangular), SOLVE_TRI, and CUMSUM operations
- Detailed explanation: These operations are needed for various model architectures and training operations. The triangular operations are used for causal masking, and cumsum is used in sampling algorithms.

- Code excerpt:
    ```cpp
    // ggml: add ops SOFTPLUS, EXPM1, TRI, SOLVE_TRI, CUMSUM
    
    // Triangular matrix extraction
    template<typename T, bool UPPER>
    __global__ void tri_kernel(
        const T * __restrict__ input,
        T * __restrict__ output,
        const int n,
        const int m,
        const int k) {  // Diagonal offset
        
        const int row = blockIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (col >= m) return;
        
        bool keep;
        if constexpr (UPPER) {
            keep = col >= row + k;  // Upper triangular
        } else {
            keep = col <= row + k;  // Lower triangular
        }
        
        output[row * m + col] = keep ? input[row * m + col] : (T)0;
    }
    
    // Cumsum with exclusive option
    template<typename T, bool EXCLUSIVE>
    __global__ void cumsum_kernel(
        const T * __restrict__ input,
        T * __restrict__ output,
        const int n) {
        
        // Implementation using parallel scan
        ...
    }
    ```

- Evidence mapping:
  - "Triangular ops" → `TRI` for matrix extraction
  - "Cumsum variants" → inclusive and exclusive options
  - "Causal masking" → upper/lower triangular support
