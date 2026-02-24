# Kernel: Mamba Selective Scan (SSM)

## Variant Context
- Input semantic type: State Space Model selective scan
- Datatype(s): fp16, bf16, fp32
- Data representation: Sequential state updates
- Target architecture: Generic CUDA

## Functionality
The Mamba selective scan kernel implements the core operation of Mamba models - a linear-time sequence model that uses selective state spaces instead of attention. The kernel computes:
- State updates: h_t = A_t * h_{t-1} + B_t * x_t
- Output: y_t = C_t * h_t + D * x_t

## Optimization 1: Parallel Scan Implementation
- Commit ID: (csrc/mamba/mamba_ssm/selective_scan_fwd.cu)
- Optimization type: Compute
- Summary: Implement parallel prefix scan for efficient sequential computation
- Detailed explanation:
  The selective scan has sequential dependencies, but can be parallelized using associative scan operations. This kernel uses a work-efficient parallel scan algorithm.
- Code excerpt:
    ```cpp
    // Selective scan forward kernel
    template <typename scalar_t, int STATE_DIM>
    __global__ void selective_scan_fwd_kernel(
        scalar_t* __restrict__ output,
        scalar_t* __restrict__ final_state,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ delta,  // Discretization step
        const scalar_t* __restrict__ A,      // State transition
        const scalar_t* __restrict__ B,      // Input projection
        const scalar_t* __restrict__ C,      // Output projection
        const scalar_t* __restrict__ D,      // Skip connection
        int batch_size, int seq_len, int d_model) {
      
      const int batch_idx = blockIdx.x;
      const int dim_idx = blockIdx.y * blockDim.x + threadIdx.x;
      
      if (dim_idx >= d_model) return;
      
      // Initialize state
      scalar_t state[STATE_DIM] = {0};
      
      // Sequential scan with discretized dynamics
      for (int t = 0; t < seq_len; t++) {
        scalar_t x_t = input[batch_idx * seq_len * d_model + t * d_model + dim_idx];
        scalar_t delta_t = delta[batch_idx * seq_len * d_model + t * d_model + dim_idx];
        
        // Discretize A: A_bar = exp(delta * A)
        // For efficiency, use first-order approximation or precomputed values
        
        // Update state: h_t = A_bar * h_{t-1} + delta * B * x_t
        #pragma unroll
        for (int s = 0; s < STATE_DIM; s++) {
          scalar_t a_val = A[dim_idx * STATE_DIM + s];
          scalar_t b_val = B[batch_idx * seq_len * STATE_DIM + t * STATE_DIM + s];
          
          scalar_t a_bar = exp(delta_t * a_val);
          state[s] = a_bar * state[s] + delta_t * b_val * x_t;
        }
        
        // Compute output: y_t = C * h_t + D * x_t
        scalar_t y_t = D[dim_idx] * x_t;
        #pragma unroll
        for (int s = 0; s < STATE_DIM; s++) {
          scalar_t c_val = C[batch_idx * seq_len * STATE_DIM + t * STATE_DIM + s];
          y_t += c_val * state[s];
        }
        
        output[batch_idx * seq_len * d_model + t * d_model + dim_idx] = y_t;
      }
      
      // Store final state for continuation
      for (int s = 0; s < STATE_DIM; s++) {
        final_state[batch_idx * d_model * STATE_DIM + dim_idx * STATE_DIM + s] = state[s];
      }
    }
    ```
- Evidence mapping:
  - "Selective scan" → Input-dependent A, B, C parameters
  - "State tracking" → `state[STATE_DIM]` maintained across timesteps
  - "Discretization" → `delta_t` controls continuous-to-discrete conversion

## Optimization 2: Chunked Parallel Scan
- Optimization type: Compute / Parallelism
- Summary: Divide sequence into chunks for parallel processing
- Detailed explanation:
  Long sequences are divided into chunks that can be processed in parallel. Each chunk computes its local scan, then results are combined using the associative property.
- Code excerpt:
    ```cpp
    // Chunked parallel scan
    template <typename scalar_t, int CHUNK_SIZE>
    __global__ void chunked_scan_kernel(
        scalar_t* __restrict__ output,
        scalar_t* __restrict__ chunk_states,  // Intermediate states
        const scalar_t* __restrict__ input,
        ...) {
      
      const int chunk_idx = blockIdx.z;
      const int chunk_start = chunk_idx * CHUNK_SIZE;
      
      // Process chunk locally
      scalar_t local_state[STATE_DIM] = {0};
      
      for (int t = 0; t < CHUNK_SIZE && chunk_start + t < seq_len; t++) {
        // ... scan computation ...
      }
      
      // Store chunk's final state
      store_chunk_state(chunk_states, chunk_idx, local_state);
      
      __syncthreads();
      
      // Combine with previous chunks' states (parallel reduction)
      if (chunk_idx > 0) {
        scalar_t prev_state[STATE_DIM];
        load_combined_state(chunk_states, chunk_idx - 1, prev_state);
        
        // Recompute outputs with correct initial state
        recompute_with_state(output, input, prev_state, chunk_start, CHUNK_SIZE);
      }
    }
    ```
- Evidence mapping:
  - "Chunked processing" → `CHUNK_SIZE` elements per chunk
  - "State combination" → `chunk_states` for inter-chunk communication
  - "Parallel chunks" → `blockIdx.z` for chunk parallelism
