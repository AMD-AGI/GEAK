# Kernel: gather_points

## Variant Context
- Input semantic type: Point gathering for point cloud processing
- Datatype(s): fp32, fp16 (template-based, supports multiple types)
- Data representation: Dense tensors (B, C, N) for points, (B, M) for indices
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel gathers points from a point cloud based on indices. The forward pass gathers features from `points` tensor at locations specified by `idx`. The backward pass (gradient kernel) accumulates gradients back to the original point locations using atomic operations.

## Optimization 1: Precomputed Base Offsets with 64-bit Arithmetic (Gradient Kernel)
- Commit ID: baseline → optimized
- Optimization type: Compute / Memory
- Summary: Precompute base offsets using size_t to prevent integer overflow for large tensors
- Detailed explanation: The baseline uses pointer arithmetic with implicit int multiplication which can overflow for large tensors. The optimized version precomputes base offsets using size_t (64-bit) arithmetic to prevent overflow, then creates dedicated base pointer variables. This ensures correctness for large batch sizes and tensor dimensions.

- Code excerpt (baseline):
    ```cpp
    grad_out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    grad_points += bs_idx * c * n + c_idx * n;

    atomicAdd(grad_points + idx[0], grad_out[0]);
    ```

- Code excerpt (optimized):
    ```cpp
    // Precompute base offsets using 64-bit to avoid overflow for large tensors
    const size_t base_grad_out   = static_cast<size_t>(bs_idx) * static_cast<size_t>(c) * static_cast<size_t>(m)
                                 + static_cast<size_t>(c_idx) * static_cast<size_t>(m);
    const size_t base_idx        = static_cast<size_t>(bs_idx) * static_cast<size_t>(m);
    const size_t base_grad_points= static_cast<size_t>(bs_idx) * static_cast<size_t>(c) * static_cast<size_t>(n)
                                 + static_cast<size_t>(c_idx) * static_cast<size_t>(n);

    // Base pointers for the (b,c) slice; keep in registers
    const scalar_t* __restrict__ grad_out_base = grad_out + base_grad_out;
    const int*      __restrict__ idx_base      = idx + base_idx;
    scalar_t*       __restrict__ grad_points_base = grad_points + base_grad_points;
    ```

- Evidence mapping:
  - "64-bit arithmetic" → `static_cast<size_t>(bs_idx) * static_cast<size_t>(c) * static_cast<size_t>(m)`
  - "Precomputed offsets" → `const size_t base_grad_out = ...`
  - "Dedicated base pointers" → `grad_out_base`, `idx_base`, `grad_points_base`

## Optimization 2: Separated Load and Atomic Operations (Gradient Kernel)
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Load values into registers before performing atomic operation
- Detailed explanation: The baseline performs the index load and gradient load inline with the atomicAdd. The optimized version first loads the index and gradient value into local variables, then performs the atomicAdd. This separation can help with instruction scheduling and makes the memory access pattern clearer.

- Code excerpt (baseline):
    ```cpp
    atomicAdd(grad_points + idx[0], grad_out[0]);
    ```

- Code excerpt (optimized):
    ```cpp
    // Coalesced loads
    const int j = idx_base[pt_idx];
    const scalar_t g = grad_out_base[pt_idx];

    // Single atomicAdd per element, preserving original accumulation semantics
    atomicAdd(grad_points_base + j, g);
    ```

- Evidence mapping:
  - "Separated loads" → `const int j = idx_base[pt_idx];` and `const scalar_t g = grad_out_base[pt_idx];`
  - "Clear atomic operation" → `atomicAdd(grad_points_base + j, g);` with preloaded values

## Optimization 3: Const Qualifiers for Index Variables (Gradient Kernel)
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added const qualifiers to index variables for better compiler optimization
- Detailed explanation: The optimized version marks the block and thread index variables as const, helping the compiler understand these values won't change and potentially enabling better register allocation.

- Code excerpt (baseline):
    ```cpp
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    ```

- Code excerpt (optimized):
    ```cpp
    const int bs_idx = blockIdx.z;
    const int c_idx  = blockIdx.y;
    const int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    ```

- Evidence mapping:
  - "Const qualifiers" → `const int bs_idx`, `const int c_idx`, `const int pt_idx`
