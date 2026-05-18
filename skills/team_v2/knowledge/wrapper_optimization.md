# Python Wrapper Optimization Patterns

When the GPU kernel itself is already fast (< 10us), the Python/C++ wrapper becomes the dominant bottleneck. On AMD MI300X, PyTorch framework overhead creates a ~14us floor per kernel call. Optimizing the wrapper can provide 2-5x additional speedup.

## Priority Order

### W0: Eliminate Unnecessary Memory Allocations

**`torch.empty()` instead of `torch.zeros()` / `new_zeros()`**
```python
# BAD: zeros() calls memset on GPU — wastes ~2us per allocation
idx = xyz.new_zeros((B, npoint, k), dtype=torch.int32)

# GOOD: empty() skips initialization — output will be fully written by kernel
idx = torch.empty((B, k, npoint), dtype=torch.int32, device=xyz.device)
```

**Remove unnecessary output buffers**: If callers don't need an intermediate result, don't allocate it.
```python
# BAD: Allocates dist2 buffer that nobody uses
dist2 = torch.empty((B, npoint, k), dtype=torch.float32, device=xyz.device)
kernel_launch(xyz, center_xyz, idx, dist2)

# GOOD: Kernel only writes idx, no dist2 allocation needed
kernel_launch(xyz, center_xyz, idx)  # Requires modifying C++ binding too
```

### W1: Eliminate Post-Kernel Copies

**Design kernel output format to match expected output**. If the caller expects `(B, K, M)` but the kernel outputs `(B, M, K)`, the Python wrapper must call `.transpose().contiguous()` which allocates a new tensor and copies all data (~3-5us).

Solution: modify the kernel to write directly in the expected output format:
```cpp
// Write in (B, K, M) format directly — no Python transpose needed
idx[bs * K * m + j * m + query] = result;  // (B, K, M) layout
// Instead of:
idx[bs * m * K + query * K + j] = result;  // (B, M, K) layout → needs transpose
```

The Python wrapper then returns `idx` directly without any post-processing.

### W2: Bypass `torch.autograd.Function` Overhead

`torch.autograd.Function.apply()` adds ~3-5us overhead per call (context creation, input checking, gradient tracking). For inference-only kernels or kernels called inside `@torch.no_grad()`, use a direct function:

```python
# BAD: autograd Function overhead (~3-5us per call)
class KNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, xyz, center_xyz):
        idx = xyz.new_zeros(...)
        knn_ext.knn_wrapper(B, N, npoint, k, xyz, center_xyz, idx, dist2)
        return idx.transpose(2, 1).contiguous()

result = KNN.apply(k, xyz, center_xyz)

# GOOD: Direct function call (~0us overhead)
@torch.no_grad()
def knn(k, xyz, center_xyz):
    idx = torch.empty(...)
    knn_ext.knn_wrapper_opt(B, N, npoint, k, xyz.contiguous(), center_xyz.contiguous(), idx)
    return idx
```

**When NOT to do this**: If the kernel has a backward pass (gradient computation), you must keep `torch.autograd.Function`.

### W3: Minimize `.contiguous()` Calls

`.contiguous()` on an already-contiguous tensor is free (returns self). But on non-contiguous tensors, it allocates + copies. Move contiguity checks to the C++ side or document input requirements.

```python
# Acceptable: .contiguous() on inputs that might not be contiguous
knn_ext.kernel(xyz.contiguous(), center_xyz.contiguous(), idx)

# Better: CHECK_CONTIGUOUS in C++ binding, require contiguous inputs
# Then in Python: just pass tensors without .contiguous()
```

### W4: Add Optimized Dispatch Paths

When the kernel has template-specialized variants (e.g., K=3,5,10), add explicit dispatch in the Python wrapper:

```python
if k in (3, 5, 10):
    # Fast path: specialized kernel, no dist2, direct output format
    idx = torch.empty((B, k, npoint), dtype=torch.int32, device=xyz.device)
    knn_ext.knn_wrapper_opt(B, N, npoint, k, transposed,
                            xyz.contiguous(), center_xyz.contiguous(), idx)
    return idx
else:
    # Fallback: generic kernel
    ...
```

### W5: Native Data Layout Support

If callers sometimes pass transposed data (e.g., (B,3,N) instead of (B,N,3)), write a kernel variant that reads the transposed layout directly instead of forcing a Python-side transpose:

```python
# BAD: Python transposes before kernel call (~20us for large tensors)
if transposed:
    xyz = xyz.transpose(2, 1).contiguous()

# GOOD: Kernel handles both layouts via template parameter
if transposed:
    knn_ext.kernel_transposed(B, N, M, k, xyz, center_xyz, idx)
else:
    knn_ext.kernel_standard(B, N, M, k, xyz, center_xyz, idx)
```

## C++ Binding Optimizations

### Reduce Binding Overhead
```cpp
// Fast path: skip CHECK_CONTIGUOUS if kernel handles non-contiguous
// Only CHECK_CUDA is strictly necessary
void knn_wrapper_opt(int b, int n, int m, int nsample, bool transposed,
    at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor idx_tensor) {
    CHECK_CUDA(xyz_tensor);
    CHECK_CUDA(new_xyz_tensor);
    // Skip CHECK_CONTIGUOUS — Python caller already ensures .contiguous()
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    knn_kernel_launcher_opt(b, n, m, nsample, transposed, xyz, new_xyz, idx, stream);
}
```

## Impact Estimates

| Optimization | Overhead Removed | Typical Savings |
|-------------|-----------------|-----------------|
| W0: empty vs zeros | GPU memset | 1-3us per alloc |
| W0: Remove unused buffers | Allocation + memset | 2-5us |
| W1: Direct output format | transpose + copy | 3-20us |
| W2: Bypass autograd | Context creation | 3-5us |
| W3: Skip contiguous | Copy if non-contiguous | 0-20us |
| W4: Specialized dispatch | Generic overhead | varies |
| W5: Native layout | transpose + copy | 3-20us |

Total potential savings: 10-50us per call. When the kernel GPU time is <5us, this is the difference between 50us and 15us per call (3.3x speedup from wrapper alone).

## When to Apply

Wrapper optimization becomes critical when:
1. All test cases run in similar time (~50us) regardless of problem size → framework overhead dominates
2. Kernel GPU time (measured by CUDA events around just the kernel) is <10us
3. Small shapes show <2x speedup while large shapes show >10x → small shapes are overhead-limited
4. Profile shows most time in PyTorch internals, not kernel compute

**TechLead**: If after Round 1 all benchmarks cluster around the same time (e.g., ~50us), assign one engineer to wrapper optimization in Round 2. This is NOT a kernel optimization — it requires modifying the Python wrapper and C++ binding files.
