# Python Wrapper Optimization Patterns

When the GPU kernel itself is already fast (< 10us), the Python/C++ wrapper becomes the dominant bottleneck. On AMD Instinct (MI300-series and newer), PyTorch framework overhead creates a roughly ~10–15us floor per kernel call (varies by card, ROCm/PyTorch version, and host — measure it on your box). Optimizing the wrapper can provide 2-5x additional speedup.

## Priority Order

### W0: Eliminate Unnecessary Memory Allocations

**`torch.empty()` instead of `torch.zeros()` / `new_zeros()`**
```python
# BAD: zeros() calls memset on GPU — wastes ~2us per allocation
out = input.new_zeros((B, M, K), dtype=torch.int32)

# GOOD: empty() skips initialization — output will be fully written by kernel
out = torch.empty((B, M, K), dtype=torch.int32, device=input.device)
```

**Remove unnecessary output buffers**: If callers don't need an intermediate result, don't allocate it.
```python
# BAD: Allocates a scratch buffer that nobody uses downstream
scratch = torch.empty((B, M, K), dtype=torch.float32, device=input.device)
my_ext.kernel(input, query, output, scratch)

# GOOD: Kernel only writes output, no scratch allocation needed
my_ext.kernel(input, query, output)  # Requires modifying C++ binding too
```

### W1: Eliminate Post-Kernel Copies

**Design kernel output format to match expected output**. If the caller expects `(B, K, M)` but the kernel outputs `(B, M, K)`, the Python wrapper must call `.transpose().contiguous()` which allocates a new tensor and copies all data (~3-5us).

Solution: modify the kernel to write directly in the expected output format:
```cpp
// Write in (B, K, M) format directly — no Python transpose needed
out[bs * K * M + j * M + query] = result;  // (B, K, M) layout
// Instead of:
out[bs * M * K + query * K + j] = result;  // (B, M, K) layout → needs transpose
```

The Python wrapper then returns `out` directly without any post-processing.

### W2: Bypass `torch.autograd.Function` Overhead

`torch.autograd.Function.apply()` adds ~3-5us overhead per call (context creation, input checking, gradient tracking). For inference-only kernels or kernels called inside `@torch.no_grad()`, use a direct function:

```python
# BAD: autograd Function overhead (~3-5us per call)
class MyKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, query):
        out = input.new_zeros(...)
        my_ext.kernel(input, query, out)
        return out.transpose(2, 1).contiguous()

result = MyKernel.apply(input, query)

# GOOD: Direct function call (~0us overhead)
@torch.no_grad()
def my_kernel(input, query):
    out = torch.empty(...)
    my_ext.kernel_opt(input.contiguous(), query.contiguous(), out)
    return out
```

**When NOT to do this**: If the kernel has a backward pass (gradient computation), you must keep `torch.autograd.Function`.

### W3: Minimize `.contiguous()` Calls

`.contiguous()` on an already-contiguous tensor is free (returns self). But on non-contiguous tensors, it allocates + copies. Move contiguity checks to the C++ side or document input requirements.

```python
# Acceptable: .contiguous() on inputs that might not be contiguous
my_ext.kernel(input.contiguous(), query.contiguous(), out)

# Better: CHECK_CONTIGUOUS in C++ binding, require contiguous inputs
# Then in Python: just pass tensors without .contiguous()
```

### W4: Add Optimized Dispatch Paths

When the kernel has template-specialized variants, add explicit dispatch in the Python wrapper:

```python
if param in SPECIALIZED_VALUES:
    # Fast path: specialized kernel, direct output format
    out = torch.empty((...), dtype=torch.int32, device=input.device)
    my_ext.kernel_opt(B, N, M, param, input.contiguous(), query.contiguous(), out)
    return out
else:
    # Fallback: generic kernel
    ...
```

### W5: Native Data Layout Support

If callers sometimes pass transposed data (e.g., (B,D,N) instead of (B,N,D)), write a kernel variant that reads the transposed layout directly instead of forcing a Python-side transpose:

```python
# BAD: Python transposes before kernel call (~20us for large tensors)
if transposed:
    input = input.transpose(2, 1).contiguous()

# GOOD: Kernel handles both layouts via template parameter
if transposed:
    my_ext.kernel_transposed(B, N, M, input, query, out)
else:
    my_ext.kernel_standard(B, N, M, input, query, out)
```

## C++ Binding Optimizations

### Reduce Binding Overhead
```cpp
// Fast path: skip CHECK_CONTIGUOUS if Python caller already ensures .contiguous()
// Only CHECK_CUDA is strictly necessary
void kernel_opt(int b, int n, int m, int param, bool transposed,
    at::Tensor input_tensor, at::Tensor query_tensor, at::Tensor out_tensor) {
    CHECK_CUDA(input_tensor);
    CHECK_CUDA(query_tensor);
    const float *input = input_tensor.data_ptr<float>();
    const float *query = query_tensor.data_ptr<float>();
    int *out = out_tensor.data_ptr<int>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kernel_launcher_opt(b, n, m, param, transposed, input, query, out, stream);
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
