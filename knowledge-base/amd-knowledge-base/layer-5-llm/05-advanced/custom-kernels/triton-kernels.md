---
layer: "5"
category: "advanced"
subcategory: "custom-kernels"
tags: ["triton", "kernels", "optimization", "hip", "mlir"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
difficulty: "expert"
estimated_time: "60min"
---

# Custom Kernels with Triton

Guide to writing custom GPU kernels for AMD using Triton.

**This documentation targets ROCm 7.0+ only.**

**Official ROCm Fork**: [https://github.com/ROCm/triton](https://github.com/ROCm/triton)  
**Upstream Repository**: [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)  
**Documentation**: [https://triton-lang.org/](https://triton-lang.org/)

> **About Triton**: Triton is a Python-based language and compiler for writing efficient custom GPU kernels. It provides automatic optimizations and works seamlessly with PyTorch on AMD GPUs via ROCm.

## Why Triton for Custom Kernels?

- **Python Syntax**: Write GPU kernels in familiar Python-like syntax
- **Automatic Optimization**: Compiler handles memory coalescing, shared memory, and other low-level details
- **AMD GPU Support**: Native ROCm support for MI300, MI250, MI200 series
- **Easy Integration**: Works directly with PyTorch tensors
- **Performance**: Often matches or exceeds hand-written HIP/CUDA kernels
- **Fused Operations**: Easy to write complex fused kernels (e.g., Flash Attention)

## Installation

```bash
# Install Triton for ROCm
pip install triton

# Verify installation
python -c "import triton; print(triton.__version__)"
```

For building from source, see the [Triton on ROCm](../../../layer-3-libraries/compilers/triton-on-rocm.md) guide.

## Basic Vector Addition Kernel

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector  
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Program ID - identifies which block this instance is processing
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform computation
    output = x + y
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)

# Python wrapper function
def add(x: torch.Tensor, y: torch.Tensor):
    # Allocate output tensor
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Define grid size (number of blocks to launch)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# Usage example
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
z = add(x, y)
print(f"Result shape: {z.shape}")
```

## Matrix Multiplication Kernel

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

## Performance Optimization

### 1. Block Size Tuning

Test different BLOCK_SIZE values for your specific workload:

```python
# Try different block sizes
for block_size in [128, 256, 512, 1024]:
    # Benchmark your kernel with different block sizes
    pass
```

### 2. Memory Coalescing

Ensure contiguous memory access patterns:

```python
# Good: Coalesced access
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)

# Bad: Strided access (slower)
offsets = pid + tl.arange(0, BLOCK_SIZE) * stride
data = tl.load(ptr + offsets)
```

### 3. Shared Memory Usage

Use `tl.dot` for efficient matrix operations that utilize shared memory:

```python
# Triton automatically uses shared memory for dot products
accumulator += tl.dot(a, b)
```

### 4. Auto-tuning

Use the `@triton.autotune` decorator to automatically find optimal configurations:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def optimized_kernel(...):
    pass
```

## AMD GPU-Specific Best Practices

1. **Target CDNA Architecture**: MI300/MI250/MI200 series have different optimal block sizes
2. **Use Block Size 256**: Often optimal for AMD CDNA GPUs
3. **Leverage LDS**: 64KB of LDS (Local Data Share) per CU on CDNA
4. **Profile with rocprof**: Use AMD's profiling tools
5. **Test on Target Hardware**: Performance characteristics vary between MI series

## Debugging Tips

```bash
# Enable debug output
export TRITON_INTERPRET=1  # Run in interpreter mode
export MLIR_ENABLE_DUMP=1  # Dump MLIR IR
export TRITON_PRINT_AUTOTUNING=1  # Show autotuning results

# Profile execution
rocprof --hip-trace python your_script.py
```

## References

### Official Documentation

- **[ROCm/triton GitHub](https://github.com/ROCm/triton)** - AMD ROCm fork
- **[Triton Documentation](https://triton-lang.org/)** - Official docs
- **[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)** - Step-by-step guides
- **[Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html)** - Complete API

### Related Guides

- [Triton on ROCm](../../../layer-3-libraries/compilers/triton-on-rocm.md) - Complete Triton guide
- [PyTorch with ROCm](../../../layer-4-frameworks/pytorch/pytorch-rocm-basics.md) - PyTorch integration
- [HIP Programming](../../../layer-2-compute-stack/hip/hip-basics.md) - Alternative kernel approach
- [GPU Optimization](../../../best-practices/performance/gpu-optimization.md) - General optimization tips

