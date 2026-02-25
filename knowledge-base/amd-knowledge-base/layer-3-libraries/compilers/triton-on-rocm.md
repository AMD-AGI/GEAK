---
layer: "3"
category: "triton"
subcategory: "compiler"
tags: ["triton", "kernel", "python", "optimization", "compiler", "mlir"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: false
last_updated: 2025-11-03
---

# Triton on AMD GPUs

Triton is a Python-based language and compiler for writing efficient GPU kernels with ROCm support.

**This documentation targets ROCm 7.0+ only.**

**Official ROCm Fork**: [https://github.com/ROCm/triton](https://github.com/ROCm/triton)  
**Upstream Repository**: [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)  
**Documentation**: [https://triton-lang.org/](https://triton-lang.org/)

> **About Triton**: Triton is a language and compiler for parallel programming. It aims to provide a Python-based programming environment to productively write custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.

## Key Features

- **Python-based DSL**: Write GPU kernels in Python syntax
- **MLIR Backend**: Built on MLIR compiler infrastructure (Version 2.0+)
- **Automatic Optimization**: Compiler handles many low-level optimizations
- **Hardware Support**: 
  - **AMD GPUs**: ROCm 5.2+ (MI300, MI250, MI200 series)
  - **NVIDIA GPUs**: Compute Capability 7.0+
  - **CPUs**: Under development
- **Easy Integration**: Works seamlessly with PyTorch
- **Auto-tuning**: Built-in performance tuning capabilities
- **Fused Operations**: Support for complex fused kernels like Flash Attention

## Installation

### Prerequisites

Before installing Triton with ROCm support, ensure you have:
- **ROCm 7.0.0 or 7.0.2** installed and configured
- **Python 3.8-3.12** (Python 3.10 or 3.11 recommended)
- **PyTorch 2.0+** with ROCm support

### Option 1: Using pip (Recommended)

```bash
# Install Triton for ROCm
pip install triton

# Verify installation
python -c "import triton; print(triton.__version__)"
python -c "import torch; import triton; print('Triton available:', triton.runtime.driver.get_current_target())"
```

### Option 2: Building from ROCm Fork

For the latest AMD GPU optimizations:

```bash
# Clone the official ROCm fork
git clone https://github.com/ROCm/triton.git
cd triton

# Install build dependencies
pip install ninja cmake wheel pybind11

# Build and install
pip install -e python

# Or with a virtualenv
python -m venv .venv --prompt triton
source .venv/bin/activate
pip install ninja cmake wheel pybind11
pip install -e python
```

### Option 3: Building with Custom LLVM

For advanced users who need specific LLVM modifications:

```bash
# Find the LLVM version Triton builds against
cat cmake/llvm-hash.txt
# Example output: 49af6502c6dcb4a7f7520178bd14df396f78240c

# Clone and checkout LLVM at the correct revision
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 49af6502c6dcb4a7f7520178bd14df396f78240c

# Build LLVM
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    ../llvm
ninja

# Build Triton with custom LLVM
cd /path/to/triton
export LLVM_BUILD_DIR=$HOME/llvm-project/build
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
pip install -e python
```

### Build Optimization Tips

```bash
# Use clang and lld for faster builds
export TRITON_BUILD_WITH_CLANG_LLD=true

# Use ccache to speed up incremental builds
export TRITON_BUILD_WITH_CCACHE=true

# Change Triton cache location
export TRITON_HOME=/path/to/custom/cache

# No build isolation for faster nop builds
pip install -e python --no-build-isolation
```

## Basic Triton Kernel

### Vector Addition

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block start offset
    block_start = pid * BLOCK_SIZE
    
    # Offsets for current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checking
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Allocate output
    output = torch.empty_like(x)
    
    # Check tensors are on GPU
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# Usage
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
z = add(x, y)
```

## Matrix Multiplication

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
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to first blocks
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Inner loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store result
    c = accumulator.to(tl.float16)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )
    
    return c
```

## Fused Operations

### Fused Softmax

```python
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Row index
    row_idx = tl.program_id(0)
    
    # Row start pointer
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load input
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Softmax computation
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    
    # Block size must be power of 2 and >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Launch kernel
    softmax_kernel[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

# Usage
x = torch.randn(1024, 512, device='cuda')
y = softmax(x)
```

## Flash Attention in Triton

```python
@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, K,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Implementation of Flash Attention
    # Simplified for demonstration
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # ... (full implementation would be here)
```

## Performance Optimization

### Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def optimized_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Kernel implementation
    pass
```

### Benchmarking

```python
import triton.testing as testing

@testing.perf_report(
    testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(10, 20)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={}
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    
    gbps = lambda ms: 3 * size * 4 / ms * 1e-6  # 3 arrays, 4 bytes, convert to GB/s
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# Run benchmark
benchmark.run(show_plots=True, print_data=True)
```

## AMD-Specific Considerations

### Architecture Awareness

```python
# Check for AMD GPU
import torch
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    
    # Adjust block sizes for AMD architecture
    if "MI" in device_name:  # AMD Instinct GPUs
        BLOCK_SIZE = 256  # Often optimal for CDNA
    else:
        BLOCK_SIZE = 128
```

### Memory Hierarchy

```python
# Leverage LDS (Local Data Share) on AMD GPUs
@triton.jit
def kernel_with_lds(
    input_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Shared memory (maps to LDS on AMD)
    shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # ... use shared memory for cooperative operations
```

## Debugging

```python
# Enable debug mode
import os
os.environ['TRITON_DEBUG'] = '1'
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'

# Print generated code
@triton.jit
def debug_kernel(...):
    pass

# Inspect generated IR
print(debug_kernel.get_compiled_kernel(...).asm['ttgir'])
```

## Testing

### Running Tests

```bash
# One-time setup
pip install scipy numpy torch pytest lit pandas matplotlib
pip install -e python

# Run Python tests
python3 -m pytest python/test/unit

# Move to build directory (adjust path for your system)
cd python/build/cmake.linux-x86_64-cpython-3.11

# Run C++ unit tests
ctest -j32

# Run lit tests
lit test
```

### Creating Symlink for Convenience

```bash
# Create symlink to build directory
ln -s python/build/cmake.linux-x86_64-cpython-3.11 build
echo build >> .git/info/exclude

# Rebuild and run lit tests
ninja -C build && (cd build ; lit test)
```

## Debugging and Development

### Helpful Environment Variables

```bash
# Dump IR before every MLIR pass
export MLIR_ENABLE_DUMP=1

# Dump IR for specific kernel only
export MLIR_ENABLE_DUMP=kernelName

# Dump LLVM IR before every pass
export LLVM_IR_ENABLE_DUMP=1

# Use Triton interpreter instead of GPU
export TRITON_INTERPRET=1

# Enable LLVM debug output
export TRITON_ENABLE_LLVM_DEBUG=1

# Limit debug output to specific passes
export TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"

# Use IR location instead of Python line numbers
export USE_IR_LOC=ttir  # or ttgir

# Print autotuning results
export TRITON_PRINT_AUTOTUNING=1

# Disable LLVM optimizations
export DISABLE_LLVM_OPT=true

# Force kernel compilation (ignore cache)
export TRITON_ALWAYS_COMPILE=1

# Dump timing information
export MLIR_ENABLE_TIMING=1
export LLVM_ENABLE_TIMING=1

# Override FP fusion behavior
export TRITON_DEFAULT_FP_FUSION=true

# Enable performance warnings
export MLIR_ENABLE_REMARK=1
```

### Debugging with VSCode

1. **Do a local build**: `pip install -e python`
2. **Find compile_commands.json**:
   ```bash
   find python/build -name 'compile_commands.json' | xargs readlink -f
   ```
3. **Configure VSCode**:
   - Install C/C++ extension
   - Open Command Palette (Shift+Cmd+P on Mac, Shift+Ctrl+P on Linux/Windows)
   - Select "C/C++: Edit Configurations (UI)"
   - Paste full path to `compile_commands.json` in "Compile Commands" textbox

## AMD-Specific Optimizations

### Architecture Awareness

```python
# Check for AMD GPU and adjust accordingly
import torch
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    
    # Adjust block sizes for AMD CDNA architecture
    if "MI300" in device_name:
        BLOCK_SIZE = 256  # Optimal for MI300 series
    elif "MI250" in device_name:
        BLOCK_SIZE = 256  # Optimal for MI250 series
    elif "MI200" in device_name:
        BLOCK_SIZE = 256  # Optimal for MI200 series
    else:
        BLOCK_SIZE = 128  # Default
```

### Memory Hierarchy on AMD GPUs

```python
# Leverage LDS (Local Data Share) on AMD GPUs
@triton.jit
def amd_optimized_kernel(
    input_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Shared memory (maps to LDS on AMD CDNA)
    # LDS is 64KB per CU on MI300/MI250
    shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Use shared memory for cooperative operations
    # This is particularly effective on AMD CDNA architecture
    pid = tl.program_id(0)
    # ... kernel implementation
```

### ROCm-Specific Profiling

```bash
# Profile Triton kernels with rocprof
rocprof --hip-trace python your_triton_script.py

# Profile with more detailed metrics
rocprof --stats python your_triton_script.py

# Use rocprof with specific counters
rocprof --input counter_input.txt python your_triton_script.py
```

## Best Practices

1. **Use autotuning** for optimal block sizes across different AMD GPU models
2. **Minimize memory transfers** between kernel calls
3. **Leverage shared memory (LDS)** for data reuse on AMD CDNA architecture
4. **Profile with rocprof** to identify bottlenecks specific to AMD GPUs
5. **Test on target hardware** - MI300, MI250, MI200 have different characteristics
6. **Use fused operations** to reduce memory bandwidth requirements
7. **Consider memory hierarchy** - LDS size is 64KB per CU on CDNA GPUs
8. **Enable debug output** during development with environment variables
9. **Use version control** for Triton cache to ensure reproducibility
10. **Benchmark against PyTorch** native operations for validation

## Version 2.0 Features

Triton 2.0 includes major improvements:

- **MLIR Backend**: Complete rewrite using MLIR compiler infrastructure
- **Better Performance**: Improved code generation and optimization
- **Bug Fixes**: Many stability and correctness improvements
- **Flash Attention Support**: Native support for back-to-back matmuls
- **Enhanced Debugging**: Better error messages and debug tools

## References

### Official Resources

- **[ROCm/triton GitHub](https://github.com/ROCm/triton)** - Official AMD ROCm fork (136 stars)
- **[triton-lang/triton GitHub](https://github.com/triton-lang/triton)** - Upstream Triton repository
- **[Triton Documentation](https://triton-lang.org/)** - Official documentation and tutorials
- **[Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html)** - Complete API reference

### Getting Started

- **[Installation Guide](https://triton-lang.org/main/getting-started/installation.html)** - Setup instructions
- **[Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)** - Step-by-step tutorials
- **[Vector Addition Tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)** - First Triton kernel
- **[Flash Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)** - Advanced fused kernels

### Community & Support

- **[Triton Discussions](https://github.com/triton-lang/triton/discussions)** - Community Q&A
- **[ROCm Community](https://community.amd.com/t5/rocm/ct-p/amd-rocm)** - AMD ROCm forums
- **[Contributing Guide](https://github.com/ROCm/triton/blob/main/CONTRIBUTING.md)** - How to contribute

### Related Guides

- [PyTorch with ROCm](../../layer-4-frameworks/pytorch/pytorch-rocm-basics.md)
- [HIP Programming](../../layer-2-compute-stack/hip/hip-basics.md)
- [GPU Optimization Best Practices](../../best-practices/performance/gpu-optimization.md)
- [Custom Kernels with Triton](../../layer-5-llm/05-advanced/custom-kernels/triton-kernels.md)

