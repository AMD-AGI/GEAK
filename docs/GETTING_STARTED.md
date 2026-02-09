# 🚀 Getting Started with Mini-Kernel Agent

This guide helps you quickly get started optimizing your GPU kernels.

## Table of Contents

1. [Quick Start (2 minutes)](#quick-start)
2. [How to Prepare Your Kernel](#how-to-prepare-your-kernel)
3. [What the Agent Does](#what-the-agent-does)
4. [Examples](#examples)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Setup (one-time)

```bash
# Clone the repository
git clone git@github.com:AMD-AGI/GEAK-agent.git
cd GEAK-agent

# Make CLI executable
chmod +x mini-kernel setup.sh

# Pull Docker image (takes a few minutes)
./setup.sh
```

### Step 2: Optimize Your Kernel

```bash
./mini-kernel optimize path/to/your/kernel.py --gpu 0
```

**That's it!** The agent will:
- ✅ Analyze your kernel
- ✅ Auto-generate comprehensive tests (LOW/MEDIUM/HIGH)
- ✅ Verify correctness
- ✅ Benchmark performance
- ✅ Apply optimizations
- ✅ Report results

---

## How to Prepare Your Kernel

### Minimum Requirements

Your kernel file needs **two functions**:

```python
# 1. Main kernel function (REQUIRED)
def triton_op(x, y):
    """Your optimized kernel - this is what gets optimized"""
    # ... your kernel code ...
    return result

# 2. Reference function (REQUIRED for correctness)
def torch_op(x, y):
    """Simple PyTorch equivalent for correctness checking"""
    return x + y  # Whatever your kernel does
```

### Recommended (Optional)

```python
# 3. Baseline runner (helps with benchmarking)
def run_baseline():
    """Creates test data and runs the kernel"""
    x = torch.randn(1024*1024, device='cuda')
    y = torch.randn(1024*1024, device='cuda')
    return triton_op(x, y)
```

### Function Naming Convention

The agent looks for functions in this order:

| Purpose | Names (in priority order) |
|---------|---------------------------|
| **Main kernel** | `triton_op`, `triton_add`, `torch_op`, `main`, `forward`, `run` |
| **Reference** | `torch_op`, `torch_add`, `ref_op`, `reference` |

> 💡 **Tip:** Name your main function `triton_op` and reference `torch_op` for best results.

---

## What the Agent Does

### 1. Kernel Analysis
```
[1/5] ANALYZING KERNEL & GENERATING TEST HARNESS...
  Kernel type: elementwise     ← Auto-detected!
  Main function: triton_op     ← Found your kernel
  Reference function: torch_op ← Found reference
  ✓ Test harness generated
```

### 2. Comprehensive Test Generation

The agent auto-generates tests across three categories:

| Category | Purpose | Example Tests |
|----------|---------|---------------|
| **LOW** | Edge cases | 1 element, 64 elements, prime sizes |
| **MEDIUM** | Typical workloads | 64K, 256K, 1M elements, FP16/BF16 |
| **HIGH** | Stress tests | 16M, 64M, 128M elements |

```
[LOW] Running 6 test cases...
  ✓ PASS | single_element  |      1 | Single element
  ✓ PASS | warp_64         |     64 | One warp size
  ✓ PASS | prime           |  1,009 | Prime number size

[MEDIUM] Running 5 test cases...
  ✓ PASS | typical_1m      | 1,048,576 | 42.3 μs | 1M elements
  ✓ PASS | fp16_1m         | 1,048,576 | 38.1 μs | FP16 precision
```

### 3. Optimization Strategies

| Strategy | What it Does | When it Helps |
|----------|--------------|---------------|
| **HIP Graph** | Captures kernel for replay | High launch overhead |
| **torch.compile** | JIT optimization | General speedup |
| **Combined** | Both above | Maximum speedup |

### 4. Final Report

```
======================================================================
  OPTIMIZATION COMPLETE
======================================================================
  Test Results:
    Total tests: 11
    Passed: 11 (100.0%)

  Performance:
    Baseline:     50.23 μs
    Best:         42.18 μs
    Speedup:      1.19x
    Best method:  HIP Graph
======================================================================
```

---

## Examples

### Example 1: Vector Addition (Elementwise)

```python
#!/usr/bin/env python3
"""my_add.py"""
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def triton_op(x, y):  # Main kernel - REQUIRED
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK']),)
    _add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    return out

def torch_op(x, y):  # Reference - REQUIRED
    return x + y
```

**Run:** `./mini-kernel optimize my_add.py --gpu 0`

### Example 2: Matrix Multiplication

```python
#!/usr/bin/env python3
"""my_matmul.py"""
import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(...):
    # Your matmul implementation
    pass

def triton_op(a, b):  # Main kernel
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    # Launch kernel...
    return c

def torch_op(a, b):  # Reference
    return torch.matmul(a, b)

def run_baseline():  # Optional but helpful
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    return triton_op(a, b)
```

### Example 3: Attention

```python
#!/usr/bin/env python3
"""my_attention.py"""
import torch
import triton
import triton.language as tl

@triton.jit
def _attention_kernel(...):
    pass

def triton_op(q, k, v, scale=None):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # Launch kernel...
    return output

def torch_op(q, k, v, scale=None):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)
```

---

## Troubleshooting

### "No main kernel function found"

**Symptom:** `Main function: None`

**Fix:** Rename your main function to `triton_op`:
```python
def triton_op(x, y):  # Use this exact name!
    return your_kernel(x, y)
```

### "No reference function found"

**Symptom:** Tests use fallback and may fail

**Fix:** Add a `torch_op` function:
```python
def torch_op(x, y):
    return x + y  # Simple PyTorch equivalent
```

### "Tests fail but kernel works"

**Symptom:** FAIL on tests, but you know the kernel is correct

**Possible causes:**
1. Input shapes don't match what your kernel expects
2. Tolerance is too strict for FP16/BF16

**Fix:** Add explicit `run_baseline()`:
```python
def run_baseline():
    # Define inputs exactly as your kernel expects
    x = torch.randn(YOUR_SHAPE, device='cuda', dtype=YOUR_DTYPE)
    return triton_op(x)
```

### "HIP Graph capture failed"

**Symptom:** `HIP error: operation failed during capture`

**This is usually fine!** Some kernels can't be graph-captured. The agent automatically falls back to other optimizations.

### "All optimizations failed"

**Symptom:** Best speedup is 1.0x

**Possible reasons:**
1. Kernel is already well-optimized
2. Kernel has dynamic shapes (can't be captured)
3. GPU error from previous runs (try restarting Docker)

---

## CLI Reference

```bash
# Basic usage
./mini-kernel optimize <kernel.py> --gpu <id>

# Examples
./mini-kernel optimize my_kernel.py --gpu 0
./mini-kernel optimize examples/add_kernel/kernel.py --gpu 3

# Test the example
./mini-kernel test examples/add_kernel/kernel.py

# Get help
./mini-kernel help
```

---

## Need Help?

1. Check `examples/add_kernel/kernel.py` for a complete working example
2. Look at the auto-generated test harness in your work directory
3. Open an issue on GitHub

Happy optimizing! 🚀
