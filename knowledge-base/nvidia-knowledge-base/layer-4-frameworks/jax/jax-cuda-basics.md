Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "4"
category: "jax"
tags: ["jax", "deep-learning", "cuda", "xla"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# JAX with CUDA

*High-performance numerical computing with JAX on Nvidia GPUs*

## Installation

```bash
# JAX with CUDA 12 support
pip install --upgrade "jax[cuda12]"

# Verify
python -c "import jax; print(jax.devices())"
```

## Basic GPU Operations

```python
import jax
import jax.numpy as jnp

# Check devices
print(jax.devices())  # Should show GPU devices

# Arrays are on GPU by default
x = jnp.array([1.0, 2.0, 3.0])  # Automatically on GPU
y = jnp.array([4.0, 5.0, 6.0])
z = x + y  # Computed on GPU

# JIT compilation
@jax.jit
def fast_function(x):
    return jnp.dot(x, x.T)

result = fast_function(jnp.ones((1000, 1000)))
```

## Automatic Differentiation

```python
# Gradient computation
def loss_fn(params, x, y):
    prediction = params['w'] @ x + params['b']
    return jnp.mean((prediction - y) ** 2)

# Compute gradient
grad_fn = jax.grad(loss_fn)
grads = grad_fn(params, x, y)

# Or use value_and_grad
value_and_grad_fn = jax.value_and_grad(loss_fn)
loss, grads = value_and_grad_fn(params, x, y)
```

## Multi-GPU with pmap

```python
# Parallel map across devices
@jax.pmap
def parallel_fn(x):
    return x ** 2

# Data replicated across GPUs
x = jnp.arange(8).reshape(4, 2)  # 4 devices, 2 elements each
result = parallel_fn(x)
```

## Best Practices

1. **Use JIT**: Compile with `@jax.jit`
2. **Use pmap for multi-GPU**: Efficient parallelism
3. **Avoid Python loops**: Use `jax.lax` control flow
4. **XLA optimization**: Automatic with JIT

## External Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GPU Performance](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)

## Related Guides

- [CUDA Basics](../../layer-2-compute-stack/cuda/cuda-basics.md)
- [PyTorch with CUDA](../pytorch/pytorch-cuda-basics.md)

