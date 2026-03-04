---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "4"
category: "frameworks"
subcategory: "jax"
title: "JAX with ROCm - Getting Started"
rocm_version: "7.0+"
rocm_verified: "7.0.1"
last_updated: "2025-11-01"
last_verified: "2025-11-01"
update_frequency: "quarterly"
status: "stable"
difficulty: "advanced"
estimated_time: "60min"
tags: ["jax", "rocm", "xla", "automatic-differentiation", "amd-gpu"]
prerequisites:
  - "layer-2/rocm/rocm-installation"
  - "layer-1/amd-gpu-arch/cdna-architecture"
  - "Python NumPy proficiency"
related:
  - "layer-4/pytorch/pytorch-rocm-basics"
  - "layer-4/tensorflow/tensorflow-rocm-basics"
  - "layer-3/compilers/triton-on-rocm"
official_docs: "https://jax.readthedocs.io/"
github_repo: "https://github.com/google/jax"
---

# JAX with ROCm - Getting Started

## Overview

JAX is a high-performance numerical computing library that combines NumPy's API with automatic differentiation and XLA compilation. With ROCm support, JAX provides cutting-edge performance on AMD GPUs.

**Key Features:**
- NumPy-like API with GPU/TPU acceleration
- Automatic differentiation (grad, vjp, jvp)
- Just-In-Time (JIT) compilation via XLA
- Vectorization (vmap) and parallelization (pmap)
- Functional programming paradigm
- Research-friendly with production capabilities

**Why JAX?**
- Fastest gradient computation
- Composable transformations
- Ideal for research and experimentation
- Growing ecosystem (Flax, Optax, Haiku)

## Prerequisites

- ROCm 7.0+ installed
- Python 3.9-3.11
- AMD GPU with gfx90a or newer (MI250X, MI300 series)
- Understanding of functional programming concepts
- NumPy proficiency

## Installation

### Method 1: Build from Source (Recommended)

JAX ROCm support requires building from source:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential

# Clone JAX repository
git clone https://github.com/google/jax.git
cd jax

# Install Bazel (build tool)
wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-linux-x86_64
chmod +x bazel-6.1.0-linux-x86_64
sudo mv bazel-6.1.0-linux-x86_64 /usr/local/bin/bazel

# Build JAX with ROCm support
python3 build/build.py \
    --enable_rocm \
    --rocm_path=/opt/rocm \
    --rocm_amdgpu_targets=gfx90a,gfx942  # MI250X, MI300

# Install
pip install dist/*.whl
pip install -e .

# Verify installation
python3 -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

### Method 2: Docker (Easiest)

```bash
# Use community ROCm JAX image
docker pull rocm/jax:latest

# Run with GPU access
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=8G \
    -v $PWD:/workspace \
    rocm/jax:latest \
    bash

# Verify
python3 -c "import jax; print(jax.devices())"
```

### Method 3: pip (Limited Support)

```bash
# Install JAX (CPU version, then configure for ROCm)
pip install jax jaxlib

# Note: Official pip packages have limited ROCm support
# Building from source is recommended for full functionality
```

## Quick Start Examples

### Basic GPU Computation

```python
import jax
import jax.numpy as jnp

# Check available devices
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Create array on GPU (automatic)
x = jnp.array([1, 2, 3, 4, 5])
print(f"Array: {x}")
print(f"Device: {x.device()}")

# Matrix multiplication
A = jnp.ones((5000, 5000))
B = jnp.ones((5000, 5000))
C = jnp.dot(A, B)
print(f"Result shape: {C.shape}")
```

### Automatic Differentiation

```python
import jax
import jax.numpy as jnp

# Define function
def f(x):
    return x**3 + 2*x**2 - 5*x + 3

# Compute derivative
df = jax.grad(f)

x = 2.0
print(f"f({x}) = {f(x)}")
print(f"f'({x}) = {df(x)}")  # Automatic differentiation

# Higher-order derivatives
ddf = jax.grad(jax.grad(f))
print(f"f''({x}) = {ddf(x)}")

# Gradient of multivariable function
def g(x):
    return jnp.sum(x**2)

grad_g = jax.grad(g)
x = jnp.array([1.0, 2.0, 3.0])
print(f"∇g = {grad_g(x)}")  # [2., 4., 6.]
```

### Just-In-Time Compilation

```python
import jax
import jax.numpy as jnp
import time

# Regular function
def slow_function(x):
    for _ in range(10):
        x = x**2 + 1
    return x

# JIT-compiled function
@jax.jit
def fast_function(x):
    for _ in range(10):
        x = x**2 + 1
    return x

x = jnp.ones((10000, 10000))

# First run (compilation)
start = time.time()
result = fast_function(x)
result.block_until_ready()  # Wait for GPU
print(f"First run (with compilation): {time.time() - start:.4f}s")

# Second run (compiled)
start = time.time()
result = fast_function(x)
result.block_until_ready()
print(f"Second run (compiled): {time.time() - start:.4f}s")

# Regular function for comparison
start = time.time()
result = slow_function(x)
result.block_until_ready()
print(f"Without JIT: {time.time() - start:.4f}s")
```

### Vectorization with vmap

```python
import jax
import jax.numpy as jnp

# Function for single input
def normalize(x):
    return x / jnp.linalg.norm(x)

# Vectorize across batch dimension
batch_normalize = jax.vmap(normalize)

# Test
batch = jnp.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

result = batch_normalize(batch)
print(f"Normalized batch:\n{result}")

# vmap with multiple dimensions
matrix_batch = jnp.ones((32, 100, 100))  # 32 matrices
vmapped_matmul = jax.vmap(lambda x: jnp.dot(x, x))
result = vmapped_matmul(matrix_batch)
print(f"Batch matmul result shape: {result.shape}")  # (32, 100, 100)
```

## Neural Network Training Example

### Simple MLP with Optax

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax

# Initialize network parameters
def init_mlp_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes))
    params = []
    for in_size, out_size, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w_key, b_key = random.split(k)
        params.append({
            'w': random.normal(w_key, (in_size, out_size)) * 0.1,
            'b': jnp.zeros(out_size)
        })
    return params

# Forward pass
@jit
def mlp_predict(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer['w'] + layer['b'])
    # Last layer (no activation)
    final_layer = params[-1]
    return x @ final_layer['w'] + final_layer['b']

# Loss function
@jit
def loss_fn(params, x, y):
    preds = mlp_predict(params, x)
    return jnp.mean((preds - y)**2)

# Training step
@jit
def train_step(params, opt_state, x, y, optimizer):
    loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

# Create model
key = random.PRNGKey(0)
params = init_mlp_params([784, 256, 128, 10], key)

# Create optimizer
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(10):
    # Generate dummy data
    x_batch = random.normal(key, (128, 784))
    y_batch = random.normal(key, (128, 10))
    
    # Train step
    params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch, optimizer)
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Distributed Training with pmap

```python
import jax
import jax.numpy as jnp
from jax import pmap, random

# Get number of devices
n_devices = jax.local_device_count()
print(f"Number of devices: {n_devices}")

# Parallel training step
@pmap
def parallel_train_step(params, x, y):
    loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Update params (simplified)
    new_params = jax.tree_map(lambda p, g: p - 0.001 * g, params, grads)
    return new_params, loss_value

# Replicate parameters across devices
params_replicated = jax.tree_map(
    lambda x: jnp.stack([x] * n_devices), 
    params
)

# Create data for each device
key = random.PRNGKey(0)
x_batch = random.normal(key, (n_devices, 128, 784))
y_batch = random.normal(key, (n_devices, 128, 10))

# Parallel training
params_replicated, losses = parallel_train_step(params_replicated, x_batch, y_batch)
print(f"Losses across devices: {losses}")
```

## ROCm-Specific Configuration

### Environment Variables

```bash
# Set visible devices
export ROCR_VISIBLE_DEVICES=0,1

# Enable XLA debug logging
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps"

# ROCm runtime configuration
export HSA_OVERRIDE_GFX_VERSION=9.0.a  # For MI250X
export GPU_MAX_HW_QUEUES=4

# JAX configuration
export JAX_PLATFORM_NAME=rocm
```

### Python Configuration

```python
import os
os.environ['JAX_PLATFORM_NAME'] = 'rocm'

import jax
import jax.numpy as jnp

# Check backend
print(f"Platform: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# Force specific device
with jax.default_device(jax.devices()[0]):
    x = jnp.array([1, 2, 3])
    print(f"Device: {x.device()}")
```

## Memory Management

### Preallocate GPU Memory

```python
import os

# Preallocate 90% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import jax
```

### Check Memory Usage

```bash
# Monitor GPU memory with rocm-smi
watch -n 1 rocm-smi

# Or use radeontop
sudo radeontop
```

## Performance Optimization

### 1. Use JIT Compilation

```python
@jax.jit
def optimized_function(x):
    return jnp.sum(x**2)
```

### 2. Use vmap for Batching

```python
# Instead of loops
results = jax.vmap(function)(batch_inputs)
```

### 3. Avoid Python Loops

```python
# Bad: Python loop
def slow(x):
    result = 0
    for i in range(len(x)):
        result += x[i]**2
    return result

# Good: Vectorized
def fast(x):
    return jnp.sum(x**2)
```

### 4. Mixed Precision

```python
# Use float16 for computation
x = jnp.array(data, dtype=jnp.float16)
result = model(x)
# Cast back to float32 for loss
result = result.astype(jnp.float32)
```

## Debugging

### Print Values Inside JIT

```python
@jax.jit
def debug_function(x):
    # Use jax.debug.print instead of print
    jax.debug.print("x value: {}", x)
    return x**2
```

### Check Compilation

```python
import jax

# Lower to see XLA HLO
lowered = jax.jit(function).lower(example_input)
print(lowered.as_text())

# Compile explicitly
compiled = lowered.compile()
```

## Troubleshooting

### Issue: GPU Not Detected

```python
import jax
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
```

**Solution:**
1. Check ROCm: `rocm-smi`
2. Verify build: Ensure JAX built with `--enable_rocm`
3. Set platform: `export JAX_PLATFORM_NAME=rocm`

### Issue: Out of Memory

```python
# Reduce memory preallocatio n
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'

# Or disable preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

### Issue: Slow First Run

JAX compiles on first run. This is normal:

```python
# Warm up
_ = jitted_function(dummy_input)  # First run (compile)
result = jitted_function(real_input)  # Fast
```

## JAX Ecosystem

### Flax (Neural Networks)

```python
import jax
import flax.linen as nn

class MLP(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        return x

model = MLP(features=10)
```

### Optax (Optimization)

```python
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(0.001)
)
```

## Comparison with Other Frameworks

| Aspect | JAX | PyTorch | TensorFlow |
|--------|-----|---------|-----------|
| **API Style** | Functional | Object-oriented | Mixed |
| **Differentiation** | Excellent | Good | Good |
| **Performance** | Excellent (XLA) | Good | Good |
| **Ease of Use** | Moderate | High | High |
| **Research** | Excellent | Excellent | Good |
| **Production** | Growing | Excellent | Excellent |
| **ROCm Support** | Build from source | Native | Official images |

## Best Practices

1. **Use JIT** for frequently called functions
2. **Use vmap** instead of Python loops
3. **Understand functional programming** - no in-place updates
4. **Profile with XLA dumps** to understand compilation
5. **Preallocate memory** for stable performance
6. **Use pure functions** for best JIT performance
7. **Build from source** for best ROCm support

## Next Steps

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [PyTorch ROCm Basics](../pytorch/pytorch-rocm-basics.md)
- [TensorFlow ROCm Basics](../tensorflow/tensorflow-rocm-basics.md)
- [Triton Compiler](../../layer-3-libraries/compilers/triton-on-rocm.md)

## References

- [JAX GitHub](https://github.com/google/jax)
- [XLA Documentation](https://www.tensorflow.org/xla)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

