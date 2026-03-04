---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "4"
category: "frameworks"
subcategory: "tensorflow"
title: "TensorFlow with ROCm - Getting Started"
rocm_version: "7.0+"
rocm_verified: "7.0.1"
last_updated: "2025-11-01"
last_verified: "2025-11-01"
update_frequency: "quarterly"
status: "stable"
difficulty: "intermediate"
estimated_time: "45min"
tags: ["tensorflow", "rocm", "deep-learning", "training", "amd-gpu"]
prerequisites:
  - "layer-2/rocm/rocm-installation"
  - "layer-1/amd-gpu-arch/cdna-architecture"
related:
  - "layer-4/pytorch/pytorch-rocm-basics"
  - "layer-4/jax/jax-rocm-basics"
  - "layer-5/llm/03-training/distributed/fsdp-training"
official_docs: "https://www.tensorflow.org/"
github_repo: "https://github.com/tensorflow/tensorflow"
---

# TensorFlow with ROCm - Getting Started

## Overview

TensorFlow is an end-to-end open-source platform for machine learning developed by Google. With ROCm support, TensorFlow can leverage AMD GPUs for accelerated training and inference.

**Key Features:**
- Comprehensive ML framework with high-level APIs (Keras)
- Production-ready serving infrastructure (TensorFlow Serving)
- Extensive ecosystem (TensorBoard, TF Lite, TF.js)
- Strong support for distributed training
- Native ROCm backend for AMD GPUs

## Prerequisites

- ROCm 7.0+ installed
- Python 3.9-3.11
- AMD GPU with gfx90a or newer (MI250X, MI300 series)
- 16GB+ system RAM

## Installation

### Method 1: Docker (Recommended)

Use AMD's official ROCm TensorFlow Docker image:

```bash
# Pull official ROCm TensorFlow image
docker pull rocm/tensorflow:rocm7.0-tf2.15-dev

# Run with GPU access
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=8G \
    -v $PWD:/workspace \
    rocm/tensorflow:rocm7.0-tf2.15-dev \
    bash

# Verify installation
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

### Method 2: pip install (Advanced)

Install TensorFlow with ROCm support:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

# Install TensorFlow ROCm variant
pip install tensorflow-rocm

# Verify installation
python3 << EOF
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with ROCm: {tf.test.is_built_with_rocm()}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
EOF
```

## Quick Start Example

### Simple GPU Computation

```python
import tensorflow as tf
import numpy as np

# Verify GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

# Create tensors on GPU
with tf.device('/GPU:0'):
    # Matrix multiplication
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    
    # Compute
    c = tf.matmul(a, b)
    print(f"Result shape: {c.shape}")
    print(f"Device: {c.device}")
```

### Simple Neural Network Training

```python
import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train on GPU (automatic)
model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    batch_size=256
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

## GPU Memory Management

### Check Available Memory

```python
import tensorflow as tf

# Get GPU devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
        # Get memory info
        details = tf.config.experimental.get_device_details(gpu)
        print(f"  Compute Capability: {details.get('compute_capability', 'N/A')}")
```

### Set Memory Growth

Prevent TensorFlow from allocating all GPU memory:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled")
    except RuntimeError as e:
        print(e)
```

### Set Memory Limit

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit to 16GB
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=16384)]
        )
    except RuntimeError as e:
        print(e)
```

## Mixed Precision Training

Leverage FP16 for faster training on AMD GPUs:

```python
from tensorflow import keras
import tensorflow as tf

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print(f"Compute dtype: {policy.compute_dtype}")
print(f"Variable dtype: {policy.variable_dtype}")

# Build model (automatic mixed precision)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax', dtype='float32')  # Output layer in FP32
])

# Compile with loss scaling
optimizer = keras.optimizers.Adam()
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Distributed Training

### Multi-GPU Strategy

```python
import tensorflow as tf
from tensorflow import keras

# Create strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Define model within strategy scope
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train (automatically distributed)
model.fit(x_train, y_train, epochs=5, batch_size=256)
```

## ROCm-Specific Configuration

### Environment Variables

```bash
# Visible devices (comma-separated GPU IDs)
export ROCR_VISIBLE_DEVICES=0,1

# Enable ROCm debug logging
export TF_CPP_MIN_LOG_LEVEL=0
export HSA_ENABLE_DEBUG=1

# Set memory pool size
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

### Python Configuration

```python
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

import tensorflow as tf
```

## Performance Optimization

### XLA Compilation

```python
import tensorflow as tf

# Enable XLA for AMD GPUs
tf.config.optimizer.set_jit(True)

@tf.function(jit_compile=True)
def train_step(x, y, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Data Pipeline Optimization

```python
# Optimize data loading
AUTOTUNE = tf.data.AUTOTUNE

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.cache()  # Cache in memory
dataset = dataset.shuffle(10000)
dataset = dataset.batch(256)
dataset = dataset.prefetch(AUTOTUNE)  # Prefetch next batch
```

## Troubleshooting

### GPU Not Detected

```python
import tensorflow as tf

# Check TensorFlow configuration
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with ROCm: {tf.test.is_built_with_rocm()}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

**Solution:**
1. Verify ROCm installation: `rocm-smi`
2. Check TensorFlow variant: `pip show tensorflow-rocm`
3. Set correct device visibility: `export ROCR_VISIBLE_DEVICES=0`

### Out of Memory Errors

```python
# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Reduce batch size
# Use gradient accumulation for effective larger batches
```

### Slow Training

1. **Use mixed precision**: Enables FP16 computation
2. **Enable XLA**: Just-In-Time compilation
3. **Optimize data pipeline**: Use prefetch and cache
4. **Check GPU utilization**: Use `rocm-smi` or `radeontop`

## Profiling with TensorBoard

```python
import tensorflow as tf
from tensorflow import keras

# Create TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    profile_batch='3,5'  # Profile batches 3-5
)

# Train with profiling
model.fit(
    x_train, y_train,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# View with TensorBoard
# tensorboard --logdir=./logs
```

## Comparison with PyTorch

| Aspect | TensorFlow | PyTorch |
|--------|-----------|---------|
| **Ease of Use** | High-level Keras API | Pythonic, flexible |
| **Production** | TF Serving, TF Lite | TorchServe |
| **Distributed** | Strong built-in | DDP, FSDP |
| **ROCm Support** | Good (official images) | Excellent (native) |
| **Debugging** | Eager mode | Dynamic graphs |

## Best Practices

1. **Use Docker images** for consistent environment
2. **Enable memory growth** to prevent OOM
3. **Use mixed precision** for 2x speedup
4. **Profile your code** to identify bottlenecks
5. **Optimize data pipeline** with prefetch
6. **Use XLA** for compute-intensive operations
7. **Check ROCm compatibility** before upgrading

## Next Steps

- [TensorFlow Distribution Strategies](https://www.tensorflow.org/guide/distributed_training)
- [PyTorch ROCm Basics](../pytorch/pytorch-rocm-basics.md)
- [JAX on ROCm](../jax/jax-rocm-basics.md)
- [Distributed Training with FSDP](../../layer-5-llm/03-training/distributed/fsdp-training.md)

## References

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [TensorFlow ROCm Docker Images](https://hub.docker.com/r/rocm/tensorflow)
- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/gpu_performance_analysis)

