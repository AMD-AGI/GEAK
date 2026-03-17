---
layer: "4"
category: "tensorflow"
tags: ["tensorflow", "deep-learning", "cuda"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# TensorFlow with CUDA

*Guide to using TensorFlow with Nvidia GPUs*

## Installation

```bash
# TensorFlow with CUDA support
pip install tensorflow[and-cuda]

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Basic GPU Operations

```python
import tensorflow as tf

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# Create tensors on GPU (automatic)
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
```

## Memory Management

```python
# Allow memory growth (recommended)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Or set memory limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])  # 8GB
```

## Mixed Precision

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build model (automatically uses FP16)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Use loss scaling
optimizer = tf.keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
```

## Multi-GPU Training

```python
# Mirror strategy (data parallelism)
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training works the same
model.fit(train_dataset, epochs=10)
```

## Best Practices

1. **Enable memory growth**: Prevents OOM errors
2. **Use mixed precision**: 2-3x speedup
3. **Use tf.data**: Efficient data pipelines
4. **XLA compilation**: Enable with `jit_compile=True`

## External Resources

- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [Mixed Precision](https://www.tensorflow.org/guide/mixed_precision)
- [Distributed Training](https://www.tensorflow.org/guide/distributed_training)

## Related Guides

- [CUDA Basics](../../layer-2-compute-stack/cuda/cuda-basics.md)
- [PyTorch with CUDA](../pytorch/pytorch-cuda-basics.md)

