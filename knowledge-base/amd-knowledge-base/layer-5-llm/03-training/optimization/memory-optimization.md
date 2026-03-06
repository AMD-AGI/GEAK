Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "5"
category: "training"
subcategory: "optimization"
tags: ["memory", "optimization", "training", "efficiency"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "advanced"
estimated_time: "45min"
---

# Memory Optimization for Training

Techniques to optimize memory usage during LLM training on AMD GPUs.

## Key Techniques

### Gradient Checkpointing

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Reduces memory by ~30% at cost of ~20% slower training
```

### Mixed Precision Training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...
    bf16=True,  # Use BF16 on MI200+
    bf16_full_eval=True,
    # Reduces memory by 2x
)
```

### Gradient Accumulation

```python
# Simulate large batch sizes
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Small batch
    gradient_accumulation_steps=32,  # Accumulate 32 steps
    # Effective batch size = 32
)
```

### CPU Offloading

```python
# DeepSpeed ZeRO-3 with CPU offloading
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}
```

### Optimizer States

```python
# Use memory-efficient optimizers
training_args = TrainingArguments(
    ...
    optim="paged_adamw_8bit",  # 8-bit optimizer
    # Reduces optimizer memory by 75%
)
```

## Best Practices

1. Enable gradient checkpointing
2. Use BF16 mixed precision
3. Accumulate gradients for large batches
4. Use DeepSpeed ZeRO for distributed training
5. Offload to CPU when necessary
6. Use quantization (QLoRA) for extreme memory constraints

## References

- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

