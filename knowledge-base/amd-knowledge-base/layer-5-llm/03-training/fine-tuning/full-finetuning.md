---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "training"
subcategory: "fine-tuning"
tags: ["full-finetuning", "training", "distributed", "deepspeed"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "advanced"
estimated_time: "50min"
---

# Full Model Fine-tuning

Complete guide to full parameter fine-tuning of large language models on AMD GPUs.

## Overview

Full fine-tuning updates all model parameters, providing:
- **Maximum quality**: Best possible adaptation to your data
- **Complete control**: Modify any aspect of the model
- **Domain adaptation**: Deep specialization for specific tasks

**Requirements:**
- Multiple high-memory GPUs (4-8x MI250X for 70B models)
- Distributed training framework (DeepSpeed, FSDP)
- Large compute budget

## DeepSpeed ZeRO Full Fine-tuning

### ZeRO Stage 1 (Optimizer Partitioning)

```python
# deepspeed_z1_config.json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-6,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}
```

Training script:
```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
)

# Training arguments with DeepSpeed
training_args = TrainingArguments(
    output_dir="./llama2-fullft",
    deepspeed="deepspeed_z1_config.json",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train
trainer.train()
```

Run with DeepSpeed:
```bash
deepspeed --num_gpus=4 train.py
```

### ZeRO Stage 2 (Optimizer + Gradient Partitioning)

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-6,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_clipping": 1.0
}
```

### ZeRO Stage 3 (Full Sharding)

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-6,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0
}
```

## PyTorch FSDP Full Fine-tuning

### Basic FSDP Setup

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
)
from transformers import LlamaDecoderLayer

# FSDP configuration
fsdp_config = {
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    "fsdp_sharding_strategy": "FULL_SHARD",  # or "SHARD_GRAD_OP", "NO_SHARD"
    "fsdp_backward_prefetch": "BACKWARD_PRE",
    "fsdp_cpu_ram_efficient_loading": True
}

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama2-fsdp",
    fsdp="full_shard auto_wrap",
    fsdp_config=fsdp_config,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-6,
    num_train_epochs=3,
    bf16=True,
    tf32=True,
    logging_steps=10,
    save_steps=500,
    gradient_checkpointing=True
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

Run with torchrun:
```bash
torchrun --nproc_per_node=8 train_fsdp.py
```

### FSDP with CPU Offloading

```python
from torch.distributed.fsdp import CPUOffload

fsdp_config = {
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    "fsdp_sharding_strategy": "FULL_SHARD",
    "fsdp_offload_params": True,  # Offload to CPU
    "fsdp_cpu_ram_efficient_loading": True
}
```

## Memory Optimization Techniques

### Gradient Checkpointing

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Configure checkpointing
from functools import partial

def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward

# Apply selective checkpointing
for layer in model.model.layers:
    layer.self_attn = torch.utils.checkpoint.checkpoint(
        create_custom_forward(layer.self_attn)
    )
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# For BF16 (recommended on MI200+)
training_args = TrainingArguments(
    ...
    bf16=True,
    bf16_full_eval=True,
)

# For FP16
training_args = TrainingArguments(
    ...
    fp16=True,
    fp16_full_eval=True,
)
```

### Activation Checkpointing

```python
# Reduce memory by recomputing activations
model.config.use_cache = False  # Disable KV cache
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False  # Better memory efficiency
    }
)
```

## Advanced Optimization

### Learning Rate Scheduling

```python
from transformers import get_cosine_schedule_with_warmup

# In TrainingArguments
training_args = TrainingArguments(
    ...
    learning_rate=5e-6,
    lr_scheduler_type="cosine",  # or "linear", "polynomial"
    warmup_ratio=0.03,  # 3% of training for warmup
    warmup_steps=100,  # Or specify exact steps
)
```

### Gradient Accumulation

```python
# Simulate larger batch size
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=1,  # Actual batch per GPU
    gradient_accumulation_steps=32,  # Effective batch = 32
    # With 8 GPUs: total effective batch = 1 * 32 * 8 = 256
)
```

### Weight Decay and Regularization

```python
training_args = TrainingArguments(
    ...
    weight_decay=0.1,  # L2 regularization
    max_grad_norm=1.0,  # Gradient clipping
    label_smoothing_factor=0.1,  # Label smoothing
)
```

## Multi-Node Training

### DeepSpeed Multi-Node

```bash
# hostfile
worker1 slots=8
worker2 slots=8

# Launch
deepspeed --hostfile=hostfile \
    --master_addr=worker1 \
    --master_port=29500 \
    train.py
```

### FSDP Multi-Node

```bash
# On master node (worker1)
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=worker1 \
    --master_port=29500 \
    train_fsdp.py

# On worker node (worker2)
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=worker1 \
    --master_port=29500 \
    train_fsdp.py
```

## Monitoring Training

### WandB Integration

```python
import wandb

wandb.init(
    project="llm-finetuning",
    config={
        "model": "llama-2-70b",
        "strategy": "deepspeed-z3",
        "learning_rate": 2e-6,
    }
)

training_args = TrainingArguments(
    ...
    report_to="wandb",
    logging_steps=10,
    run_name="llama2-fullft-run1"
)
```

### Custom Callbacks

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log custom metrics
            if torch.distributed.get_rank() == 0:
                print(f"Step {state.global_step}:")
                print(f"  Loss: {logs.get('loss', 0):.4f}")
                print(f"  LR: {logs.get('learning_rate', 0):.2e}")
                print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

trainer = Trainer(..., callbacks=[CustomCallback()])
```

## Saving and Checkpointing

### Save Full Model

```python
training_args = TrainingArguments(
    ...
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 latest checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# After training
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
```

### Resume from Checkpoint

```python
# Resume training
trainer.train(resume_from_checkpoint="./checkpoint-1000")

# Or load specific checkpoint
model = AutoModelForCausalLM.from_pretrained("./checkpoint-1000")
```

## Best Practices

### For 7B Models (Single Node)

```python
# 4-8 GPUs, DeepSpeed ZeRO-2
training_args = TrainingArguments(
    deepspeed="ds_z2_config.json",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True
)
```

### For 70B Models (Multi-Node)

```python
# 32+ GPUs, DeepSpeed ZeRO-3 with offloading
training_args = TrainingArguments(
    deepspeed="ds_z3_offload_config.json",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-6,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    save_steps=100,
    logging_steps=1
)
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=32

# Enable all memory optimizations
gradient_checkpointing=True
model.config.use_cache = False

# Use ZeRO-3 with CPU offloading
deepspeed="ds_z3_offload_config.json"
```

### Training Divergence

```python
# Reduce learning rate
learning_rate=1e-6  # Lower

# Increase warmup
warmup_ratio=0.05  # 5% warmup

# Clip gradients more aggressively
max_grad_norm=0.5
```

## Performance Comparison

| Model Size | Method | GPUs | Memory/GPU | Time (epoch) |
|------------|--------|------|------------|--------------|
| 7B | Standard | 1 | 28GB | 12h |
| 7B | ZeRO-2 | 4 | 14GB | 4h |
| 70B | ZeRO-2 | 16 | 40GB | 48h |
| 70B | ZeRO-3 | 32 | 20GB | 72h |

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)

