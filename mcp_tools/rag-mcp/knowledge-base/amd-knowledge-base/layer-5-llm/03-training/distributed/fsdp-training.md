---
layer: "5"
category: "training"
tags: ["fsdp", "distributed", "training", "pytorch"]
rocm_version: "7.0+"
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "30min"
---

# FSDP Training on AMD GPUs

Fully Sharded Data Parallel (FSDP) enables training large models that don't fit on a single GPU.

## Setup

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

# Initialize distributed training
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
```

## Basic FSDP Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
)

# Mixed precision policy
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Wrap model with FSDP
model = FSDP(
    model,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=local_rank,
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

model.train()
for batch in dataloader:
    inputs = batch['input_ids'].to(local_rank)
    labels = batch['labels'].to(local_rank)
    
    outputs = model(inputs, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Transformer-Specific Wrapping

```python
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Auto-wrap policy for transformers
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=local_rank,
)
```

## Gradient Checkpointing

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Enable gradient checkpointing
def check_fn(submodule):
    return isinstance(submodule, LlamaDecoderLayer)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
        m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=check_fn,
)
```

## CPU Offloading

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=mixed_precision_policy,
    device_id=local_rank,
)
```

## Sharding Strategies

```python
# FULL_SHARD: Shard parameters, gradients, and optimizer states
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)

# SHARD_GRAD_OP: Shard gradients and optimizer states only
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
)

# NO_SHARD: No sharding (equivalent to DDP)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.NO_SHARD,
)

# HYBRID_SHARD: Shard within node, replicate across nodes
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
)
```

## Saving and Loading

```python
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

# Save checkpoint
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    save_policy,
):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, "checkpoint.pt")

# Load checkpoint
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
):
    state_dict = torch.load("checkpoint.pt")
    model.load_state_dict(state_dict)
```

## Complete Training Script

```python
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # FSDP configuration
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Training loop
    model.train()
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            inputs = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if local_rank == 0 and step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
    
    # Save checkpoint
    if local_rank == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pt')
    
    cleanup()

if __name__ == '__main__':
    main()
```

Launch:
```bash
torchrun --standalone --nproc_per_node=8 train_fsdp.py
```

## Integration with Hugging Face Trainer

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=1e-5,
    bf16=True,
    # FSDP configuration
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    },
    # Gradient checkpointing
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## Performance Tips

1. **Use BF16** on CDNA2+ GPUs for best performance
2. **Enable gradient checkpointing** for large models
3. **Tune batch size** and gradient accumulation
4. **Use HYBRID_SHARD** for multi-node training
5. **Profile memory** with `torch.cuda.memory_summary()`

## Monitoring

```python
# Memory usage
if local_rank == 0:
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [HuggingFace FSDP Guide](https://huggingface.co/docs/transformers/main_classes/trainer#pytorch-fully-sharded-data-parallel)

