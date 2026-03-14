---
layer: "5"
category: "training"
subcategory: "fine-tuning"
tags: ["qlora", "quantization", "peft", "memory-efficient"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "40min"
---

# QLoRA: Quantized LoRA Fine-tuning

Train large language models with 4-bit quantization and LoRA on AMD GPUs with minimal memory.

## Overview

QLoRA (Quantized Low-Rank Adaptation) combines:
- 4-bit NormalFloat (NF4) quantization of base model
- LoRA adapters trained in higher precision
- Double quantization for optimizer states

**Benefits:**
- Fine-tune 70B models on single 48GB GPU
- 4x memory reduction vs standard LoRA
- Maintains ~99% of full fine-tuning quality

## Quick Start

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-llama2-70b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    bf16=True,
    optim="paged_adamw_32bit",
    save_steps=100
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
model.save_pretrained("./qlora-final")
```

## Quantization Configuration

### NF4 Quantization (Recommended)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 (optimal for normal distributions)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=True  # Quantize quantization constants
)

# Memory reduction: 70B model ~140GB → ~35GB
```

### FP4 Quantization

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Float4 (standard floating point)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

### Nested Quantization

```python
# Double quantization: quantize the quantization constants
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Saves ~0.4 bits per parameter
)
```

## QLoRA Training Strategies

### Large Models (70B+)

```python
# Memory-optimized configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

lora_config = LoraConfig(
    r=64,  # Higher rank for large models
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # Include MLP
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 16
    learning_rate=1e-4,  # Lower LR for large models
    warmup_steps=100,
    num_train_epochs=1,  # Fewer epochs
    bf16=True,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    max_grad_norm=0.3
)
```

### Medium Models (7B-13B)

```python
# Balanced configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    optim="paged_adamw_32bit"
)
```

## Advanced Techniques

### Mixed Precision Training

```python
# Use different precisions for different operations
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,  # Base model in BF16
    device_map="auto"
)

# LoRA adapters train in FP32 for stability
lora_config = LoraConfig(
    ...
    init_lora_weights="gaussian",  # Better initialization
)

# Training uses mixed precision
training_args = TrainingArguments(
    ...
    bf16=True,  # Enable BF16 training
    bf16_full_eval=True,  # BF16 for evaluation too
)
```

### Gradient Checkpointing

```python
# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Disable KV cache during training

# Further reduce memory
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

### Custom Quantization Blocks

```python
# Selectively quantize layers
from peft import prepare_model_for_kbit_training

# Don't quantize embedding and LM head
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Custom preparation
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

## Memory Optimization

### Multi-GPU QLoRA

```python
# Distribute across multiple GPUs
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute
    max_memory={0: "40GB", 1: "40GB"}  # Specify per-GPU limits
)

# Or manual distribution
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0-15": 0,
    "model.layers.16-31": 1,
    "model.layers.32-47": 2,
    "model.layers.48-63": 3,
    "model.norm": 3,
    "lm_head": 3
}
```

### CPU Offloading

```python
# Offload to CPU when needed
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="offload",  # Directory for CPU offload
    offload_state_dict=True
)
```

## Inference with QLoRA

### Loading QLoRA Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load quantized base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load QLoRA adapters
model = PeftModel.from_pretrained(base_model, "./qlora-final")

# Generate
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### Merging for Deployment

```python
# Merge adapters (requires dequantization)
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("./merged-model", safe_serialization=True)
tokenizer.save_pretrained("./merged-model")
```

## Monitoring and Debugging

### Memory Monitoring

```python
import torch

def print_memory_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# During training
class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            print_memory_usage()

trainer = Trainer(..., callbacks=[MemoryCallback()])
```

### Quality Monitoring

```python
# Evaluate perplexity during training
from torch.nn import CrossEntropyLoss

def compute_perplexity(model, dataset):
    model.eval()
    loss_fct = CrossEntropyLoss()
    total_loss = 0
    
    for batch in dataset:
        with torch.no_grad():
            outputs = model(**batch)
            loss = loss_fct(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                batch["labels"].view(-1)
            )
            total_loss += loss.item()
    
    perplexity = torch.exp(torch.tensor(total_loss / len(dataset)))
    return perplexity.item()

# Log periodically
class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        ppl = compute_perplexity(model, eval_dataset)
        metrics["perplexity"] = ppl
        print(f"Perplexity: {ppl:.2f}")
```

## Best Practices

### Hyperparameter Selection

```python
# For 70B models (single 48GB GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1
)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    warmup_steps=100,
    num_train_epochs=1,
    max_grad_norm=0.3
)
```

### Common Issues

```python
# Issue: Out of memory
# Solution 1: Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=32

# Solution 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Reduce LoRA rank
lora_config = LoraConfig(r=32, ...)  # Instead of 64

# Issue: Training instability
# Solution: Lower learning rate and use warmup
training_args = TrainingArguments(
    learning_rate=5e-5,  # Lower LR
    warmup_steps=200,  # More warmup
    max_grad_norm=0.3  # Gradient clipping
)

# Issue: Poor quality
# Solution: Increase LoRA capacity
lora_config = LoraConfig(
    r=128,  # Higher rank
    target_modules=[...],  # More modules
)
```

## Performance Comparison

| Model | Method | Memory | Speed | Quality |
|-------|--------|--------|-------|---------|
| Llama-2-70B | Full FT | 560GB | 1x | 100% |
| Llama-2-70B | LoRA | 140GB | 1.2x | 99% |
| Llama-2-70B | QLoRA | 35GB | 0.8x | 97% |

## Troubleshooting

### Quantization Errors

```python
# If quantization fails, try:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

### Slow Training

```python
# Optimize data loading
training_args = TrainingArguments(
    ...
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False
)
```

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [PEFT QLoRA Guide](https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning)

