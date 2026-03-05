---
layer: "5"
category: "llm"
subcategory: "training"
tags: ["qlora", "quantization", "finetuning", "4bit"]
cuda_version: "13.0+"
difficulty: "intermediate"
estimated_time: "45min"
last_updated: 2025-11-17
---

# QLoRA: Quantized LoRA Fine-tuning

*Train large language models with 4-bit quantization and LoRA on Nvidia GPUs*

## Overview

QLoRA combines 4-bit quantization with LoRA to enable fine-tuning of very large models (70B+) on consumer GPUs. Reduces memory by up to 75% compared to standard fine-tuning.

## Installation

```bash
pip install transformers peft datasets accelerate bitsandbytes
```

## Basic QLoRA Fine-tuning

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=True  # Double quantization
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    bf16=True,  # Use BF16
    logging_steps=10,
    optim="paged_adamw_8bit"  # 8-bit optimizer
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text"
)

trainer.train()
model.save_pretrained("./qlora-model")
```

## Memory Requirements

QLoRA dramatically reduces memory:

| Model Size | Full FP32 | LoRA FP16 | QLoRA 4-bit | GPU |
|------------|-----------|-----------|-------------|-----|
| 7B | 28GB | 16GB | 6GB | RTX 3090 |
| 13B | 52GB | 28GB | 10GB | RTX 4090 |
| 33B | 132GB | 70GB | 24GB | A100-40GB |
| 65-70B | 260GB | 140GB | 48GB | A100-80GB |

## QLoRA on Single Consumer GPU

```python
# Fine-tune 13B model on RTX 4090 (24GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "22GB"}  # Leave 2GB for overhead
)

# Rest of training code...
```

## Double Quantization

Saves additional ~3GB memory:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Quantize quantization constants
)
```

## Paged Optimizer

For training very large models:

```python
training_args = TrainingArguments(
    ...
    optim="paged_adamw_32bit",  # Or paged_adamw_8bit
    gradient_checkpointing=True  # Save more memory
)
```

## Loading QLoRA Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model in 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./qlora-model")

# Generate
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
```

## Best Practices

1. **Use BF16 compute**: Better than FP16 for QLoRA
2. **Enable double quantization**: Extra memory savings
3. **Use paged optimizer**: Handles memory spikes
4. **Gradient checkpointing**: Trades compute for memory
5. **Small batch size + gradient accumulation**: Effective batch size

## QLoRA vs LoRA Comparison

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| Memory | 2x less than full | 4x less than full |
| Speed | Fast | Slightly slower (~20%) |
| Quality | High | Nearly identical |
| GPU Required | High-end | Consumer GPUs OK |
| 70B on single GPU | No | Yes (with 48GB+) |

## Common Issues

### CUDA Out of Memory

```python
# Solution 1: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 2: Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=32

# Solution 3: Use double quantization
bnb_4bit_use_double_quant=True
```

### Slow Training

```python
# Normal - QLoRA is ~20% slower than LoRA
# To improve speed:
# 1. Use larger batch sizes if memory allows
# 2. Enable Flash Attention 2
# 3. Use faster storage for dataset
```

## External Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [QLoRA Examples](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

## Related Guides

- [LoRA Fine-tuning](lora-finetuning.md)
- [Memory Optimization](../optimization/memory-optimization.md)
- [Quickstart Fine-tuning](../../00-quickstart/quickstart-finetuning.md)

