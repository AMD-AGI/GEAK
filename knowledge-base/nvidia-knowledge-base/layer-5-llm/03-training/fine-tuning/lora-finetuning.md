---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "llm"
subcategory: "training"
tags: ["lora", "finetuning", "peft", "training"]
cuda_version: "13.0+"
difficulty: "intermediate"
estimated_time: "45min"
last_updated: 2025-11-17
---

# LoRA Fine-tuning on Nvidia GPUs

*Complete guide to efficient fine-tuning using Low-Rank Adaptation (LoRA)*

## Overview

LoRA enables efficient fine-tuning of large language models by training only small adapter matrices, reducing memory requirements by up to 3x while maintaining performance.

## Installation

```bash
pip install transformers peft datasets accelerate bitsandbytes
```

## Basic LoRA Fine-tuning

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Load model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("your-dataset")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    dataset_text_field="text",
)

trainer.train()

# Save LoRA weights
model.save_pretrained("./lora-model")
```

## Advanced Configuration

```python
lora_config = LoraConfig(
    r=32,  # Higher rank = more parameters
    lora_alpha=64,  # Typically 2x rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head"]  # Also train output layer
)
```

## Multi-GPU Training

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

Or use DeepSpeed:

```bash
# deepspeed_config.json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": true},
    "zero_optimization": {"stage": 2}
}

# Run
deepspeed --num_gpus=8 train.py --deepspeed deepspeed_config.json
```

## Loading and Using Fine-tuned Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./lora-model")

# Merge for faster inference (optional)
model = model.merge_and_unload()

# Generate
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## LoRA Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `r` | 8-64 | Rank (higher = more parameters) |
| `lora_alpha` | 16-128 | Scaling factor (often 2x rank) |
| `lora_dropout` | 0.05-0.1 | Dropout rate |
| `target_modules` | ["q_proj", "v_proj"] | Which layers to adapt |
| `learning_rate` | 1e-4 to 3e-4 | Higher than full fine-tuning |

## Memory Requirements

Approximate GPU memory for LoRA fine-tuning:

| Model Size | Base Memory | LoRA Memory | GPU |
|------------|-------------|-------------|-----|
| 7B | 14GB | 16GB | A10, RTX 4090 |
| 13B | 26GB | 28GB | A100-40GB |
| 30B | 60GB | 64GB | A100-80GB |
| 65-70B | 130GB | 140GB | 2x A100-80GB |

## Best Practices

1. **Start with small rank**: r=8 or r=16
2. **Target attention layers**: q_proj, v_proj minimum
3. **Use gradient accumulation**: Simulate larger batches
4. **Monitor training loss**: Ensure convergence
5. **Evaluate on validation set**: Prevent overfitting

## Common Issues

### Out of Memory

```python
# Solution 1: Reduce batch size
per_device_train_batch_size=2

# Solution 2: Increase gradient accumulation
gradient_accumulation_steps=8

# Solution 3: Use QLoRA (4-bit)
```

### Model Not Learning

```python
# Solution 1: Increase learning rate
learning_rate=3e-4

# Solution 2: Increase LoRA rank
r=32

# Solution 3: Train more layers
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## External Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [LoRA Best Practices](https://huggingface.co/docs/peft/conceptual_guides/lora)

## Related Guides

- [Quickstart Fine-tuning](../../00-quickstart/quickstart-finetuning.md)
- [QLoRA Fine-tuning](qlora-finetuning.md)
- [Dataset Preparation](../preparation/dataset-preparation.md)
- [Memory Optimization](../optimization/memory-optimization.md)

