---
layer: "5"
category: "quickstart"
subcategory: "training"
tags: ["quickstart", "fine-tuning", "lora", "training", "llm"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "beginner"
estimated_time: "15min"
---

# Fine-tune an LLM in 15 Minutes

Quick guide to fine-tuning a 7B parameter model on AMD GPUs using LoRA.

## Prerequisites

- ROCm 7.0+ installed
- Python 3.10+
- At least 24GB GPU memory (for 7B models)
- A dataset (we'll use a sample one)

## Quick Start

### Step 1: Install Dependencies

```bash
# Install required packages
pip install torch transformers datasets peft accelerate bitsandbytes
```

### Step 2: Prepare Your Script

Create `finetune.py`:

```python
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:1000]")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama2-lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    warmup_steps=10,
    optim="paged_adamw_8bit"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Start training
print("Starting training...")
trainer.train()

# Save model
trainer.save_model("./llama2-lora-final")
print("Training complete! Model saved to ./llama2-lora-final")
```

### Step 3: Run Training

```bash
python finetune.py
```

Expected output:
```
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
Starting training...
Step 10/180 | Loss: 2.34 | Time: 5.2s
Step 20/180 | Loss: 1.98 | Time: 5.1s
...
Training complete! Model saved to ./llama2-lora-final
```

### Step 4: Test Your Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./llama2-lora-final")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate
prompt = "### Human: What is AMD ROCm?\n### Assistant:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## What Just Happened?

1. **LoRA (Low-Rank Adaptation)**: Fine-tuned only 0.06% of parameters
2. **Memory Efficient**: Used BF16 and 8-bit optimizer
3. **Quick Training**: 3 epochs on 1000 examples in ~15 minutes
4. **Portable Model**: LoRA adapters are small (~8MB vs 13GB full model)

## Configuration Explained

### LoRA Parameters
```python
r=16          # Rank: Higher = more capacity, more memory
lora_alpha=32 # Scaling factor: Usually 2x rank
lora_dropout=0.05  # Regularization
```

### Training Parameters
```python
batch_size=4               # Per GPU batch size
gradient_accumulation=4    # Effective batch = 4x4 = 16
learning_rate=2e-4        # Standard for LoRA
bf16=True                 # Use BF16 on MI200+
```

## Next Steps

- **Multi-GPU Training**: Use [FSDP for distributed training](../03-training/distributed/fsdp-training.md)
- **Advanced LoRA**: Try [QLoRA for larger models](../03-training/fine-tuning/qlora-finetuning.md)
- **Your Own Data**: Learn [dataset preparation](../03-training/preparation/dataset-preparation.md)

## Common Issues

### Out of Memory
```python
# Reduce batch size or use QLoRA
per_device_train_batch_size=2
gradient_accumulation_steps=8

# Or use 4-bit quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### Slow Training
```bash
# Check GPU utilization
watch -n 1 rocm-smi

# Increase batch size if memory allows
per_device_train_batch_size=8
```

### Loss Not Decreasing
```python
# Adjust learning rate
learning_rate=1e-4  # Lower if unstable
learning_rate=5e-4  # Higher if too slow
```

## Performance Tips

1. **Use BF16**: Better than FP16 on AMD MI200+ series
2. **Optimize Batch Size**: Maximize GPU utilization
3. **Gradient Checkpointing**: For larger models
4. **Multi-GPU**: Use all available GPUs

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Multi-GPU with DeepSpeed
training_args = TrainingArguments(
    ...
    deepspeed="ds_config.json"
)
```

## Time Breakdown

- Installation: ~3 minutes
- Model download: ~15 GB, varies by connection
- Training setup: ~1 minute
- Training (3 epochs, 1000 examples): ~10-12 minutes
- Testing: ~30 seconds

## Resources Used

- **GPU Memory**: ~20GB for 7B model with LoRA
- **Disk Space**: ~15GB for model + ~8MB for LoRA weights
- **Training Time**: ~4 minutes per epoch (1000 examples)

Ready for production? Check out [full fine-tuning guide](../03-training/fine-tuning/full-finetuning.md) and [distributed training](../03-training/distributed/fsdp-training.md)!

