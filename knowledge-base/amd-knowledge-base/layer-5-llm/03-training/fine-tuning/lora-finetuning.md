---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "training"
subcategory: "fine-tuning"
tags: ["lora", "peft", "fine-tuning", "efficient-training"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "45min"
---

# LoRA Fine-tuning on AMD GPUs

Complete guide to efficient fine-tuning using Low-Rank Adaptation (LoRA) on AMD hardware.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:
- Trains only ~0.1% of model parameters
- Reduces memory requirements by 3x
- Maintains comparable quality to full fine-tuning
- Produces small adapter files (~10MB vs 13GB)

### How LoRA Works

```python
# Standard fine-tuning: Update all weights
W_new = W_old + ΔW  # ΔW is full rank

# LoRA: Decompose ΔW into low-rank matrices
W_new = W_old + B × A  # B: (d×r), A: (r×k), where r << d, k
```

## Quick Start

### Installation

```bash
pip install peft transformers datasets accelerate bitsandbytes
```

### Basic LoRA Fine-tuning

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# Load and prepare dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

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
    output_dir="./llama2-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    optim="paged_adamw_8bit"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train
trainer.train()

# Save LoRA adapters
model.save_pretrained("./llama2-lora-final")
```

## LoRA Configuration

### Key Parameters

```python
lora_config = LoraConfig(
    r=16,  # Rank: Higher = more capacity, more memory
           # Typical values: 8, 16, 32, 64
    
    lora_alpha=32,  # Scaling factor: Usually 2×rank
                     # Controls influence of LoRA adapters
    
    target_modules=[  # Which layers to adapt
        "q_proj",     # Query projection
        "k_proj",     # Key projection (optional)
        "v_proj",     # Value projection
        "o_proj",     # Output projection (optional)
        "gate_proj",  # For gated architectures
        "up_proj",
        "down_proj"
    ],
    
    lora_dropout=0.05,  # Dropout for LoRA layers
                         # Typical: 0.05-0.1
    
    bias="none",  # Bias handling: "none", "all", "lora_only"
    
    task_type="CAUSAL_LM",  # Task type
    
    modules_to_save=None,  # Additional modules to train fully
)
```

### Target Module Selection

```python
# Minimal (fastest, least memory)
target_modules=["q_proj", "v_proj"]

# Balanced (recommended)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Maximum (best quality, more memory)
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Find all linear layers automatically
import re

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    
    # Remove output layer
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

target_modules = find_all_linear_names(model)
```

## Advanced Techniques

### Multi-Rank LoRA

```python
# Use different ranks for different layers
from peft import LoraConfig, get_peft_model

# Higher rank for attention, lower for FFN
lora_config = LoraConfig(
    r=32,  # Default rank
    target_modules={
        "q_proj": 32,  # Attention gets higher rank
        "v_proj": 32,
        "up_proj": 16,  # FFN gets lower rank
        "down_proj": 16
    },
    lora_alpha=64,
    task_type="CAUSAL_LM"
)
```

### LoRA with Gradient Checkpointing

```python
# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

### Layer-Specific LoRA

```python
# Apply LoRA only to specific layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[20, 21, 22, 23],  # Only last 4 layers
    task_type="CAUSAL_LM"
)
```

## Inference with LoRA

### Loading LoRA Adapters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./llama2-lora-final")

# Merge adapters into base model (optional, for deployment)
model = model.merge_and_unload()

# Inference
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("What is machine learning?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Multiple LoRA Adapters

```python
# Load base model once
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto"
)

# Load different adapters for different tasks
coding_model = PeftModel.from_pretrained(base_model, "./lora-coding")
math_model = PeftModel.from_pretrained(base_model, "./lora-math")
chat_model = PeftModel.from_pretrained(base_model, "./lora-chat")

# Switch adapters
model.set_adapter("coding")  # Use coding adapter
model.set_adapter("math")  # Switch to math adapter
```

### Merging Multiple LoRA Adapters

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load and merge multiple adapters
model = PeftModel.from_pretrained(base_model, "./lora-adapter-1")
model.load_adapter("./lora-adapter-2", adapter_name="adapter2")

# Merge adapters with weights
model.add_weighted_adapter(
    adapters=["default", "adapter2"],
    weights=[0.7, 0.3],
    adapter_name="merged",
    combination_type="linear"
)

model.set_adapter("merged")
```

## Optimization Techniques

### Memory Optimization

```python
from transformers import BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Memory usage: ~70GB → ~10GB for 70B model
```

### Speed Optimization

```python
# Use larger batch size with gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Increase batch size
    gradient_accumulation_steps=2,   # Reduce accumulation
    bf16=True,  # Use BF16 on MI200+
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    gradient_checkpointing=True,  # Enable checkpointing
    max_grad_norm=0.3,  # Gradient clipping
)
```

## Monitoring and Debugging

### Training Metrics

```python
from transformers import TrainerCallback

class LoRAMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log LoRA-specific metrics
            print(f"Step {state.global_step}:")
            print(f"  Loss: {logs.get('loss', 0):.4f}")
            print(f"  Learning rate: {logs.get('learning_rate', 0):.2e}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[LoRAMetricsCallback()]
)
```

### Inspecting LoRA Weights

```python
# Print LoRA parameters
for name, param in model.named_parameters():
    if 'lora' in name:
        print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# Get LoRA weight statistics
def print_lora_stats(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            lora_a = module.lora_A['default'].weight
            lora_b = module.lora_B['default'].weight
            print(f"\n{name}:")
            print(f"  LoRA A: {lora_a.shape}, std={lora_a.std():.4f}")
            print(f"  LoRA B: {lora_b.shape}, std={lora_b.std():.4f}")

print_lora_stats(model)
```

## Best Practices

### Hyperparameter Selection

```python
# For 7B models on single GPU
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_steps=100,
    bf16=True
)

# For 70B models with quantization
lora_config = LoraConfig(
    r=64,  # Higher rank for larger models
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1
)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    num_train_epochs=1,
    bf16=True,
    optim="paged_adamw_8bit"
)
```

### Common Pitfalls

```python
# ❌ Wrong: Using FP32 (wastes memory)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✓ Right: Use BF16/FP16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)

# ❌ Wrong: Too high learning rate
learning_rate=5e-3  # Too high for LoRA

# ✓ Right: Appropriate for LoRA
learning_rate=2e-4

# ❌ Wrong: Adapting too few layers
target_modules=["q_proj"]  # Insufficient

# ✓ Right: Balance coverage and efficiency
target_modules=["q_proj", "v_proj"]  # or more
```

## Troubleshooting

### Low Training Speed

```python
# Enable optimizations
training_args = TrainingArguments(
    ...
    bf16=True,  # Use BF16
    optim="paged_adamw_8bit",  # Efficient optimizer
    gradient_checkpointing=False,  # Disable if memory allows
    dataloader_num_workers=4,  # Parallel data loading
)
```

### Poor Quality

```python
# Increase LoRA capacity
lora_config = LoraConfig(
    r=64,  # Increase rank
    lora_alpha=128,
    target_modules=[  # Add more modules
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# Adjust training
training_args = TrainingArguments(
    ...
    num_train_epochs=5,  # More epochs
    learning_rate=1e-4,  # Lower LR
)
```

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [LoRA Best Practices](https://huggingface.co/docs/peft/conceptual_guides/lora)

