---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "foundations"
subcategory: "transformers"
tags: ["transformers", "huggingface", "models", "inference", "training"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "45min"
---

# Hugging Face Transformers on AMD GPUs

Complete guide to using the Transformers library with AMD ROCm for model inference and training.

## Installation

### Basic Installation

```bash
# Install transformers and dependencies
pip install transformers accelerate datasets

# Install PyTorch for ROCm 7.x
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# For production, consider building from source for ROCm 7.x:
# git clone --recursive https://github.com/ROCm/pytorch
# cd pytorch && pip install -r requirements.txt && python setup.py install
```

### Optional Dependencies

```bash
# For tokenizers
pip install tokenizers

# For evaluation
pip install evaluate scikit-learn

# For quantization
pip install bitsandbytes optimum

# For ONNX export
pip install optimum[onnxruntime]
```

## Model Loading and Inference

### Loading Pre-trained Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load model on GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Automatically distribute across GPUs
)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Device Placement Strategies

```python
# Strategy 1: Single GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
).to("cuda")

# Strategy 2: Auto device map (recommended for large models)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Automatically splits across available GPUs
)

# Strategy 3: Manual device map
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0-15": 0,
    "model.layers.16-31": 1,
    "model.layers.32-47": 2,
    "model.layers.48-63": 3,
    "model.norm": 3,
    "lm_head": 3
}
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map=device_map
)

# Strategy 4: Offload to CPU/disk
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)
```

## Text Generation

### Basic Generation

```python
from transformers import pipeline

# Create generation pipeline
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device=0,  # GPU 0
    torch_dtype=torch.bfloat16
)

# Generate
outputs = generator(
    "Once upon a time",
    max_length=100,
    num_return_sequences=3,
    temperature=0.8
)

for i, output in enumerate(outputs):
    print(f"Generated {i+1}: {output['generated_text']}")
```

### Advanced Generation Parameters

```python
from transformers import GenerationConfig

# Configure generation
generation_config = GenerationConfig(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Generate with config
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

# Create streamer
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Generate in thread
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Print tokens as they're generated
for new_text in streamer:
    print(new_text, end='', flush=True)

thread.join()
```

## Quantization

### 8-bit Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Memory usage: ~70GB → ~35GB
```

### 4-bit Quantization (QLoRA)

```python
# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Memory usage: ~70GB → ~18GB
```

### GPTQ Quantization

```python
from optimum.gptq import GPTQQuantizer

# Quantize model (one-time process)
quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize="model.layers")
quantizer.quantize_model(model, tokenizer)
model.save_pretrained("llama2-70b-gptq-4bit")

# Load quantized model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "llama2-70b-gptq-4bit",
    device_map="auto"
)
```

## Model Types and Tasks

### Causal Language Models (GPT-style)

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
# For: text generation, completion
```

### Sequence Classification

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
# For: sentiment analysis, text classification
```

### Question Answering

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
# For: extractive QA
```

### Token Classification

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=9
)
# For: NER, POS tagging
```

## Training with Transformers

### Basic Training Loop

```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    bf16=True,  # Use BF16 on AMD MI200+
    gradient_checkpointing=True
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()
```

### Custom Training Loop

```python
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Setup
optimizer = AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
    print(f"Epoch {epoch+1} complete")
```

## Model Parallelism

### Pipeline Parallelism

```python
from transformers import AutoModelForCausalLM

# Automatically distribute layers across GPUs
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Manual pipeline parallelism
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
model.parallelize()  # Simple API for pipeline parallelism
```

### DeepSpeed Integration

```python
# Training with DeepSpeed
training_args = TrainingArguments(
    output_dir="./results",
    deepspeed="ds_config.json",
    bf16=True,
    ...
)

trainer = Trainer(
    model=model,
    args=training_args,
    ...
)
trainer.train()
```

DeepSpeed config (`ds_config.json`):
```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

## Performance Optimization

### Memory Optimization

```python
# 1. Gradient checkpointing
model.gradient_checkpointing_enable()

# 2. Mixed precision training
training_args = TrainingArguments(
    bf16=True,  # Use BF16 on MI200+
    bf16_full_eval=True
)

# 3. Gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8  # Effective batch size = 32
)

# 4. Optimizer state offloading
from transformers import Trainer
from torch.optim import AdamW

trainer = Trainer(
    model=model,
    optimizers=(AdamW(model.parameters(), lr=5e-5), None)
)
```

### Speed Optimization

```python
# 1. Compile model (PyTorch 2.0+)
import torch
model = torch.compile(model)

# 2. Flash Attention (when available)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)

# 3. Better TransformerEngine integration
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
```

## Saving and Loading Models

### Save Full Model

```python
# Save model and tokenizer
model.save_pretrained("./my-finetuned-model")
tokenizer.save_pretrained("./my-finetuned-model")

# Load
model = AutoModelForCausalLM.from_pretrained("./my-finetuned-model")
tokenizer = AutoTokenizer.from_pretrained("./my-finetuned-model")
```

### Save Only Weights

```python
# Save checkpoint
torch.save(model.state_dict(), "model_checkpoint.pt")

# Load
model.load_state_dict(torch.load("model_checkpoint.pt"))
```

### Push to Hub

```python
# Login to Hugging Face
from huggingface_hub import login
login(token="your_token_here")

# Push model
model.push_to_hub("username/my-model")
tokenizer.push_to_hub("username/my-model")

# Load from hub
model = AutoModelForCausalLM.from_pretrained("username/my-model")
```

## Troubleshooting

### Out of Memory

```python
# 1. Reduce batch size
per_device_train_batch_size=2

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Use quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 4. Offload to CPU
device_map="auto"
offload_folder="offload"
```

### Slow Inference

```python
# 1. Use better dtype
torch_dtype=torch.bfloat16  # Faster than float32

# 2. Batch inputs
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# 3. Use pipelines (optimized)
generator = pipeline("text-generation", model=model, batch_size=8)
```

### Model Not Loading

```python
# Check available memory
import torch
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Use device_map auto for large models
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

## References

- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Transformers on AMD ROCm](https://huggingface.co/docs/transformers/perf_hardware)
- [Model Hub](https://huggingface.co/models)
- [Quantization Guide](https://huggingface.co/docs/transformers/main_classes/quantization)

