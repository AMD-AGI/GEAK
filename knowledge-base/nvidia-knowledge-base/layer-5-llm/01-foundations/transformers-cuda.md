---
layer: "5"
category: "llm"
subcategory: "foundations"
tags: ["transformers", "huggingface", "llm"]
cuda_version: "13.0+"
difficulty: "intermediate"
estimated_time: "45min"
last_updated: 2025-11-17
---

# Hugging Face Transformers on Nvidia GPUs

*Complete guide to using Transformers library with CUDA*

## Installation

```bash
pip install transformers accelerate bitsandbytes
```

## Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model on GPU
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatic GPU placement
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate
prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Multi-GPU with device_map

```python
# Automatically distribute across GPUs
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    torch_dtype=torch.float16,
    device_map="auto"  # Splits across available GPUs
)
```

## 8-bit Quantization

```python
# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Saves ~50% memory
```

## 4-bit Quantization (QLoRA)

```python
from transformers import BitsAndBytesConfig

# 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Saves ~75% memory
```

## Pipeline API

```python
from transformers import pipeline

# Text generation pipeline
generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

# Generate
outputs = generator("Once upon a time", max_new_tokens=100)
print(outputs[0]['generated_text'])
```

## Batch Inference

```python
# Process multiple prompts efficiently
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Machine learning is"
]

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)

for output in outputs:
    print(tokenizer.decode(output))
```

## Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer)

# Generate in thread
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Stream output
for text in streamer:
    print(text, end="", flush=True)
```

## Flash Attention 2

```python
# Install flash-attn
# pip install flash-attn --no-build-isolation

# Enable Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # 2-3x faster
    device_map="auto"
)
```

## Best Practices

1. **Use `device_map="auto"`**: Automatic GPU placement
2. **Use FP16**: `torch_dtype=torch.float16`
3. **Enable Flash Attention**: Faster inference
4. **Quantization**: 8-bit/4-bit for memory savings
5. **Batch processing**: Higher throughput

## Common Issues

### Out of Memory

```python
# Solution 1: Use quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Solution 2: Use smaller model
# Solution 3: Reduce batch size
```

### Slow Generation

```python
# Enable Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Use FP16
model = model.half()
```

## External Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Quantization Guide](https://huggingface.co/docs/transformers/main_classes/quantization)
- [GPU Inference](https://huggingface.co/docs/transformers/perf_infer_gpu_one)

## Related Guides

- [PyTorch with CUDA](../../layer-4-frameworks/pytorch/pytorch-cuda-basics.md)
- [vLLM Serving](../02-inference/serving-engines/vllm-serving.md)
- [LoRA Fine-tuning](../03-training/fine-tuning/lora-finetuning.md)

