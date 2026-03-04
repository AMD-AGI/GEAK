---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "models"
subcategory: "llama"
tags: ["llama", "llama-2", "optimization", "performance"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "advanced"
estimated_time: "45min"
---

# LLaMA Model Optimization on AMD GPUs

Comprehensive guide to optimizing LLaMA and LLaMA-2 models on AMD hardware.

## Model Overview

**LLaMA Family:**
- LLaMA-2-7B, 13B, 70B
- LLaMA-2-Chat variants
- Code LLaMA variants

**Architecture:** Transformer decoder with:
- Grouped-Query Attention (GQA) in larger models
- RMSNorm instead of LayerNorm
- SwiGLU activation
- Rotary Position Embeddings (RoPE)

## Inference Optimization

### vLLM Configuration

```python
from vllm import LLM, SamplingParams

# Optimal vLLM settings for LLaMA-2-70B on MI250X
llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,  # 4 GPUs
    dtype="bfloat16",  # Best on MI200+
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    # PagedAttention settings
    block_size=16,
    max_num_batched_tokens=8192,
    max_num_seqs=256,
    # Enable optimizations
    trust_remote_code=True,
    disable_custom_all_reduce=False
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    max_tokens=512
)

# Generate
outputs = llm.generate(prompts, sampling_params)
```

### Flash Attention Integration

```python
from transformers import AutoModelForCausalLM
import torch

# Enable Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Verify
print(f"Using attention: {model.config._attn_implementation}")
```

### Quantization Strategies

#### AWQ (4-bit)

```python
from vllm import LLM

# Use AWQ quantized model
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    dtype="float16",
    tensor_parallel_size=2,  # Reduced GPU requirement
    gpu_memory_utilization=0.95
)

# Memory: 140GB → 35GB
# Speed: ~2x faster
# Quality: ~98% of original
```

#### GPTQ (4-bit)

```python
llm = LLM(
    model="TheBloke/Llama-2-70B-GPTQ",
    quantization="gptq",
    dtype="float16",
    tensor_parallel_size=2
)
```

## Training Optimization

### LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Optimal LoRA config for LLaMA
lora_config = LoraConfig(
    r=64,  # Higher rank for 70B
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # MLP (important for LLaMA)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### QLoRA Fine-tuning

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Fine-tune 70B on single 48GB GPU
```

### Full Fine-tuning with DeepSpeed

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
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true
    }
}
```

## Model-Specific Optimizations

### RoPE Scaling for Longer Context

```python
from transformers import AutoModelForCausalLM, AutoConfig

# Extend context from 4K to 8K tokens
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config.rope_scaling = {
    "type": "linear",
    "factor": 2.0  # 2x context length
}

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.bfloat16
)
```

### GQA Optimization

```python
# LLaMA-2-70B uses GQA (num_key_value_heads=8)
# Optimize KV cache size

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    # KV cache is 8x smaller due to GQA
    max_num_seqs=512,  # Can handle more sequences
    block_size=16,
    gpu_memory_utilization=0.95
)
```

### SwiGLU Activation Tuning

```python
# SwiGLU is memory-intensive
# Enable gradient checkpointing for training

model.gradient_checkpointing_enable()
model.config.use_cache = False

# Checkpoint only MLP layers (SwiGLU)
def checkpoint_mlp(module):
    if hasattr(module, 'mlp'):
        module.mlp = torch.utils.checkpoint.checkpoint(module.mlp)

model.apply(checkpoint_mlp)
```

## Multi-GPU Strategies

### Tensor Parallelism

```python
# Split model across GPUs
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,  # 8-way parallelism
    dtype="bfloat16"
)

# Layer distribution:
# - Embedding: GPU 0
# - Layers 0-9: GPU 0-1
# - Layers 10-19: GPU 2-3
# - Layers 20-29: GPU 4-5
# - Layers 30-39: GPU 6-7
# - Output: GPU 7
```

### Pipeline Parallelism

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",  # Automatic pipeline parallelism
    torch_dtype=torch.bfloat16,
    max_memory={
        0: "40GB", 1: "40GB", 2: "40GB", 3: "40GB",
        4: "40GB", 5: "40GB", 6: "40GB", 7: "40GB"
    }
)
```

## Prompt Engineering

### Chat Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Apply chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is quantum computing?"}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Format:
# <s>[INST] <<SYS>>
# You are a helpful assistant.
# <</SYS>>
#
# What is quantum computing? [/INST]
```

### System Prompt Optimization

```python
# Good system prompts for LLaMA-2-Chat
SYSTEM_PROMPTS = {
    "general": "You are a helpful, respectful and honest assistant.",
    "coding": "You are an expert programmer. Provide clear, efficient code.",
    "analysis": "You are a data analyst. Provide detailed, accurate analysis."
}

def create_prompt(user_message, task="general"):
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user", "content": user_message}
    ]
```

## Performance Benchmarks

### Throughput

```python
import time
from vllm import LLM, SamplingParams

def benchmark_throughput(model_size="7b", num_prompts=100):
    model_name = f"meta-llama/Llama-2-{model_size}-hf"
    llm = LLM(model=model_name, dtype="bfloat16")
    
    prompts = [f"Prompt {i}" for i in range(num_prompts)]
    sampling_params = SamplingParams(max_tokens=100)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    print(f"LLaMA-2-{model_size}:")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Latency: {elapsed/num_prompts:.2f}s/request")

benchmark_throughput("7b")
benchmark_throughput("70b")
```

### Memory Usage

| Model | Precision | Memory | GPUs (MI250X) |
|-------|-----------|--------|---------------|
| 7B | BF16 | 14GB | 1 |
| 13B | BF16 | 26GB | 1 |
| 70B | BF16 | 140GB | 3-4 |
| 70B | AWQ 4-bit | 35GB | 1 |

## Production Deployment

### Docker Configuration

```dockerfile
FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_2.1.1

RUN pip install vllm transformers

ENV MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
ENV TENSOR_PARALLEL_SIZE=4
ENV MAX_MODEL_LEN=4096

CMD python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95
```

### Load Balancing

```nginx
upstream llama_backends {
    least_conn;
    server llama1:8000 max_fails=3 fail_timeout=30s;
    server llama2:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location /v1/ {
        proxy_pass http://llama_backends;
        proxy_read_timeout 600s;
    }
}
```

## Troubleshooting

### Out of Memory

```python
# Solution 1: Reduce context length
max_model_len=2048  # Instead of 4096

# Solution 2: Use quantization
quantization="awq"

# Solution 3: More GPUs
tensor_parallel_size=8
```

### Slow Generation

```python
# Solution 1: Increase batch size
max_num_seqs=256
max_num_batched_tokens=8192

# Solution 2: Use BF16
dtype="bfloat16"

# Solution 3: Optimize KV cache
block_size=16
```

## References

- [LLaMA-2 Paper](https://arxiv.org/abs/2307.09288)
- [LLaMA-2 Model Card](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [vLLM LLaMA Guide](https://docs.vllm.ai/en/latest/models/supported_models.html)

