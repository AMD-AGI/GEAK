---
layer: "5"
category: "quickstart"
subcategory: "inference"
tags: ["quickstart", "inference", "vllm", "llm", "getting-started"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "beginner"
estimated_time: "5min"
---

# LLM Inference in 5 Minutes

Get started with LLM inference on AMD GPUs in just 5 minutes using vLLM.

## Prerequisites

- ROCm 7.0+ installed
- Python 3.10+
- At least 16GB GPU memory (for 7B models)

## Quick Start

### Step 1: Install vLLM

```bash
# Install vLLM for ROCm
pip install vllm
```

### Step 2: Run Your First Inference

```python
from vllm import LLM, SamplingParams

# Initialize model (first run will download the model)
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Generate
prompts = ["Explain what is AMD ROCm in one sentence:"]
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
```

### Step 3: Verify It Works

```bash
python inference.py
```

Expected output:
```
AMD ROCm (Radeon Open Compute) is an open-source software platform 
that enables GPU computing for high-performance applications on AMD GPUs...
```

## What Just Happened?

1. **vLLM**: A high-performance inference engine optimized for AMD GPUs
2. **Model Loading**: Automatically downloaded Llama-2-7B from Hugging Face
3. **Inference**: Generated text using PagedAttention for efficient memory usage

## Next Steps

- **Serve an API**: Learn how to [deploy an OpenAI-compatible API](../02-inference/serving-engines/vllm-serving.md)
- **Use Different Models**: Try larger models or different architectures
- **Fine-tune**: Check out [fine-tuning quickstart](quickstart-finetuning.md)

## Common Issues

### Out of Memory
```bash
# Use smaller model or quantization
llm = LLM(model="meta-llama/Llama-2-7b-hf", quantization="awq")
```

### Model Download Fails
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/storage
```

### ROCm Not Detected
```bash
# Verify ROCm installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Performance Tips

For best performance:
- Use BF16 precision on MI200+ series
- Enable tensor parallelism for multi-GPU setups
- Adjust `gpu_memory_utilization` based on your GPU memory

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    gpu_memory_utilization=0.90
)
```

## Time Breakdown

- Installation: ~2 minutes
- First model download: ~15 GB, varies by connection
- First inference: ~30 seconds (model loading)
- Subsequent inference: <1 second per prompt

Ready for more? Check out the [complete vLLM guide](../02-inference/serving-engines/vllm-serving.md)!

