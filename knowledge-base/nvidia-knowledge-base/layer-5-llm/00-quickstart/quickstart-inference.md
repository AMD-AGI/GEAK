Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "5"
category: "llm"
subcategory: "quickstart"
tags: ["llm", "inference", "quickstart", "vllm"]
cuda_version: "13.0+"
difficulty: "beginner"
estimated_time: "5min"
last_updated: 2025-11-17
---

# LLM Inference in 5 Minutes

*Get LLM inference running on Nvidia GPUs in minutes*

## Prerequisites

- Nvidia GPU with 16GB+ VRAM
- CUDA 13.0+ installed
- Python 3.9+

## Quick Start with vLLM

```bash
# Install vLLM
pip install vllm

# Run inference
python -c "
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model='meta-llama/Llama-2-7b-hf')

# Generate
prompts = ['Hello, my name is']
outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95))

for output in outputs:
    print(output.outputs[0].text)
"
```

## With Transformers

```bash
# Install
pip install transformers accelerate

# Run
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

prompt = 'Hello, my name is'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
"
```

## Docker

```bash
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-hf
```

## Next Steps

- [vLLM Serving Guide](../02-inference/serving-engines/vllm-serving.md)
- [Production Deployment](../02-inference/deployment/production-serving.md)

