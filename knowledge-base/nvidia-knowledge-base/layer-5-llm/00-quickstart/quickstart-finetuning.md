---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "llm"
subcategory: "quickstart"
tags: ["llm", "finetuning", "lora", "quickstart"]
cuda_version: "13.0+"
difficulty: "beginner"
estimated_time: "15min"
last_updated: 2025-11-17
---

# Fine-tune an LLM in 15 Minutes

*Quick start guide for LLM fine-tuning with LoRA*

## Prerequisites

- Nvidia GPU with 16GB+ VRAM
- CUDA 13.0+
- Python 3.9+

## Quick LoRA Fine-tuning

```bash
# Install dependencies
pip install transformers peft datasets accelerate bitsandbytes

# Create training script
cat > finetune.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
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
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("Abirate/english_quotes", split="train[:1000]")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="quote",
)

trainer.train()
model.save_pretrained("./lora-model")
EOF

# Run training
python finetune.py
```

## Test the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./lora-model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate
prompt = "The best way to"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Next Steps

- [Full LoRA Guide](../03-training/fine-tuning/lora-finetuning.md)
- [QLoRA for larger models](../03-training/fine-tuning/qlora-finetuning.md)
- [Dataset Preparation](../03-training/preparation/dataset-preparation.md)

