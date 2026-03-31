# GPT Models Optimization on ROCm

*Comprehensive guide to running and optimizing GPT models on AMD GPUs*

## Overview

GPT (Generative Pre-trained Transformer) models are autoregressive language models that excel at text generation, completion, and various NLP tasks. This guide covers optimizing GPT models (GPT-2, GPT-3, GPT-4, and variants) on AMD GPUs using ROCm.

## Supported GPT Model Variants

### GPT-2 Family
- **GPT-2 Small**: 124M parameters, 12 layers
- **GPT-2 Medium**: 355M parameters, 24 layers  
- **GPT-2 Large**: 774M parameters, 36 layers
- **GPT-2 XL**: 1.5B parameters, 48 layers

### GPT-3/3.5 Style Models
- **GPT-J-6B**: 6B parameters (EleutherAI)
- **GPT-NeoX-20B**: 20B parameters (EleutherAI)
- **CodeGen**: Code generation focused variants

### Recent Architectures
- **GPT-4 Style**: Multi-modal capabilities
- **ChatGPT Style**: Instruction-tuned variants
- **Code Models**: GitHub Copilot style models

## Installation and Setup

### Environment Setup
```bash
# Install PyTorch with ROCm 7.x support
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# For production environments, build from source:
# git clone --recursive https://github.com/ROCm/pytorch
# cd pytorch && pip install -r requirements.txt && python setup.py install

# Install transformers and related libraries
pip install transformers accelerate bitsandbytes-rocm

# Install optimized attention libraries
pip install flash-attn --no-build-isolation
pip install xformers
```

### Model Loading with Transformers
```python
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    GPTJForCausalLM, GPTNeoXForCausalLM,
    AutoModelForCausalLM, AutoTokenizer
)

# Check ROCm availability
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

# Load GPT-2 model
def load_gpt2(model_size="gpt2-medium", device="cuda"):
    tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{model_size}")
    model = GPT2LMHeadModel.from_pretrained(f"gpt2-{model_size}")
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move to AMD GPU
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

# Load larger models with memory optimization
def load_large_gpt(model_name="EleutherAI/gpt-j-6B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load with 16-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatic device placement
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer
```

## Text Generation Optimization

### Basic Generation
```python
def generate_text(model, tokenizer, prompt, max_length=100, device="cuda"):
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate with optimized parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Usage
model, tokenizer = load_gpt2("medium")
prompt = "The future of AI in healthcare is"
result = generate_text(model, tokenizer, prompt)
print(result)
```

### Advanced Generation Strategies
```python
def advanced_generation(model, tokenizer, prompt, strategy="nucleus"):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generation_config = {
        "max_new_tokens": 200,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    
    if strategy == "greedy":
        generation_config.update({
            "do_sample": False,
        })
    elif strategy == "beam_search":
        generation_config.update({
            "num_beams": 5,
            "early_stopping": True,
        })
    elif strategy == "nucleus":
        generation_config.update({
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
        })
    elif strategy == "contrastive":
        generation_config.update({
            "penalty_alpha": 0.6,
            "top_k": 4,
        })
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Batched Generation
```python
def batch_generation(model, tokenizer, prompts, batch_size=4):
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode batch
        batch_results = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            batch_results.append(text)
        
        results.extend(batch_results)
    
    return results
```

## Memory Optimization

### Gradient Checkpointing
```python
# Enable gradient checkpointing for training
model.gradient_checkpointing_enable()

# For inference, use torch.no_grad()
def memory_efficient_inference(model, tokenizer, prompt):
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clear cache after generation
        torch.cuda.empty_cache()
        
        return result
```

### Model Sharding for Large Models
```python
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from transformers.utils import hub

def load_sharded_model(model_name, device_map="auto"):
    # Initialize empty model
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name)
        )
    
    # Load and dispatch across devices
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_name,
        device_map=device_map,
        offload_folder="offload",
        offload_state_dict=True,
    )
    
    return model

# Usage for very large models
def load_gpt_neox_20b():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Custom device map for multi-GPU setup
    device_map = {
        "gpt_neox.embed_in": 0,
        "gpt_neox.layers.0": 0,
        "gpt_neox.layers.1": 0,
        # ... distribute layers across GPUs
        "gpt_neox.layers.43": 1,  
        "gpt_neox.final_layer_norm": 1,
        "embed_out": 1,
    }
    
    model = load_sharded_model("EleutherAI/gpt-neox-20b", device_map)
    return model, tokenizer
```

## Performance Optimization

### Flash Attention Integration
```python
# Enable Flash Attention for supported models
def enable_flash_attention(model):
    if hasattr(model, "config"):
        # Enable for models that support it
        model.config.use_flash_attention_2 = True
    
    return model

# Alternative: Use attention slicing for memory reduction
def enable_attention_slicing(model, slice_size="auto"):
    if hasattr(model, "enable_attention_slicing"):
        model.enable_attention_slicing(slice_size)
    return model
```

### Optimized Attention Patterns
```python
import torch.nn.functional as F

class OptimizedGPTAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-head attention parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        
        # Linear layers
        self.qkv_proj = torch.nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_values=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use ROCm-optimized scaled dot product attention
        if torch.cuda.is_available() and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=True if attention_mask is None else False
            )
        else:
            # Fallback implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                scores += attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        output = self.out_proj(attn_output)
        
        return output
```

### Quantization for Inference
```python
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

def load_quantized_gpt(model_name, quantization_config=None):
    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    return model

# 4-bit quantization for extreme memory savings
def load_4bit_gpt(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    return load_quantized_gpt(model_name, quantization_config)
```

## Fine-tuning GPT Models

### LoRA Fine-tuning
```python
from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_finetuning(model, target_modules=None):
    if target_modules is None:
        # Default target modules for GPT models
        target_modules = ["c_attn", "c_proj", "c_fc"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# Fine-tuning loop
def finetune_gpt_lora(model, tokenizer, train_dataset, num_epochs=3):
    from torch.utils.data import DataLoader
    from transformers import AdamW, get_linear_schedule_with_warmup
    
    # Setup data loader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in train_loader:
            # Tokenize batch
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    return model
```

### Full Fine-tuning with DeepSpeed
```python
import deepspeed
from deepspeed.ops.adam import FusedAdam

def setup_deepspeed_gpt(model, config_path="ds_config.json"):
    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        }
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    return model_engine, optimizer
```

## Multi-GPU Deployment

### Tensor Parallelism
```python
def setup_tensor_parallel_gpt(model, world_size=2):
    """Setup tensor parallelism for large GPT models"""
    
    # Split attention layers across GPUs
    for layer in model.transformer.h:
        # Split attention weights
        layer.attn.c_attn.weight = torch.nn.Parameter(
            layer.attn.c_attn.weight.chunk(world_size, dim=0)[torch.distributed.get_rank()]
        )
        
        # Split MLP weights
        layer.mlp.c_fc.weight = torch.nn.Parameter(
            layer.mlp.c_fc.weight.chunk(world_size, dim=0)[torch.distributed.get_rank()]
        )
    
    return model

# Pipeline parallelism setup
def setup_pipeline_parallel_gpt(model, num_stages=2):
    """Setup pipeline parallelism for GPT models"""
    
    layers_per_stage = len(model.transformer.h) // num_stages
    
    # Assign layers to pipeline stages
    for i, layer in enumerate(model.transformer.h):
        stage = i // layers_per_stage
        layer.to(f"cuda:{stage}")
    
    return model
```

### Distributed Inference
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def distributed_gpt_inference(model, tokenizer, prompts):
    """Run distributed inference across multiple GPUs"""
    
    # Initialize distributed processing
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Wrap model with DDP
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Split prompts across GPUs
    local_prompts = prompts[rank::world_size]
    
    results = []
    for prompt in local_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{rank}")
        
        with torch.no_grad():
            outputs = model.module.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(result)
    
    # Gather results from all GPUs
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)
    
    # Flatten results
    final_results = []
    for gpu_results in all_results:
        final_results.extend(gpu_results)
    
    return final_results
```

## Specialized Applications

### Code Generation (GitHub Copilot Style)
```python
class CodeGPT:
    def __init__(self, model_name="microsoft/CodeGPT-small-py"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add special tokens for code
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def complete_code(self, code_prompt, max_length=200):
        """Complete code given a prompt"""
        inputs = self.tokenizer(
            code_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=0.2,  # Lower temperature for code
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the new tokens
        generated_code = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_code
    
    def explain_code(self, code):
        """Generate explanation for code"""
        prompt = f"# Explain this code:\n{code}\n# Explanation:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation.split("# Explanation:\n")[-1].strip()

# Usage
code_gpt = CodeGPT()
prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
completion = code_gpt.complete_code(prompt)
print(f"Code completion: {completion}")
```

### Conversational AI
```python
class ConversationalGPT:
    def __init__(self, model_name="microsoft/DialoGPT-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.chat_history = []
        
        # Add special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def chat(self, user_input):
        """Generate response in a conversation"""
        # Add user input to history
        self.chat_history.append(f"User: {user_input}")
        
        # Create conversation context
        context = "\n".join(self.chat_history[-10:])  # Last 10 exchanges
        context += "\nAssistant:"
        
        inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Add to history
        self.chat_history.append(f"Assistant: {response}")
        
        return response
    
    def reset_conversation(self):
        """Reset chat history"""
        self.chat_history = []

# Usage
chatbot = ConversationalGPT()
response = chatbot.chat("Hello, how are you?")
print(f"Bot: {response}")
```

## Benchmarking and Evaluation

### Performance Benchmarking
```python
import time
from torch.profiler import profile, ProfilerActivity

def benchmark_gpt_inference(model, tokenizer, prompts, num_runs=10):
    """Benchmark GPT model inference performance"""
    
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
            model.generate(**inputs, max_new_tokens=50)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            for prompt in prompts[:5]:  # Benchmark on first 5 prompts
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                model.generate(**inputs, max_new_tokens=50, use_cache=True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    # Statistics
    avg_time = sum(times) / len(times)
    throughput = (len(prompts[:5]) * 50) / avg_time  # tokens/second
    
    print(f"Average time per batch: {avg_time:.3f}s")
    print(f"Throughput: {throughput:.1f} tokens/second")
    
    return avg_time, throughput

def profile_gpt_model(model, tokenizer, prompt):
    """Profile GPU usage during inference"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=100)
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    return prof
```

### Model Quality Evaluation
```python
def evaluate_generation_quality(model, tokenizer, test_prompts, reference_texts=None):
    """Evaluate generation quality using various metrics"""
    
    generated_texts = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text[len(prompt):])  # Remove prompt
    
    # Calculate metrics
    metrics = {}
    
    # Perplexity
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    for text in generated_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    
    metrics["perplexity"] = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    
    # Diversity metrics
    unique_tokens = set()
    total_tokens = 0
    
    for text in generated_texts:
        tokens = tokenizer.tokenize(text)
        unique_tokens.update(tokens)
        total_tokens += len(tokens)
    
    metrics["diversity"] = len(unique_tokens) / total_tokens if total_tokens > 0 else 0
    
    # Length statistics
    lengths = [len(tokenizer.tokenize(text)) for text in generated_texts]
    metrics["avg_length"] = sum(lengths) / len(lengths)
    metrics["length_std"] = torch.tensor(lengths).std().item()
    
    return metrics, generated_texts
```

## Production Deployment

### Model Serving with FastAPI
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="GPT Model API")

# Global model variables
model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    # Load model on startup
    model_name = "gpt2-medium"  # Configure as needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Deployment
```dockerfile
# Dockerfile for GPT model serving
FROM rocm/pytorch:latest

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Download model (or mount as volume)
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('gpt2-medium'); AutoTokenizer.from_pretrained('gpt2-medium')"

EXPOSE 8000

CMD ["python", "serve_gpt.py"]
```

## Best Practices

### Memory Management
```python
class GPTMemoryManager:
    def __init__(self, model):
        self.model = model
        self.initial_memory = torch.cuda.memory_allocated()
    
    def clear_cache_periodically(self, every_n_calls=10):
        """Clear CUDA cache every N generations"""
        if hasattr(self, 'call_count'):
            self.call_count += 1
        else:
            self.call_count = 1
            
        if self.call_count % every_n_calls == 0:
            torch.cuda.empty_cache()
    
    def monitor_memory(self):
        """Monitor memory usage"""
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"Current memory: {current_memory / 1e9:.2f} GB")
        print(f"Peak memory: {peak_memory / 1e9:.2f} GB")
        print(f"Memory increase: {(current_memory - self.initial_memory) / 1e9:.2f} GB")
        
        return current_memory, peak_memory
```

### Error Handling
```python
def robust_generation(model, tokenizer, prompt, max_retries=3):
    """Generate text with error handling and retries"""
    
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM error on attempt {attempt + 1}, clearing cache...")
            torch.cuda.empty_cache()
            
            if attempt == max_retries - 1:
                raise RuntimeError("Out of memory after all retries")
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            
            if attempt == max_retries - 1:
                raise
    
    return None
```

## Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [GPT Model Papers](https://openai.com/research)
- [ROCm Programming Guide](https://rocmdocs.amd.com/)

### Related Guides
- [LLaMA Optimization](../llama/llama-optimization.md)
- [Mistral Optimization](../mistral/mistral-optimization.md)
- [Memory Optimization](../../03-training/optimization/memory-optimization.md)

### Tools and Libraries
- [vLLM Serving](../../02-inference/serving-engines/vllm-serving.md)
- [DeepSpeed](https://deepspeed.ai/)
- [PEFT Library](https://github.com/huggingface/peft)

---
*Tags: gpt, language-models, text-generation, rocm, optimization, inference, fine-tuning*
*Estimated reading time: 60 minutes*