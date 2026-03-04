---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "inference"
subcategory: "optimization"
tags: ["serving", "optimization", "throughput", "latency", "batching"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "advanced"
estimated_time: "55min"
---

# Serving Optimization for LLM Inference

Comprehensive guide to optimizing LLM serving performance on AMD GPUs.

## Performance Metrics

### Key Metrics

1. **Throughput**: Tokens/second across all requests
2. **Latency**: Time from request to first/complete token
3. **Time to First Token (TTFT)**: Prompt processing latency
4. **Time Per Output Token (TPOT)**: Generation speed
5. **Concurrent Users**: Number of simultaneous requests

### Measuring Performance

```python
import time
from vllm import LLM, SamplingParams

def benchmark_llm(model_name, prompts, max_tokens=100):
    """Comprehensive LLM benchmark"""
    llm = LLM(model=model_name, dtype="bfloat16")
    sampling_params = SamplingParams(max_tokens=max_tokens)
    
    # Measure total time
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start
    
    # Calculate metrics
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / total_time
    avg_latency = total_time / len(prompts)
    
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Requests: {len(prompts)}")
    
    return {
        'throughput': throughput,
        'latency': avg_latency,
        'total_tokens': total_tokens,
        'num_requests': len(prompts)
    }

# Run benchmark
prompts = [f"Tell me about topic {i}" for i in range(100)]
results = benchmark_llm("meta-llama/Llama-2-7b-hf", prompts)
```

## Batching Strategies

### Continuous Batching

```python
# vLLM automatically uses continuous batching
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=256,  # Maximum sequences in batch
    max_num_batched_tokens=8192,  # Maximum tokens per batch
)

# Requests are dynamically batched
prompts = ["Prompt 1", "Prompt 2", ...]
outputs = llm.generate(prompts, sampling_params)
```

### Static Batching

```python
def static_batch_inference(model, prompts, batch_size=32):
    """Process prompts in fixed-size batches"""
    outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_outputs = model.generate(batch, sampling_params)
        outputs.extend(batch_outputs)
    
    return outputs
```

### Dynamic Batching

```python
from collections import deque
import asyncio

class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        
    async def add_request(self, prompt):
        """Add request to batch queue"""
        future = asyncio.Future()
        self.queue.append((prompt, future))
        
        # Trigger batch if full
        if len(self.queue) >= self.max_batch_size:
            await self.process_batch()
        
        return await future
    
    async def process_batch(self):
        """Process accumulated batch"""
        if not self.queue:
            return
        
        # Extract batch
        batch = []
        futures = []
        while self.queue and len(batch) < self.max_batch_size:
            prompt, future = self.queue.popleft()
            batch.append(prompt)
            futures.append(future)
        
        # Process batch
        outputs = self.model.generate(batch, sampling_params)
        
        # Return results
        for future, output in zip(futures, outputs):
            future.set_result(output)
```

## GPU Memory Optimization

### Memory Configuration

```python
from vllm import LLM

# Optimize GPU memory utilization
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,  # Use 95% of GPU memory
    max_model_len=4096,  # Limit context length
    swap_space=4,  # GB of CPU swap for overflow
    quantization="awq",  # Use quantization
)
```

### KV Cache Optimization

```python
# Configure KV cache size
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    block_size=16,  # PagedAttention block size
    max_num_seqs=256,  # Max concurrent sequences
    max_num_batched_tokens=8192,  # Max tokens in batch
)

# Monitor KV cache usage
import torch
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## Tensor Parallelism

### Multi-GPU Configuration

```python
# Distribute model across GPUs
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,  # Use 8 GPUs
    dtype="bfloat16",
)

# GPU selection
import os
os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
```

### Pipeline Parallelism

```python
# Combine tensor and pipeline parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4-way tensor parallelism
    pipeline_parallel_size=2,  # 2-way pipeline parallelism
    # Total: 4 * 2 = 8 GPUs
)
```

## Quantization for Throughput

### AWQ Quantization

```python
# Use AWQ 4-bit quantization
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    dtype="float16",
)

# Benefits:
# - 4x memory reduction
# - ~2x throughput increase
# - Minimal quality loss
```

### GPTQ Quantization

```python
# Use GPTQ quantization
llm = LLM(
    model="TheBloke/Llama-2-70B-GPTQ",
    quantization="gptq",
    dtype="float16",
)
```

### FP8 Quantization

```python
# FP8 quantization (when available)
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    quantization="fp8",
    dtype="float16",
)
```

## Caching Strategies

### Prompt Caching

```python
class PromptCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, prompt):
        """Get cached result"""
        return self.cache.get(prompt)
    
    def put(self, prompt, result):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest
            self.cache.pop(next(iter(self.cache)))
        self.cache[prompt] = result
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()

# Usage
cache = PromptCache()

def generate_with_cache(prompt):
    # Check cache
    cached = cache.get(prompt)
    if cached:
        return cached
    
    # Generate
    result = llm.generate([prompt], sampling_params)[0]
    
    # Cache result
    cache.put(prompt, result)
    return result
```

### Prefix Caching

```python
# vLLM prefix caching (automatic)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
)

# Repeated prefixes are cached automatically
prompts = [
    "System: You are a helpful assistant.\nUser: Question 1",
    "System: You are a helpful assistant.\nUser: Question 2",
    "System: You are a helpful assistant.\nUser: Question 3",
]
# "System: You are a helpful assistant." is computed once
```

## Request Prioritization

### Priority Queue

```python
import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    timestamp: float = field(compare=False)
    prompt: str = field(compare=False)
    params: Any = field(compare=False)

class PriorityScheduler:
    def __init__(self, model):
        self.model = model
        self.queue = []
        
    def add_request(self, prompt, params, priority=1):
        """Add request with priority (lower = higher priority)"""
        request = PrioritizedRequest(
            priority=priority,
            timestamp=time.time(),
            prompt=prompt,
            params=params
        )
        heapq.heappush(self.queue, request)
    
    def process_batch(self, batch_size=32):
        """Process highest priority requests"""
        batch = []
        for _ in range(min(batch_size, len(self.queue))):
            request = heapq.heappop(self.queue)
            batch.append(request)
        
        if not batch:
            return []
        
        prompts = [r.prompt for r in batch]
        params = batch[0].params  # Assume same params
        
        outputs = self.model.generate(prompts, params)
        return outputs
```

## Load Balancing

### Round-Robin Balancer

```python
class RoundRobinBalancer:
    def __init__(self, backends):
        self.backends = backends
        self.index = 0
        
    def get_backend(self):
        """Get next backend in round-robin"""
        backend = self.backends[self.index]
        self.index = (self.index + 1) % len(self.backends)
        return backend
    
    async def generate(self, prompt, params):
        """Generate using load balanced backend"""
        backend = self.get_backend()
        return await backend.generate(prompt, params)

# Usage
backends = [
    LLM(model="llama-2-7b", tensor_parallel_size=2),
    LLM(model="llama-2-7b", tensor_parallel_size=2),
]
balancer = RoundRobinBalancer(backends)
```

### Least-Connections Balancer

```python
class LeastConnectionsBalancer:
    def __init__(self, backends):
        self.backends = backends
        self.connections = {id(b): 0 for b in backends}
        
    def get_backend(self):
        """Get backend with least connections"""
        backend = min(self.backends, 
                     key=lambda b: self.connections[id(b)])
        self.connections[id(backend)] += 1
        return backend
    
    def release_backend(self, backend):
        """Release backend after request"""
        self.connections[id(backend)] -= 1
```

## Monitoring and Profiling

### Performance Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter('llm_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('llm_latency_seconds', 'Request latency')
THROUGHPUT = Gauge('llm_throughput_tokens_per_sec', 'Throughput')
QUEUE_SIZE = Gauge('llm_queue_size', 'Queue size')

def monitored_generate(model, prompt, params):
    """Generate with monitoring"""
    REQUEST_COUNT.inc()
    QUEUE_SIZE.set(len(model.queue))
    
    start = time.time()
    output = model.generate([prompt], params)[0]
    latency = time.time() - start
    
    REQUEST_LATENCY.observe(latency)
    tokens = len(output.outputs[0].token_ids)
    THROUGHPUT.set(tokens / latency)
    
    return output
```

### Profiling

```python
import torch.profiler as profiler

def profile_inference(model, prompts):
    """Profile inference performance"""
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        outputs = model.generate(prompts, sampling_params)
    
    # Print results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    # Export trace
    prof.export_chrome_trace("inference_trace.json")
    
    return outputs
```

## Advanced Optimization Techniques

### Speculative Decoding

```python
def speculative_decoding(large_model, small_model, prompt, k=5):
    """
    Use small model to speculate k tokens,
    verify with large model
    """
    context = prompt
    outputs = []
    
    while len(outputs) < max_tokens:
        # Small model generates k candidate tokens
        candidates = small_model.generate(
            [context],
            SamplingParams(max_tokens=k)
        )[0]
        
        # Large model verifies candidates
        for token in candidates.outputs[0].token_ids:
            context_with_token = context + tokenizer.decode([token])
            
            # Verify with large model
            verification = large_model.generate(
                [context_with_token],
                SamplingParams(max_tokens=1)
            )[0]
            
            if verification.outputs[0].token_ids[0] == token:
                # Accepted
                outputs.append(token)
                context = context_with_token
            else:
                # Rejected, use large model token
                outputs.append(verification.outputs[0].token_ids[0])
                break
    
    return outputs
```

### Model Merging

```python
# Serve multiple LoRA adapters on same base model
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=8,  # Support 8 LoRA adapters
)

# Generate with different adapters
outputs1 = llm.generate(prompts, sampling_params, lora_request="adapter1")
outputs2 = llm.generate(prompts, sampling_params, lora_request="adapter2")
```

## Best Practices

### Configuration Tuning

```python
# For maximum throughput
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    max_num_seqs=256,  # High batch size
    max_num_batched_tokens=8192,  # High token limit
    gpu_memory_utilization=0.95,  # Maximum memory
)

# For minimum latency
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    max_num_seqs=1,  # Single request at a time
    max_num_batched_tokens=2048,  # Lower limit
    gpu_memory_utilization=0.90,  # Leave headroom
)
```

### Production Checklist

- [ ] Enable BF16/FP16 precision
- [ ] Configure tensor parallelism for large models
- [ ] Set appropriate max_num_seqs and max_num_batched_tokens
- [ ] Use quantization for memory-constrained scenarios
- [ ] Enable prefix caching for repeated prompts
- [ ] Implement request prioritization
- [ ] Set up monitoring and alerting
- [ ] Configure load balancing for multiple instances
- [ ] Implement graceful degradation
- [ ] Set up health checks
- [ ] Configure timeouts appropriately
- [ ] Enable logging and tracing

## Troubleshooting

### Low Throughput

```python
# Check GPU utilization
watch -n 1 rocm-smi

# Increase batch size
max_num_seqs=512
max_num_batched_tokens=16384

# Enable quantization
quantization="awq"

# Use more GPUs
tensor_parallel_size=8
```

### High Latency

```python
# Reduce batch size
max_num_seqs=32

# Reduce context length
max_model_len=2048

# Use smaller/quantized model
quantization="awq"

# Optimize network
# Use local model cache
HF_HOME=/local/fast/storage
```

### Out of Memory

```python
# Reduce memory usage
gpu_memory_utilization=0.85
max_model_len=2048

# Enable CPU swapping
swap_space=8

# Use quantization
quantization="awq"

# Use more GPUs
tensor_parallel_size=4
```

## References

- [vLLM Performance Tuning](https://docs.vllm.ai/en/stable/)
- [Continuous Batching Paper](https://arxiv.org/abs/2209.05996)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)

