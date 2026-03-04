---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "inference"
subcategory: "optimization"
tags: ["attention", "optimization", "flashattention", "pagedattention", "performance"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "advanced"
estimated_time: "50min"
---

# Attention Mechanism Optimization for AMD GPUs

Advanced techniques for optimizing attention mechanisms in LLM inference on AMD hardware.

## Attention Mechanisms Overview

### Standard Attention

```python
import torch

def standard_attention(Q, K, V, mask=None):
    """
    Standard scaled dot-product attention
    Memory: O(n²) where n is sequence length
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

**Limitations:**
- O(n²) memory complexity
- Materializes full attention matrix
- Becomes prohibitive for long sequences

## Flash Attention

### What is Flash Attention?

Flash Attention is an IO-aware attention algorithm that:
- Reduces HBM accesses by using SRAM/cache efficiently
- Maintains O(n²) time but reduces memory to O(n)
- Achieves 2-4x speedup over standard attention

### Implementation with AMD GPUs

```python
from transformers import AutoModelForCausalLM
import torch

# Load model with Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Enable FA2
    device_map="auto"
)

# Verify Flash Attention is enabled
print(model.config.attn_implementation)
```

### Manual Flash Attention Implementation

```python
import torch
import torch.nn.functional as F

def flash_attention_forward(q, k, v, causal=True, sm_scale=None):
    """
    Flash Attention forward pass optimized for ROCm
    
    Args:
        q, k, v: [batch, heads, seq_len, head_dim]
        causal: Use causal masking
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
    """
    batch, heads, seq_len, head_dim = q.shape
    
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Block sizes optimized for AMD GPUs
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Output tensor
    out = torch.zeros_like(q)
    l = torch.zeros((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    m = torch.full((batch, heads, seq_len), float('-inf'), device=q.device, dtype=torch.float32)
    
    # Process in blocks
    for start_m in range(0, seq_len, BLOCK_M):
        end_m = min(start_m + BLOCK_M, seq_len)
        q_block = q[:, :, start_m:end_m, :]
        
        # Block-wise computation
        for start_n in range(0, seq_len, BLOCK_N):
            end_n = min(start_n + BLOCK_N, seq_len)
            
            # Skip if causal masking excludes this block
            if causal and start_n > end_m:
                continue
            
            k_block = k[:, :, start_n:end_n, :]
            v_block = v[:, :, start_n:end_n, :]
            
            # Compute attention scores for block
            qk = torch.matmul(q_block, k_block.transpose(-2, -1)) * sm_scale
            
            # Apply causal mask
            if causal:
                mask = torch.tril(torch.ones((end_m - start_m, end_n - start_n), 
                                            device=q.device, dtype=torch.bool))
                qk = qk.masked_fill(~mask, float('-inf'))
            
            # Online softmax
            m_new = torch.maximum(m[:, :, start_m:end_m], qk.max(dim=-1, keepdim=True).values)
            p = torch.exp(qk - m_new)
            l_new = torch.exp(m[:, :, start_m:end_m].unsqueeze(-1) - m_new) * \
                    l[:, :, start_m:end_m].unsqueeze(-1) + p.sum(dim=-1, keepdim=True)
            
            # Update output
            out[:, :, start_m:end_m, :] = \
                (out[:, :, start_m:end_m, :] * torch.exp(m[:, :, start_m:end_m].unsqueeze(-1) - m_new) * \
                 l[:, :, start_m:end_m].unsqueeze(-1) + torch.matmul(p, v_block)) / l_new
            
            # Update statistics
            m[:, :, start_m:end_m] = m_new.squeeze(-1)
            l[:, :, start_m:end_m] = l_new.squeeze(-1)
    
    return out
```

### Performance Comparison

```python
import torch
import time

def benchmark_attention(batch_size, seq_len, num_heads, head_dim, num_iters=100):
    """Compare standard vs flash attention"""
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(10):
        _ = standard_attention(q, k, v)
        _ = flash_attention_forward(q, k, v)
    
    torch.cuda.synchronize()
    
    # Benchmark standard attention
    start = time.time()
    for _ in range(num_iters):
        _ = standard_attention(q, k, v)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    
    # Benchmark flash attention
    start = time.time()
    for _ in range(num_iters):
        _ = flash_attention_forward(q, k, v)
    torch.cuda.synchronize()
    flash_time = time.time() - start
    
    print(f"Sequence length: {seq_len}")
    print(f"Standard Attention: {standard_time/num_iters*1000:.2f}ms")
    print(f"Flash Attention: {flash_time/num_iters*1000:.2f}ms")
    print(f"Speedup: {standard_time/flash_time:.2f}x")
    print()

# Run benchmarks
for seq_len in [512, 1024, 2048, 4096]:
    benchmark_attention(1, seq_len, 32, 128)
```

## Paged Attention (vLLM)

### Concept

PagedAttention manages KV cache in fixed-size blocks (pages), similar to virtual memory:

```python
┌─────────────────────────────────────┐
│  Logical KV Cache (Continuous)     │
│  [Seq1][Seq2][Seq3][Seq4]         │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Physical KV Cache (Blocks)         │
│  [B1][B3][B7][B2][B5]...           │
└─────────────────────────────────────┘
```

Benefits:
- Near-zero waste (<4% fragmentation)
- Dynamic sharing between sequences
- Memory-efficient batching

### vLLM with PagedAttention

```python
from vllm import LLM, SamplingParams

# Configure block size for PagedAttention
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    # PagedAttention configuration
    block_size=16,  # Tokens per block (16 is optimal for most cases)
    max_num_blocks_per_seq=2048,  # Maximum blocks per sequence
    gpu_memory_utilization=0.95,
    swap_space=4,  # GB of CPU swap for KV cache
)

# PagedAttention automatically handles batching
prompts = [f"Question {i}: " for i in range(100)]
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

# Efficient batching with shared memory
outputs = llm.generate(prompts, sampling_params)
```

### Custom PagedAttention Implementation

```python
import torch
import torch.nn.functional as F

class PagedAttentionCache:
    def __init__(self, num_blocks, block_size, num_heads, head_dim, dtype=torch.bfloat16):
        """
        Paged KV cache for efficient memory management
        
        Args:
            num_blocks: Total number of blocks in cache
            block_size: Number of tokens per block
            num_heads: Number of attention heads
            head_dim: Dimension of each head
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Allocate physical blocks
        self.k_cache = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            dtype=dtype, device='cuda'
        )
        
        # Block allocation tracking
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # seq_id -> list of block indices
    
    def allocate_sequence(self, seq_id, num_tokens):
        """Allocate blocks for a new sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of memory: insufficient blocks")
        
        allocated_blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated_blocks
        
        return allocated_blocks
    
    def free_sequence(self, seq_id):
        """Free blocks for a sequence"""
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)
    
    def write_kv(self, seq_id, position, k, v):
        """Write K, V to cache at position"""
        blocks = self.block_tables[seq_id]
        block_idx = position // self.block_size
        offset = position % self.block_size
        
        physical_block = blocks[block_idx]
        self.k_cache[physical_block, :, offset, :] = k
        self.v_cache[physical_block, :, offset, :] = v
    
    def read_kv(self, seq_id, length):
        """Read K, V from cache up to length"""
        blocks = self.block_tables[seq_id]
        num_blocks = (length + self.block_size - 1) // self.block_size
        
        # Gather blocks
        k_blocks = [self.k_cache[blocks[i]] for i in range(num_blocks)]
        v_blocks = [self.v_cache[blocks[i]] for i in range(num_blocks)]
        
        # Concatenate and trim
        k = torch.cat(k_blocks, dim=1)[:, :length, :]
        v = torch.cat(v_blocks, dim=1)[:, :length, :]
        
        return k, v

def paged_attention(q, cache, seq_id, seq_length):
    """
    Attention using paged KV cache
    
    Args:
        q: Query tensor [num_heads, head_dim]
        cache: PagedAttentionCache
        seq_id: Sequence identifier
        seq_length: Current sequence length
    """
    # Read KV from paged cache
    k, v = cache.read_kv(seq_id, seq_length)
    
    # Standard attention computation
    scale = 1.0 / (q.size(-1) ** 0.5)
    scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v).squeeze(1)
    
    return output
```

## Multi-Query Attention (MQA)

### Concept

Multi-Query Attention shares K and V across all heads, reducing KV cache size:

```python
Standard Attention:
- Q: [batch, num_heads, seq_len, head_dim]
- K: [batch, num_heads, seq_len, head_dim]
- V: [batch, num_heads, seq_len, head_dim]

Multi-Query Attention:
- Q: [batch, num_heads, seq_len, head_dim]
- K: [batch, 1, seq_len, head_dim]  ← Single head
- V: [batch, 1, seq_len, head_dim]  ← Single head
```

**Benefits:**
- Reduces KV cache by num_heads factor
- Enables longer sequences
- Minimal quality degradation

### Implementation

```python
import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head query, single-head K/V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states):
        batch, seq_len, hidden_size = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape Q for multi-head
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V remain single-head but broadcast across heads
        k = k.view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        
        # Broadcast K, V across all heads
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)
        
        # Standard attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        
        return output
```

## Grouped-Query Attention (GQA)

### Concept

GQA is a hybrid between MHA and MQA:
- Groups multiple query heads to share K/V
- Reduces KV cache while maintaining quality

```python
Llama-2-70B example:
- Query heads: 64
- KV heads: 8
- Group size: 64 / 8 = 8 queries per KV head
```

### Implementation

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size)
    
    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K, V for each group
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        output = self.out_proj(attn_output)
        
        return output
```

## ROCm-Specific Optimizations

### Using rocBLAS for Attention

```python
import torch

# Enable rocBLAS optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use BF16 for better performance on MI200+
def optimized_attention(q, k, v):
    """Attention optimized for AMD GPUs"""
    # Ensure BF16 dtype
    q = q.to(dtype=torch.bfloat16)
    k = k.to(dtype=torch.bfloat16)
    v = v.to(dtype=torch.bfloat16)
    
    # Scaled dot-product (uses rocBLAS GEMM)
    scale = 1.0 / (q.size(-1) ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # Output (uses rocBLAS GEMM)
    output = torch.matmul(attn_weights, v)
    
    return output
```

### Kernel Fusion

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    """
    Fused attention kernel for AMD GPUs using Triton
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load Q block
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)
    
    # Initialize output accumulators
    acc = tl.zeros([BLOCK_M, K], dtype=tl.float32)
    
    # Loop over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        # Load K block
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
                 (start_n + offs_n)[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[None, :] < N, other=0.0)
        
        # Compute attention scores
        qk = tl.dot(q, k)
        qk = qk * (1.0 / tl.sqrt(K.to(tl.float32)))
        
        # Softmax
        qk_max = tl.max(qk, axis=1, keep_dims=True)
        qk_exp = tl.exp(qk - qk_max)
        qk_sum = tl.sum(qk_exp, axis=1, keep_dims=True)
        attn = qk_exp / qk_sum
        
        # Load V block
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
                 (start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N, other=0.0)
        
        # Accumulate output
        acc += tl.dot(attn.to(v.dtype), v)
    
    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < M)
```

## Performance Tuning

### Block Size Selection

```python
def find_optimal_block_size(seq_len, num_heads, head_dim):
    """Find optimal block size for PagedAttention"""
    # Test different block sizes
    block_sizes = [8, 16, 32, 64]
    best_time = float('inf')
    best_size = 16
    
    for block_size in block_sizes:
        llm = LLM(
            model="meta-llama/Llama-2-7b-hf",
            block_size=block_size,
            dtype="bfloat16"
        )
        
        # Benchmark
        start = time.time()
        prompts = ["Test prompt"] * 10
        outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
        elapsed = time.time() - start
        
        if elapsed < best_time:
            best_time = elapsed
            best_size = block_size
        
        del llm
        torch.cuda.empty_cache()
    
    return best_size
```

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Triton Documentation](https://triton-lang.org/)

