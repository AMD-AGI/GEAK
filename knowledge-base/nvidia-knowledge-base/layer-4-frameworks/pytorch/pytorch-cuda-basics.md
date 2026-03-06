Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "4"
category: "pytorch"
tags: ["pytorch", "deep-learning", "cuda", "training"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# PyTorch with CUDA

*Complete guide to using PyTorch with Nvidia GPUs*

## Installation

```bash
# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## Basic GPU Operations

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Move tensors to GPU
x = torch.randn(1000, 1000).cuda()  # Method 1
y = torch.randn(1000, 1000, device='cuda')  # Method 2
z = torch.randn(1000, 1000).to('cuda')  # Method 3

# Operations on GPU tensors
result = x @ y  # Matrix multiplication on GPU

# Move back to CPU
cpu_result = result.cpu()
```

## Training on GPU

```python
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()  # Move model to GPU

# Data and optimizer
data = torch.randn(32, 784).cuda()
target = torch.randint(0, 10, (32,)).cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training step
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward pass with scaled gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Multi-GPU Training

### DataParallel (Simple)

```python
model = MyModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()

# Training loop works the same
```

### DistributedDataParallel (Recommended)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# Training loop
for data, target in dataloader:
    data = data.cuda()
    target = target.cuda()
    # ... train as usual
```

Run with:
```bash
torchrun --nproc_per_node=8 train.py
```

## Performance Optimization

```python
# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Disable debugging features
torch.autograd.set_detect_anomaly(False)

# Pin memory for faster transfers
dataloader = DataLoader(dataset, pin_memory=True)

# Non-blocking transfers
data = data.cuda(non_blocking=True)

# Use TF32 on Ampere+ (default)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## Memory Management

```python
# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()

# Gradient checkpointing for memory savings
from torch.utils.checkpoint import checkpoint

def forward(x):
    x = checkpoint(layer1, x)
    x = checkpoint(layer2, x)
    return x
```

## Best Practices

1. **Use mixed precision**: 2x speedup on Volta+
2. **Enable cuDNN benchmark**: Faster convolutions
3. **Pin memory**: Faster data loading
4. **Use DDP**: Better than DataParallel
5. **Check Tensor Core usage**: Ensure FP16/TF32 active

## External Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)

## Related Guides

- [CUDA Basics](../../layer-2-compute-stack/cuda/cuda-basics.md)
- [cuDNN Usage](../../layer-3-libraries/dnn/cudnn-usage.md)
- [LLM Training](../../layer-5-llm/03-training/fine-tuning/lora-finetuning.md)

