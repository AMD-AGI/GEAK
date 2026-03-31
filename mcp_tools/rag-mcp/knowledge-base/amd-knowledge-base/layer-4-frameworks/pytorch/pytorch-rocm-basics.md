---
layer: "4"
category: "pytorch"
subcategory: "framework"
tags: ["pytorch", "training", "inference", "rocm", "ml-framework"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
---

# PyTorch with ROCm

Complete guide to using PyTorch with AMD GPUs via ROCm.

**This documentation targets ROCm 7.0+ only.**

**Official Repository**: [https://github.com/ROCm/pytorch](https://github.com/ROCm/pytorch)  
**Upstream Fork**: [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)  
**Latest ROCm Release**: v2.3.0 + ROCm optimizations (August 2024)

> **Note**: PyTorch is now part of the ROCm ecosystem with dedicated ROCm-specific optimizations and features. The ROCm/pytorch repository is a fork of the upstream PyTorch with AMD GPU enhancements.

## Installation

### Prerequisites

Before installing PyTorch with ROCm support, ensure you have:

- **ROCm 7.0.0 or 7.0.2** installed and configured
- **Python 3.8+** (Python 3.10 recommended)
- **CMake 3.13+**
- **Git with LFS** support
- **Build essentials**: gcc/g++, make

### Option 1: Using Pre-built Wheels (Recommended)

```bash
# Install PyTorch for ROCm (nightly builds)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# For stable releases (when available)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'ROCm version: {torch.version.hip}')"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Option 2: Building from Source (Advanced)

For the latest ROCm 7.x optimizations and features:

```bash
# Clone the official ROCm/pytorch repository
git clone --recursive https://github.com/ROCm/pytorch
cd pytorch

# Checkout the develop branch (or specific release tag)
git checkout develop
git submodule sync
git submodule update --init --recursive

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables for ROCm
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_ROCM=1
export USE_CUDA=0

# Build and install (this may take 30-60 minutes)
python setup.py develop
# or for production installation:
# pip install -v -e .

# Verify installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
```

#### Build Configuration Options

You can customize the build with environment variables:

```bash
# Build with specific ROCm version
export ROCM_PATH=/opt/rocm

# Enable/disable features
export USE_CUDNN=0              # Disable cuDNN (use MIOpen instead)
export USE_DISTRIBUTED=1        # Enable distributed training
export USE_MKLDNN=1            # Enable oneDNN optimizations
export MAX_JOBS=$(nproc)       # Parallel build jobs

# Build with optimizations
export USE_NINJA=1             # Use Ninja build system (faster)
export BUILD_CAFFE2=0          # Disable Caffe2 build

# Build PyTorch
python setup.py install
```

### Option 3: Using Conda (if available)

```bash
# Create conda environment
conda create -n pytorch_rocm python=3.10
conda activate pytorch_rocm

# Install PyTorch with ROCm
# Note: Check conda-forge for latest ROCm-enabled packages
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

### Option 4: Using Docker (Recommended for Quick Start)

```bash
# Pull official PyTorch ROCm image (latest)
docker pull pytorch/pytorch:latest

# Or pull from ROCm Docker Hub for ROCm-specific images
docker pull rocm/pytorch:latest

# Run container with GPU access
docker run --gpus all --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    pytorch/pytorch:latest

# For ROCm 7.0.2 specific image (if available)
docker run --gpus all --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    rocm/pytorch:rocm7.0.2_ubuntu22.04_py3.10_pytorch_latest

# Verify inside container
python -c "import torch; print(torch.cuda.is_available())"
```

#### Building Custom Docker Image

```dockerfile
# Dockerfile for custom PyTorch with ROCm
FROM rocm/dev-ubuntu-22.04:7.0.2

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git cmake \
    && rm -rf /var/lib/apt/lists/*

# Clone and build PyTorch
RUN git clone --recursive https://github.com/ROCm/pytorch /opt/pytorch
WORKDIR /opt/pytorch
RUN pip install -r requirements.txt
RUN USE_ROCM=1 USE_CUDA=0 python setup.py install

# Set working directory
WORKDIR /workspace
```

Build and run:
```bash
docker build -t pytorch-rocm:custom .
docker run --gpus all --rm -it --ipc=host pytorch-rocm:custom
```

> **Note**: PyTorch uses shared memory to share data between processes. When using multiprocessing (e.g., multi-threaded data loaders), increase shared memory with `--ipc=host` or `--shm-size` option.

## Basic Usage

### GPU Detection and Setup

```python
import torch

# Check ROCm availability
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"ROCm version: {torch.version.hip}")

# GPU information
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Basic Tensor Operations

```python
# Create tensors on GPU
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')

# Operations
z = torch.matmul(x, y)
z = x + y
z = torch.nn.functional.relu(x)

# Move between devices
cpu_tensor = torch.randn(100, 100)
gpu_tensor = cpu_tensor.to('cuda')
back_to_cpu = gpu_tensor.cpu()

# Memory management
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
torch.cuda.empty_cache()
```

## Training a Simple Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Setup
device = torch.device('cuda')
model = SimpleNN(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
X = torch.randn(10000, 784)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for mixed precision
scaler = GradScaler()

model.train()
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast(dtype=torch.float16):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Distributed Training

### Single Machine Multi-GPU

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = SimpleNN(784, 256, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=128, sampler=sampler)
    
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(ddp_model.parameters())
    
    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)
        ddp_model.train()
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)
            
            outputs = ddp_model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch
import torch.multiprocessing as mp

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

### Using torchrun

```python
# train.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Get distributed training parameters from environment
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    # Create model
    model = SimpleNN(784, 256, 10).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Training code...
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

Launch:
```bash
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py

# Multi-node
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=10.0.0.1 --master_port=29500 train.py
```

## ROCm-Specific Optimizations

### Enable TF32/BF16 Operations

```python
# Enable TF32 for matrix multiplications (CDNA2+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use BF16 for training
model = model.to(dtype=torch.bfloat16)
```

### Optimize DataLoader

```python
# Use pin_memory for faster transfers
dataloader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

### GPU Selection

```python
# Select specific GPU
os.environ['HIP_VISIBLE_DEVICES'] = '0'

# Or in code
torch.cuda.set_device(0)

# Multi-GPU selection
os.environ['HIP_VISIBLE_DEVICES'] = '0,1,3'
```

## Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(10):
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
```

## Troubleshooting

### Check ROCm Installation

```python
import torch
print(torch.cuda.is_available())
print(torch.version.hip)
print(torch.cuda.get_arch_list())
```

### Memory Issues

```python
# Clear cache
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()

# Gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint
```

### Performance Issues

```bash
# Check GPU utilization
watch -n 1 rocm-smi

# Profile with rocprof
rocprof --hip-trace python train.py
```

## References

### Official Resources

- **[ROCm/pytorch GitHub Repository](https://github.com/ROCm/pytorch)** - Official AMD ROCm PyTorch fork
- **[PyTorch Official Website](https://pytorch.org/)** - Upstream PyTorch project
- **[PyTorch Getting Started](https://pytorch.org/get-started/locally/)** - Installation guide
- **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)** - Complete API reference

### Docker Images

- **[PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch)** - Official PyTorch images
- **[ROCm PyTorch Docker Hub](https://hub.docker.com/r/rocm/pytorch)** - ROCm-specific PyTorch images

### Training & Deployment

- **[PyTorch Tutorials](https://pytorch.org/tutorials/)** - Official tutorials
- **[PyTorch Examples](https://github.com/pytorch/examples)** - Code examples
- **[PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)** - Distributed training guide
- **[TorchServe](https://github.com/pytorch/serve)** - Model serving framework

### ROCm Documentation

- **[ROCm Installation Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)** - ROCm setup
- **[ROCm Documentation](https://rocm.docs.amd.com/en/latest/)** - Complete ROCm docs
- **[HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)** - HIP API reference

### Community

- **[PyTorch Forums](https://discuss.pytorch.org/)** - Community discussions
- **[PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)** - Bug reports and feature requests
- **[ROCm/pytorch Issues](https://github.com/ROCm/pytorch/issues)** - ROCm-specific issues

### Related Guides

- [Distributed Training with RCCL](../../layer-3-libraries/communications/rccl-usage.md)
- [MIOpen for Deep Learning](../../layer-3-libraries/ml-primitives/miopen-usage.md)
- [GPU Optimization Best Practices](../../best-practices/performance/gpu-optimization.md)

