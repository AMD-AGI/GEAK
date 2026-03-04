---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "training"
subcategory: "preparation"
tags: ["environment", "setup", "training", "configuration"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "30min"
---

# Training Environment Setup

Complete guide to setting up an optimal training environment for LLMs on AMD GPUs.

## Hardware Requirements

### Minimum (7B-13B models)
- 1-2x AMD MI250X (64GB each) or MI300X (128GB+)
- 128GB system RAM
- 1TB NVMe SSD
- 10Gbps network

### Recommended (70B+ models)
- 4-8x AMD MI250X or MI300X
- 512GB+ system RAM  
- 2TB+ NVMe SSD
- 100Gbps network (for multi-node)

## System Setup

### ROCm Installation

```bash
# Install ROCm 7.0
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_latest_all.deb
sudo dpkg -i amdgpu-install_latest_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocm-smi
rocminfo

# Set environment variables
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Python Environment

```bash
# Create virtual environment
python3.10 -m venv ~/venvs/llm-training
source ~/venvs/llm-training/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch for ROCm 7.x
# Using nightly build for best ROCm 7.x compatibility
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# Alternative: Build from official ROCm/pytorch for production
# git clone --recursive https://github.com/ROCm/pytorch
# cd pytorch && pip install -r requirements.txt && python setup.py install

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### Essential Libraries

```bash
# Core ML libraries
pip install transformers datasets accelerate evaluate

# Training frameworks
pip install deepspeed
pip install flash-attn --no-build-isolation

# Parameter-efficient fine-tuning
pip install peft bitsandbytes

# Monitoring and logging
pip install wandb tensorboard

# Utilities
pip install huggingface_hub tqdm sentencepiece protobuf
```

## Storage Configuration

### Model Cache Setup

```bash
# Create cache directories
export MODEL_CACHE=/data/models/cache
export HF_HOME=$MODEL_CACHE
export TRANSFORMERS_CACHE=$MODEL_CACHE
export HF_DATASETS_CACHE=/data/datasets

# Create directories
sudo mkdir -p $MODEL_CACHE $HF_DATASETS_CACHE
sudo chown -R $USER:$USER /data/models /data/datasets

# Add to ~/.bashrc
cat << 'EOF' >> ~/.bashrc
export MODEL_CACHE=/data/models/cache
export HF_HOME=$MODEL_CACHE
export TRANSFORMERS_CACHE=$MODEL_CACHE
export HF_DATASETS_CACHE=/data/datasets
EOF
```

### Optimizing Storage

```bash
# Use NVMe for training data
sudo mkdir -p /nvme/training
sudo mount /dev/nvme0n1 /nvme/training
sudo chown -R $USER:$USER /nvme/training

# Configure tmpfs for fast temporary storage
sudo mkdir -p /mnt/tmpfs
sudo mount -t tmpfs -o size=64G tmpfs /mnt/tmpfs
```

## Network Configuration

### Multi-GPU Communication

```bash
# Verify RCCL
python -c "import torch; import torch.distributed as dist; print('RCCL available:', torch.cuda.nccl.version())"

# Test GPU-to-GPU bandwidth
/opt/rocm/bin/rocm-bandwidth-test

# For InfiniBand (multi-node)
sudo apt-get install -y infiniband-diags ibverbs-utils
ibstat  # Verify IB connectivity
```

### Environment Variables

```bash
# Optimize RCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# ROCm optimizations
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8

# Add to training script
cat << 'EOF' >> ~/.bashrc
export NCCL_DEBUG=INFO
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
EOF
```

## Docker Setup (Optional)

### Build Training Container

```dockerfile
FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_2.1.1

# Install training dependencies
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    datasets==2.14.0 \
    accelerate==0.24.0 \
    deepspeed==0.11.0 \
    peft==0.6.0 \
    bitsandbytes==0.41.0 \
    wandb==0.15.0 \
    tensorboard==2.14.0

# Set up workspace
WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/models /workspace/outputs

# Environment variables
ENV HF_HOME=/workspace/models/cache
ENV TRANSFORMERS_CACHE=/workspace/models/cache
ENV HF_DATASETS_CACHE=/workspace/data

CMD ["/bin/bash"]
```

Build and run:
```bash
docker build -t llm-training:latest -f Dockerfile.training .

docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host --shm-size 64G \
    -v $(pwd):/workspace \
    -v /data/models:/workspace/models \
    -v /data/datasets:/workspace/data \
    llm-training:latest
```

## Monitoring Tools

### System Monitoring

```bash
# Install monitoring tools
sudo apt-get install -y htop iotop nethogs

# GPU monitoring
watch -n 1 rocm-smi

# Create monitoring script
cat << 'EOF' > monitor.sh
#!/bin/bash
while true; do
    clear
    echo "=== GPU Status ==="
    rocm-smi
    echo -e "\n=== System Memory ==="
    free -h
    echo -e "\n=== Disk Usage ==="
    df -h | grep -E '/$|/data|/nvme'
    sleep 2
done
EOF
chmod +x monitor.sh
```

### Weights & Biases

```bash
# Install and login
pip install wandb
wandb login

# Test logging
python << 'EOF'
import wandb

wandb.init(project="test-project")
wandb.log({"metric": 1.0})
wandb.finish()
EOF
```

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir=./runs --bind_all --port=6006

# Access at http://localhost:6006
```

## Configuration Files

### Training Config Template

```yaml
# config/training_config.yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  dtype: "bfloat16"
  gradient_checkpointing: true

training:
  output_dir: "./outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  save_total_limit: 3
  
optimization:
  bf16: true
  optim: "adamw_torch"
  max_grad_norm: 1.0
  weight_decay: 0.01
  
distributed:
  strategy: "fsdp"  # or "deepspeed"
  world_size: 4
```

### DeepSpeed Config

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}
```

## Validation Script

```python
# validate_setup.py
import torch
import sys

def validate_environment():
    """Validate training environment setup"""
    checks = []
    
    # Check PyTorch
    checks.append(("PyTorch installed", True))
    print(f"PyTorch version: {torch.__version__}")
    
    # Check ROCm
    rocm_available = torch.cuda.is_available()
    checks.append(("ROCm available", rocm_available))
    if rocm_available:
        print(f"ROCm version: {torch.version.hip}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check distributed
    try:
        import torch.distributed as dist
        checks.append(("Distributed available", True))
        print(f"NCCL version: {torch.cuda.nccl.version()}")
    except:
        checks.append(("Distributed available", False))
    
    # Check transformers
    try:
        import transformers
        checks.append(("Transformers installed", True))
        print(f"Transformers version: {transformers.__version__}")
    except:
        checks.append(("Transformers installed", False))
    
    # Check accelerate
    try:
        import accelerate
        checks.append(("Accelerate installed", True))
        print(f"Accelerate version: {accelerate.__version__}")
    except:
        checks.append(("Accelerate installed", False))
    
    # Check DeepSpeed
    try:
        import deepspeed
        checks.append(("DeepSpeed installed", True))
        print(f"DeepSpeed version: {deepspeed.__version__}")
    except:
        checks.append(("DeepSpeed installed", False))
    
    # Check PEFT
    try:
        import peft
        checks.append(("PEFT installed", True))
        print(f"PEFT version: {peft.__version__}")
    except:
        checks.append(("PEFT installed", False))
    
    # Summary
    print("\n" + "="*50)
    print("Setup Validation Summary")
    print("="*50)
    failed = []
    for check, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
        if not passed:
            failed.append(check)
    
    if failed:
        print(f"\n❌ {len(failed)} check(s) failed:")
        for check in failed:
            print(f"  - {check}")
        return False
    else:
        print("\n✅ All checks passed!")
        return True

if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
```

Run validation:
```bash
python validate_setup.py
```

## Troubleshooting

### ROCm Not Detected

```bash
# Check ROCm installation
ls /opt/rocm

# Verify drivers
dmesg | grep amdgpu

# Reinstall if needed
sudo amdgpu-install --usecase=rocm --uninstall
sudo amdgpu-install --usecase=rocm
```

### Out of Disk Space

```bash
# Clean up cache
rm -rf ~/.cache/huggingface/hub/*
rm -rf ~/.cache/torch/*

# Check disk usage
du -sh ~/.cache/*
df -h
```

### Slow Network Downloads

```bash
# Use mirror
export HF_ENDPOINT=https://hf-mirror.com

# Pre-download models
huggingface-cli download meta-llama/Llama-2-7b-hf
```

## References

- [ROCm Installation Guide](https://rocm.docs.amd.com/)
- [PyTorch ROCm Setup](https://pytorch.org/get-started/locally/)
- [Transformers Installation](https://huggingface.co/docs/transformers/installation)
- [DeepSpeed Setup](https://www.deepspeed.ai/getting-started/)

