---
layer: "2"
category: "cuda"
subcategory: "installation"
tags: ["cuda", "installation", "setup", "drivers"]
cuda_version: "13.0+"
cuda_verified: "13.0"
last_updated: 2025-11-17
---

# CUDA Installation Guide

*Complete guide to installing CUDA Toolkit and Nvidia drivers*

## Overview

CUDA Toolkit provides the complete development environment for C/C++ GPU programming, including compiler, libraries, and debugging tools. This guide covers CUDA 13.0+ installation on Ubuntu/Debian and RHEL/CentOS systems.

**Official Documentation**: [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## Prerequisites

### System Requirements

**Minimum:**
- Nvidia GPU with compute capability 5.0+ (Maxwell or newer)
- 64-bit Linux distribution
- GCC compiler
- Kernel headers and development packages

**Recommended:**
- Nvidia GPU with compute capability 7.0+ (Volta or newer)
- Ubuntu 22.04 or RHEL 8/9
- Latest kernel

### Check GPU Compatibility

```bash
# Check if Nvidia GPU is present
lspci | grep -i nvidia

# Example output:
# 01:00.0 3D controller: NVIDIA Corporation Device 2330 (rev a1)

# Check compute capability (if driver is installed)
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Installation Methods

### Method 1: Distribution Package Managers (Recommended)

Easiest method with automatic dependency management.

#### Ubuntu/Debian

```bash
# Remove old CUDA/Nvidia installations (if any)
sudo apt-get --purge remove "*cuda*" "*nvidia*"
sudo apt-get autoremove

# Update package list
sudo apt-get update

# Install CUDA Toolkit (includes drivers)
sudo apt-get install -y cuda-toolkit-12-6

# Or install with drivers
sudo apt-get install -y cuda-12-6

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### RHEL/CentOS/Fedora

```bash
# Install kernel headers
sudo dnf install kernel-devel-$(uname -r) kernel-headers-$(uname -r)

# Add CUDA repository
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install CUDA
sudo dnf clean all
sudo dnf -y install cuda-toolkit-12-6

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Method 2: Runfile Installer

For custom installations or when package manager is unavailable.

```bash
# Download CUDA installer
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

# Make executable
chmod +x cuda_12.6.0_560.28.03_linux.run

# Run installer
sudo sh cuda_12.6.0_560.28.03_linux.run

# Follow prompts:
# - Accept EULA
# - Install driver (if needed)
# - Install toolkit
# - Install samples (optional)

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Method 3: Docker (Development/Testing)

For isolated environments or cloud deployments.

```bash
# Pull official CUDA image
docker pull nvidia/cuda:13.0.0-devel-ubuntu22.04

# Run with GPU access
docker run --gpus all -it nvidia/cuda:13.0.0-devel-ubuntu22.04 bash

# Verify CUDA inside container
nvcc --version
nvidia-smi
```

See [CUDA Docker Images](cuda-docker-images.md) for more details.

## Driver Installation

### Separate Driver Installation

If you need just the driver without CUDA Toolkit:

```bash
# Ubuntu
sudo apt-get install -y nvidia-driver-560

# Or use nvidia-driver-latest
sudo apt-get install -y nvidia-driver-latest

# RHEL/CentOS
sudo dnf install -y nvidia-driver-latest-dkms
```

### Check Driver Version

```bash
# Check installed driver
nvidia-smi

# Expected output shows driver version
# Driver Version: 560.28.03    CUDA Version: 12.6
```

### Driver Compatibility

CUDA Toolkit version requires minimum driver version:

| CUDA Version | Min Linux Driver | Recommended |
|--------------|------------------|-------------|
| 12.6 | 560.28.03 | 560.x |
| 12.5 | 555.42.02 | 555.x |
| 12.4 | 550.54.14 | 550.x |
| 12.3 | 545.23.06 | 545.x |
| 12.0 | 525.60.13 | 525.x+ |
| 11.8 | 520.61.05 | 520.x+ |

**Note**: Newer drivers are backward compatible with older CUDA versions.

## Verification

### Verify Installation

```bash
# Check CUDA compiler
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.6, V12.6.20

# Check driver and GPU
nvidia-smi

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Compile and run sample
cd /usr/local/cuda-12.6/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# Should show:
# Result = PASS
```

### Test CUDA Program

```cpp
// test.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Compile and run:

```bash
nvcc test.cu -o test
./test

# Should print:
# Hello from GPU thread 0
# Hello from GPU thread 1
# ...
```

## Post-Installation Configuration

### Disable Nouveau Driver

Nouveau (open-source Nvidia driver) conflicts with proprietary driver:

```bash
# Create blacklist file
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"

# Regenerate initramfs
sudo update-initramfs -u

# Reboot
sudo reboot
```

### Persistence Mode

Enable persistence mode for faster GPU initialization:

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Make persistent across reboots
sudo systemctl enable nvidia-persistenced
```

### Power Management

Set optimal power settings:

```bash
# Set to maximum performance mode
sudo nvidia-smi -pl 300  # Set power limit (watts)

# For all GPUs
for i in {0..7}; do
    sudo nvidia-smi -i $i -pm 1
    sudo nvidia-smi -i $i -pl 400
done
```

### Multi-GPU Configuration

```bash
# Check topology
nvidia-smi topo -m

# Example output shows NVLink connections:
#         GPU0    GPU1    GPU2    GPU3
# GPU0     X      NV12    NV12    NV12
# GPU1    NV12     X      NV12    NV12
# GPU2    NV12    NV12     X      NV12
# GPU3    NV12    NV12    NV12     X

# Set compute mode (optional)
sudo nvidia-smi -c DEFAULT  # or EXCLUSIVE_PROCESS
```

## Multiple CUDA Versions

### Installing Multiple Versions

```bash
# Install CUDA 12.6
sudo apt-get install cuda-toolkit-12-6

# Install CUDA 12.0
sudo apt-get install cuda-toolkit-12-0

# Both installed in:
# /usr/local/cuda-12.6
# /usr/local/cuda-12.0
```

### Switching Between Versions

```bash
# Create symbolic link to default version
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.6 /usr/local/cuda

# Or switch to 12.0
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.0 /usr/local/cuda

# Verify
nvcc --version
```

### Environment Modules

For managing multiple versions:

```bash
# Install environment modules
sudo apt-get install environment-modules

# Create module file
sudo mkdir -p /usr/share/modules/modulefiles/cuda
sudo vi /usr/share/modules/modulefiles/cuda/12.6

# Module file content:
#%Module1.0
proc ModulesHelp { } {
    puts stderr "CUDA 12.6"
}
module-whatis "CUDA 12.6"
setenv CUDA_HOME /usr/local/cuda-12.6
prepend-path PATH /usr/local/cuda-12.6/bin
prepend-path LD_LIBRARY_PATH /usr/local/cuda-12.6/lib64

# Load module
module load cuda/12.6
```

## Common Issues and Solutions

### Issue: "No CUDA-capable device detected"

**Cause**: Driver not loaded or GPU not recognized

**Solution:**
```bash
# Check if driver is loaded
lsmod | grep nvidia

# If not, load manually
sudo modprobe nvidia

# Check for errors
dmesg | grep -i nvidia

# Reinstall driver if needed
sudo apt-get install --reinstall nvidia-driver-560
sudo reboot
```

### Issue: "version `GLIBC_X.XX' not found"

**Cause**: Incompatible CUDA version with system libraries

**Solution:**
```bash
# Install compatible CUDA version
sudo apt-get install cuda-toolkit-12-6

# Or update system
sudo apt-get update && sudo apt-get upgrade
```

### Issue: "nvcc: command not found"

**Cause**: CUDA not in PATH

**Solution:**
```bash
# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue: Compilation errors with new GCC

**Cause**: CUDA requires specific GCC versions

**Solution:**
```bash
# CUDA 12.6 supports GCC up to 13.x
# Install compatible GCC version
sudo apt-get install gcc-12 g++-12

# Use with nvcc
nvcc -ccbin gcc-12 program.cu
```

### Issue: "CUDA error: out of memory"

**Cause**: GPU memory exhausted

**Solution:**
```bash
# Check GPU memory
nvidia-smi

# Reset GPU (careful: kills all processes)
sudo nvidia-smi --gpu-reset

# Or reboot
sudo reboot
```

## Uninstallation

### Remove CUDA and Drivers

```bash
# Ubuntu
sudo apt-get --purge remove "*cuda*" "*nvidia*"
sudo apt-get autoremove
sudo rm -rf /usr/local/cuda*

# RHEL
sudo dnf remove "*cuda*" "*nvidia*"
sudo rm -rf /usr/local/cuda*

# Reboot
sudo reboot
```

## Best Practices

1. **Use latest stable driver**: Better performance and bug fixes
2. **Enable persistence mode**: Faster GPU initialization
3. **Install via package manager**: Easier updates
4. **Keep drivers updated**: Important for new GPU support
5. **Use Docker for development**: Isolated environments
6. **Test after installation**: Run `deviceQuery` and samples

## Cloud Providers

### AWS EC2

```bash
# p3/p4 instances come with drivers
# Install CUDA Toolkit only
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

### Google Cloud Platform

```bash
# GCP VMs with GPUs have drivers pre-installed
# Install CUDA Toolkit
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
```

### Azure

```bash
# Use GPU-enabled VM images
# Or install manually
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

## External Resources

- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [CUDA Installation Guide (Official)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [Nvidia Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda)

## Related Guides

- [CUDA Programming Basics](cuda-basics.md)
- [CUDA Docker Images](cuda-docker-images.md)
- [CUDA Profiling](cuda-profiling.md)
- [PyTorch with CUDA](../../layer-4-frameworks/pytorch/pytorch-cuda-basics.md)

