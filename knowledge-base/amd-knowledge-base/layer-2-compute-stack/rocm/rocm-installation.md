Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "2"
category: "rocm"
subcategory: "installation"
tags: ["rocm", "installation", "setup", "ubuntu", "rhel", "compute-stack"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
---

# ROCm Installation Guide

Comprehensive guide for installing ROCm (Radeon Open Compute), AMD's open-source compute stack for GPU computing, machine learning, and HPC workloads.

**This guide covers ROCm 7.0+ only.**

**Latest Version**: ROCm 7.0.2 (October 2024)  
**Official Repository**: [https://github.com/ROCm/ROCm](https://github.com/ROCm/ROCm)  
**Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)

> **Important**: This project requires ROCm 7.0 or later. For older versions, refer to official ROCm documentation.

## System Requirements

### Supported GPUs

**CDNA Architecture** (Data Center):
- **MI300 Series**: MI300X, MI300A (gfx942) - Instinct™ accelerators
- **MI200 Series**: MI250X, MI250, MI210 (gfx90a)
- **MI100 Series**: MI100 (gfx908)

**RDNA Architecture** (Consumer):
- **RDNA 3**: RX 7900 XTX, RX 7900 XT (gfx110X-all) - Limited support
- **RDNA 2**: RX 6900 XT, RX 6800 XT (gfx103X-all) - Limited support

Check the [official GPU support matrix](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html) for complete compatibility information.

### Supported Operating Systems

- **Ubuntu**: 22.04 LTS, 20.04 LTS
- **RHEL/Rocky/AlmaLinux**: 8.x, 9.x
- **SLES**: 15 SP4, SP5
- **CentOS**: 7 (limited support)

### Prerequisites

```bash
# Check kernel version (minimum 5.15+ recommended)
uname -r

# Verify AMD GPU is detected
lspci | grep -i amd

# Check for existing ROCm installation
ls /opt/rocm 2>/dev/null && echo "ROCm already installed" || echo "No ROCm installation found"
```

## Installation Methods

### Method 1: Package Manager (Recommended)

#### Ubuntu 22.04

```bash
# Download and install the installer package
# For ROCm 7.0.2 (check https://rocm.docs.amd.com for latest)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
sudo apt install ./amdgpu-install_*.deb

# Update package list
sudo apt update

# Install ROCm (core components)
sudo amdgpu-install --usecase=rocm

# Or install ROCm with ML/AI libraries
sudo amdgpu-install --usecase=rocm --usecase=ml

# Install specific version
sudo amdgpu-install --usecase=rocm --rocmrelease=7.0.2
```

#### Ubuntu 20.04

```bash
# Download installer for Ubuntu 20.04 (focal)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_6.3.60300-1_all.deb
sudo apt install ./amdgpu-install_*.deb
sudo apt update
sudo amdgpu-install --usecase=rocm
```

#### RHEL/Rocky/AlmaLinux 8.x

```bash
# Add ROCm repository
sudo tee /etc/yum.repos.d/amdgpu.repo <<'EOF'
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/latest/rhel/8.10/main/x86_64
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

sudo tee /etc/yum.repos.d/rocm.repo <<'EOF'
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/el8/latest/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

# Install ROCm
sudo yum install rocm-dev rocm-libs
```

#### RHEL/Rocky/AlmaLinux 9.x

```bash
# For RHEL 9.x
sudo tee /etc/yum.repos.d/rocm.repo <<'EOF'
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/el9/latest/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

sudo yum install rocm-dev rocm-libs
```

### Method 2: Docker (Recommended for Development)

Use official ROCm Docker images for isolated, reproducible environments:

```bash
# Pull latest ROCm 7.x base image (Ubuntu 22.04)
docker pull rocm/dev-ubuntu-22.04:7.1

# Pull latest ROCm 7.x base image (Ubuntu 24.04)
docker pull rocm/dev-ubuntu-24.04:7.1

# Pull PyTorch with ROCm (check Docker Hub for specific ROCm 7.1 tags)
docker pull rocm/pytorch:latest

# Pull TensorFlow with ROCm (check Docker Hub for specific ROCm 7.1 tags)
docker pull rocm/tensorflow:latest

# Run container with GPU access
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    rocm/dev-ubuntu-22.04:7.1

# ⚠️ Always check Docker Hub for the latest available tags:
# https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
# https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags
# https://hub.docker.com/r/rocm/pytorch/tags
```

**Custom Dockerfile Example**:

```dockerfile
# Dockerfile for ROCm 7.1 development
FROM rocm/dev-ubuntu-22.04:7.1

# Install additional ROCm libraries
RUN apt-get update && apt-get install -y \
    rocblas \
    rocfft \
    rocsolver \
    rocsparse \
    rocrand \
    miopen-hip \
    rccl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV HIP_VISIBLE_DEVICES=0

WORKDIR /workspace
CMD ["/bin/bash"]
```

```bash
# Build and run
docker build -t my-rocm-dev:7.1 .
docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    -v $(pwd):/workspace \
    my-rocm-dev:7.1
```

> **📖 For complete Docker image reference**, see: [ROCm Docker Images Guide](./rocm-docker-images.md)

### Method 3: Build from Source (Advanced)

For developers who need custom builds or latest development features:

```bash
# Use TheRock build system
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock

# Setup ccache for faster rebuilds (optional)
eval "$(./build_tools/setup_ccache.py)"

# Configure for your GPU
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

# Build
cmake --build build

# Install (optional)
sudo cmake --install build --prefix /opt/rocm
```

See [TheRock documentation](../therock/therock-overview.md) for detailed build instructions.

## Post-Installation Setup

### 1. Add User to Groups

```bash
sudo usermod -a -G render,video $LOGNAME
# Logout and login for changes to take effect
```

### 2. Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# ROCm paths
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.0  # If needed for unsupported GPUs

# HIP configuration
export HIP_PLATFORM=amd
export HIP_VISIBLE_DEVICES=0  # GPU selection
```

### 3. Verify Installation

```bash
# Check ROCm installation
rocminfo

# Check HIP
hipconfig

# Check available GPUs
rocm-smi

# Run simple test
/opt/rocm/bin/rocminfo | grep "Name:"
```

## Remote Development Setup

### SSH with X11 Forwarding

```bash
# On remote server
sudo apt install xauth

# SSH connection
ssh -X user@remote-gpu-server

# Verify display
echo $DISPLAY
```

### VS Code Remote Development

```json
// .vscode/settings.json
{
    "remote.SSH.defaultExtensions": [
        "ms-python.python",
        "ms-vscode.cpptools"
    ],
    "remote.SSH.remotePlatform": {
        "gpu-server": "linux"
    },
    "terminal.integrated.env.linux": {
        "PATH": "/opt/rocm/bin:${env:PATH}",
        "LD_LIBRARY_PATH": "/opt/rocm/lib:${env:LD_LIBRARY_PATH}"
    }
}
```

## Troubleshooting

### Issue: GPU not detected

```bash
# Check kernel driver
lsmod | grep amdgpu

# Reload driver if needed
sudo modprobe -r amdgpu
sudo modprobe amdgpu

# Check dmesg for errors
sudo dmesg | grep amdgpu
```

### Issue: Permission denied

```bash
# Check device permissions
ls -l /dev/kfd /dev/dri/render*

# Verify group membership
groups $USER

# Should see: render, video
```

### Issue: HIP not found

```bash
# Verify HIP installation
dpkg -l | grep hip

# Reinstall if needed
sudo apt install hip-runtime-amd hip-dev
```

## Version-Specific Notes (ROCm 7.0+ Only)

> **Note**: This project targets ROCm 7.0+ exclusively. Earlier versions are not covered.

### ROCm 7.0.2 (Latest Stable - October 2024) ⭐ **Recommended**
- **Status**: Latest stable release
- Bug fixes and stability improvements
- Enhanced vLLM 0.4.x+ support
- Improved PyTorch 2.4+ performance
- Better multi-GPU scaling with RCCL
- Production-ready
- **Use for**: All new deployments and production workloads

### ROCm 7.0.0 (Major Release)
- **Status**: Major release with significant updates
- Significant performance improvements over ROCm 6.x
- Flash Attention 2 support
- Enhanced ML framework integration
- Improved Windows support via TheRock
- Better vLLM integration
- **Use for**: Stable base if 7.0.2 has issues

### Future ROCm 7.x Releases
- Follow [ROCm Releases](https://github.com/ROCm/ROCm/releases) for upcoming 7.x versions
- This project will update to support new 7.x releases as they become available

**Version Selection Guide for ROCm 7.0+**:
- **✅ Production**: ROCm 7.0.2 (latest stable)
- **✅ Development**: ROCm 7.0.2 or build from TheRock
- **✅ Bleeding edge**: Build from TheRock develop branch
- **❌ ROCm < 7.0**: Not supported by this project

Check [ROCm Releases](https://github.com/ROCm/ROCm/releases) for complete changelog and release notes.

## Next Steps

After installation:

1. Install ML frameworks (PyTorch, TensorFlow)
2. Set up development environment
3. Run benchmark tests
4. Configure for your specific workload

## References

### Official Resources

- **ROCm GitHub**: [https://github.com/ROCm/ROCm](https://github.com/ROCm/ROCm)
- **ROCm Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)
- **TheRock Build System**: [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock)
- **ROCm Libraries**: [https://github.com/ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)

### Installation Documentation

- **Linux Installation Guide**: [https://rocm.docs.amd.com/projects/install-on-linux](https://rocm.docs.amd.com/projects/install-on-linux)
- **Windows Installation Guide**: [https://rocm.docs.amd.com/projects/install-on-windows](https://rocm.docs.amd.com/projects/install-on-windows)
- **GPU Support Matrix**: [https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
- **Docker Guide**: [https://rocm.docs.amd.com/en/latest/deploy/docker.html](https://rocm.docs.amd.com/en/latest/deploy/docker.html)

### Release Information

- **Release Notes**: [https://github.com/ROCm/ROCm/releases](https://github.com/ROCm/ROCm/releases)
- **Changelog**: [https://github.com/ROCm/ROCm/blob/develop/CHANGELOG.md](https://github.com/ROCm/ROCm/blob/develop/CHANGELOG.md)
- **Known Issues**: Check release-specific documentation

### Community Support

- **GitHub Discussions**: [https://github.com/ROCm/ROCm/discussions](https://github.com/ROCm/ROCm/discussions)
- **GitHub Issues**: [https://github.com/ROCm/ROCm/issues](https://github.com/ROCm/ROCm/issues)
- **AMD Community**: [https://community.amd.com](https://community.amd.com)

