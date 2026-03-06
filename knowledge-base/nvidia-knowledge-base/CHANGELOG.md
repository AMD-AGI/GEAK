Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Nvidia Knowledge Base Changelog

## 2025-11-17 - Focus on Hopper/Blackwell and CUDA 13.x

### Major Changes

#### Removed Content
- ❌ **Deleted Pascal Architecture** (pascal-architecture.md) - Outdated (2016)
- ❌ **Deleted Volta Architecture** (volta-architecture.md) - Outdated (2017)
- ❌ **Deleted Ampere Architecture** (ampere-architecture.md) - Outdated (2020)

#### Added Content
- ✅ **Blackwell Architecture** (blackwell-architecture.md) - Latest gen with FP4, 5th gen Tensor Cores
  - B200/B100 GPUs
  - 192GB HBM3e memory
  - 8 TB/s bandwidth
  - FP4/FP6/FP8 support
  - NVLink 5.0
  - GB200 Superchip

#### Updated Content
- 🔄 **All CUDA versions updated**: 12.x → 13.0+
  - Metadata: `cuda_version: "13.0+"`
  - Docker images: `cuda:13.0`
  - PyTorch wheels: `cu130`
  - Installation guides
  
- 🔄 **GPU Comparison** updated to focus on:
  - Blackwell (B200/B100) vs Hopper (H200/H100)
  - Removed all references to A100, V100, P100
  - Updated use case recommendations for trillion-parameter models
  
- 🔄 **README.md** updated:
  - Focus: CUDA 13.0 with Hopper/Blackwell
  - Total: 28+ focused guides (was 45+)
  
- 🔄 **INDEX.md** updated:
  - Title reflects modern GPUs only
  - Layer 1 now shows only Hopper and Blackwell
  - CUDA 13.x ecosystem emphasis

### Current Focus

**GPU Architectures:**
- Blackwell (2024+): B200, B100, GB200
- Hopper (2022+): H200, H100

**CUDA Version:**
- 13.0+ only
- No legacy support

**Key Features:**
- FP4, FP6, FP8 precision support
- 5th generation Tensor Cores
- Transformer Engine v2
- NVLink 5.0 (1.8 TB/s)
- Up to 192GB HBM3e memory

### File Count
- **Before**: 28+ guides (included all architectures)
- **After**: 28 focused guides (Hopper/Blackwell only)
- **Quality over quantity**: Focused, modern content

### Migration Notes

**For users with older GPUs:**
- A100/A30/A10 (Ampere): Consider legacy documentation or AMD alternatives
- V100 (Volta): No longer supported in this knowledge base
- P100 (Pascal): No longer supported in this knowledge base

**Recommended upgrade path:**
- Current: Any Ampere or older → Upgrade to: H100 or wait for B200
- Focus on latest features: FP8, Transformer Engine, large memory

### Benefits of This Change

1. **Reduced Complexity**: Simpler navigation, less outdated information
2. **Modern Focus**: Latest GPU features and optimizations
3. **CUDA 13.x**: Access to newest CUDA toolkit features
4. **Future-Ready**: Prepared for trillion-parameter models
5. **Cleaner Documentation**: No confusion about which GPU to target

### External Resources

- [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)
- [Hopper Architecture](https://www.nvidia.com/en-us/data-center/h100/)
- [CUDA 13.x Documentation](https://docs.nvidia.com/cuda/)
- [GB200 Superchip](https://www.nvidia.com/en-us/data-center/grace-blackwell-superchip/)

