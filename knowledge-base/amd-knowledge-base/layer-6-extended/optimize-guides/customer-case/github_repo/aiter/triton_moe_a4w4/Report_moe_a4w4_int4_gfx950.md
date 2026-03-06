Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: moe_gemm_a4w4

## Variant Context
- Input semantic type: Mixture of Experts (MoE) GEMM
- Datatype(s): INT4/FP4 activation and weight
- Data representation: 4-bit quantized with DeepSeek FP4 format
- Target architecture: gfx950 (MI350) - skipped on MI300 due to hardware limitations

## Functionality
This kernel implements INT4/FP4 GEMM for Mixture of Experts layers. It handles:
1. 4-bit quantized matrix multiplication
2. Expert-parallel computation
3. Token routing and gathering
4. DeepSeek-style FP4 quantization format

## Optimization 1: DeepSeek FP4 Quantization Integration
- Commit ID: 9eecdecb0
- Optimization type: Precision / Memory
- Summary: Integrated DeepSeek FP4 quantization format for activation quantization.

- Detailed explanation:
  The kernel uses DeepSeek's FP4 quantization format which provides:
  1. Better accuracy than naive INT4 for neural network weights
  2. Efficient packing (2 values per byte)
  3. Compatible with DeepSeek R1 model shapes

- Code excerpt:
    ```python
    # refactor activation quant to use deepseek fp4 quant
    # tune testcase for a4w4 based on deepseek r1 shapes
    ```

- Evidence mapping:
  - DeepSeek FP4 → Commit message mentions "deepseek fp4 quant"
  - DeepSeek R1 shapes → Tuned for specific model configurations

## Optimization 2: Layer-Specific Profiling Support
- Commit ID: 9eecdecb0
- Optimization type: Debugging / Profiling
- Summary: Added layer1/layer2 suffix for easier performance profiling.

- Detailed explanation:
  Similar to the A8W8 blockscale kernel, this optimization adds suffixes to distinguish between the two GEMM operations in MoE layers for profiling purposes.

- Code excerpt:
    ```python
    # Add layer1/layer2 suffix for easier profiling
    ```

- Evidence mapping:
  - Layer naming → `_layer1` and `_layer2` suffixes
  - Profiling support → Easier identification in profiling tools

## Optimization 3: Architecture-Specific Execution
- Commit ID: 9eecdecb0
- Optimization type: Architecture
- Summary: Kernel is optimized for gfx950 (MI350) and skipped on MI300 due to hardware limitations.

- Detailed explanation:
  The A4W4 kernel requires specific hardware support for 4-bit operations that is available on gfx950 (MI350) but not on gfx942 (MI300). The implementation includes:
  1. Architecture detection
  2. Graceful fallback or skip on unsupported hardware

- Code excerpt:
    ```python
    # skip a4w4 unit tests on MI300
    ```

- Evidence mapping:
  - Architecture check → Tests skipped on MI300
  - gfx950 optimization → Designed for CDNA4 architecture
