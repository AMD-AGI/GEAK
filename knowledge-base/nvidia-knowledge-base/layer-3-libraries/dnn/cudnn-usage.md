Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "3"
category: "dnn"
tags: ["cudnn", "deep-learning", "neural-networks", "convolution"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# cuDNN Usage Guide

*GPU-accelerated library for deep neural networks*

## Overview

cuDNN (CUDA Deep Neural Network library) provides highly optimized implementations of deep learning primitives: convolutions, pooling, normalization, activation functions, and more.

**Official Documentation**: [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)

## Installation

```bash
# Check installation
ls /usr/local/cuda/lib64/libcudnn*

# Check version
cat /usr/local/cuda/include/cudnn_version.h | grep "CUDNN_MAJOR\|CUDNN_MINOR\|CUDNN_PATCHLEVEL"
```

Download from [Nvidia Developer](https://developer.nvidia.com/cudnn) if not installed.

## Basic Usage

### Initialization

```cpp
#include <cudnn.h>

cudnnHandle_t cudnn;
cudnnCreate(&cudnn);

// Perform operations...

cudnnDestroy(cudnn);
```

## Common Operations

### Convolution

```cpp
// Create tensor descriptors
cudnnTensorDescriptor_t input_desc, output_desc;
cudnnFilterDescriptor_t filter_desc;
cudnnConvolutionDescriptor_t conv_desc;

cudnnCreateTensorDescriptor(&input_desc);
cudnnCreateTensorDescriptor(&output_desc);
cudnnCreateFilterDescriptor(&filter_desc);
cudnnCreateConvolutionDescriptor(&conv_desc);

// Set input: NCHW format (batch, channels, height, width)
cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                           batch, in_channels, height, width);

// Set filter: NCHW format
cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                          out_channels, in_channels, kernel_h, kernel_w);

// Set convolution parameters
cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                dilation_h, dilation_w,
                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

// Get output dimensions
int out_n, out_c, out_h, out_w;
cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                      &out_n, &out_c, &out_h, &out_w);

// Set output descriptor
cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                           out_n, out_c, out_h, out_w);

// Find best algorithm
cudnnConvolutionFwdAlgoPerf_t algo_perf;
int returned_algo_count;
cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc,
                                      output_desc, 1, &returned_algo_count, &algo_perf);

// Get workspace size
size_t workspace_size;
cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc,
                                        output_desc, algo_perf.algo, &workspace_size);

void *workspace;
cudaMalloc(&workspace, workspace_size);

// Perform convolution
float alpha = 1.0f, beta = 0.0f;
cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input,
                        filter_desc, d_filter, conv_desc, algo_perf.algo,
                        workspace, workspace_size,
                        &beta, output_desc, d_output);
```

### Activation Functions

```cpp
cudnnActivationDescriptor_t activation_desc;
cudnnCreateActivationDescriptor(&activation_desc);

// ReLU
cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU,
                              CUDNN_PROPAGATE_NAN, 0.0);

// Apply activation
cudnnActivationForward(cudnn, activation_desc, &alpha, input_desc, d_input,
                       &beta, output_desc, d_output);
```

### Pooling

```cpp
cudnnPoolingDescriptor_t pooling_desc;
cudnnCreatePoolingDescriptor(&pooling_desc);

// Max pooling 2x2
cudnnSetPooling2dDescriptor(pooling_desc, CUDNN_POOLING_MAX,
                            CUDNN_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);

// Perform pooling
cudnnPoolingForward(cudnn, pooling_desc, &alpha, input_desc, d_input,
                    &beta, output_desc, d_output);
```

### Batch Normalization

```cpp
cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;

// Forward
cudnnBatchNormalizationForwardTraining(
    cudnn, mode, &alpha, &beta,
    input_desc, d_input, output_desc, d_output,
    bn_param_desc, d_scale, d_bias,
    exponential_average_factor,
    d_running_mean, d_running_variance,
    epsilon, d_save_mean, d_save_inv_variance);
```

## Tensor Cores

Enable Tensor Core acceleration:

```cpp
// Enable Tensor Cores for convolutions
cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

// Allow TF32 on Ampere+ (default)
cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
```

## Complete Example

```cpp
#include <cudnn.h>
#include <cuda_runtime.h>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    // Input: 1 image, 3 channels, 32x32
    int batch = 1, in_c = 3, h = 32, w = 32;
    // Filter: 64 filters, 3 input channels, 3x3
    int out_c = 64, k_h = 3, k_w = 3;
    int pad = 1, stride = 1;
    
    // Create descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    
    // Set descriptors
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, in_c, h, w);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                              out_c, in_c, k_h, k_w);
    cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, 1, 1,
                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // Get output dimensions
    int out_n, out_c_actual, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                         &out_n, &out_c_actual, &out_h, &out_w);
    
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               out_n, out_c_actual, out_h, out_w);
    
    // Allocate memory
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, batch * in_c * h * w * sizeof(float));
    cudaMalloc(&d_filter, out_c * in_c * k_h * k_w * sizeof(float));
    cudaMalloc(&d_output, out_n * out_c_actual * out_h * out_w * sizeof(float));
    
    // Find algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_count;
    cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc,
                                        conv_desc, output_desc, 1,
                                        &returned_count, &algo_perf);
    
    // Allocate workspace
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc,
                                           conv_desc, output_desc,
                                           algo_perf.algo, &workspace_size);
    void *workspace;
    cudaMalloc(&workspace, workspace_size);
    
    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input,
                           filter_desc, d_filter, conv_desc, algo_perf.algo,
                           workspace, workspace_size,
                           &beta, output_desc, d_output);
    
    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(workspace);
    
    return 0;
}
```

Compile:
```bash
nvcc -lcudnn conv_example.cu -o conv_example
```

## Python Usage (PyTorch/TensorFlow)

cuDNN is automatically used by deep learning frameworks:

```python
import torch
import torch.nn as nn

# cuDNN automatically used for Conv2d
model = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
input = torch.randn(1, 3, 32, 32).cuda()

# Uses cuDNN internally
output = model(input)

# Enable cuDNN benchmarking for performance
torch.backends.cudnn.benchmark = True

# Disable if you need deterministic results
torch.backends.cudnn.deterministic = True
```

## Performance Optimization

### Algorithm Selection

```cpp
// Get all algorithms
int returned_count;
cudnnConvolutionFwdAlgoPerf_t algo_perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc,
                                     conv_desc, output_desc,
                                     CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                     &returned_count, algo_perf);

// Select fastest
cudnnConvolutionFwdAlgo_t best_algo = algo_perf[0].algo;
```

### Tensor Core Usage

```cpp
// Enable for maximum performance on Volta+
cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
```

### Workspace Management

```cpp
// Limit workspace size if needed
size_t workspace_limit = 256 * 1024 * 1024;  // 256 MB
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &max_count);

// Find algorithm with workspace limit
cudnnGetConvolutionForwardAlgorithm_v7(
    cudnn, input_desc, filter_desc, conv_desc, output_desc,
    max_count, &returned_count, algo_perf);

// Pick algorithm within workspace limit
for (int i = 0; i < returned_count; i++) {
    if (algo_perf[i].memory <= workspace_limit) {
        best_algo = algo_perf[i].algo;
        break;
    }
}
```

## Best Practices

1. **Enable Tensor Cores**: Use `CUDNN_TENSOR_OP_MATH`
2. **Benchmark algorithms**: Use `cudnnFindConvolutionForwardAlgorithm`
3. **Reuse descriptors**: Create once, use many times
4. **Async execution**: Use CUDA streams
5. **FP16 for training**: Use mixed precision on Volta+

## External Resources

- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [cuDNN API Reference](https://docs.nvidia.com/deeplearning/cudnn/api/)

## Related Guides

- [cuBLAS Usage](../blas/cublas-usage.md)
- [PyTorch with CUDA](../../layer-4-frameworks/pytorch/pytorch-cuda-basics.md)
- [Tensor Core Programming](../../layer-5-llm/05-advanced/custom-kernels/cuda-kernels.md)

