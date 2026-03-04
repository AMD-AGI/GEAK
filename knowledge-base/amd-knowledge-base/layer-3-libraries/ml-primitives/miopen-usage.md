---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "3"
category: "ml-primitives"
subcategory: "deep-learning"
tags: ["miopen", "convolution", "deep-learning", "primitives"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# MIOpen Usage Guide

MIOpen is AMD's library for high-performance deep learning primitives, providing optimized implementations of convolutions, pooling, normalization, and activation functions.

## Installation

```bash
sudo apt install miopen-hip miopen-hip-dev
```

## Convolution Example

```cpp
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>

void convolution_example() {
    miopenHandle_t handle;
    miopenCreate(&handle);
    
    // Input: N=1, C=3, H=224, W=224
    // Filter: K=64, C=3, H=3, W=3
    
    // Create tensor descriptors
    miopenTensorDescriptor_t inputDesc, outputDesc;
    miopenCreateTensorDescriptor(&inputDesc);
    miopenCreateTensorDescriptor(&outputDesc);
    
    // Set input format: NCHW
    int input_dims[] = {1, 3, 224, 224};
    miopenSet4dTensorDescriptor(inputDesc, miopenFloat, 
                                input_dims[0], input_dims[1],
                                input_dims[2], input_dims[3]);
    
    // Create convolution descriptor
    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);
    miopenInitConvolutionDescriptor(convDesc,
                                    miopenConvolution,  // mode
                                    0, 0,              // pad_h, pad_w
                                    1, 1,              // stride_h, stride_w
                                    1, 1);             // dilation_h, dilation_w
    
    // Filter descriptor
    miopenTensorDescriptor_t filterDesc;
    miopenCreateTensorDescriptor(&filterDesc);
    int filter_dims[] = {64, 3, 3, 3};  // K, C, H, W
    miopenSet4dTensorDescriptor(filterDesc, miopenFloat,
                                filter_dims[0], filter_dims[1],
                                filter_dims[2], filter_dims[3]);
    
    // Get output dimensions
    int n, c, h, w;
    miopenGetConvolutionForwardOutputDim(convDesc, inputDesc, filterDesc,
                                        &n, &c, &h, &w);
    
    miopenSet4dTensorDescriptor(outputDesc, miopenFloat, n, c, h, w);
    
    // Find best algorithm
    miopenConvAlgoPerf_t perf;
    int returnedAlgoCount;
    size_t workspace_size = 0;
    
    miopenFindConvolutionForwardAlgorithm(
        handle, inputDesc, d_input,
        filterDesc, d_filter,
        convDesc, outputDesc, d_output,
        1, &returnedAlgoCount, &perf,
        nullptr, workspace_size, false);
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (perf.memory > 0) {
        hipMalloc(&workspace, perf.memory);
    }
    
    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    miopenConvolutionForward(handle, &alpha,
                            inputDesc, d_input,
                            filterDesc, d_filter,
                            convDesc, perf.fwd_algo,
                            &beta, outputDesc, d_output,
                            workspace, perf.memory);
    
    // Cleanup
    if (workspace) hipFree(workspace);
    miopenDestroyTensorDescriptor(inputDesc);
    miopenDestroyTensorDescriptor(outputDesc);
    miopenDestroyTensorDescriptor(filterDesc);
    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroy(handle);
}
```

## PyTorch Integration

MIOpen is automatically used by PyTorch on AMD GPUs:

```python
import torch
import torch.nn as nn

# Convolution uses MIOpen
model = nn.Conv2d(3, 64, kernel_size=3, padding=1).to('cuda')
input = torch.randn(1, 3, 224, 224, device='cuda')

# Uses MIOpen convolution primitives
output = model(input)

# Batch normalization uses MIOpen
bn = nn.BatchNorm2d(64).to('cuda')
normalized = bn(output)

# Pooling uses MIOpen
pool = nn.MaxPool2d(2, 2).to('cuda')
pooled = pool(normalized)
```

## Performance Tips

1. **Use FP16/BF16 for training**
   ```python
   model = model.half()  # FP16
   ```

2. **Enable TensorCore-like ops**
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

3. **Tune workspace size**
   ```bash
   export MIOPEN_USER_DB_PATH=/tmp/miopen_cache
   ```

## References

- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [MIOpen GitHub](https://github.com/ROCmSoftwarePlatform/MIOpen)
