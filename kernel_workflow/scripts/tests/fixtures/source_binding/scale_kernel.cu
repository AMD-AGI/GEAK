// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// A tiny executable fixture: use hipcc on a GPU or c++ -x c++ in CPU-only CI.
#include <cstdlib>
#include <iostream>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

HOST_DEVICE float transform(float x) {
    return x * 2.0f;
}

#ifdef __HIPCC__
__global__ void scale(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = transform(input[i]);
}

void check_hip(hipError_t status) {
    if (status != hipSuccess) {
        std::cerr << hipGetErrorString(status) << '\n';
        std::exit(90);
    }
}
#endif

int main() {
    float input[] = {1.0f, 2.0f, -3.0f, 0.5f};
    float output[4] = {};
#ifdef __HIPCC__
    float *device_input, *device_output;
    check_hip(hipMalloc(&device_input, sizeof(input)));
    check_hip(hipMalloc(&device_output, sizeof(output)));
    check_hip(hipMemcpy(device_input, input, sizeof(input), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(scale, dim3(1), dim3(64), 0, 0, device_input, device_output, 4);
    check_hip(hipGetLastError());
    check_hip(hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    check_hip(hipFree(device_input));
    check_hip(hipFree(device_output));
    const char* backend = "hip";
#else
    for (int i = 0; i < 4; ++i) output[i] = transform(input[i]);
    const char* backend = "host";
#endif
    std::cout << "{\"backend\":\"" << backend << "\",\"output\":[";
    for (int i = 0; i < 4; ++i) {
        if (i) std::cout << ',';
        std::cout << output[i];
    }
    std::cout << "]}\n";
}
