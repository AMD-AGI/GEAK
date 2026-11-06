// FROZEN ABI — the harness calls this by name/signature. Do NOT change it.
// The kernel body behind it (in kernel_src/rmsnorm_kernel.hip) is the
// editable optimization surface.
#pragma once

// Row-wise RMSNorm over device pointers (no mean subtraction, no bias):
//   y[r,i] = x[r,i] * rsqrt(mean_i(x[r,i]^2) + eps) * gamma[i]
// x,y : [M, N] row-major fp32 device buffers
// gamma : [N] fp32 device buffer
// Launches on the default stream; caller synchronizes.
extern "C" void rmsnorm_launch(const float* d_x, const float* d_gamma,
                               float* d_y, int M, int N, float eps);
