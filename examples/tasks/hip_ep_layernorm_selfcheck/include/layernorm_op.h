// FROZEN ABI — the harness calls this by name/signature. Do NOT change it.
// The kernel body behind it (in kernel_src/layernorm_kernel.hip) is the
// editable optimization surface.
#pragma once

// Row-wise LayerNorm over device pointers:
//   y[r,i] = (x[r,i] - mean_r) * rsqrt(var_r + eps) * gamma[i] + beta[i]
// x,y : [M, N] row-major fp32 device buffers
// gamma,beta : [N] fp32 device buffers
// Launches on the default stream; caller synchronizes.
extern "C" void layernorm_launch(const float* d_x, const float* d_gamma,
                                 const float* d_beta, float* d_y, int M, int N,
                                 float eps);
