# Torch fp32 reference for RMSNorm — the author-HIP oracle's truth source + denominator.
# Receives the in_tensors in declared registry order; returns the single output.
#   y = x * rsqrt(mean(x^2, dim=-1) + eps) * gamma   (no mean subtraction, no bias)
import torch

EPS = 1.0e-6


def rmsnorm(args):
    x, gamma = args                      # d_x [M,N], d_gamma [N]
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + EPS) * gamma.float()).to(x.dtype)
