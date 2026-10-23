# Torch fp32 reference for LayerNorm — the author-HIP oracle's truth source + denominator.
# Receives the in_tensors in declared registry order; returns the single output.
#   y = (x - mean) * rsqrt(var + eps) * gamma + beta
import torch

EPS = 1.0e-6


def layernorm(args):
    x, gamma, beta = args                # d_x [M,N], d_gamma [N], d_beta [N]
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + EPS)
    return (y * gamma.float() + beta.float()).to(x.dtype)
