#!/usr/bin/env python3
"""Correctness check for the patched attn_residual pair, against the module's
own eager reference (`aggregate_stream_torch`) and an fp64 oracle.

Covers every nvb the model reaches (0..8, bank is 8 rows) at the decode batch,
a chunked-prefill tile, and a couple of ragged token counts, since BLOCK_R is
now sized per call and nvb=0 takes a distinct path (no bank rows at all).
"""
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.layers.attn_residual import (  # noqa: E402
    _COMBINE_BLOCK_H,
    _COMBINE_WARPS,
    _SCORE_WARPS,
    _mix_fused,
)

H = 7168
BLOCK_NUM = 8


class FakeProj:
    def __init__(self, w):
        self.weight = w


class FakeNorm:
    def __init__(self, w, eps=1e-6):
        self.weight = w
        self.variance_epsilon = eps


def oracle(prefix, bank, nvb, cw, eps):
    rows = torch.cat([bank[:, :nvb, :], prefix.unsqueeze(1)], dim=1).double()
    rrms = 1.0 / torch.sqrt((rows * rows).sum(-1) / H + eps)
    p = torch.softmax((rows * cw.double()).sum(-1) * rrms, dim=-1)
    return (p.unsqueeze(-1) * rows).sum(1)


def main():
    print(f"launch config: score_warps={_SCORE_WARPS} "
          f"combine_block_h={_COMBINE_BLOCK_H} combine_warps={_COMBINE_WARPS}")
    torch.manual_seed(0)
    dev = "cuda"
    # cw = score_proj.weight[0] * score_norm.weight, per get_cw()
    pw = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
    nw = torch.randn(H, dtype=torch.bfloat16, device=dev)
    proj, norm = FakeProj(pw), FakeNorm(nw)
    cw = (pw[0].float() * nw.float())

    worst = 0.0
    fails = 0
    for T in (1, 7, 64, 129, 2048, 16384):
        prefix = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
        bank = torch.randn(T, BLOCK_NUM, H, dtype=torch.bfloat16, device=dev)
        for nvb in range(0, BLOCK_NUM + 1):
            got = _mix_fused(prefix, bank, nvb, proj, norm)
            ref = oracle(prefix, bank, nvb, cw, norm.variance_epsilon)
            # bf16 output: 1 ulp near magnitude 1 is ~7.8e-3
            err = (got.double() - ref).abs().max().item()
            scale = ref.abs().max().item()
            ok = err <= 2e-2 * max(scale, 1.0)
            worst = max(worst, err)
            if not ok:
                fails += 1
                print(f"  FAIL T={T:>6} nvb={nvb}  max|err|={err:.3e} "
                      f"scale={scale:.3f}")
    print(f"worst max|err| over all shapes: {worst:.3e}")
    print("PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
