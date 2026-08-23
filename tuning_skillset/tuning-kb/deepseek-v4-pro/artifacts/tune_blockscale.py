#!/usr/bin/env python3
"""Offline tuner for aiter's non-preshuffled FP8 block-scale GEMM (a8w8_blockscale).

Why this exists: DeepSeek-V4-Pro dispatches three dense GEMM shapes through
`aiter.gemm_a8w8_blockscale` that have no row in any shipped tuned CSV, so they
fall through to `gemm_a8w8_blockscale_ck(...)` with kernelName="" (the CK
heuristic default).  aiter ships no tuner in the pip wheel, so this is a
stand-in: it enumerates the AOT-compiled kernel instances out of the .so,
times each one at each padded-M bucket, checks numerics, and emits a CSV in
aiter's `a8w8_blockscale_tuned_gemm` schema.

Run with the vLLM server DOWN -- it needs the GPU to itself or the timings are
meaningless.

  python3 analysis/tune_blockscale.py --out /tmp/dsv4_blockscale.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import torch

AITER_JIT = "/usr/local/lib/python3.12/dist-packages/aiter/jit"

# Lookup in get_CKGEMM_config tries M, then get_padded_m(gl=0) (round up to a
# 16/32/.. bucket), then get_padded_m(gl=1) (round up to a power of two).  Rows
# at the powers of two therefore cover every possible M.
M_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# (N, K) -> what it is in the model, at TP=8.  Derived from the four shapes that
# log "not found tuned config in a8w8_blockscale_tuned_gemm.csv" on a live
# MTP server, cross-referenced against the layer definitions.
SHAPES = {
    (2048, 7168): "attention.fused_wqa_wkv, [q_lora 1536 | head_dim 512], disable_tp -> replicated; x61 layers",
    (768, 7168): "shared_experts.gate_up_proj, 2*3072/8; x61 layers",
    (7168, 384): "shared_experts.down_proj, 3072/8; x61 layers",
    (7168, 7168): "mtp.e_proj and mtp.h_proj, ReplicatedLinear(hidden, hidden); x2 per draft step",
}


def kernel_names(so_name, prefix):
    out = subprocess.run(
        ["strings", "-n", "20", os.path.join(AITER_JIT, so_name)],
        capture_output=True, text=True, check=True).stdout
    names = sorted({l.strip() for l in out.splitlines() if l.startswith(prefix)})
    return names


def make_inputs(M, N, K, device="cuda"):
    """Match what vLLM's Fp8BlockScaledMMLinearKernel hands aiter.

    A  : [M, K]     float8_e4m3fn   (per-token-group 1x128 quantised activation)
    B  : [N, K]     float8_e4m3fn   (128x128 block quantised weight)
    As : [M, K/128] float32         (QuantFP8 use_ue8m0=False -> fp32 scales)
    Bs : [N/128, K/128] float32
    """
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(M, K, generator=g, device=device, dtype=torch.float32) / 4
    b = torch.randn(N, K, generator=g, device=device, dtype=torch.float32) / 4
    A = a.to(torch.float8_e4m3fn)
    B = b.to(torch.float8_e4m3fn)
    As = torch.rand(M, (K + 127) // 128, generator=g, device=device,
                    dtype=torch.float32) * 0.02 + 0.01
    Bs = torch.rand((N + 127) // 128, (K + 127) // 128, generator=g,
                    device=device, dtype=torch.float32) * 0.02 + 0.01
    return A, B, As, Bs


def reference(A, B, As, Bs):
    M, K = A.shape
    N = B.shape[0]
    af = A.to(torch.float32) * As.repeat_interleave(128, dim=1)[:, :K]
    bf = B.to(torch.float32) * Bs.repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1)[:N, :K]
    return (af @ bf.T).to(torch.bfloat16)


def err_ratio(got, ref):
    got = got.to(torch.float32)
    ref = ref.to(torch.float32)
    denom = ref.abs().mean().clamp_min(1e-6)
    return ((got - ref).abs().mean() / denom).item()


def time_call(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dsv4_blockscale_tuned.csv")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--err-tol", type=float, default=0.02)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--ms", default="")
    args = ap.parse_args()

    from aiter.jit.utils.chip_info import get_gfx, get_cu_num
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale_ck,
        gemm_a8w8_blockscale_cktile,
        compute_gemm_SplitK,
    )

    gfx, cu = get_gfx(), get_cu_num()
    ck = kernel_names("module_gemm_a8w8_blockscale.so", "a8w8_blockscale_1x128x128_")
    tile = kernel_names("module_gemm_a8w8_blockscale_cktile.so",
                        "a8w8_blockscale_cktile_")
    print(f"[tune] gfx={gfx} cu={cu}  ck={len(ck)} cktile={len(tile)}", flush=True)

    shapes = list(SHAPES)
    if args.shapes:
        shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    ms = M_BUCKETS
    if args.ms:
        ms = [int(v) for v in args.ms.split(",")]

    rows = []
    fout = open(args.out, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["gfx", "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK",
                "us", "kernelName", "tflops", "bw", "errRatio"])

    for (N, K) in shapes:
        for M in ms:
            A, B, As, Bs = make_inputs(M, N, K)
            ref = reference(A, B, As, Bs)
            Y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

            cands = []  # (libtype, name, splitK, callable)
            cands.append(("default", "", 0,
                          lambda: gemm_a8w8_blockscale_ck(A, B, As, Bs, Y)))
            # splitK matters more than the tile choice for these shapes: they are
            # all skinny (M<=512 in decode) against a fat K=7168, so without a
            # K-split the grid is a handful of tiles and B gets re-read once per
            # M-tile.  aiter stores splitK as a log2 exponent (see
            # compute_gemm_SplitK), so 0..4 spans 1..16 way.
            for n in ck:
                for sk in (0, 1, 2, 3, 4):
                    cands.append(("ck", n, sk,
                                  (lambda n=n, sk=sk: gemm_a8w8_blockscale_ck(
                                      A, B, As, Bs, Y, splitK=sk, kernelName=n))))
            for n in tile:
                cands.append(("cktile", n, 0,
                              (lambda n=n: gemm_a8w8_blockscale_cktile(
                                  A, B, As, Bs, Y, False, 0, n))))

            # correctness screen first, then time only the survivors
            live = []
            for lib, name, sk, fn in cands:
                try:
                    Y.zero_()
                    fn()
                    torch.cuda.synchronize()
                    er = err_ratio(Y, ref)
                except Exception as ex:
                    continue
                if er != er or er > args.err_tol:
                    continue
                live.append((lib, name, sk, fn, er))

            # Keep each timing loop at a roughly constant amount of work so the
            # small-M buckets are not measured over a handful of microseconds.
            iters = int(max(20, min(args.iters, 5e13 / (2 * M * N * K))))

            # Rule 6b: interleave the candidates across reps rather than running
            # each to completion back-to-back -- gfx950 drifts badly over a long
            # sweep and a back-to-back ordering aliases drift onto kernel choice.
            best = {}
            for rep in range(args.reps):
                for lib, name, sk, fn, er in live:
                    us = time_call(fn, iters, warmup=5 if rep else 20)
                    k = (lib, name, sk)
                    best[k] = min(best.get(k, 1e9), us)

            ranked = sorted(best.items(), key=lambda kv: kv[1])
            if not ranked:
                print(f"[tune] N={N} K={K} M={M}: NO valid kernel", flush=True)
                continue
            (lib, name, sk), us = ranked[0]
            dflt = best.get(("default", "", 0), float("nan"))
            flops = 2 * M * N * K
            byts = M * K + N * K + M * N * 2
            print(f"[tune] N={N:5d} K={K:5d} M={M:5d}  default={dflt:8.2f}us  "
                  f"best={us:8.2f}us ({dflt / us if us else 0:.2f}x)  {lib} {name[:60]}",
                  flush=True)
            for (l2, n2, s2), u2 in ranked[:5]:
                print(f"          {u2:8.2f}us  {l2:8s} {n2}", flush=True)
            if lib == "default":
                # nothing beat the CK heuristic -- emit nothing for this M
                rows.append((N, K, M, None, dflt, dflt))
                continue
            er = [e for l, n, s, f, e in live if (l, n, s) == (lib, name, sk)][0]
            w.writerow([gfx, cu, M, N, K, lib, 0, sk, round(us, 4), name,
                        round(flops / us / 1e6, 2), round(byts / us / 1e3, 2),
                        round(er, 6)])
            fout.flush()
            rows.append((N, K, M, name, dflt, us))
            del A, B, As, Bs, ref, Y
            torch.cuda.empty_cache()

    fout.close()
    print(f"[tune] wrote {args.out}", flush=True)
    tot_d = sum(r[4] for r in rows)
    tot_b = sum(r[5] for r in rows)
    print(f"[tune] summed default {tot_d:.1f}us vs best {tot_b:.1f}us "
          f"({tot_d / tot_b:.3f}x over the swept buckets)")


if __name__ == "__main__":
    main()
