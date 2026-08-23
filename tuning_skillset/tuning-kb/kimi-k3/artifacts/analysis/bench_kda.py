#!/usr/bin/env python3
"""Graph-captured sweep of the KDA packed-decode kernel's launch geometry.

`kda_packed_decode_kernel<8, false>` is 5.93% of arm-C decode GPU time -- 69
calls, one per KDA layer, ~31 us each.  At the frozen shape it moves
B*HV*V*K*4*2 = 99 MB of fp32 state per call, so 31 us is ~3.2 TB/s on a part
whose HBM does ~8, and the launch is the reason to suspect: the block is
`kWarps*32` threads with `threadIdx.x >> 5` warps, i.e. a 32-lane-warp shape,
and at kWarps=8 the grid is only B*HV = 756 blocks of 4 wave64s -> about 3 waves
per SIMD, which is thin for a pure streaming kernel.

kWarps is a template parameter of the JIT module, so a sweep is free: build one
module per value and race them.  Timing is graph-captured
(`tuning-core/graph_captured_benchmarking.md`) and every result asserts
`mode == "cudagraph"`.

    python3 analysis/bench_kda.py                # sweep at the decode shape
    python3 analysis/bench_kda.py --batch 8 32 63 64
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, "/work/tuning_skillset/benchmark")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_harness import cuda_graph_bench  # noqa: E402

from sglang.kernels.jit.utils import (  # noqa: E402
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

# Kimi-K3 linear_attn_config: num_heads 96, head_dim 128; TP=8 -> 12 per rank.
H = HV = 12
K = V = 128
SLOTS = 64  # max_mamba_cache_size from the frozen launch


def build(warps):
    args = make_cpp_args(warps, is_arch_support_pdl())
    return load_jit(
        "kda_packed_decode_" + str(warps),
        *args,
        cuda_files=["attention/kda_packed_decode.cuh"],
        cuda_wrappers=[("run", f"KdaPackedDecodeKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def make(B, dev="cuda"):
    torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(0)
    t = dict(
        mixed_qkv=torch.randn(B, 2 * H * K + HV * V, dtype=torch.bfloat16,
                              device=dev, generator=g),
        a=torch.randn(B, HV * K, dtype=torch.bfloat16, device=dev, generator=g),
        b=torch.randn(B, HV, dtype=torch.bfloat16, device=dev, generator=g),
        A_log=torch.randn(HV, dtype=torch.float32, device=dev, generator=g),
        dt_bias=torch.randn(HV * K, dtype=torch.float32, device=dev, generator=g),
        o=torch.zeros(B, HV, V, dtype=torch.bfloat16, device=dev),
        state0=torch.randn(SLOTS, HV, V, K, dtype=torch.float32, device=dev,
                           generator=g),
        # decode slots are the first B rows of the pool, as the graph pads them
        idx=torch.arange(B, dtype=torch.int32, device=dev),
    )
    return t


def call(mod, t, state):
    mod.run(t["mixed_qkv"], t["a"], t["b"], t["A_log"], t["dt_bias"],
            t["o"], state, t["idx"], 0.08838834764831845, 0.0, False, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warps", type=int, nargs="*", default=[2, 4, 8, 16, 32])
    ap.add_argument("--batch", type=int, nargs="*", default=[63])
    ap.add_argument("--reps", type=int, default=16)
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name}  CU={props.multi_processor_count}  "
          f"H=HV={H} K=V={K} slots={SLOTS}\n")

    mods = {}
    for w in args.warps:
        try:
            mods[w] = build(w)
        except Exception as e:
            print(f"  kWarps={w}: build failed: {type(e).__name__}: {str(e)[:90]}")
    print()

    for B in args.batch:
        t = make(B)
        mb = B * HV * V * K * 4 * 2 / 1e6  # fp32 state read + write
        # reference: shipped kWarps=8, one call from the pristine state
        ref_state = t["state0"].clone()
        call(mods[8], t, ref_state)
        ref_o = t["o"].clone()

        print(f"B={B}  grid=B*HV={B*HV} blocks  state traffic {mb:.1f} MB/call")
        print(f"{'kWarps':>7} {'threads':>8} {'us':>8} {'TB/s':>7} {'vs 8':>7} "
              f"{'max|do|':>9} {'max|dh|':>9}")
        base = None
        for w in sorted(mods):
            state = t["state0"].clone()
            call(mods[w], t, state)          # correctness, one call
            do = (t["o"].float() - ref_o.float()).abs().max().item()
            dh = (state - ref_state).abs().max().item()

            # timing: state mutates in place, so no verify hook -- the guard
            # here is the correctness call above, not the timed replay.
            timing_state = t["state0"].clone()

            def multi(w=w, s=timing_state):
                for _ in range(args.reps):
                    call(mods[w], t, s)

            r = cuda_graph_bench(multi, warmup=10, iters=30)
            assert r["mode"] == "cudagraph", f"kWarps={w}: {r['mode']}"
            us = statistics.median(r["times_ms"]) * 1e3 / args.reps
            if w == 8:
                base = us
            print(f"{w:>7} {w*32:>8} {us:>8.2f} {mb/us/1e3:>7.2f} "
                  f"{(base/us if base else float('nan')):>6.3f}x "
                  f"{do:>9.2e} {dh:>9.2e}")


if __name__ == "__main__":
    main()
