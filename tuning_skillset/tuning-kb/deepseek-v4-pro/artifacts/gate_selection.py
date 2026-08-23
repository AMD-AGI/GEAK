#!/usr/bin/env python3
"""`tuning-aiter` §2b gate: engagement is not selection.

A row in the CSV that matches the shape only proves the *lookup* hits.  It does
not prove the tuned kernel is what the production entry point runs -- the
dispatch can silently drop `kernelName`, or the row can be shadowed by a
duplicate shape with a lower `us` after the merge.  The only proof is to call
the same entry point vLLM calls and see the tuned time come back.

  python3 analysis/gate_selection.py --csv /tmp/dsv4_blockscale_tuned.csv

Prints, per (N,K,M): the CSV's tuned `us`, the time of the *production* call
`aiter.gemm_a8w8_blockscale(...)`, and the untuned `gemm_a8w8_blockscale_ck`
default for scale.  Run with the server down.
"""

import argparse
import csv
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tune_blockscale import make_inputs, reference, err_ratio, time_call  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/dsv4_blockscale_tuned.csv")
    ap.add_argument("--probe", default="2048x7168x128,7168x7168x256,7168x384x2048,768x7168x4096")
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    import aiter
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck, get_CKGEMM_config
    from aiter.jit.core import AITER_CONFIGS

    want = {}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            want[(int(r["N"]), int(r["K"]), int(r["M"]))] = (
                float(r["us"]), r["libtype"], int(r["splitK"]), r["kernelName"])

    print(f"{'N':>6} {'K':>6} {'M':>6} {'csv_us':>9} {'prod_us':>9} {'dflt_us':>9} "
          f"{'ratio':>7}  verdict")
    ok = True
    for spec in args.probe.split(","):
        N, K, M = (int(v) for v in spec.split("x"))
        A, B, As, Bs = make_inputs(M, N, K)
        ref = reference(A, B, As, Bs)
        Y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

        cfg = get_CKGEMM_config(M, N, K,
                                AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
        prod = lambda: aiter.gemm_a8w8_blockscale(A, B, As, Bs)  # noqa: E731
        out = prod()
        er = err_ratio(out, ref)
        t_prod = time_call(prod, args.iters, warmup=20)
        t_dflt = time_call(lambda: gemm_a8w8_blockscale_ck(A, B, As, Bs, Y),
                           args.iters, warmup=20)
        csv_us, lib, sk, name = want.get((N, K, M), (float("nan"), "-", -1, "-"))
        ratio = t_prod / csv_us if csv_us == csv_us else float("nan")
        good = ratio < 1.25
        ok &= good
        print(f"{N:6d} {K:6d} {M:6d} {csv_us:9.2f} {t_prod:9.2f} {t_dflt:9.2f} "
              f"{ratio:7.2f}  {'SELECTED' if good else 'NOT SELECTED'}  err={er:.2e}")
        print(f"       lookup -> {cfg}")
        print(f"       csv    -> {lib} splitK={sk} {name}")
        del A, B, As, Bs, ref, Y
        torch.cuda.empty_cache()

    print("\nGATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
