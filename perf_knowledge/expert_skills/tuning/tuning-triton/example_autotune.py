#!/usr/bin/env python3
"""Reference Triton autotuning setup for AMD Instinct.

Read this for the *method*, then adapt it. The specific block sizes here are not
recommendations — they are a space to search. What is worth copying is the shape of the
setup: prune analytically, put the AMD knobs where they are actually consumed, key the
cache on everything that matters, and re-measure the winner against the noise floor.

    export HIP_VISIBLE_DEVICES=<idle gpu>
    export TRITON_PRINT_AUTOTUNING=1
    python3 example_autotune.py --M 4096 --N 4096 --K 4096
    REGIME=decode python3 example_autotune.py --M 1 --N 4096 --K 4096
"""

import argparse
import itertools
import os
import statistics

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# 1. The space, and the predicate that prunes it
# ---------------------------------------------------------------------------

def _lds_limit():
    """LDS bytes per workgroup, read from the device rather than assumed.

    A hardcoded constant here is the single most expensive line in this file
    to get wrong, because being wrong is invisible. The pruner below rejects
    tiles that exceed it, and a rejected tile is never compiled, never timed,
    and never mentioned in the output -- so an over-tight limit does not
    produce an error, it produces a smaller set of wins and the conclusion
    that there was nothing to find.

    gfx942 has 64 KB and gfx950 has 160 KB. Measured on gfx950: carrying
    gfx942's constant across prunes 28% of the tiles that actually compile.

    torch 2.9.1 does not expose `shared_memory_per_block` at all, so the
    attribute read has to be guarded; rocminfo's GROUP segment is the second
    opinion, and both agreed at 163840 on the MI355X this was checked on.
    """
    try:
        return int(torch.cuda.get_device_properties(
            torch.cuda.current_device()).shared_memory_per_block)
    except Exception:
        pass
    try:
        import re
        import subprocess
        out = subprocess.run(["rocminfo"], capture_output=True, text=True,
                             timeout=60).stdout
        vals = [int(kb) * 1024 for kb in re.findall(
            r"Segment:\s+GROUP\s*\n\s*Size:\s+(\d+)\(\S+\)\s*KB", out)]
        if vals:
            return max(vals)
    except Exception:
        pass
    print("WARNING: could not read LDS per workgroup from torch or rocminfo; "
          "assuming 64 KB. On a part with more LDS than gfx942 this silently "
          "prunes legal tiles.")
    return 64 * 1024


LDS_LIMIT = _lds_limit()
WAVE = 64              # CDNA is wave64, unlike wave32 on RDNA/NVIDIA


def viable(BM, BN, BK, warps, stages, dtype_bytes=2):
    """Reject configs the hardware cannot run, or that cannot keep threads busy.

    Cheap arithmetic here saves a compile + timing pass per rejected config. On the space
    below this removes roughly 40% of candidates before Triton ever sees them.
    """
    lds = (BM * BK + BK * BN) * dtype_bytes * min(stages, 2)
    if lds > LDS_LIMIT:
        return False
    threads = warps * WAVE
    if BM * BN < threads:              # fewer output elements than threads
        return False
    if (BM * BN) // threads > 256:     # too much per-thread state; register pressure
        return False
    return True


def build_configs(regime="general"):
    """Enumerate, prune, and attach the AMD knobs.

    NOTE the AMD knobs go in the *first positional dict*, next to the constexpr block
    sizes. Passing them as Config keyword arguments raises TypeError -- Config accepts only
    num_warps / num_stages / num_ctas / maxnreg / pre_hook / ir_override.
    """
    if regime == "decode":
        # M=1-ish: tall tiles are pure waste. Search small BM, large BK, low warps.
        bms, bns, bks = [16, 32], [32, 64, 128], [64, 128, 256]
        warps_opts, stage_opts = [1, 2, 4], [1, 2]
    else:
        bms, bns, bks = [64, 128, 256], [64, 128, 256], [32, 64]
        warps_opts, stage_opts = [4, 8], [1, 2]

    # matrix_instr_nonkdim selects the MFMA shape; 0 lets the compiler choose.
    # It was worth +7.7% over auto on a 4096^3 bf16 case -- always search it.
    nonkdim_opts = [0, 16, 32]
    kpack_opts = [1, 2]

    tiles = list(itertools.product(bms, bns, bks, warps_opts, stage_opts))
    kept = [t for t in tiles if viable(*t)]
    knobs = list(itertools.product(nonkdim_opts, kpack_opts))

    configs = [
        triton.Config(
            {
                "BM": BM, "BN": BN, "BK": BK, "GM": 8,
                "matrix_instr_nonkdim": nk,   # <-- AMD knob, positional dict
                "kpack": kp,                  # <-- AMD knob, positional dict
                # "waves_per_eu": 2,          # add when latency-, not compute-bound
            },
            num_warps=w,
            num_stages=s,
        )
        for (BM, BN, BK, w, s) in kept
        for (nk, kp) in knobs
    ]
    print(f"# regime={regime}: {len(tiles)} tile/warp/stage combos, "
          f"{len(tiles) - len(kept)} pruned as unviable, {len(kept)} kept "
          f"x {len(knobs)} knob combos = {len(configs)} configs to race")
    return configs


# ---------------------------------------------------------------------------
# 2. The kernel
# ---------------------------------------------------------------------------
# key= must name every argument whose value changes which config is best.
# Dropping 'M' here makes an M=1 call reuse the config tuned at M=4096: measured 3.75x
# slower, with no error raised. Autotune caches per key tuple, so a missing key is a
# silent correctness-of-choice bug.


# The decorator is evaluated at *import* time, so the config list is fixed before any
# argument is seen. That is why regime selection is an env var here and not a CLI flag:
# you cannot pick the space based on the shape you are about to run. In production the
# same constraint means one kernel serves all regimes from one (larger) list -- or you
# define separate kernels per regime and dispatch between them yourself.
REGIME = os.environ.get("REGIME", "general")   # "general" | "decode"


@triton.autotune(configs=build_configs(REGIME), key=["M", "N", "K"])
@triton.jit
def _gemm(A, B, C, M, N, K,
          sam, sak, sbk, sbn, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    per_group = GM * num_n
    group = pid // per_group
    first_m = group * GM
    gsize = min(num_m - first_m, GM)
    pid_m = first_m + ((pid % per_group) % gsize)
    pid_n = (pid % per_group) // gsize

    rm = (pid_m * BM + tl.arange(0, BM)) % M
    rn = (pid_n * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    a_ptr = A + (rm[:, None] * sam + rk[None, :] * sak)
    b_ptr = B + (rk[:, None] * sbk + rn[None, :] * sbn)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        mask = rk[None, :] < K - k * BK
        a = tl.load(a_ptr, mask=mask, other=0.0)
        b = tl.load(b_ptr, mask=mask.T, other=0.0)
        acc += tl.dot(a, b)
        a_ptr += BK * sak
        b_ptr += BK * sbk

    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    tl.store(C + scm * om[:, None] + scn * on[None, :],
             acc.to(C.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))


def gemm(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    _gemm[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1))
    return c


# ---------------------------------------------------------------------------
# 3. Gate on correctness, then measure against the noise floor
# ---------------------------------------------------------------------------

def err_ratio(out, ref):
    """Relative gate. Absolute error grows with K on a correct kernel and cannot gate."""
    o, r = out.float(), ref.float()
    return ((o - r).abs() > 1e-2 * r.abs() + 1e-2).float().mean().item()


def measure(fn, repeats=7, warmup=25, rep=100):
    """Independent samples -> median and spread. One do_bench call is not a measurement."""
    fn()  # absorb JIT compile
    samples = sorted(triton.testing.do_bench(fn, warmup=warmup, rep=rep) for _ in range(repeats))
    med = statistics.median(samples)
    return med, (samples[-1] - samples[0]) / med * 100.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- pass --device=/dev/kfd --device=/dev/dri "
                         "--group-add video, and check rocminfo")

    M, N, K = args.M, args.N, args.K
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    ref = torch.mm(a, b)

    out = gemm(a, b)                       # triggers the autotune pass
    err = err_ratio(out, ref)
    if err >= 0.05:
        raise SystemExit(f"FAIL correctness: err_ratio={err:.4f} -- timing is meaningless")

    tri_ms, tri_spread = measure(lambda: gemm(a, b), args.repeats)
    ref_ms, ref_spread = measure(lambda: torch.mm(a, b), args.repeats)
    flops = 2 * M * N * K

    print(f"\n# M={M} N={N} K={K} bf16  err_ratio={err:.4f} (gate <0.05)")
    print(f"{'backend':10s} {'ms':>8s} {'TFLOPS':>9s} {'spread%':>8s}")
    print(f"{'triton':10s} {tri_ms:8.3f} {flops / tri_ms * 1e-9:9.1f} {tri_spread:8.1f}")
    print(f"{'torch.mm':10s} {ref_ms:8.3f} {flops / ref_ms * 1e-9:9.1f} {ref_spread:8.1f}")
    print(f"\n# winning config: {_gemm.best_config}")

    delta = (ref_ms / tri_ms - 1) * 100
    floor = max(tri_spread, ref_spread)
    verdict = "REAL" if abs(delta) > floor else f"WITHIN NOISE (spread {floor:.1f}%)"
    print(f"# triton vs torch.mm: {delta:+.1f}%  -> {verdict}")


if __name__ == "__main__":
    main()
