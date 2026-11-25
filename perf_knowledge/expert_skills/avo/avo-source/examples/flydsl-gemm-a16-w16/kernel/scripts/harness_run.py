#!/usr/bin/env python3
"""Correctness and CUDA-event benchmark harness for the FlyDSL GEMM case."""

import importlib.util
import json
import os

import torch
import triton
import triton.language as tl

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton__gemm_a16_w16_kernel.py")
GOLDEN_FILE = os.path.join(
    TASK_DIR, "source_golden", "triton__gemm_a16_w16_kernel.py"
)
KERNEL_NAME = "_gemm_a16_w16_kernel"
TEST_CASES = os.path.join(TASK_DIR, "test_cases.json")

K_DIM = 2880
N_DIM = 5120
SEQLEN = 1024
CONCURRENCY = [2, 32, 64]
META = dict(
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=128,
    GROUP_SIZE_M=1,
    NUM_KSPLIT=1,
    SPLITK_BLOCK_SIZE=K_DIM,
    num_warps=4,
    num_stages=2,
    waves_per_eu=2,
    matrix_instr_nonkdim=16,
    cache_modifier=".cg",
)


def regime_cases():
    return [
        {
            "test_case_id": f"c{b}",
            "B": b,
            "M": b * SEQLEN,
            "N": N_DIM,
            "K": K_DIM,
            "params": {
                "B": b,
                "seqlen": SEQLEN,
                "M": b * SEQLEN,
                "N": N_DIM,
                "K": K_DIM,
            },
        }
        for b in CONCURRENCY
    ]


def write_test_cases():
    with open(TEST_CASES, "w", encoding="utf-8") as f:
        json.dump(regime_cases(), f, indent=2)


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@triton.jit
def pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 1):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def _load_kernel(path):
    spec = importlib.util.spec_from_file_location("k_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__dict__.update(remap_xcd=remap_xcd, pid_grid=pid_grid)
    kern = getattr(mod, KERNEL_NAME)
    try:
        kern.fn.__globals__.update(
            remap_xcd=remap_xcd, pid_grid=pid_grid, tl=tl, triton=triton
        )
    except AttributeError:
        pass
    return kern


def build_inputs(m, n, k, seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((m, k), generator=g, dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), generator=g, dtype=torch.bfloat16, device="cuda")
    b = w.T
    bias = torch.randn((n,), generator=g, dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    return a, b, bias, c


def _grid(m, n, meta):
    return (
        meta["NUM_KSPLIT"]
        * triton.cdiv(m, meta["BLOCK_SIZE_M"])
        * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )


def launch(kern, a, b, bias, c, m, n, k, meta):
    kern[_grid(m, n, meta)](
        a,
        b,
        bias,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        0,
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
        GROUP_SIZE_M=meta["GROUP_SIZE_M"],
        NUM_KSPLIT=meta["NUM_KSPLIT"],
        SPLITK_BLOCK_SIZE=meta["SPLITK_BLOCK_SIZE"],
        cache_modifier=meta["cache_modifier"],
        activation="",
        use_activation=False,
        ADD_BIAS=True,
        SKIP_REDUCE=False,
        num_warps=meta["num_warps"],
        num_stages=meta["num_stages"],
        waves_per_eu=meta["waves_per_eu"],
        matrix_instr_nonkdim=meta["matrix_instr_nonkdim"],
    )
    return c


def run_correctness():
    edit = _load_kernel(SOURCE_FILE)
    gold = _load_kernel(GOLDEN_FILE)
    for tc in regime_cases():
        m, n, k = tc["M"], tc["N"], tc["K"]
        out_e = launch(edit, *build_inputs(m, n, k), m, n, k, META)
        out_g = launch(gold, *build_inputs(m, n, k), m, n, k, META)
        torch.cuda.synchronize()
        cosine = torch.nn.functional.cosine_similarity(
            out_e.float().flatten(), out_g.float().flatten(), dim=0
        ).item()
        denom = out_g.float().abs().max().item()
        max_rel = (out_e.float() - out_g.float()).abs().max().item() / (
            denom + 1e-6
        )
        if cosine < 0.99 or max_rel >= 1e-2:
            return False, (
                f"{tc['test_case_id']}: cosine={cosine:.5f} "
                f"max_rel={max_rel:.4e}"
            )
    return True, None


def run_performance(warmup=10, iters=100):
    edit = _load_kernel(SOURCE_FILE)
    results = []
    for tc in regime_cases():
        m, n, k = tc["M"], tc["N"], tc["K"]
        args = build_inputs(m, n, k)
        for _ in range(warmup):
            launch(edit, *args, m, n, k, META)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends):
            start.record()
            launch(edit, *args, m, n, k, META)
            end.record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters
        results.append(
            {
                "test_case_id": tc["test_case_id"],
                "execution_time_ms": avg,
                "params": tc["params"],
            }
        )
    return results
