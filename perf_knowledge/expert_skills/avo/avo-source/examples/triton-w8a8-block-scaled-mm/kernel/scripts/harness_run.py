"""Real launcher + benchmark harness for the raw @triton.jit kernel
``_w8a8_triton_block_scaled_mm`` (vLLM fp8 block-scaled GEMM).

Workload regime (campaign20): input seqlen = output seqlen = 1024;
concurrency B in {2, 32, 64}. This is a token-parallel GEMM, so the
token dimension M = B * 1024 (prefill token count). Model dims are kept
from the captured base case:
    K = 7168, N = 2112, group_n = group_k = 128.

The kernel is a plain @triton.jit kernel with NO python wrapper, so this
harness:
  * builds seeded, deterministic, valid fp8 + per-block-scale inputs,
  * recomputes the 1-D grid FORMULA from the regime dims,
  * launches the jit kernel (output C is written in place),
  * compares the editable source vs a frozen golden copy of the kernel,
  * times each case with CUDA events (10 warmup + 100 timed).
"""
import os, sys, json, importlib.util
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(THIS_DIR)
SRC_EDIT = os.path.join(TASK_DIR, "source", "triton__w8a8_triton_block_scaled_mm.py")
SRC_GOLD = os.path.join(TASK_DIR, "source_golden", "triton__w8a8_triton_block_scaled_mm.py")
KERNEL_NAME = "_w8a8_triton_block_scaled_mm"

# ---- captured model dims (kept) ----
K = 7168
N = 2112
GROUP_N = 128
GROUP_K = 128
SEQLEN = 1024            # campaign20 regime
FP8_DTYPE = torch.float8_e4m3fnuz
OUT_DTYPE = torch.bfloat16

# ---- launch meta from captured base case ----
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128
GROUP_SIZE_M = 32
NUM_WARPS = 4
NUM_STAGES = 2

CONCURRENCIES = [2, 32, 64]


def _load_kernel(path):
    spec = importlib.util.spec_from_file_location("k_" + os.path.basename(os.path.dirname(path)), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, KERNEL_NAME)


def cdiv(a, b):
    return (a + b - 1) // b


def build_inputs(B, seed=42):
    """Build deterministic fp8 block-scaled GEMM inputs for concurrency B.

    M = B * SEQLEN tokens. Returns (args, meta) where args is the positional
    arg list matching the kernel signature and meta carries grid/launch params.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    M = B * SEQLEN
    fp8_max = torch.finfo(FP8_DTYPE).max

    # high-precision reference operands in [-1,1]
    a_hi = (torch.rand((M, K), generator=g, device="cuda", dtype=torch.float32) * 2 - 1)
    b_hi = (torch.rand((N, K), generator=g, device="cuda", dtype=torch.float32) * 2 - 1)

    # per-token (row) scale for A: shape [M, K//GROUP_K]
    k_blocks = cdiv(K, GROUP_K)            # 56
    n_blocks = cdiv(N, GROUP_N)            # 17
    # quantize A per (token, k-block); produce fp8 + fp32 scales As[M, k_blocks]
    A_fp8 = torch.empty((M, K), device="cuda", dtype=FP8_DTYPE)
    As = torch.empty((M, k_blocks), device="cuda", dtype=torch.float32)
    for kb in range(k_blocks):
        ks = kb * GROUP_K
        ke = min(ks + GROUP_K, K)
        blk = a_hi[:, ks:ke]
        amax = blk.abs().amax(dim=1).clamp_min(1e-12)
        scale = amax / fp8_max
        As[:, kb] = scale
        A_fp8[:, ks:ke] = (blk / scale[:, None]).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)

    # quantize B per (n-block, k-block); Bs[n_blocks, k_blocks]
    B_fp8 = torch.empty((N, K), device="cuda", dtype=FP8_DTYPE)
    Bs = torch.empty((n_blocks, k_blocks), device="cuda", dtype=torch.float32)
    for nb in range(n_blocks):
        ns = nb * GROUP_N
        ne = min(ns + GROUP_N, N)
        for kb in range(k_blocks):
            ks = kb * GROUP_K
            ke = min(ks + GROUP_K, K)
            blk = b_hi[ns:ne, ks:ke]
            amax = blk.abs().amax().clamp_min(1e-12)
            scale = amax / fp8_max
            Bs[nb, kb] = scale
            B_fp8[ns:ne, ks:ke] = (blk / scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)

    C = torch.empty((M, N), device="cuda", dtype=OUT_DTYPE)

    # strides (contiguous layouts)
    stride_am, stride_ak = A_fp8.stride()
    stride_bk = B_fp8.stride(1)        # over K
    stride_bn = B_fp8.stride(0)        # over N
    stride_cm, stride_cn = C.stride()
    stride_As_m, stride_As_k = As.stride()
    # kernel indexes Bs as Bs + offs_bsn*stride_Bs_n + offs_ks*stride_Bs_k
    # with Bs laid out [n_blocks, k_blocks]: stride_Bs_n=k_blocks, stride_Bs_k=1
    stride_Bs_n, stride_Bs_k = Bs.stride()  # (k_blocks, 1)

    args = [
        A_fp8, B_fp8, C, As, Bs,
        M, N, K,
        GROUP_N, GROUP_K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_As_m, stride_As_k,
        stride_Bs_k, stride_Bs_n,
    ]
    grid = (cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N),)
    meta = dict(
        grid=grid,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return args, meta, C


def launch(kern, args, meta):
    grid = meta["grid"]
    kern[grid](
        *args,
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"], BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"], GROUP_SIZE_M=meta["GROUP_SIZE_M"],
        num_warps=meta["num_warps"], num_stages=meta["num_stages"],
    )


def _case_id(B):
    return f"c{B}"


def _params(B):
    return {"B": B, "M": B * SEQLEN, "N": N, "K": K,
            "group_n": GROUP_N, "group_k": GROUP_K, "seqlen": SEQLEN}


def run_correctness():
    kern_e = _load_kernel(SRC_EDIT)
    kern_g = _load_kernel(SRC_GOLD)
    for B in CONCURRENCIES:
        args_e, meta_e, C_e = build_inputs(B, seed=42)
        launch(kern_e, args_e, meta_e)
        args_g, meta_g, C_g = build_inputs(B, seed=42)
        launch(kern_g, args_g, meta_g)
        torch.cuda.synchronize()
        a = C_e.float()
        b = C_g.float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        denom = b.abs().clamp_min(1e-6)
        max_rel = ((a - b).abs() / denom).max().item()
        if not (cos >= 0.99 and max_rel < 1e-2):
            return False, f"{_case_id(B)}: cos={cos:.6f} max_rel={max_rel:.4e}"
    return True, None


def run_performance():
    kern = _load_kernel(SRC_EDIT)
    results = []
    for B in CONCURRENCIES:
        args, meta, C = build_inputs(B, seed=42)
        for _ in range(10):
            launch(kern, args, meta)
        torch.cuda.synchronize()
        n_iter = 100
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for j in range(n_iter):
            starts[j].record(); launch(kern, args, meta); ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
        results.append({"test_case_id": _case_id(B), "execution_time_ms": avg, "params": _params(B)})
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok, err = run_correctness()
        print("Correctness:", "PASS" if ok else "FAIL", err or "")
    else:
        for r in run_performance():
            print(f"Performance: {r['execution_time_ms']:.4f} ms ({r['test_case_id']})")
