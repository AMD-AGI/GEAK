"""
Unit test for Triton sparse MLA decode kernel (FP8 in-kernel dequant).

The kernel reads FP8 KV cache directly and dequantizes in-kernel — no Python-side
dequant. Golden outputs come from ref.py (PyTorch reference on dequanted bf16).
FP8 tolerance: atol=2e-2, rtol=2e-2 (looser than bf16 due to quantization error).

Test configs are identical to reference/test_flash_mla_sparse_decoding.py.
"""
import sys
import time
import math
import dataclasses
from typing import Tuple, List

import torch

import kernelkit as kk
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import lib
import ref

from triton_flash_mla_decode import run_triton_decode


def gen_testcase() -> List[RawTestParam]:
    """
    Test cases based on REAL DS_v4 serving shapes:
    - h_q=128, h_kv=1, d_qk=d_v=512, d_rope=64, s_q=1 (decode)
    - Dual-scope: main topk=128 block_size=128 (SWA), extra topk=1024 block_size=256 (c4 sparse)
    - Decode batch: b=32..256 in production
    - FP8 MODEL1 (E8M0)

    Correctness: ~60 cases covering the real shape + key edges (single/dual, topk_length,
    attn_sink, varlen, zero-seqlen, all-invalid). Not a cartesian explosion.
    Perf: 15 cases = real dual-scope shape at 5 batch sizes × 3 topk_length combos.
    """
    d_qk = 512

    # --- Correctness: real shape + key edges (20 cases) ---
    correctness_cases = []

    # Real dual-scope (production path): 4 cases (2 batch × 2 topk_len combos)
    for b in [32, 128]:
        correctness_cases.append(RawTestParam(
            b, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
            enable_attn_sink=True, extra_s_k=16384, extra_topk=1024,
            block_size=128, extra_block_size=256,
            check_correctness=True, num_runs=0))
        correctness_cases.append(RawTestParam(
            b, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
            have_topk_length=True, enable_attn_sink=True,
            extra_s_k=16384, extra_topk=1024,
            block_size=128, extra_block_size=256,
            have_extra_topk_length=True,
            check_correctness=True, num_runs=0))

    # Single-scope (robustness): 2 cases
    for b in [32, 128]:
        correctness_cases.append(RawTestParam(
            b, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
            block_size=128, enable_attn_sink=True,
            check_correctness=True, num_runs=0))

    # Small batch b=4 dual-scope: 1 case
    correctness_cases.append(RawTestParam(
        4, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
        enable_attn_sink=True, extra_s_k=16384, extra_topk=1024,
        block_size=128, extra_block_size=256,
        check_correctness=True, num_runs=0))

    # Large batch b=256 dual-scope: 1 case
    correctness_cases.append(RawTestParam(
        256, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
        enable_attn_sink=True, extra_s_k=16384, extra_topk=1024,
        block_size=128, extra_block_size=256,
        check_correctness=True, num_runs=0))

    # s_q=2 (multi-token): 1 case
    correctness_cases.append(RawTestParam(
        32, 128, 2, 1, 16384, True, topk=128, d_qk=d_qk,
        extra_s_k=16384, extra_topk=1024,
        block_size=128, extra_block_size=256,
        enable_attn_sink=True,
        check_correctness=True, num_runs=0))

    # Varlen=False: 1 case
    correctness_cases.append(RawTestParam(
        32, 128, 1, 1, 16384, False, topk=128, d_qk=d_qk,
        extra_s_k=16384, extra_topk=1024,
        block_size=128, extra_block_size=256,
        enable_attn_sink=True,
        check_correctness=True, num_runs=0))
    # Subtotal correctness: 10

    # --- Corner cases (10 cases) ---
    corner_cases = []
    # All-invalid indices (dual + single)
    corner_cases.append(RawTestParam(
        32, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        is_all_indices_invalid=True, block_size=128, enable_attn_sink=True,
        extra_s_k=1024, extra_topk=1024, extra_block_size=256,
        check_correctness=True, num_runs=0))
    corner_cases.append(RawTestParam(
        32, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        is_all_indices_invalid=True, block_size=128, enable_attn_sink=True,
        check_correctness=True, num_runs=0))
    # Zero seqlens
    corner_cases.append(RawTestParam(
        32, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        have_zero_seqlen_k=True, block_size=128, enable_attn_sink=True,
        extra_s_k=1024, extra_topk=1024, extra_block_size=256,
        check_correctness=True, num_runs=0))
    corner_cases.append(RawTestParam(
        4, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        have_zero_seqlen_k=True, block_size=128, enable_attn_sink=True,
        check_correctness=True, num_runs=0))
    # No attn_sink
    corner_cases.append(RawTestParam(
        32, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        block_size=128, enable_attn_sink=False,
        extra_s_k=1024, extra_topk=1024, extra_block_size=256,
        check_correctness=True, num_runs=0))
    # Lonely query (all-invalid + zero-seqlen + no-sink)
    corner_cases.append(RawTestParam(
        32, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        is_all_indices_invalid=True, have_zero_seqlen_k=True,
        block_size=128, enable_attn_sink=False,
        check_correctness=True, num_runs=0))
    corner_cases.append(RawTestParam(
        4, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        is_all_indices_invalid=True, have_zero_seqlen_k=True,
        block_size=128, enable_attn_sink=False,
        check_correctness=True, num_runs=0))
    # Dual-scope with topk_length edge
    corner_cases.append(RawTestParam(
        32, 128, 1, 1, 512, True, topk=128, d_qk=d_qk,
        have_topk_length=True, block_size=128, enable_attn_sink=True,
        extra_s_k=1024, extra_topk=1024, extra_block_size=256,
        have_extra_topk_length=True,
        check_correctness=True, num_runs=0))
    # Small b=2 dual-scope
    corner_cases.append(RawTestParam(
        2, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
        enable_attn_sink=True, extra_s_k=16384, extra_topk=1024,
        block_size=128, extra_block_size=256,
        check_correctness=True, num_runs=0))
    # b=4 single-scope small s_k
    corner_cases.append(RawTestParam(
        4, 128, 1, 1, 256, True, topk=128, d_qk=d_qk,
        block_size=128, enable_attn_sink=True,
        check_correctness=True, num_runs=0))

    # --- Performance cases: REAL DS_v4 serving shapes (15 cases) ---
    # All dual-scope, h_q=128, main topk=128 bs=128, extra topk=1024 bs=256.
    base_and_bszs = [
        (RawTestParam(0, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
                      extra_s_k=16384, extra_topk=1024, block_size=128, extra_block_size=256),
         [2, 32, 64, 128, 256]),
        (RawTestParam(0, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
                      extra_s_k=16384, extra_topk=1024, block_size=128, extra_block_size=256,
                      have_extra_topk_length=True),
         [2, 32, 64, 128, 256]),
        (RawTestParam(0, 128, 1, 1, 16384, True, topk=128, d_qk=d_qk,
                      extra_s_k=16384, extra_topk=1024, block_size=128, extra_block_size=256,
                      have_topk_length=True, have_extra_topk_length=True),
         [2, 32, 64, 128, 256]),
    ]
    performance_cases = [
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ]

    return correctness_cases + corner_cases + performance_cases


@dataclasses.dataclass
class Result:
    is_correct: bool
    triton_us: float = 0.0
    ref_us: float = 0.0
    speedup: float = 0.0
    tflops: float = 0.0
    gBps: float = 0.0


_counter = kk.Counter()


@torch.inference_mode()
def test_triton_decode(p: TestParam) -> Result:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()
    assert p.decode

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase_for_decode(p)

    def run_triton():
        return run_triton_decode(p, t)

    def run_ref_fn():
        return ref.ref_sparse_attn_decode(p, t)

    if p.check_correctness:
        torch.cuda.synchronize()
        out_ans, lse_ans = run_triton()
        torch.cuda.synchronize()

    result = Result(is_correct=True)

    if p.num_runs > 0:
        triton_time = kk.bench_by_cuda_events(run_triton, num_warmups_each=15, num_runs_each=p.num_runs)
        ref_time = kk.bench_by_cuda_events(run_ref_fn, num_warmups_each=15, num_runs_each=p.num_runs)
        flops_and_mem_vol = lib.count_flop_and_mem_vol_for_decode(p, t)
        tflops = flops_and_mem_vol.flop / triton_time / 1e12
        gBps = flops_and_mem_vol.mem_vol / triton_time / 1e9
        speedup = ref_time / triton_time
        result.triton_us = triton_time * 1e6
        result.ref_us = ref_time * 1e6
        result.speedup = speedup
        result.tflops = tflops
        result.gBps = gBps
        print(f"  Triton: {result.triton_us:7.0f} us  Ref: {result.ref_us:7.0f} us  "
              f"Speedup: {speedup:.2f}x  TFlops: {tflops:.1f}  GB/s: {gBps:.0f}")

    if p.check_correctness:
        torch.cuda.synchronize()
        out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)

        # FP8 tolerance (quantization error)
        out_atol, out_rtol = 2e-2, 2e-2
        lse_atol, lse_rtol = 1e-2, 1e-2

        is_correct = True
        is_correct &= kk.check_is_allclose("out", out_ans, out_ref,
                                            abs_tol=out_atol, rel_tol=out_rtol, cos_diff_tol=5e-6)
        is_correct &= kk.check_is_allclose("lse", lse_ans, lse_ref,
                                            abs_tol=lse_atol, rel_tol=lse_rtol)
        result.is_correct = is_correct

    return result


def run_capture_safety_probe() -> bool:
    """Gate the #1 e2e blocker: the kernel must be CUDA-GRAPH CAPTURE-SAFE.

    A kernel that is correct + fast in isolation still scores e2e_delta=null /
    engagement_hits=0 ("hung on first capture batch, never healthy") if its first
    Triton JIT compile lands inside the HIP graph capture region (under TP=8 the
    in-capture compile desyncs NCCL -> server never healthy). The kernel MUST
    expose `ensure_warmed(device)` and, after it runs, a cold capture (no prior
    real forward) must be COMPILE-FREE and correct. See optimize.md §"CRITICAL #0".
    """
    import time
    print("\n\033[46m[capture-safety probe] warmup -> cold graph capture -> replay\033[0m")
    dev = torch.device("cuda:0")

    try:
        from triton_flash_mla_decode import ensure_warmed
    except ImportError:
        print("\033[31mCAPTURE_SAFE=0  reason=kernel_missing_ensure_warmed "
              "(required for precompile-before-capture; see optimize.md #0)\033[0m")
        return False

    # real serving shape: bs=256 dual-scope + attn_sink + topk_length
    p = RawTestParam(256, 128, 1, 1, 16384, True, topk=128, d_qk=512,
                     enable_attn_sink=True, extra_s_k=16384, extra_topk=1024,
                     block_size=128, extra_block_size=256,
                     have_topk_length=True, have_extra_topk_length=True,
                     check_correctness=True, num_runs=0).to_test_param()
    p.seed = 4242
    t = lib.generate_testcase_for_decode(p)

    try:
        ensure_warmed(dev)                       # MUST compile every specialization here
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        t0 = time.time()
        with torch.inference_mode(), torch.cuda.graph(g):   # NO prior real forward
            out, lse = run_triton_decode(p, t)
        cap_s = time.time() - t0
        torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize()
        with torch.inference_mode():
            out_ref, _ = ref.ref_sparse_attn_decode(p, t)
        err = (out.float() - out_ref.float()).abs().max().item()
    except Exception as e:
        print(f"\033[31mCAPTURE_SAFE=0  reason=capture_raised: {e}\033[0m")
        return False

    ok = cap_s < 1.0 and err < 5e-2   # <1s => compile happened in ensure_warmed, not in capture
    color = "\033[32m" if ok else "\033[31m"
    print(f"{color}CAPTURE_SAFE={int(ok)}  capture_s={cap_s:.2f} (compile-free<<1s)  "
          f"replay_max_abs_err={err:.2e}\033[0m")
    if not ok and cap_s >= 1.0:
        print("  -> capture triggered a compile: ensure_warmed did not cover this "
              "specialization (or autotune benchmarks inside capture). Fix per optimize.md #0.")
    return ok


# STRICT-enforcement gate: the authored kernel must implement every mandatory SPEC (see
# docs/optimize.md §"SPEC IDs") or attach a measured SPEC_SKIP_JUSTIFICATION. Mirrors the
# capture_safe gate: reads SPECS_IMPLEMENTED / SPEC_SKIP_JUSTIFICATION from the kernel module.
_MANDATORY_SPECS = (
    "fp8_fused_dequant",
    "capture_safety_ensure_warmed",
    "dual_scope_fused",
    "autotune_or_bucket_dispatch",
    "shape_specialized_constexpr",
)


def run_specs_coverage_gate() -> bool:
    try:
        import triton_flash_mla_decode as _tk
    except Exception as e:  # kernel not importable -> other gates already fail
        print(f"\033[31mSPECS_OK=0  reason=kernel_import_failed: {e}\033[0m")
        return False
    impl = getattr(_tk, "SPECS_IMPLEMENTED", None)
    just = getattr(_tk, "SPEC_SKIP_JUSTIFICATION", {}) or {}
    if not isinstance(impl, dict):
        print("\033[31mSPECS_OK=0  reason=kernel_missing_SPECS_IMPLEMENTED "
              "(strict flash_mla: export SPECS_IMPLEMENTED{spec_id:bool}; see optimize.md §SPEC IDs)\033[0m")
        return False
    missing = [s for s in _MANDATORY_SPECS
               if not impl.get(s, False) and not str(just.get(s, "")).strip()]
    ok = not missing
    color = "\033[32m" if ok else "\033[31m"
    print(f"{color}SPECS_OK={int(ok)}  implemented="
          f"{[s for s in _MANDATORY_SPECS if impl.get(s, False)]}  "
          f"missing_unjustified={missing}\033[0m")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true', help='Skip correctness, only run perf cases')
    args, _ = parser.parse_known_args()

    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(32)

    raw_testcases = gen_testcase()
    testcases = [t.to_test_param() for t in raw_testcases]

    if args.benchmark_only:
        testcases = [t for t in testcases if t.num_runs > 0]
        for t in testcases:
            t.check_correctness = False
        print(f"Benchmark-only mode: running {len(testcases)} perf cases")

    if args.quick:
        testcases = [t for t in testcases if t.check_correctness][:12]
        print(f"Quick mode: running {len(testcases)} cases")

    print(f"\033[46m{len(testcases)} testcases to run (FP8 in-kernel dequant)\033[0m")

    is_no_cooldown = lib.is_no_cooldown()
    num_testcases_len = len(str(len(testcases)))
    failed_cases = []
    skipped_cases = []
    results: List[Tuple[TestParam, Result]] = []
    for testcase_idx, testcase in enumerate(testcases):
        if testcase != testcases[0] and testcase.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        print(f"[{testcase_idx+1:{num_testcases_len}d}/{len(testcases)}, {testcase_idx/len(testcases)*100:3.0f}%]  ", end='')
        try:
            result = test_triton_decode(testcase)
            results.append((testcase, result))
            if not result.is_correct:
                failed_cases.append(testcase)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_str = str(e)
            if 'out of memory' in err_str.lower() or 'OutOfResources' in err_str:
                print(f"  SKIPPED (OOM)")
                skipped_cases.append(testcase)
                torch.cuda.empty_cache()
            else:
                raise

    # === Summary ===
    print()
    num_correctness = sum(1 for t, r in results if t.check_correctness)
    num_correct = sum(1 for t, r in results if t.check_correctness and r.is_correct)

    if skipped_cases:
        print(f"\033[33m\033[1m{len(skipped_cases)} cases skipped (OOM)\033[0m")

    if num_correctness > 0:
        if num_correct == num_correctness:
            print(f"\033[32m\033[1m{num_correct}/{num_correctness} correctness cases passed\033[0m")
        else:
            print(f"\033[31m\033[1m{num_correct}/{num_correctness} correctness cases passed\033[0m")
            for t in failed_cases:
                print(f"\t{t},")

    # Speedup geomean
    speedups = [r.speedup for _, r in results if r.speedup > 0]
    if speedups:
        geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"Speedup vs ref geomean: {geomean_speedup:.2f}x  ({len(speedups)} perf cases)")

    tflops_list = [r.tflops for _, r in results if r.tflops > 0]
    if tflops_list:
        geomean_tflops = math.exp(sum(math.log(t) for t in tflops_list) / len(tflops_list))
        print(f"TFlops geomean:         {geomean_tflops:.1f}")

    # Capture-safety gate (the #1 e2e blocker — a correct+fast kernel that is not
    # capture-safe scores e2e_delta=null / hangs the server). Skipped in benchmark-only.
    capture_safe = True
    specs_ok = True
    if not args.benchmark_only:
        capture_safe = run_capture_safety_probe()
        specs_ok = run_specs_coverage_gate()   # STRICT: mandatory SPEC coverage

    # Machine-readable output
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else 1.0
    print(f"\nUNITTEST_RESULT correctness={num_correct}/{num_correctness} "
          f"geomean_speedup={geomean:.4f} capture_safe={int(capture_safe)} specs_ok={int(specs_ok)}")

    if failed_cases or not capture_safe or not specs_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
