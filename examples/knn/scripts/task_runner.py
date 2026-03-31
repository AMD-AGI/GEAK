#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/knn"""
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch

TASK_NAME = "hip2hip/knn"

# 5 test shapes: (B, N, M, k)
TEST_SHAPES = [
    (2, 64, 16, 3),
    (4, 256, 64, 5),
    (8, 1024, 128, 5),
    (2, 2048, 256, 10),
    (4, 512, 512, 3),
]


def cpu_reference(k, xyz, center_xyz):
    """Find k-nearest neighbors using brute-force pairwise distance."""
    # xyz: (B,N,3), center_xyz: (B,M,3) -> (B,k,M)
    dist = torch.cdist(center_xyz.float(), xyz.float())  # (B, M, N)
    _, idx = dist.topk(k, dim=2, largest=False)  # (B, M, k)
    return idx.transpose(2, 1).contiguous().int()  # (B, k, M)


def run_compile():
    try:
        from kernel_loader import knn_ext  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    from knn_wrapper import knn

    for i, (B, N, M, k) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + i)
        xyz = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
        center_xyz = torch.randn(B, M, 3, device="cuda", dtype=torch.float32)

        gpu_idx = knn(k, xyz, center_xyz)
        cpu_idx = cpu_reference(k, xyz.cpu(), center_xyz.cpu())

        # Sort along k dimension to handle tie-breaking differences
        gpu_sorted = torch.sort(gpu_idx.cpu(), dim=1)[0]
        cpu_sorted = torch.sort(cpu_idx, dim=1)[0]

        if not torch.all(gpu_sorted == cpu_sorted):
            mismatch = (gpu_sorted != cpu_sorted).sum().item()
            total = gpu_sorted.numel()
            return False, f"Shape {i+1} failed (B={B},N={N},M={M},k={k}): {mismatch}/{total} mismatches"

    return True, None


def _time_kernel(fn, n_warmup=10, n_iter=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def run_performance():
    from knn_wrapper import knn

    test_cases = []
    
    for shape_idx, (B, N, M, k) in enumerate(TEST_SHAPES):
        torch.manual_seed(42 + shape_idx)
        xyz = torch.randn(B, N, 3, device="cuda", dtype=torch.float32)
        center_xyz = torch.randn(B, M, 3, device="cuda", dtype=torch.float32)

        # Perf1: standard layout knn (B, N, 3) query
        ms_standard = _time_kernel(lambda: knn(k, xyz, center_xyz))

        # Perf2: transposed layout knn (B, 3, N) query
        xyz_t = xyz.transpose(1, 2).contiguous()
        center_xyz_t = center_xyz.transpose(1, 2).contiguous()
        ms_transposed = _time_kernel(lambda: knn(k, xyz_t, center_xyz_t, True))

        # Perf3: self-query knn (center_xyz = xyz)
        ms_self = _time_kernel(lambda: knn(k, xyz, xyz))

        test_cases.append({
            "test_case_id": f"shape_{shape_idx}_standard",
            "execution_time_ms": ms_standard,
            "params": {
                "B": B,
                "N": N,
                "M": M,
                "k": k,
                "layout": "standard"
            }
        })
        test_cases.append({
            "test_case_id": f"shape_{shape_idx}_transposed",
            "execution_time_ms": ms_transposed,
            "params": {
                "B": B,
                "N": N,
                "M": M,
                "k": k,
                "layout": "transposed"
            }
        })
        test_cases.append({
            "test_case_id": f"shape_{shape_idx}_self_query",
            "execution_time_ms": ms_self,
            "params": {
                "B": B,
                "N": N,
                "M": M,
                "k": k,
                "layout": "self_query"
            }
        })
    
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        report = {"test_cases": test_cases}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        for case in test_cases:
            print(f"Perf: {case['execution_time_ms']:.4f} ms ({case['test_case_id']})")
        sys.exit(0)


if __name__ == "__main__":
    main()
