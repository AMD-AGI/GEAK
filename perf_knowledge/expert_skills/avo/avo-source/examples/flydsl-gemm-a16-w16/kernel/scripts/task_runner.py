#!/usr/bin/env python3
"""Task runner for the FlyDSL-capable gemm_a16_w16 example."""

import argparse
import ast
import importlib.util
import json
import os
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(
    TASK_DIR, "source", "triton__gemm_a16_w16_kernel.py"
)
KERNEL_NAME = "_gemm_a16_w16_kernel"


def run_compile():
    try:
        with open(SOURCE_FILE, encoding="utf-8") as f:
            ast.parse(f.read())
        spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getattr(mod, KERNEL_NAME)
        return True, None
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, error = run_compile()
        with open(
            os.path.join(build_dir, "compile_report.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"status": "ok" if ok else "fail", "error": error}, f)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if error:
            print("Error:", error)
        return 0 if ok else 1

    import harness_run

    harness_run.write_test_cases()
    if args.mode == "correctness":
        ok, error = harness_run.run_correctness()
        with open(
            os.path.join(build_dir, "correctness_report.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({"status": "ok" if ok else "fail", "error": error}, f)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if error:
            print("Error:", error)
        return 0 if ok else 1

    cases = harness_run.run_performance(warmup=10, iters=100)
    with open(
        os.path.join(build_dir, "performance_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"test_cases": cases}, f, indent=2)
    for case in cases:
        print(
            f"Performance: {case['execution_time_ms']:.4f} ms "
            f"({case['test_case_id']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
