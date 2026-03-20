#!/usr/bin/env python3
"""Regenerate baseline files for a preprocess directory with a new harness.

Usage: python regen_baselines.py <preprocess_dir> [--gpu 0]

Regenerates: harness_results.json, benchmark_baseline.txt,
full_benchmark_baseline.txt, profile.json, baseline_metrics.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EVAL_BENCHMARK_ITERATIONS = 200


def run_harness(harness_path: str, mode: str, env: dict, repo_root: str,
                extra_args: str = "") -> dict:
    cmd = f"python3 {harness_path} --{mode}"
    if extra_args:
        cmd += f" {extra_args}"
    t0 = time.time()
    proc = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=600, env=env, cwd=repo_root,
    )
    elapsed = round(time.time() - t0, 2)
    return {
        "mode": mode,
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[:1000] if proc.stderr else "",
        "duration_s": elapsed,
    }


def run_kernel_profile(harness_path: str, env: dict, repo_root: str,
                       gpu_id: str) -> dict | None:
    warmup_cmd = f"python3 {harness_path} --profile"
    for _ in range(2):
        subprocess.run(
            warmup_cmd, shell=True, capture_output=True, timeout=300,
            env=env, cwd=repo_root,
        )

    profile_cmd = (
        f'kernel-profile "python3 {harness_path} --profile"'
        f' --gpu-devices {gpu_id} --replays 5'
    )
    proc = subprocess.run(
        profile_cmd, shell=True, capture_output=True, text=True,
        timeout=600, env=env, cwd=repo_root,
    )
    if proc.returncode != 0:
        print(f"  kernel-profile failed: {proc.stderr[:300]}", file=sys.stderr)
        return None

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocess_dir")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    pp = Path(args.preprocess_dir).resolve()
    harness_path = (pp / "harness_path.txt").read_text().strip()
    resolved = json.loads((pp / "resolved.json").read_text())
    repo_root = resolved["local_repo_path"]

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

    print(f"Preprocess dir: {pp}")
    print(f"Harness: {harness_path}")
    print(f"Repo root: {repo_root}")
    print(f"GPU: {args.gpu}")
    print()

    modes = ["correctness", "profile", "benchmark", "full-benchmark"]
    extra = f"--iterations {EVAL_BENCHMARK_ITERATIONS}"
    results = []
    for mode in modes:
        print(f"Running --{mode}...", flush=True)
        mode_extra = extra if mode in ("benchmark", "full-benchmark") else ""
        r = run_harness(harness_path, mode, env, repo_root, mode_extra)
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {status} ({r['duration_s']}s)")
        if not r["success"] and mode == "correctness":
            print(f"  STDERR: {r['stderr'][:300]}", file=sys.stderr)
            print("CORRECTNESS FAILED - aborting", file=sys.stderr)
            sys.exit(1)
        results.append(r)

    (pp / "harness_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote harness_results.json ({len(results)} modes)")

    for r in results:
        if r["mode"] == "benchmark" and r["success"]:
            (pp / "benchmark_baseline.txt").write_text(r["stdout"])
            print("Wrote benchmark_baseline.txt")
        if r["mode"] == "full-benchmark" and r["success"]:
            (pp / "full_benchmark_baseline.txt").write_text(r["stdout"])
            print("Wrote full_benchmark_baseline.txt")

    print("\nRunning kernel-profile...", flush=True)
    profiling = run_kernel_profile(harness_path, env, repo_root, args.gpu)
    if profiling:
        (pp / "profile.json").write_text(json.dumps(profiling, indent=2))
        print("Wrote profile.json")

        try:
            from minisweagent.run.preprocess.baseline import build_baseline_metrics
            baseline_metrics = build_baseline_metrics(profiling, include_all=True)
        except Exception as exc:
            print(f"  build_baseline_metrics failed: {exc}, using empty dict")
            baseline_metrics = {}
    else:
        print("  kernel-profile failed, keeping existing profile.json")
        try:
            baseline_metrics = json.loads((pp / "baseline_metrics.json").read_text())
        except Exception:
            baseline_metrics = {}

    bb_path = pp / "benchmark_baseline.txt"
    if bb_path.exists():
        bb_text = bb_path.read_text()
        lat_m = re.search(
            r"GEAK_RESULT_LATENCY_MS=([\d.]+)", bb_text
        ) or re.search(
            r"Geometric mean latency:\s*([\d.]+)\s*ms", bb_text
        )
        if lat_m:
            baseline_metrics["benchmark_duration_us"] = float(lat_m.group(1)) * 1000.0
        shape_m = re.search(r"(\d+)\s+shapes", bb_text, re.IGNORECASE)
        if shape_m:
            baseline_metrics["benchmark_shape_count"] = int(shape_m.group(1))

    (pp / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2))
    print(f"Wrote baseline_metrics.json")
    print(f"  duration_us: {baseline_metrics.get('duration_us', 'N/A')}")
    print(f"  bottleneck: {baseline_metrics.get('bottleneck', 'N/A')}")
    print(f"  benchmark_duration_us: {baseline_metrics.get('benchmark_duration_us', 'N/A')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
