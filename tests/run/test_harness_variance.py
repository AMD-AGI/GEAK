#!/usr/bin/env python3
"""Harness variance test.

Runs geak-preprocess N times for each kernel and measures variance:
  - Are --benchmark shape lines identical across runs?
  - Are --full-benchmark shape lines identical?
  - Shape count consistency
  - Harness source diff (what changes between runs?)

Usage (inside Docker):
    python tests/run/test_harness_variance.py --runs 5
    python tests/run/test_harness_variance.py --runs 5 --kernel topk fused_rms_fp8
    python tests/run/test_harness_variance.py --runs 5 -o /workspace/variance_test/

After running, review the report at <output_dir>/variance_report.json
Then ask the agent to validate the harnesses (test 2).
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KERNELS = {
    "fused_rms_fp8": {
        "url": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/quant/fused_fp8_quant.py#L24",
        "local": "aiter/ops/triton/quant/fused_fp8_quant.py",
    },
    "topk": {
        "url": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/topk.py#L167",
        "local": "aiter/ops/triton/topk.py",
    },
    "fused_qkv_rope": {
        "url": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/rope/fused_qkv_split_qk_rope.py#L8",
        "local": "aiter/ops/triton/rope/fused_qkv_split_qk_rope.py",
    },
    "lean_atten_paged": {
        "url": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/attention/lean_atten_paged.py",
        "local": "aiter/ops/triton/attention/lean_atten_paged.py",
    },
    "moe_routing_sigmoid_top1": {
        "url": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py#L16",
        "local": "aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py",
    },
}


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_preprocess(kernel_url, output_dir, gpu_id=0, aiter_repo=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "minisweagent.run.preprocessor",
        kernel_url, "-o", str(out), "--gpu", str(gpu_id),
    ]
    if aiter_repo:
        cmd.extend(["--repo", aiter_repo])
    env = os.environ.copy()
    env["GEAK_BENCHMARK_ITERATIONS"] = "5"
    env["GEAK_HARNESS_ONLY"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_REPO_ROOT / 'src'}:{existing}" if existing else str(_REPO_ROOT / "src")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    return result.returncode == 0, result.stdout, result.stderr


def find_harness(output_dir):
    out = Path(output_dir)
    sel_file = out / "testcase_selection.json"
    if sel_file.exists():
        sel = json.loads(sel_file.read_text())
        hp = sel.get("harness_path", "")
        if hp:
            if Path(hp).is_file():
                return Path(hp)
            local = out / Path(hp).name
            if local.is_file():
                return local
    for pattern in ["test_*_harness.py", "*_harness.py", "test_*_focused.py"]:
        matches = list(out.glob(pattern))
        if matches:
            return matches[0]
    return None


def run_mode(harness_path, mode, repo_root=None, gpu_id=0, timeout=300):
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["GEAK_BENCHMARK_ITERATIONS"] = "5"
    env["PYTHONUNBUFFERED"] = "1"
    if repo_root:
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    cmd = [sys.executable, str(harness_path), f"--{mode}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"TIMEOUT after {timeout}s"


def extract_shapes_used(stdout):
    """Parse GEAK_SHAPES_USED=[(a,b,c), ...] from stdout.

    Returns a sorted list of tuples, or None if not found.
    """
    m = re.search(r"GEAK_SHAPES_USED=(\[.*\])", stdout)
    if not m:
        return None
    try:
        import ast
        shapes = ast.literal_eval(m.group(1))
        return sorted(tuple(s) for s in shapes)
    except (ValueError, SyntaxError):
        return None


def extract_latency(stdout):
    m = re.search(r"GEAK_RESULT_LATENCY_MS=([0-9.eE\-+]+)", stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def get_repo_root(output_dir):
    resolved = Path(output_dir) / "resolved.json"
    if resolved.exists():
        r = json.loads(resolved.read_text())
        return r.get("local_repo_path")
    return None


def test_kernel_variance(kernel_name, kernel_url, base_dir, num_runs=5, gpu_id=0, aiter_repo=None):
    """Run preprocessing N times and collect variance data."""
    print(f"\n{'='*60}")
    print(f"  {kernel_name} ({num_runs} runs)")
    if aiter_repo:
        print(f"  Using pre-cloned repo: {aiter_repo}")
    print(f"{'='*60}")

    runs = []

    for i in range(num_runs):
        run_dir = base_dir / kernel_name / f"run_{i}"
        print(f"\n  --- Run {i+1}/{num_runs} ---")

        run_data = {
            "run": i,
            "preprocess_ok": False,
            "harness_path": None,
            "harness_source": None,
            "modes": {},
            "bench_shapes": None,
            "full_shapes": None,
            "correctness_shapes": None,
            "profile_shapes": None,
            "bench_latency": None,
            "full_latency": None,
        }

        ok, _, stderr = run_preprocess(kernel_url, str(run_dir), gpu_id=gpu_id, aiter_repo=aiter_repo)
        run_data["preprocess_ok"] = ok
        if not ok:
            print(f"    Preprocessing FAILED")
            print(f"    {stderr[-300:]}")
            runs.append(run_data)
            continue

        harness = find_harness(str(run_dir))
        if harness is None:
            print(f"    No harness found")
            runs.append(run_data)
            continue

        run_data["harness_path"] = str(harness)
        run_data["harness_source"] = harness.read_text()
        repo_root = get_repo_root(str(run_dir))

        mode_to_key = {
            "correctness": "correctness_shapes",
            "profile": "profile_shapes",
            "benchmark": "bench_shapes",
            "full-benchmark": "full_shapes",
        }
        for mode in ["correctness", "profile", "benchmark", "full-benchmark"]:
            ok, stdout, stderr = run_mode(harness, mode, repo_root, gpu_id)
            shapes = extract_shapes_used(stdout) if ok else None
            run_data["modes"][mode] = {
                "ok": ok,
                "shapes_count": len(shapes) if shapes else 0,
                "stderr_tail": stderr[-200:] if not ok else "",
            }
            run_data[mode_to_key[mode]] = shapes

            if mode == "benchmark" and ok:
                run_data["bench_latency"] = extract_latency(stdout)
            if mode == "full-benchmark" and ok:
                run_data["full_latency"] = extract_latency(stdout)

            status = "OK" if ok else "FAIL"
            count = len(shapes) if shapes else "?"
            print(f"    --{mode}: {status} ({count} shapes)")

        bc = len(run_data["bench_shapes"]) if run_data["bench_shapes"] else 0
        fc = len(run_data["full_shapes"]) if run_data["full_shapes"] else 0
        print(f"    Benchmark: {bc} shapes, Full: {fc} shapes")
        runs.append(run_data)

    # ── Analyze variance ──
    report = analyze_variance(kernel_name, runs)
    return runs, report


def analyze_variance(kernel_name, runs):
    """Compare actual shape tuples across runs."""
    report = {
        "kernel": kernel_name,
        "total_runs": len(runs),
        "preprocess_success_rate": sum(1 for r in runs if r["preprocess_ok"]) / len(runs),
    }

    successful = [r for r in runs if r["preprocess_ok"] and r["harness_source"]]
    if len(successful) < 2:
        report["error"] = f"Only {len(successful)} successful runs, need >= 2 for comparison"
        return report

    # Compare actual shape tuples across runs
    for key, label in [
        ("bench_shapes", "benchmark"),
        ("full_shapes", "full_benchmark"),
        ("correctness_shapes", "correctness"),
        ("profile_shapes", "profile"),
    ]:
        shapes_per_run = [r[key] for r in successful]
        non_none = [s for s in shapes_per_run if s is not None]

        report[f"{label}_parsed_count"] = len(non_none)
        report[f"{label}_shape_counts"] = [len(s) if s else 0 for s in shapes_per_run]

        if len(non_none) >= 2:
            all_same = all(s == non_none[0] for s in non_none)
            report[f"{label}_shapes_deterministic"] = all_same

            if all_same:
                report[f"{label}_shapes"] = [list(t) for t in non_none[0]]
            else:
                # Show what differs
                intersection = set(map(tuple, non_none[0]))
                union = set(map(tuple, non_none[0]))
                for s in non_none[1:]:
                    s_set = set(map(tuple, s))
                    intersection &= s_set
                    union |= s_set
                report[f"{label}_shapes_intersection"] = len(intersection)
                report[f"{label}_shapes_union"] = len(union)
                report[f"{label}_shapes_only_in_some_runs"] = len(union - intersection)
        else:
            report[f"{label}_shapes_deterministic"] = None

    # Subset relationship: benchmark ⊂ full_benchmark
    bench_runs = [r["bench_shapes"] for r in successful if r["bench_shapes"]]
    full_runs = [r["full_shapes"] for r in successful if r["full_shapes"]]
    if bench_runs and full_runs:
        all_subsets = all(
            set(map(tuple, b)).issubset(set(map(tuple, f)))
            for b, f in zip(bench_runs, full_runs)
        )
        report["benchmark_subset_of_full"] = all_subsets

    # ≤25 check
    if bench_runs:
        report["benchmark_all_leq_25"] = all(len(b) <= 25 for b in bench_runs)

    # Mode pass rates
    for mode in ["correctness", "profile", "benchmark", "full-benchmark"]:
        pass_count = sum(1 for r in successful if r["modes"].get(mode, {}).get("ok", False))
        report[f"{mode}_pass_rate"] = pass_count / len(successful)

    # Latency variance
    bench_lats = [r["bench_latency"] for r in successful if r["bench_latency"] is not None]
    if bench_lats:
        report["benchmark_latencies_ms"] = bench_lats

    # Harness source diff
    base_source = successful[0]["harness_source"]
    diffs = []
    for i, r in enumerate(successful[1:], 1):
        diff = list(difflib.unified_diff(
            base_source.splitlines(keepends=True),
            r["harness_source"].splitlines(keepends=True),
            fromfile=f"run_0", tofile=f"run_{i}", n=1,
        ))
        if diff:
            diffs.append({"run_0_vs": i, "diff_lines": len(diff), "diff": "".join(diff[:50])})
    report["harness_source_identical"] = len(diffs) == 0
    if diffs:
        report["harness_diffs"] = diffs

    # Print summary
    print(f"\n  ── Variance Report: {kernel_name} ──")
    print(f"  Preprocess success: {report['preprocess_success_rate']*100:.0f}%")
    for key, label in [("benchmark", "benchmark"), ("full_benchmark", "full-benchmark"),
                        ("correctness", "correctness"), ("profile", "profile")]:
        det = report.get(f"{key}_shapes_deterministic")
        counts = report.get(f"{key}_shape_counts", [])
        det_str = "YES" if det else ("NO" if det is False else "N/A")
        print(f"  --{label}: deterministic={det_str}, counts={counts}")
        if det is False:
            inter = report.get(f"{key}_shapes_intersection", "?")
            union = report.get(f"{key}_shapes_union", "?")
            diff = report.get(f"{key}_shapes_only_in_some_runs", "?")
            print(f"    intersection={inter}, union={union}, varying={diff}")

    sub = report.get("benchmark_subset_of_full")
    if sub is not None:
        print(f"  benchmark ⊂ full-benchmark: {sub}")
    leq = report.get("benchmark_all_leq_25")
    if leq is not None:
        print(f"  benchmark ≤ 25 shapes: {leq}")
    print(f"  Harness source identical: {report['harness_source_identical']}")
    for mode in ["correctness", "profile", "benchmark", "full-benchmark"]:
        print(f"  --{mode} pass rate: {report.get(f'{mode}_pass_rate', 0)*100:.0f}%")

    return report


def _run_kernel_batch(kernel_names, kernels_dict, base_dir, num_runs, gpu_id, aiter_repo):
    """Run variance test for a single kernel (called in a worker thread)."""
    results = {}
    for name in kernel_names:
        kinfo = kernels_dict[name]
        if aiter_repo:
            kernel_ref = str(Path(aiter_repo) / kinfo["local"])
        else:
            kernel_ref = kinfo["url"]
        runs, report = test_kernel_variance(
            name, kernel_ref, base_dir, num_runs=num_runs, gpu_id=gpu_id,
            aiter_repo=aiter_repo,
        )
        results[name] = (runs, report)

        kernel_report_path = base_dir / name / "variance_report.json"
        kernel_report_path.parent.mkdir(parents=True, exist_ok=True)
        kernel_report_path.write_text(json.dumps(report, indent=2, default=str))

        for r in runs:
            if r["harness_source"]:
                review_path = base_dir / name / f"harness_run_{r['run']}.py"
                review_path.write_text(r["harness_source"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Harness variance test")
    parser.add_argument(
        "--kernel", "-k", nargs="*", default=list(KERNELS.keys()),
        choices=list(KERNELS.keys()),
    )
    parser.add_argument("--runs", "-n", type=int, default=5)
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated GPU IDs for parallel execution (e.g. '4,5'). "
             "Runs one kernel per GPU concurrently.",
    )
    parser.add_argument(
        "--aiter-repo", default=None,
        help="Path to a pre-cloned aiter repo (skips git clone each run)",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir).resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="variance_")).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",")] if args.gpus else [args.gpu]
    parallel = len(gpu_ids)

    print(f"Output: {base_dir}")
    print(f"Kernels: {args.kernel}")
    print(f"Runs per kernel: {args.runs}")
    print(f"Parallel: {parallel} (GPUs: {gpu_ids})")

    all_reports = {}

    if parallel <= 1:
        for name in args.kernel:
            kinfo = KERNELS[name]
            if args.aiter_repo:
                kernel_ref = str(Path(args.aiter_repo) / kinfo["local"])
            else:
                kernel_ref = kinfo["url"]
            runs, report = test_kernel_variance(
                name, kernel_ref, base_dir, num_runs=args.runs, gpu_id=gpu_ids[0],
                aiter_repo=args.aiter_repo,
            )
            all_reports[name] = report

            kernel_report_path = base_dir / name / "variance_report.json"
            kernel_report_path.parent.mkdir(parents=True, exist_ok=True)
            kernel_report_path.write_text(json.dumps(report, indent=2, default=str))

            for r in runs:
                if r["harness_source"]:
                    review_path = base_dir / name / f"harness_run_{r['run']}.py"
                    review_path.write_text(r["harness_source"])
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Split kernels into batches, one per GPU
        batches = [[] for _ in range(parallel)]
        for i, name in enumerate(args.kernel):
            batches[i % parallel].append(name)

        print(f"Batches: {batches}")

        futures = {}
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for batch_idx, batch in enumerate(batches):
                if not batch:
                    continue
                gpu = gpu_ids[batch_idx]
                future = pool.submit(
                    _run_kernel_batch, batch, KERNELS, base_dir,
                    args.runs, gpu, args.aiter_repo,
                )
                futures[future] = (batch, gpu)

            for future in as_completed(futures):
                batch, gpu = futures[future]
                try:
                    results = future.result()
                    for name, (runs, report) in results.items():
                        all_reports[name] = report
                except Exception as exc:
                    print(f"  Batch {batch} on GPU {gpu} FAILED: {exc}")
                    for name in batch:
                        all_reports[name] = {"error": str(exc)}

    # Save combined report
    combined = base_dir / "variance_report.json"
    combined.write_text(json.dumps(all_reports, indent=2, default=str))

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    all_deterministic = True
    for name, report in all_reports.items():
        bench_det = report.get("benchmark_shapes_deterministic", False)
        src_ident = report.get("harness_source_identical", False)
        if not bench_det:
            all_deterministic = False
        status = "PASS" if bench_det else "FAIL"
        print(f"  [{status}] {name}: shapes_deterministic={bench_det}, source_identical={src_ident}")

    print(f"\nReports saved to: {base_dir}")
    print(f"Harness files saved as harness_run_N.py for manual review (test 2)")
    sys.exit(0 if all_deterministic else 1)


if __name__ == "__main__":
    main()
