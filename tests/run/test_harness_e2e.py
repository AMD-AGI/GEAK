#!/usr/bin/env python3
"""End-to-end harness validation test.

Runs geak-preprocess on real kernels, then validates the generated harness
WITHOUT knowing what shapes to expect. Tests only observable behavior:
  1. All 4 modes run and exit 0
  2. --correctness passes (kernel output matches reference)
  3. --benchmark and --full-benchmark print GEAK_RESULT_LATENCY_MS=<number>
  4. --benchmark uses ≤ 25 shapes
  5. --full-benchmark uses ≥ as many shapes as --benchmark
  6. --benchmark is deterministic (same shapes every run)

Run inside Docker:
    python tests/run/test_harness_e2e.py --kernel topk
    python tests/run/test_harness_e2e.py                           # all kernels
    python tests/run/test_harness_e2e.py --skip-preprocess -o /path/to/existing/output --kernel topk

Environment:
    AMD_LLM_API_KEY or LLM_GATEWAY_KEY for preprocessing.
    GPU required for harness execution.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KERNELS = {
    "fused_rms_fp8": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/quant/fused_fp8_quant.py#L24",
    "topk": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/topk.py#L167",
    "fused_qkv_rope": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/rope/fused_qkv_split_qk_rope.py#L8",
    "lean_atten_paged": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/attention/lean_atten_paged.py",
    "moe_routing_sigmoid_top1": "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py#L16",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_preprocess(kernel_url, output_dir, gpu_id=0):
    """Run geak-preprocess. Returns True on success."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "minisweagent.run.preprocessor",
        kernel_url, "-o", str(out), "--gpu", str(gpu_id),
    ]
    print(f"    cmd: {' '.join(cmd)}")
    env = os.environ.copy()
    env["GEAK_BENCHMARK_ITERATIONS"] = "5"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_REPO_ROOT / 'src'}:{existing}" if existing else str(_REPO_ROOT / "src")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    if result.returncode != 0:
        print(f"    STDERR (last 1500 chars):\n{result.stderr[-1500:]}")
    return result.returncode == 0


def find_harness(output_dir):
    """Find the harness .py file."""
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
    """Run a harness mode. Returns (exit_ok, stdout, stderr)."""
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


def extract_shape_lines(stdout):
    """Extract per-shape result lines from benchmark stdout.

    Shape lines contain a latency value (e.g. "0.1234ms" or "0.1234 ms")
    and are NOT the final GEAK_RESULT line or header lines.
    """
    lines = []
    for line in stdout.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("GEAK_RESULT"):
            continue
        if re.search(r"\d+\.\d+\s*ms", stripped):
            lines.append(stripped)
    return lines


def extract_latency(stdout):
    """Extract GEAK_RESULT_LATENCY_MS value from stdout."""
    m = re.search(r"GEAK_RESULT_LATENCY_MS=([0-9.eE\-+]+)", stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition, description):
        if condition:
            self.passed += 1
            print(f"    PASS: {description}")
        else:
            self.failed += 1
            self.errors.append(description)
            print(f"    FAIL: {description}")

    def summary(self):
        status = "PASSED" if self.failed == 0 else "FAILED"
        return f"  [{status}] {self.name}: {self.passed} passed, {self.failed} failed"


def test_kernel(kernel_name, kernel_url, output_dir, skip_preprocess=False, gpu_id=0):
    """Full e2e test for one kernel."""
    result = TestResult(kernel_name)
    out = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"  {kernel_name}")
    print(f"{'='*60}")

    # ── 1. Preprocessing ──
    if not skip_preprocess:
        print("\n  [Preprocessing]")
        ok = run_preprocess(kernel_url, str(out), gpu_id=gpu_id)
        result.check(ok, "geak-preprocess exits 0")
        if not ok:
            return result

    # ── 2. Find harness ──
    print("\n  [Find harness]")
    harness = find_harness(str(out))
    result.check(harness is not None, f"Harness file found: {harness}")
    if harness is None:
        return result

    # Get repo root for PYTHONPATH
    repo_root = None
    resolved = out / "resolved.json"
    if resolved.exists():
        r = json.loads(resolved.read_text())
        repo_root = r.get("local_repo_path")

    # ── 3. Run all 4 modes ──
    print("\n  [Mode: correctness]")
    ok, stdout, stderr = run_mode(harness, "correctness", repo_root, gpu_id)
    result.check(ok, "--correctness exits 0 (kernel output matches reference)")
    if not ok:
        print(f"      stderr: {stderr[-500:]}")

    print("\n  [Mode: profile]")
    ok, stdout, stderr = run_mode(harness, "profile", repo_root, gpu_id)
    result.check(ok, "--profile exits 0")
    if not ok:
        print(f"      stderr: {stderr[-500:]}")

    print("\n  [Mode: benchmark]")
    ok_bench, stdout_bench, stderr = run_mode(harness, "benchmark", repo_root, gpu_id)
    result.check(ok_bench, "--benchmark exits 0")
    if not ok_bench:
        print(f"      stderr: {stderr[-500:]}")

    print("\n  [Mode: full-benchmark]")
    ok_full, stdout_full, stderr = run_mode(harness, "full-benchmark", repo_root, gpu_id)
    result.check(ok_full, "--full-benchmark exits 0")
    if not ok_full:
        print(f"      stderr: {stderr[-500:]}")

    # ── 4. Output format ──
    print("\n  [Output format]")
    if ok_bench:
        lat = extract_latency(stdout_bench)
        result.check(
            lat is not None and lat > 0,
            f"--benchmark has GEAK_RESULT_LATENCY_MS (got: {lat})",
        )
    if ok_full:
        lat = extract_latency(stdout_full)
        result.check(
            lat is not None and lat > 0,
            f"--full-benchmark has GEAK_RESULT_LATENCY_MS (got: {lat})",
        )

    # ── 5. Shape counts ──
    print("\n  [Shape counts]")
    bench_shapes = extract_shape_lines(stdout_bench) if ok_bench else []
    full_shapes = extract_shape_lines(stdout_full) if ok_full else []

    if ok_bench:
        result.check(
            0 < len(bench_shapes) <= 25,
            f"--benchmark runs ≤ 25 shapes (got: {len(bench_shapes)})",
        )
    if ok_full:
        result.check(
            len(full_shapes) > 0,
            f"--full-benchmark runs > 0 shapes (got: {len(full_shapes)})",
        )
    if ok_bench and ok_full:
        result.check(
            len(full_shapes) >= len(bench_shapes),
            f"--full-benchmark shapes ({len(full_shapes)}) ≥ --benchmark shapes ({len(bench_shapes)})",
        )

    # ── 6. Determinism: run --benchmark again, same shape lines ──
    print("\n  [Determinism]")
    if ok_bench:
        ok2, stdout2, _ = run_mode(harness, "benchmark", repo_root, gpu_id)
        bench_shapes_2 = extract_shape_lines(stdout2) if ok2 else []
        result.check(
            bench_shapes == bench_shapes_2,
            f"--benchmark deterministic: {len(bench_shapes)} shapes identical across 2 runs",
        )
        if bench_shapes != bench_shapes_2:
            print(f"      Run 1: {bench_shapes[:3]}...")
            print(f"      Run 2: {bench_shapes_2[:3]}...")

    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end harness validation")
    parser.add_argument(
        "--kernel", "-k", nargs="*", default=list(KERNELS.keys()),
        choices=list(KERNELS.keys()), help="Kernels to test (default: all)",
    )
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.output_dir).resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="harness_e2e_")).resolve()
    print(f"Output: {base_dir}")

    results = []
    single_kernel = len(args.kernel) == 1
    for name in args.kernel:
        kernel_dir = base_dir if (args.skip_preprocess and single_kernel) else base_dir / name
        r = test_kernel(
            name, KERNELS[name], str(kernel_dir),
            skip_preprocess=args.skip_preprocess, gpu_id=args.gpu,
        )
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_pass = sum(r.passed for r in results)
    total_fail = sum(r.failed for r in results)
    for r in results:
        print(r.summary())
    print(f"\nTotal: {total_pass} passed, {total_fail} failed")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
