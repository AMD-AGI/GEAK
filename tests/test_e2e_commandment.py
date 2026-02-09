#!/usr/bin/env python3
"""
End-to-end test: COMMANDMENT evaluation on add_kernel inside Docker.
Tests the full pipeline: load kernel -> generate COMMANDMENT -> evaluate.
"""

import json
import os
import sys
import tempfile

# Add project root to path (works from repo root or tests/ directory)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from openevolve.database import load_program_from_directory
from openevolve.commandment_evaluator import CommandmentEvaluator, CommandmentGenerator


# Use env var or auto-detect common locations
ADD_KERNEL_DIR = os.environ.get("GEAK_ADD_KERNEL_DIR", "")
if not ADD_KERNEL_DIR:
    for candidate in [
        os.path.join(_PROJECT_ROOT, "..", "GEAK-msa", "examples", "add_kernel"),
        "/workspace/examples/add_kernel",
        os.path.join(os.path.expanduser("~"), "GEAK-msa", "examples", "add_kernel"),
    ]:
        if os.path.isdir(candidate):
            ADD_KERNEL_DIR = os.path.abspath(candidate)
            break


def test_e2e_commandment_evaluation():
    """Full pipeline: load -> generate COMMANDMENT -> evaluate baseline -> evaluate candidate."""

    # Step 1: Load the kernel
    print("[Step 1] Loading add_kernel...")
    files, main_file = load_program_from_directory(ADD_KERNEL_DIR)
    print(f"  Loaded {len(files)} files, main_file={main_file}")

    # Step 2: Create a working directory with test/profile scripts
    tmpdir = tempfile.mkdtemp(prefix="geak_e2e_")
    print(f"[Step 2] Working dir: {tmpdir}")

    # Write kernel files
    for rel_path, content in files.items():
        path = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    # Write a correctness test
    test_script = os.path.join(tmpdir, "test_correctness.py")
    with open(test_script, "w") as f:
        f.write("""
import sys
sys.path.insert(0, '.')
import torch
from kernel import triton_op, torch_op

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
result = triton_op(x, y)
expected = torch_op(x, y)
assert torch.allclose(result, expected, atol=1e-5), "Correctness check failed!"
print("Correctness: PASSED")
""")

    # Write a profiling script that outputs JSON metrics
    profile_script = os.path.join(tmpdir, "profile_kernel.py")
    with open(profile_script, "w") as f:
        f.write("""
import json
import time
import torch
import sys
sys.path.insert(0, '.')
from kernel import triton_op

# Warmup
x = torch.randn(1024*1024, device='cuda')
y = torch.randn(1024*1024, device='cuda')
for _ in range(10):
    triton_op(x, y)
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    triton_op(x, y)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times.append((end - start) * 1e6)  # microseconds

latency_avg = sum(times) / len(times)
latency_min = min(times)
latency_max = max(times)

metrics = {
    "latency_us": latency_avg,
    "latency_min_us": latency_min,
    "latency_max_us": latency_max,
    "latency_avg_us": latency_avg,
}
print(json.dumps(metrics))
""")

    # Step 3: Generate COMMANDMENT.md from validated commands
    print("[Step 3] Generating COMMANDMENT.md...")
    gen = CommandmentGenerator()
    commandment_path = os.path.join(tmpdir, "COMMANDMENT.md")
    gen.save(
        output_path=commandment_path,
        kernel_dir=tmpdir,
        setup_commands=[
            f"export PYTHONPATH={tmpdir}:$PYTHONPATH",
        ],
        correctness_commands=[
            f"cd {tmpdir} && python3 test_correctness.py",
        ],
        profile_commands=[
            f"cd {tmpdir} && python3 profile_kernel.py",
        ],
    )
    print(f"  COMMANDMENT.md written to {commandment_path}")

    # Step 4: Run baseline evaluation
    print("[Step 4] Running baseline evaluation...")
    evaluator = CommandmentEvaluator(
        commandment_path=commandment_path,
        kernel_dir=tmpdir,
    )
    baseline_result = evaluator.evaluate(
        program_files=files,
        program_id="baseline_001",
        work_dir=tmpdir,
    )
    print(f"  Success: {baseline_result.success}")
    print(f"  Correctness: {baseline_result.correctness_passed}")
    print(f"  Metrics: {baseline_result.metrics}")
    assert baseline_result.success, f"Baseline evaluation failed: {baseline_result.error}"
    assert baseline_result.correctness_passed, "Baseline correctness failed"

    # Save baseline metrics
    baseline_metrics_path = os.path.join(tmpdir, "baseline_metrics.json")
    with open(baseline_metrics_path, "w") as f:
        json.dump(baseline_result.metrics, f)
    print(f"  Baseline metrics saved to {baseline_metrics_path}")

    # Step 5: Evaluate a "candidate" (the same kernel - should get speedup ~1.0)
    print("[Step 5] Evaluating candidate (same kernel)...")
    evaluator_with_baseline = CommandmentEvaluator(
        commandment_path=commandment_path,
        baseline_metrics_path=baseline_metrics_path,
        kernel_dir=tmpdir,
    )
    candidate_result = evaluator_with_baseline.evaluate(
        program_files=files,
        program_id="candidate_001",
        work_dir=tmpdir,
    )
    print(f"  Success: {candidate_result.success}")
    print(f"  Speedup: {candidate_result.speedup:.2f}x")
    print(f"  Metrics: {candidate_result.metrics}")
    assert candidate_result.success, f"Candidate evaluation failed: {candidate_result.error}"
    # Speedup should be close to 1.0 (same kernel)
    assert 0.5 < candidate_result.speedup < 2.0, \
        f"Speedup should be ~1.0 for same kernel, got {candidate_result.speedup}"

    print("\n" + "=" * 60)
    print("E2E COMMANDMENT evaluation test PASSED!")
    print(f"  Baseline latency: {baseline_result.metrics.get('latency_avg_us', 'N/A'):.2f} us")
    print(f"  Candidate speedup: {candidate_result.speedup:.2f}x")
    return True


if __name__ == "__main__":
    try:
        test_e2e_commandment_evaluation()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
