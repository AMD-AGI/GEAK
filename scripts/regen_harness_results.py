#!/usr/bin/env python3
"""Regenerate harness_results.json for the self-contained kernel ablation."""
import subprocess, json, time, os

PP = "/home/sdubagun/work/repos/GEAK/outputs/eval_3kernels/topk/preprocess_selfcontained"
KERNEL = PP + "/kernel.py"

modes = [
    ("correctness", ["--correctness"]),
    ("profile", ["--profile"]),
    ("benchmark", ["--benchmark"]),
    ("full-benchmark", ["--full-benchmark"]),
]

results = []
env = os.environ.copy()
env["HIP_VISIBLE_DEVICES"] = "0"

for mode_name, args in modes:
    print(f"Running --{mode_name}...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        ["python3", KERNEL] + args,
        capture_output=True, text=True, timeout=300,
        env=env,
    )
    elapsed = round(time.time() - t0, 1)
    ok = proc.returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'} ({elapsed}s)")
    results.append({
        "mode": mode_name,
        "success": ok,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[:500] if proc.stderr else "",
        "duration_s": elapsed,
    })

with open(PP + "/harness_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote harness_results.json ({len(results)} modes)")
