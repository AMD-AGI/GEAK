# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
GEAK Agent Test Suite Runner

Runs the full pipeline on selected AITER kernels to measure agent capabilities.
Results are saved to eval_suite/results/ (gitignored).
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
MSA_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIG_FILE = SCRIPT_DIR / "config.json"


def load_config():
    """Load test suite configuration."""
    with open(CONFIG_FILE) as f:
        return json.load(f)


def ensure_aiter_cloned(aiter_path: Path, repo_url: str):
    """Clone AITER if not present."""
    if not aiter_path.exists():
        print(f"Cloning AITER to {aiter_path}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(aiter_path)],
            check=True
        )
    return aiter_path


def run_pipeline(kernel_path: str, kernel_id: str, max_iterations: int = 10) -> dict:
    """Run the full GEAK agent pipeline on a kernel."""
    result = {
        "kernel_id": kernel_id,
        "kernel_path": kernel_path,
        "start_time": datetime.now().isoformat(),
        "discovery_success": False,
        "test_generation_success": False,
        "benchmark_baseline_created": False,
        "optimization_completed": False,
        "speedup_ratio": None,
        "error": None,
        "duration_seconds": 0
    }
    
    start = time.time()
    
    task = f"""Complete GEAK Agent Pipeline for {kernel_path}

1. DISCOVER: Analyze the kernel
2. TEST GENERATION: Create test cases
3. BENCHMARKING: Save baseline metrics
4. OPTIMIZATION: Use OpenEvolve MCP with max_iterations={max_iterations}
5. Save optimized kernel and metrics"""
    
    cmd = [
        "python3", "-m", "minisweagent.run.mini",
        "-m", "claude-sonnet-4.5",
        "-t", task,
        "--yolo"
    ]
    
    # Set up environment with correct PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MSA_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(MSA_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout per kernel
        )
        
        output = proc.stdout + proc.stderr
        
        # Parse results from output
        if "DISCOVER" in output and "kernel" in output.lower():
            result["discovery_success"] = True
        if "test" in output.lower() and ("created" in output.lower() or "generated" in output.lower()):
            result["test_generation_success"] = True
        if "baseline" in output.lower() and "metrics" in output.lower():
            result["benchmark_baseline_created"] = True
        if "optimiz" in output.lower() and ("complete" in output.lower() or "success" in output.lower()):
            result["optimization_completed"] = True
            
        # Try to extract speedup
        import re
        speedup_match = re.search(r"speedup[:\s]+(\d+\.?\d*)", output, re.IGNORECASE)
        if speedup_match:
            result["speedup_ratio"] = float(speedup_match.group(1))
            
        # Check for errors in output
        if proc.returncode != 0:
            result["error"] = f"Exit code {proc.returncode}: {proc.stderr[:500] if proc.stderr else 'No stderr'}"
            
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (30 min)"
    except Exception as e:
        result["error"] = str(e)
    
    result["duration_seconds"] = round(time.time() - start, 2)
    result["end_time"] = datetime.now().isoformat()
    
    return result


def run_suite(kernels_to_run: list = None, max_iterations: int = 10):
    """Run the test suite."""
    config = load_config()
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Clone AITER inside MSA_ROOT so it's accessible in Docker
    # (Docker mounts MSA_ROOT as /workspace)
    aiter_path = MSA_ROOT / "aiter"
    ensure_aiter_cloned(aiter_path, config["aiter_repo"])
    
    # Select kernels
    kernels = config["kernels"]
    if kernels_to_run:
        kernels = [k for k in kernels if k["id"] in kernels_to_run]
    
    print(f"\n{'='*60}")
    print(f"GEAK Agent Test Suite - {len(kernels)} kernels")
    print(f"{'='*60}\n")
    
    results = {
        "suite_name": config["name"],
        "run_date": datetime.now().isoformat(),
        "total_kernels": len(kernels),
        "max_iterations": max_iterations,
        "kernel_results": []
    }
    
    for i, kernel in enumerate(kernels, 1):
        # Use relative path (works in Docker where MSA_ROOT is mounted as /workspace)
        kernel_rel_path = f"aiter/{kernel['path']}"
        str(aiter_path / kernel["path"])
        print(f"\n[{i}/{len(kernels)}] {kernel['id']} ({kernel['category']})")
        print(f"    Path: {kernel_rel_path}")
        print(f"    {kernel['description']}")
        print("-" * 40)
        
        result = run_pipeline(kernel_rel_path, kernel["id"], max_iterations)
        results["kernel_results"].append(result)
        
        # Print status
        status = []
        if result["discovery_success"]:
            status.append("Discovery")
        if result["test_generation_success"]:
            status.append("Tests")
        if result["benchmark_baseline_created"]:
            status.append("Benchmark")
        if result["optimization_completed"]:
            status.append("Optimized")
        
        print(f"    Status: {', '.join(status) if status else 'Failed'}")
        print(f"    Duration: {result['duration_seconds']}s")
        if result["error"]:
            print(f"    Error: {result['error']}")
        if result["speedup_ratio"]:
            print(f"    Speedup: {result['speedup_ratio']}x")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_counts = {
        "discovery": sum(1 for r in results["kernel_results"] if r["discovery_success"]),
        "test_gen": sum(1 for r in results["kernel_results"] if r["test_generation_success"]),
        "benchmark": sum(1 for r in results["kernel_results"] if r["benchmark_baseline_created"]),
        "optimization": sum(1 for r in results["kernel_results"] if r["optimization_completed"])
    }
    
    total = len(kernels)
    print(f"Discovery:      {success_counts['discovery']}/{total} ({100*success_counts['discovery']/total:.0f}%)")
    print(f"Test Gen:       {success_counts['test_gen']}/{total} ({100*success_counts['test_gen']/total:.0f}%)")
    print(f"Benchmark:      {success_counts['benchmark']}/{total} ({100*success_counts['benchmark']/total:.0f}%)")
    print(f"Optimization:   {success_counts['optimization']}/{total} ({100*success_counts['optimization']/total:.0f}%)")
    
    results["summary"] = success_counts
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"run_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GEAK Agent Test Suite")
    parser.add_argument("--kernels", nargs="+", help="Specific kernel IDs to run")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max optimization iterations")
    parser.add_argument("--list", action="store_true", help="List available kernels")
    
    args = parser.parse_args()
    
    if args.list:
        config = load_config()
        print("\nAvailable kernels:")
        for k in config["kernels"]:
            print(f"  {k['id']:30} - {k['description']}")
        return
    
    run_suite(args.kernels, args.max_iterations)


if __name__ == "__main__":
    main()
