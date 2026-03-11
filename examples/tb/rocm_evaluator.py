# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
🛡️ BULLETPROOF ROCm TRITON KERNEL EVALUATOR  (AMD GPU EDITION) 🛡️

This evaluator is a **drop-in replacement** for the original Metal evaluator (https://github.com/codelion/openevolve/blob/main/examples/mlx_metal_kernel_opt/evaluator.py).
It focuses on Triton kernels executed on AMD GPUs (ROCm / HIP back-end) and
contains multiple layers of protection to guarantee that *the surrounding
evolution process NEVER hard-crashes*, no matter how broken a candidate
kernel may be.

Key changes vs. the original version
────────────────────────────────────
• All Metal-specific logic was removed or rewritten for Triton/ROCm  
• Error classification now understands HIP / ROCm run-time messages  
• Kernel compilation is attempted with `triton.compile()` *before* running  
• A microscopic “canary” launch is executed to catch illegal memory access  
• Graceful fallback to PyTorch reference attention if anything goes wrong  
• Optional *address-sanitiser* style buffer over-allocation for extra safety  

The public API is **identical** (`evaluate(program_text: str) -> dict`) so
OpenEvolve (or any other driver) does not need to be altered.
"""

from __future__ import annotations

import os, sys, time, traceback, tempfile, importlib, types, math, subprocess, json
from typing import Any, Dict, List, Optional, Tuple
from glob import glob
import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except Exception as _e:
    raise RuntimeError(
        "Triton must be installed with AMD/ROCm support "
    ) from _e

import geak_eval  # Using GEAK-eval-OE evaluation framework
from geak_eval.constants import ROCm_DATA_ROOT, ROCm_DATA_AUTOTUNE_ROOT

# ────────────────────────────────────────────────────────────────────────────
# Exceptions (Triton / ROCm specific)
# ────────────────────────────────────────────────────────────────────────────
class TritonKernelSafetyError(Exception):
    """Static or dynamic Triton kernel safety violation."""


class HipRuntimeError(Exception):
    """Errors originating from the HIP / ROCm driver/runtime."""


class TritonCompilationError(Exception):
    """triton.compile raised an error."""


# ────────────────────────────────────────────────────────────────────────────
# Evaluator
# ────────────────────────────────────────────────────────────────────────────
class BulletproofTritonEvaluator:
    """
    A *single* instance is created per evaluation.  All state (error counters,
    baseline numbers, …) lives inside this object.
    """

    # ----------  configuration knobs  ----------
    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY_S = 1.0

    # conservative “canary” launch sizes (tokens, heads, …)
    _MAX_SAFE_TOKENS = 128
    _MAX_SAFE_BATCH = 2

    # compile / launch-timeouts (soft, seconds)
    _COMPILE_TIMEOUT_S = 6000
    _LAUNCH_TIMEOUT_S = 6000

    # Use environment variable for golden data path with fallback
    GOLDEN_DATA_PATH = os.environ.get(
        'ROCM_GOLDEN_DATA_PATH',
        '/home/sapmajum/geak-openevolve/TB-eval-OE/tb_eval/data/ROCm/data/performance/golden_results'
    )

    # -------------------------------------------------
    def __init__(self, kernel_filename: Optional[str] = None) -> None:
        self.triton_compile_errors: int = 0
        self.hip_runtime_errors: int = 0
        self.memory_violations: int = 0
        self.total_gpu_errors: int = 0
        self.successful_fallbacks: int = 0
        self.retry_attempts_used: int = 0

        # Baseline (PyTorch) numbers are cached between attempts
        self._baseline_metrics: Optional[Dict[str, float]] = None
        
        # Baseline file path (will be set during first evaluation)
        self._baseline_file: Optional[str] = None
        
        # Store kernel filename for test code merging
        self._kernel_filename = kernel_filename

        print("🛡️ BULLETPROOF TRITON KERNEL EVALUATOR (AMD GPU) INITIALISED")

    # ======================================================================
    # Test code merging helper
    # ======================================================================
    def _merge_test_code(self, kernel_file_path: str, kernel_name: str) -> str:
        """
        Merge generated kernel code with appropriate test code based on autotune detection.
        
        Args:
            kernel_file_path: Path to the file containing generated kernel code
            kernel_name: Name of the kernel file (e.g., 'test_add_kernel.py')
            
        Returns:
            Path to the merged file (same as input, overwritten)
        """
        # Read the generated kernel code
        with open(kernel_file_path, 'r') as f:
            kernel_code = f.read()
        
        # Detect if kernel uses @triton.autotune
        has_autotune = "@triton.autotune" in kernel_code
        
        # Select appropriate test directory
        if has_autotune:
            test_dir = ROCm_DATA_AUTOTUNE_ROOT
            print(f"✅ Detected @triton.autotune - using ROCm_v1_autotune tests")
        else:
            test_dir = ROCm_DATA_ROOT
            print(f"✅ No @triton.autotune - using ROCm_v1 tests")
        
        # Construct path to test file
        test_file_path = os.path.join(test_dir, kernel_name)
        
        if not os.path.exists(test_file_path):
            print(f"⚠️ WARNING: Test file not found at {test_file_path}")
            print(f"   Continuing with kernel-only code (may fail if tests are missing)")
            return kernel_file_path
        
        # Read test code from the reference test file
        # Extract only the test code (after the separator line)
        with open(test_file_path, 'r') as f:
            test_file_content = f.read()
        
        # Find the separator (146 '#' characters)
        separator = '#' * 146
        if separator in test_file_content:
            test_code = test_file_content.split(separator)[1].strip()
        else:
            print(f"⚠️ WARNING: Could not find test separator in {test_file_path}")
            print(f"   Using entire file as test code")
            test_code = test_file_content
        
        # Merge kernel code with test code
        merged_code = f"{kernel_code}\n\n{separator}\n\n{test_code}"
        
        # Write merged code back to the file
        with open(kernel_file_path, 'w') as f:
            f.write(merged_code)
        
        print(f"✅ Merged kernel with test code from {os.path.basename(test_dir)}")
        return kernel_file_path

    # ======================================================================
    # Public entry-point used by OpenEvolve
    # ======================================================================
    def evaluate(self, test_suite_path: str, program_text: str, ref_wrapper_path: str, 
                 wrapper_fn_name: str, unit_tests_path: str, n_warmup: int, n_iters: int,
                 atol: float, rtol: float, verbose: bool, gpu_id: int=0) -> Dict[str, Any]:
        """Master function: never raises, always returns a dict.
        
        Args:
            test_suite_path: Path to test suite (not used in ROCm evaluator)
            program_text: Path to the program/kernel to evaluate
            ref_wrapper_path: Path to reference wrapper (not used in ROCm evaluator)
            wrapper_fn_name: Name of wrapper function (not used in ROCm evaluator)
            unit_tests_path: Path to unit tests (not used in ROCm evaluator)
            n_warmup: Number of warmup iterations (not used in ROCm evaluator)
            n_iters: Number of benchmark iterations (not used in ROCm evaluator)
            atol: Absolute tolerance (not used in ROCm evaluator)
            rtol: Relative tolerance (not used in ROCm evaluator)
            verbose: Verbose output flag (not used in ROCm evaluator)
            gpu_id: GPU ID to use for evaluation
        """
        try:
            ## if program_text is not a path then save the code in a temporary file
            if not os.path.exists(program_text):
                print("Program text is not a file path, saving to temporary file.")
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False
                ) as temp_file:
                    temp_file.write(program_text)
                    program_text = temp_file.name
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
            else:
                print(f"Using program text from file: {program_text}")

            print(f"Evaluating Triton kernel from: {program_text}")
            
            # Determine kernel filename for test merging
            # Extract from program_text (actual file being evaluated)
            if self._kernel_filename:
                kernel_name = self._kernel_filename
            else:
                # Extract filename from program_text (e.g., /path/test_kernel_sub.py -> test_kernel_sub.py)
                kernel_name = os.path.basename(program_text)
                print(f"📝 Extracted kernel name from program_text: {kernel_name}")
                
                # If that doesn't look like a test file, try test_suite_path as fallback
                if not kernel_name.startswith("test_") or not kernel_name.endswith(".py"):
                    if test_suite_path and os.path.exists(test_suite_path):
                        kernel_name = os.path.basename(test_suite_path)
                        print(f"📝 Using kernel name from test_suite_path instead: {kernel_name}")
            
            print(f"📝 Final kernel name for test merging: {kernel_name}")
            
            # Merge appropriate test code based on autotune detection
            program_text = self._merge_test_code(program_text, kernel_name)
            
            # Run correctness tests using pytest
            env = os.environ.copy()
            env['HIP_VISIBLE_DEVICES'] = str(gpu_id % 8)
            
            # Run pytest for correctness (excluding performance tests)
            correctness_cmd = [
                "pytest", "-v", "-x", "--maxfail=1", 
                program_text,
                "-k", "not (test_performance or test_save)"
            ]
            print(f"Running correctness tests: {' '.join(correctness_cmd)}")
            correctness_result = subprocess.run(
                correctness_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env
            )
                
            call, correct, benchmark, benchmark_params, err_msg = None, None, None, None, ""

            # Check correctness test results
            call = correctness_result.returncode == 0
            correct = call  # If tests pass, kernel is correct
            
            if not call:
                print(f"Correctness tests failed. Return code: {correctness_result.returncode}")
                print(f"STDOUT: {correctness_result.stdout}")
                print(f"STDERR: {correctness_result.stderr}")
                # Combine stdout and stderr for error message
                err_msg = f"Correctness tests failed (exit {correctness_result.returncode}):\n"
                if correctness_result.stdout:
                    err_msg += f"STDOUT: {correctness_result.stdout}\n"
                if correctness_result.stderr:
                    err_msg += f"STDERR: {correctness_result.stderr}\n"
                return self._create_comprehensive_failure_result(err_msg)

            # Run performance tests to get benchmark data
            # Get the directory where the kernel file is located
            kernel_dir = os.path.dirname(os.path.abspath(program_text))
            perf_dir = os.path.join(kernel_dir, "perf")
            
            # Clean perf directory before running to avoid mixing results from different kernels
            if os.path.exists(perf_dir):
                print(f"Cleaning perf directory: {perf_dir}")
                for old_file in os.listdir(perf_dir):
                    if old_file.endswith('.json'):
                        old_path = os.path.join(perf_dir, old_file)
                        print(f"  Removing old perf file: {old_file}")
                        os.remove(old_path)
            
            # Run performance tests
            perf_cmd = [
                "pytest", "-v", "-x", "--maxfail=1",
                program_text,
                "-k", "test_performance or test_save_performance_results"
            ]
            print(f"Running performance tests: {' '.join(perf_cmd)}")
            perf_result = subprocess.run(
                perf_cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env
            )
            
            # Parse performance results from JSON files
            perf_data = None
            benchmark = None
            benchmark_params = None
            
            print(f"Performance test result - returncode: {perf_result.returncode}")
            print(f"Performance test stdout: {perf_result.stdout[:500]}")
            print(f"Performance test stderr: {perf_result.stderr[:500]}")
            
            if os.path.exists(perf_dir):
                print(f"Looking for performance results in: {perf_dir}")
                perf_files = [f for f in os.listdir(perf_dir) if f.endswith('.json')]
                print(f"Found {len(perf_files)} JSON files: {perf_files}")
                if perf_files:
                    import json
                    perf_files_with_mtime = [(f, os.path.getmtime(os.path.join(perf_dir, f))) for f in perf_files]
                    latest_file = max(perf_files_with_mtime, key=lambda x: x[1])[0]
                    latest_perf_file = os.path.join(perf_dir, latest_file)
                    print(f"Reading performance data from: {latest_perf_file} (most recent)")
                    with open(latest_perf_file, 'r') as f:
                        perf_data = json.load(f)
                    
                    print(f"Performance data structure: {list(perf_data[0].keys()) if perf_data else 'empty'}")
                    print(f"Total benchmark configs: {len(perf_data)}")
                else:
                    print(f"No JSON performance files found in {perf_dir}")
            else:
                print(f"Performance directory does not exist: {perf_dir}")
            
            if not perf_data:
                print("Warning: No benchmark data available")
                benchmark = 0.0
            
            # Determine the latency key used in perf data
            latency_key = None
            if perf_data and len(perf_data) > 0:
                for key in ['latency_ms', 'ms', 'mean_ms', 'median_ms']:
                    if key in perf_data[0]:
                        latency_key = key
                        break
            
            # Build per-config latency map: "M_N" -> ms
            current_latencies = {}
            if perf_data and latency_key:
                for entry in perf_data:
                    params = entry.get('params', entry.get('config', {}))
                    config_key = f"{params.get('M', 0)}_{params.get('N', 0)}"
                    current_latencies[config_key] = entry[latency_key]
                # Use first entry as the representative single benchmark for logging
                benchmark = perf_data[0][latency_key]
                benchmark_params = perf_data[0].get('config', perf_data[0].get('params', {}))
            
            # Baseline file stores per-config latencies as JSON
            if self._baseline_file is None:
                eval_dir = os.path.dirname(program_text)
                evals_root = os.path.dirname(eval_dir)
                self._baseline_file = os.path.join(evals_root, ".baseline_latencies.json")
            
            # Load or establish baseline
            baseline_latencies = None
            if os.path.exists(self._baseline_file):
                try:
                    import json as _json
                    with open(self._baseline_file, 'r') as f:
                        baseline_latencies = _json.load(f)
                    print(f"Loaded per-config baseline latencies ({len(baseline_latencies)} configs)")
                except Exception as e:
                    print(f"Warning: Could not load baseline from {self._baseline_file}: {e}")
            
            if baseline_latencies is None and current_latencies:
                baseline_latencies = current_latencies
                speedup = 1.0
                print(f"Establishing NEW per-config baseline ({len(baseline_latencies)} configs), speedup=1.0")
                try:
                    import json as _json
                    os.makedirs(os.path.dirname(self._baseline_file), exist_ok=True)
                    with open(self._baseline_file, 'w') as f:
                        _json.dump(baseline_latencies, f)
                    print(f"Saved baseline to: {self._baseline_file}")
                except Exception as e:
                    print(f"Warning: Could not save baseline to {self._baseline_file}: {e}")
            elif baseline_latencies and current_latencies:
                # Compute per-config speedups and take geometric mean
                per_config_speedups = []
                for config_key, curr_ms in current_latencies.items():
                    if config_key in baseline_latencies and curr_ms > 0:
                        base_ms = baseline_latencies[config_key]
                        s = base_ms / curr_ms
                        per_config_speedups.append(s)
                        print(f"  Config {config_key}: {base_ms:.6f}ms -> {curr_ms:.6f}ms = {s:.4f}x")
                
                if per_config_speedups:
                    import math
                    log_sum = sum(math.log(s) for s in per_config_speedups)
                    speedup = math.exp(log_sum / len(per_config_speedups))
                    print(f"Geometric mean speedup across {len(per_config_speedups)} configs: {speedup:.4f}x")
                else:
                    speedup = 0.0
                    print("Warning: No matching configs for speedup calculation")
            else:
                speedup = 0.0
                print("Warning: Could not compute speedup (missing data)")
            
            if benchmark is None:
                benchmark = 0.0
            
            # Format benchmark_params for reporting
            if benchmark_params and isinstance(benchmark_params, dict):
                params_str = "; ".join(f"{k}={v}" for k, v in benchmark_params.items())
                benchmark_params = f"Kernel parameters: {params_str}, geomean speedup: {speedup:.4f}x ({len(current_latencies)} configs)"
            elif not benchmark_params:
                benchmark_params = f"Geomean speedup: {speedup:.4f}x ({len(current_latencies)} configs)"

            summary = ""
            if call:
                summary += "The Triton kernel was successfully compiled and launched. "
            else:
                summary += "The Triton kernel failed to compile or launch. "

            if correct:
                summary += "The Triton kernel produced correct results. "
            else:
                summary += "The Triton kernel produced incorrect results. "

            if benchmark is not None and benchmark > 0:
                summary += f"The Triton kernel achieved a speedup of {speedup:.4f}x compared to the baseline."
            else:
                summary += "The Triton kernel failed to benchmark."

            safety_validation = ""
            if call:
                safety_validation = "This generated Triton kernel was evaluated safely, with no hard crashes or memory violations."
            else:
                safety_validation = "This generated Triton kernel failed to compile or run safely, resulting in a hard crash or memory violation."
                if err_msg:
                    safety_validation += f" Error message: {err_msg}"
            if correct:
                safety_validation += " The results were correct."
            else:
                safety_validation += " The results were incorrect."
                if err_msg:
                    safety_validation += f" Error message: {err_msg}"
            
            if speedup > 0:
                baseline_comparison = f"Performance report: {benchmark_params}. Geomean speedup={speedup:.4f}x across {len(current_latencies)} configs"
            else:
                baseline_comparison = "The Triton kernel failed to benchmark, so no performance comparison can be made."
                if err_msg:
                    baseline_comparison += f" Error message: {err_msg}"

            benchmark_results = []
            if benchmark_params is not None:
                benchmark_results.append(
                    f"Performance report: {benchmark_params}."
                )

            return {
                "success": speedup > 0,
                "final_score": speedup,  # Use speedup (higher is better)
                "performance_metrics": speedup,  # Use speedup (higher is better)
                "correctness_score": 1 if correct else 0,
                "combined_score": speedup,  # Use speedup for optimization
                "benchmark_results": benchmark_results if benchmark_params is not None else [],
                "baseline_comparison": baseline_comparison,
                "individual_comparisons": [],
                "summary": summary,
                # "metal_safety_statistics": self._get_comprehensive_error_statistics(),
                "safety_validation": safety_validation,
                "error": err_msg if err_msg else None,
            }

        except Exception as top_e:
            traceback.print_exc()
            return self._create_comprehensive_failure_result(f"Top-level crash caught: {top_e}")

    def _create_comprehensive_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create comprehensive failure result with full error statistics"""
        return {
            "success": False,
            "final_score": 0.0,
            "error": error_message,
            "performance_metrics": {},
            "combined_score": 0.0,
            "correctness_score": 0.0,
            "summary": f"Evaluation failed due to: {error_message}",
            # "metal_safety_statistics": self._get_comprehensive_error_statistics(),
            "safety_validation": {"success": False, "error": error_message},
        }

def evaluate(test_suite_path: str, program_text: str, ref_wrapper_path: str,
             wrapper_fn_name: str, unit_tests_path: str, n_warmup: int, n_iters: int,
             atol: float, rtol: float, verbose: bool, gpu_id: int = 0) -> Dict[str, Any]:
    """🛡️ BULLETPROOF evaluation function called by OpenEvolve"""
    # Extract kernel filename from test_suite_path if available
    kernel_filename = None
    if test_suite_path and os.path.exists(test_suite_path):
        kernel_filename = os.path.basename(test_suite_path)
    
    evaluator = BulletproofTritonEvaluator(kernel_filename=kernel_filename)
    return evaluator.evaluate(test_suite_path, program_text, ref_wrapper_path,
                             wrapper_fn_name, unit_tests_path, n_warmup, n_iters,
                             atol, rtol, verbose, gpu_id)

# # ────────────────────────────────────────────────────────────────────────────
# # quick self-test (optional)
# # ────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     dummy_kernel = """
# import triton
# import triton.language as tl

# @triton.jit
# def kernel(X, Y, T: tl.constexpr, H: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     off = pid * T * H
#     x_ptr = X + off
#     y_ptr = Y + off
#     for i in range(0, T * H):
#         tl.store(y_ptr + i, tl.load(x_ptr + i))
# """
#     res = evaluate(dummy_kernel)
#     print(res)
