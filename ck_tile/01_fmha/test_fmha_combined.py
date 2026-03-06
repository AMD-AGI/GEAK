#!/usr/bin/env python3
"""
Combined test and benchmark script for FMHA kernel.
Tests both correctness and performance.
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")
        return False
    print(f"SUCCESS")
    return True

def main():
    os.chdir("/workspace/GEAK/ck_tile/01_fmha")
    
    # Build step
    build_cmd = (
        "hipcc -I/opt/rocm/include -I/workspace/GEAK/ck_tile "
        "-std=c++17 --offload-arch=gfx942 -O3 "
        "example_fmha_fwd.cpp -o /tmp/example_fmha_fwd"
    )
    
    if not run_command(build_cmd, "Building FMHA Forward Example"):
        print("\nBuild failed. Exiting.")
        return 1
    
    # Correctness test - small config
    correctness_cmd = (
        "/tmp/example_fmha_fwd "
        "-mode=0 -b=1 -h=2 -s=128 -d=64 "
        "-v=1 -prec=fp16 -repeat=5"
    )
    
    if not run_command(correctness_cmd, "Correctness Test (small config)"):
        print("\nCorrectness test failed. Exiting.")
        return 1
    
    # Benchmark test - standard config
    benchmark_cmd = (
        "/tmp/example_fmha_fwd "
        "-mode=0 -b=2 -h=8 -s=512 -d=128 "
        "-v=0 -prec=fp16 -repeat=20"
    )
    
    if not run_command(benchmark_cmd, "Benchmark Test (standard config)"):
        print("\nBenchmark test failed. Exiting.")
        return 1
    
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")
    print(f"{'='*60}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
