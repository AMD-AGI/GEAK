#!/usr/bin/env python3
"""
Test harness for CK-tile elementwise kernel
Provides --correctness, --profile, and --benchmark modes
"""
import subprocess
import sys
import argparse

def run_test(mode, m=4096, n=4096):
    """Run the elementwise test with specified mode"""
    test_bin = "/workspace/GEAK/ck_tile/21_elementwise/test_isolated/build/elementwise_test"
    
    if mode == "correctness":
        # Run with validation
        result = subprocess.run([test_bin, str(m), str(n), "1", "5", "10"], 
                              capture_output=True, text=True)
        if "PASSED" in result.stdout:
            print("Correctness check: PASSED")
            return 0
        else:
            print("Correctness check: FAILED")
            print(result.stdout)
            print(result.stderr)
            return 1
            
    elif mode == "profile":
        # Run for profiling (minimal output)
        result = subprocess.run([test_bin, str(m), str(n), "0", "0", "1"], 
                              capture_output=False, text=True)
        return result.returncode
        
    elif mode == "benchmark":
        # Run full benchmark
        result = subprocess.run([test_bin, str(m), str(n), "1", "10", "100"], 
                              capture_output=False, text=True)
        return result.returncode
        
    return 1

def main():
    parser = argparse.ArgumentParser(description="Test harness for CK-tile elementwise kernel")
    parser.add_argument("--correctness", action="store_true", help="Run correctness check")
    parser.add_argument("--profile", action="store_true", help="Run for profiling")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--m", type=int, default=4096, help="M dimension")
    parser.add_argument("--n", type=int, default=4096, help="N dimension")
    
    args = parser.parse_args()
    
    if args.correctness:
        return run_test("correctness", args.m, args.n)
    elif args.profile:
        return run_test("profile", args.m, args.n)
    elif args.benchmark:
        return run_test("benchmark", args.m, args.n)
    else:
        # Default: run all
        print("Running correctness check...")
        ret = run_test("correctness", args.m, args.n)
        if ret != 0:
            return ret
        print("\nRunning benchmark...")
        return run_test("benchmark", args.m, args.n)

if __name__ == "__main__":
    sys.exit(main())
