#!/usr/bin/env python3
"""
Test harness for GEMM+Add CK kernel
Provides --correctness, --profile, and --benchmark modes for GEAK pipeline
"""

import argparse
import subprocess
import sys
import os
import json

# Fixed problem sizes for consistent profiling
PROBLEM_SIZES = {
    'small': {'M': 1024, 'N': 1024, 'K': 1024},
    'medium': {'M': 2048, 'N': 2048, 'K': 2048},
    'default': {'M': 3840, 'N': 4096, 'K': 4096},
}

# Use medium size for profiling (good balance of accuracy and speed)
PROFILE_SIZE = PROBLEM_SIZES['medium']

class GemmAddTester:
    def __init__(self, kernel_variant='xdl_fp16'):
        self.kernel_variant = kernel_variant
        self.repo_dir = os.path.dirname(os.path.abspath(__file__))
        self.binary_path = os.path.join(
            self.repo_dir, 
            'build', 
            f'example_gemm_add_{kernel_variant}'
        )
        
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(
                f"Binary not found: {self.binary_path}\n"
                f"Run ./build_and_test.sh first to build the kernels"
            )
    
    def run_kernel(self, verify=True, init_method=1, time_kernel=False, 
                   M=None, N=None, K=None):
        """Run the kernel with specified parameters"""
        # Use default sizes if not specified
        if M is None:
            size = PROFILE_SIZE
            M, N, K = size['M'], size['N'], size['K']
        
        # Strides match dimensions for row-major layout
        StrideA = K
        StrideB = N
        StrideD = N
        StrideE = N
        
        cmd = [
            self.binary_path,
            str(int(verify)),      # do_verification
            str(init_method),       # init_method
            str(int(time_kernel)),  # time_kernel
            str(M), str(N), str(K),
            str(StrideA), str(StrideB), str(StrideD), str(StrideE)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.repo_dir
        )
        
        return result
    
    def correctness(self):
        """Run correctness test"""
        print(f"Running correctness test for {self.kernel_variant}...")
        print(f"Problem size: M={PROFILE_SIZE['M']}, N={PROFILE_SIZE['N']}, K={PROFILE_SIZE['K']}")
        
        result = self.run_kernel(verify=True, time_kernel=False)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            print(f"✗ Correctness test FAILED", file=sys.stderr)
            return False
        
        print(f"✓ Correctness test PASSED")
        return True
    
    def profile(self):
        """Run kernel once for profiling (minimal overhead)"""
        # Run with timing disabled but verification enabled
        # The profiler will capture the kernel execution
        result = self.run_kernel(verify=True, time_kernel=False)
        
        # Silent mode for profiler - just run the kernel
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
    
    def benchmark(self):
        """Run performance benchmark"""
        print(f"Benchmarking {self.kernel_variant}...")
        print(f"Problem size: M={PROFILE_SIZE['M']}, N={PROFILE_SIZE['N']}, K={PROFILE_SIZE['K']}")
        
        result = self.run_kernel(verify=True, time_kernel=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            print(f"✗ Benchmark FAILED", file=sys.stderr)
            return False
        
        # Parse performance from output
        for line in result.stdout.split('\n'):
            if 'Perf:' in line:
                print(f"\n{line}")
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='Test harness for GEMM+Add CK kernel'
    )
    parser.add_argument(
        '--correctness', 
        action='store_true',
        help='Run correctness verification'
    )
    parser.add_argument(
        '--profile', 
        action='store_true',
        help='Run kernel for profiling (single execution)'
    )
    parser.add_argument(
        '--benchmark', 
        action='store_true',
        help='Run performance benchmark'
    )
    parser.add_argument(
        '--variant',
        default='xdl_fp16',
        choices=['xdl_fp16', 'xdl_bf16', 'wmma_fp16', 'wmma_bf16'],
        help='Kernel variant to test'
    )
    
    args = parser.parse_args()
    
    # Default to benchmark if no mode specified
    if not (args.correctness or args.profile or args.benchmark):
        args.benchmark = True
    
    try:
        tester = GemmAddTester(kernel_variant=args.variant)
        
        if args.correctness:
            success = tester.correctness()
            sys.exit(0 if success else 1)
        
        elif args.profile:
            tester.profile()
            sys.exit(0)
        
        elif args.benchmark:
            success = tester.benchmark()
            sys.exit(0 if success else 1)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
