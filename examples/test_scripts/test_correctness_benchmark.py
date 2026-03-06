# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""
Test correctness benchmark script.

NOTE: ParallelAgent automatically captures all output to stdout/stderr using tee.
If you use subprocess.PIPE to capture output internally, you MUST print it to stdout/stderr
afterwards so ParallelAgent can capture it. This script demonstrates the pattern:
1. Capture output with subprocess.PIPE for local checking
2. Print the captured output to stdout/stderr so ParallelAgent can capture it

If you create similar test scripts:
- If using subprocess.PIPE, always print the captured output to stdout/stderr
- Use flush=True for print statements to ensure immediate output
- ParallelAgent will automatically capture all output via tee
"""
import os
import subprocess
import sys

if len(sys.argv) < 3:
    print("Usage: python test_correctness_benchmark.py <bench_name> <workdir>")
    print("Example: python test_correctness_benchmark.py benchmark_device_merge_sort WORK_REPO")
    sys.exit(1)

bench_name = sys.argv[1]
test_name = bench_name.replace("benchmark", "test")
workdir = sys.argv[2]

if not os.path.exists(workdir):
    os.makedirs(workdir)

build_dir = os.path.join(workdir, "build")
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

commands = [
    # Newer CMake versions error out when dependencies use very old
    # cmake_minimum_required(<3.5). This policy minimum unblocks googletest.
    "ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DBUILD_TEST=ON "
    "-DAMDGPU_TARGETS=gfx942 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ../.",
    f"make -j {test_name}",
    f"./test/rocprim/{test_name}",
    # save git patch first
    f"make -j8 {bench_name}",
    f"./benchmark/{bench_name} --trials 20"
]

for cmd in commands:
    print(f"running: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=build_dir)
    stdout_text = result.stdout.decode('utf-8', errors='ignore')
    stderr_text = result.stderr.decode('utf-8', errors='ignore')
    # Print output to stdout/stderr so ParallelAgent can capture it via tee
    if stdout_text:
        print(stdout_text, flush=True)
    if result.returncode != 0 or "FAIL" in stdout_text:
        print(f"fail: {cmd}", flush=True)
        sys.exit(result.returncode if result.returncode != 0 else 1)