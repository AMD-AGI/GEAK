import subprocess
import sys
import os
import re

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
    "ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DBUILD_TEST=ON -DAMDGPU_TARGETS=gfx942 ../.",
    f"make -j {test_name}",
    f"./test/rocprim/{test_name}",
    # save git patch first
    f"make -j8 {bench_name}",
    f"./benchmark/{bench_name} --trials 20"
]

for cmd in commands:
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=build_dir)
    stdout_text = result.stdout.decode('utf-8', errors='ignore')
    if result.returncode != 0 or "FAIL" in stdout_text:
        print(f"fail: {cmd}")
        break
