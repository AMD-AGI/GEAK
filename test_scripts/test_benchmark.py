import subprocess
import sys
import os
import re

if len(sys.argv) < 3:
    print("Usage: python test_correctness_benchmark.py <bench_name> <workdir>")
    print("Example: python test_correctness_benchmark.py benchmark_device_merge_sort WROR_REPO")
    sys.exit(1)

bench_name = sys.argv[1]
workdir = sys.argv[2]

commands = [
    "ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DBUILD_TEST=ON -DAMDGPU_TARGETS=gfx942 ../.",
    # save git patch first
    f"make -j8 {bench_name}",
    f"./benchmark/{bench_name} --trials 20"
]

if not os.path.exists(workdir):
    os.makedirs(workdir)

build_dir = os.path.join(workdir, "build")
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

for cmd in commands:
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=build_dir)
    if result.returncode != 0:
        print(f"fail: {cmd}")
        break
