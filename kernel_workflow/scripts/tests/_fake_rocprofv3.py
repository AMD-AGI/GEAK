# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fake rocprofv3 + ATT decoder fixtures for the kernel_tools profiling tests (CPU-only)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]

FAKE = r"""#!/usr/bin/env bash
case "${1:-}" in
  --version) echo 'fixture sdk'; exit 0;;
  --help) echo '--att --att-library-path --kernel-trace --pmc'; exit 0;;
esac
out=""; mode=""; pmc=()
while [ $# -gt 0 ]; do
  case "$1" in
    -d) out="$2"; shift 2;;
    --kernel-trace) mode=trace; shift;;
    --att) mode=att; shift;;
    --pmc) mode=pmc; shift; while [ $# -gt 0 ] && [[ "$1" != -* ]]; do pmc+=("$1"); shift; done;;
    --) shift; break;;
    *) shift;;
  esac
done
mkdir -p "$out"
case "$mode" in
  trace) printf 'Kernel_Name,Start_Timestamp,End_Timestamp\n_gemm_fwd_kernel,0,5000\nat::native::elementwise_kernel,0,10\n' > "$out/r_kernel_trace.csv";;
  pmc) for c in "${pmc[@]}"; do [ "$c" = BAD_COUNTER ] && exit 134; done
       { echo "Dispatch_ID,Kernel_Name,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp"
         for c in "${pmc[@]}"; do echo "1,_gemm_fwd_kernel,$c,50,1000,6000"; done; } > "$out/r_counter_collection.csv";;
  att) mkdir -p "$out/ui_output_agent_1_dispatch_1"
       printf 'Kernel_Name,Dispatch_Id,VGPR_Count,Accum_VGPR_Count,SGPR_Count,LDS_Block_Size,Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z\n_gemm_fwd_kernel,1,128,0,32,65536,256,1,1\n' > "$out/r_kernel_trace.csv"
       python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import _att_fixture; _att_fixture.write(sys.argv[2])" \
         "$KT_DIR_FOR_FAKE" "$out/ui_output_agent_1_dispatch_1";;
esac
"$@"
"""


def _decoder(directory):
    compiler = shutil.which('cc')
    if not compiler:
        pytest.skip('C compiler needed to exercise the decoder loader with a synthetic V1 library')
    subprocess.run([compiler, '-shared', '-fPIC', '-x', 'c', '-', '-o',
                    str(directory / 'librocprof-trace-decoder.so')],
                   input='void rocprof_trace_decoder_parse_data(void) {}', text=True, check=True)


def _env(tmp_path, arch="gfx950", **extra):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "rocprofv3").write_text(FAKE)
    (bindir / "rocprofv3").chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", PYTORCH_ROCM_ARCH=arch,
               GEAK_GPU_LOCK_DIR=str(tmp_path / "locks"), GEAK_GPU_REQUIRE_IDLE="0",
               KERNEL_ENV_KEEP_ARCH="1", KERNEL_ENV_SKIP_ENUM_REAP="1", WARMUP_RUNS="0",
               PROFILER_PRIORITY="rocprofv3", KT_ATT_SYSTEM_DIRS=str(tmp_path / "none"),
               KT_DIR_FOR_FAKE=str(SCRIPTS / "kernel_tools"))
    for k in ("GEAK_GPU_LOCK_HELD", "GEAK_GPU_BROKER", "ROCPROF_ATT_LIBRARY_PATH", "PROFILE_PMC_GROUPS",
              "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(k, None)
    env.update(extra)
    return env
