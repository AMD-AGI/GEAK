# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""profile_kernel.sh optional layers (--pmc / --derived / --att / --spi) with a fake rocprofv3.

CPU-only. The default run (no flags) keeps GEAK's contract: profile_report.txt + `Profiler used:`.
The optional layers add subdirs, bisect a counter group that aborts, and DEGRADE -- never fail --
when the arch is not CDNA or the ATT decoder is absent.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
PROFILE = SCRIPTS / "profile_kernel.sh"

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


def _profile(tmp_path, out, *flags, **env):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    r = subprocess.run(["bash", str(PROFILE), "0", "true", str(out), *flags], cwd=work,
                       env=_env(tmp_path, **env), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r, json.loads((out / "profile_layers.json").read_text())


def test_default_run_contract_unchanged(tmp_path):
    out = tmp_path / "o"
    r, layers = _profile(tmp_path, out)
    assert "Profiler used: rocprofv3" in r.stdout
    assert (out / "profile_report.txt").is_file()
    assert layers == {"profiler": "rocprofv3", "pmc": "off", "att": "off", "spi": "off"}
    assert not (out / "pmc").exists() and not (out / "att").exists()


def test_pmc_and_att_layers_collect(tmp_path):
    dec = tmp_path / "dec"
    dec.mkdir()
    _decoder(dec)
    out = tmp_path / "o"
    r, layers = _profile(tmp_path, out, "--pmc", "--att", ROCPROF_ATT_LIBRARY_PATH=str(dec))
    assert layers["pmc"] == "collected:pmc" and layers["att"] == "collected", layers
    assert (out / "kernel_select" / "selected_kernel.txt").read_text().strip() == "_gemm_fwd_kernel"
    for group in ("memory", "memory_ea", "sol", "stall", "waitbusy", "lds_raw"):
        assert (out / "pmc" / f"pmc_{group}").is_dir(), group
    summary = (out / "pmc" / "pmc_summary.txt").read_text()
    assert "MfmaUtil" in summary and "_fetch_size_caveat" in summary   # gfx950 caveat stamped
    assert "TCC_EA0_RDREQ_sum" in (out / "pmc" / "pmc_memory_ea" / "r_counter_collection.csv").read_text()
    hotspots = (out / "att" / "hotspots.txt").read_text()
    assert "VMEM-wait" in hotspots
    assert "Architecture:   gfx950" in hotspots
    assert "occupancy:      2 waves/SIMD" in hotspots
    report = (out / "profile_report.txt").read_text()
    assert "PMC layer" in report and "ATT layer" in report
    assert "Profiler used: rocprofv3" in r.stdout


def test_att_analysis_failure_is_partial(tmp_path):
    dec = tmp_path / 'dec'
    dec.mkdir()
    _decoder(dec)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    python = bindir / 'python3'
    python.write_text(f'''#!/usr/bin/env bash
case "$1" in */hotspot_analyzer.py) exit 6;; esac
exec "{sys.executable}" "$@"
''')
    python.chmod(0o755)
    _r, layers = _profile(tmp_path, tmp_path / 'o', '--att', ROCPROF_ATT_LIBRARY_PATH=str(dec))
    assert layers['att'] == 'partial:analysis_failed(rc=6)'


def test_aborting_counter_is_bisected_out(tmp_path):
    out = tmp_path / "o"
    _r, layers = _profile(tmp_path, out, "--pmc", "--kernel", "_gemm",
                          PROFILE_PMC_GROUPS="g1:SQ_WAVES BAD_COUNTER GRBM_COUNT")
    rec = json.loads((out / "pmc" / "pmc_collection.json").read_text())
    assert rec["groups"]["g1"]["dropped"] == ["BAD_COUNTER"]
    assert sorted(c for p in rec["groups"]["g1"]["passes"] for c in p) == ["GRBM_COUNT", "SQ_WAVES"]
    assert layers["pmc"] == "partial:pmc"


def test_layers_degrade_instead_of_failing(tmp_path):
    out = tmp_path / "o"
    _r, layers = _profile(tmp_path, out, "--derived", "--att", "--spi", arch="gfx1201")
    assert layers["pmc"] == "degraded:non_cdna_arch"
    assert layers["att"] == "degraded:decoder_absent"
    assert layers["spi"] == "degraded:no_source"
    report = (out / "profile_report.txt").read_text()
    assert "rocprofv3 -L" in report and "profiling-dependencies.md" in report
    assert json.loads((out / 'att/att_dependencies.json').read_text())['att']['status'] == 'decoder_absent'


def test_unknown_option_is_refused(tmp_path):
    r = subprocess.run(["bash", str(PROFILE), "0", "true", str(tmp_path / "o"), "--bogus"],
                       cwd=tmp_path, env=_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "unknown option" in r.stderr
