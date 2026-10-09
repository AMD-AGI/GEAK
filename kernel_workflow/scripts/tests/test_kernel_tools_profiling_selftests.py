# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every profiling tool in kernel_workflow/scripts/kernel_tools passes --selftest, both from its GEAK
location and through the Gluon pack's shim at the old scripts/ path (CPU-only; no GPU, profiler or
ATT decoder needed -- the tools degrade, and their selftests use synthetic fixtures)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
KT = SCRIPTS / "kernel_tools"
PACK = SCRIPTS.parents[1] / "perf_knowledge/expert_skills/skills/gluon_authoring/scripts"

TOOLS = [
    "capture.sh", "rocprof_compute_probe.sh", "rocprofv3_safe.sh",
    "parse_pmc.py", "parse_rc.py", "kernel_breakdown.py", "hotspot_analyzer.py",
    "att_to_perfetto.py", "att_merge_perfetto.py", "att_timeline.py", "att_opclass.py",
    "tile_trace.py", "serve_traces.py",
]


def _run(path, tmp_path):
    cmd = (["bash", str(path)] if path.suffix == ".sh" else [sys.executable, str(path)]) + ["--selftest"]
    env = dict(os.environ, TMPDIR=str(tmp_path), MPLBACKEND="Agg")
    env.pop("GEAK_GPU_LOCK_HELD", None)
    return subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("where", ["kernel_tools", "pack_shim"])
def test_selftest_passes(tool, where, tmp_path):
    path = (KT if where == "kernel_tools" else PACK) / tool
    assert path.is_file(), path
    r = _run(path, tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "PASS" in out and "FAIL" not in out.replace("FAILED", ""), out[-2000:]


@pytest.mark.parametrize("tool", TOOLS)
def test_pack_path_is_a_shim(tool):
    text = (PACK / tool).read_text()
    assert f"kernel_workflow/scripts/kernel_tools/{tool}" in text and len(text.splitlines()) < 30


def test_profiler_wrappers_follow_geak_conventions():
    for name in ("capture.sh", "rocprof_compute_probe.sh", "rocprofv3_safe.sh"):
        code = [ln for ln in (KT / name).read_text().splitlines() if not ln.lstrip().startswith("#")]
        joined = "\n".join(code)
        assert "kt_ensure_lock" in joined, f"{name}: GPU work not routed through gpu_lock.sh"
        assert not any("export HIP_VISIBLE_DEVICES=" in ln or "env HIP_VISIBLE_DEVICES=" in ln
                       for ln in code), f"{name}: inline HIP_VISIBLE_DEVICES pin"
        assert not any(ln.strip().startswith("rm ") or " rm -" in ln for ln in code
                       if "grep" not in ln), f"{name}: rm (move aside instead)"
        assert "--cmd" in joined, f"{name}: no bash -c command-string form"


def test_pmc_counter_names_match_the_parser():
    sys.path.insert(0, str(KT))
    import parse_pmc
    names = {c for g in parse_pmc.PMC_GROUPS.values() for c in g}
    assert {"TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"} <= names
    assert "TCC_EA0_RDREQ_DRAM_sum" not in names
    # profile_kernel.sh --pmc takes its groups from parse_pmc (single source), never a literal list
    pk = (SCRIPTS / "profile_kernel.sh").read_text()
    assert "parse_pmc.py\" --print-groups" in pk and "TCC_EA0_RDREQ_DRAM" not in pk
