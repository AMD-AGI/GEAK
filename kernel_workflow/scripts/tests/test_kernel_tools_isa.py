# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the asm / ISA / occupancy kernel tools in kernel_workflow/scripts/kernel_tools.

Every tool's offline --selftest must pass, its data must resolve via _hwdata
(perf_knowledge/hardware/data), and a tool that needs an arch must refuse rather than default.
No GPU, no network, no torch.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
KT = REPO / "kernel_workflow" / "scripts" / "kernel_tools"
HW_DATA = REPO / "perf_knowledge" / "hardware" / "data"

PY_TOOLS = ["gfx950_isa", "layout_facts", "amd_occupancy", "probe"]
SH_TOOLS = ["hw_sources.sh", "dump_ir.sh"]


def run(*args, env=None, cwd=None, timeout=300):
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, check=False,
                          env=full_env, cwd=cwd or "/tmp", timeout=timeout)


def _ok(result):
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert "FAIL" not in "\n".join(ln for ln in out.splitlines()
                                   if ln.strip().startswith(("FAIL", "[FAIL"))), out[-3000:]


@pytest.mark.parametrize("tool", PY_TOOLS)
def test_python_tool_selftest(tool):
    path = KT / f"{tool}.py"
    _ok(run(sys.executable, path, "--selftest"))


@pytest.mark.parametrize("tool", SH_TOOLS)
def test_shell_tool_selftest(tool):
    path = KT / tool
    result = run("bash", path, "--selftest")
    _ok(result)
    assert "SELFTEST PASS" in result.stdout


def test_hwdata_selftest():
    result = run(sys.executable, KT / "_hwdata.py", "--selftest")
    _ok(result)
    assert "SELFTEST PASS" in result.stdout


def test_data_resolves_through_hwdata_and_override():
    """hw_constants.json comes from perf_knowledge/hardware/data; $GEAK_HW_DATA_DIR wins."""
    code = "import sys; sys.path.insert(0, %r); import amd_occupancy as o; print(o._find_hw_constants())" % str(KT)
    result = run(sys.executable, "-c", code)
    _ok(result)
    assert Path(result.stdout.strip()).resolve() == (HW_DATA / "hw_constants.json").resolve()


def test_hwdata_override_wins(tmp_path):
    (tmp_path / "hw_constants.json").write_text(json.dumps({"arch": {}}))
    code = "import sys; sys.path.insert(0, %r); import amd_occupancy as o; print(o._find_hw_constants())" % str(KT)
    result = run(sys.executable, "-c", code, env={"GEAK_HW_DATA_DIR": str(tmp_path)})
    _ok(result)
    assert Path(result.stdout.strip()) == tmp_path / "hw_constants.json"


def test_gfx950_isa_reads_moved_databases():
    result = run(sys.executable, KT / "gfx950_isa.py", "facts", "v_mfma_f32_32x32x16_bf16")
    _ok(result)
    assert "gfx950" in result.stdout
    result = run(sys.executable, KT / "gfx950_isa.py", "--arch", "cdna3", "encoding",
                 "v_mfma_f32_32x32x8_f16")
    _ok(result)
    assert "CDNA3" in result.stdout


# ---- a tool that needs an arch refuses when it is not given --------------------------------------
def test_probe_plan_requires_arch():
    result = run(sys.executable, KT / "probe.py", "plan", "--warps", "4", "--acc", "acc=128x128")
    assert result.returncode == 2 and "--arch" in result.stderr
    result = run(sys.executable, KT / "probe.py", "plan", "--arch", "gfx950", "--warps", "4",
                 "--acc", "acc=128x128")
    _ok(result)
    assert "163840 B/CU (gfx950)" in result.stdout


def test_probe_measure_withholds_unnamed_target(tmp_path):
    (tmp_path / "k.s").write_text("  .vgpr_count: 128\n  .agpr_count: 0\n")
    result = run(sys.executable, KT / "probe.py", "measure", "--dir", tmp_path)
    _ok(result)
    assert "UNNAMED target" in result.stdout and "WITHHELD" in result.stdout
    result = run(sys.executable, KT / "probe.py", "measure", "--dir", tmp_path, "--arch", "gfx950")
    _ok(result)
    assert "[gfx950]" in result.stdout and "waves/SIMD=4" in result.stdout


def test_amd_occupancy_requires_arch():
    result = run(sys.executable, KT / "amd_occupancy.py", "--vgpr", "128")
    assert result.returncode != 0 and "--arch" in (result.stdout + result.stderr)
    result = run(sys.executable, KT / "amd_occupancy.py", "--vgpr", "176", "--arch", "gfx950")
    _ok(result)
    assert "waves/SIMD by VGPR = 2" in result.stdout


def test_layout_facts_requires_arch():
    assert run(sys.executable, KT / "layout_facts.py").returncode == 2


def test_no_rm_in_moved_shell_tools():
    for tool in SH_TOOLS:
        for ln in (KT / tool).read_text().splitlines():
            code = ln.split("#", 1)[0]
            assert not any(code.strip().startswith(p) for p in ("rm -", "rm  -")), (tool, ln)
            assert "; rm -" not in code and "&& rm -" not in code, (tool, ln)
