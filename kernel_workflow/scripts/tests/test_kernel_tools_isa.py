# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the asm / ISA / occupancy kernel tools in kernel_workflow/scripts/kernel_tools.

Every tool's offline --selftest must pass, its data must resolve via _hwdata
(perf_knowledge/hardware/data), and a tool that needs an arch must refuse rather than default.
No GPU, no network, no torch.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
KT = REPO / "kernel_workflow" / "scripts" / "kernel_tools"
HW_DATA = REPO / "perf_knowledge" / "hardware" / "data"


@pytest.mark.parametrize("cadence", [16, 32])
def test_mfma_report_preserves_measurements_without_decision_dispatch(tmp_path, cadence):
    wave = tmp_path / "wave.json"
    code = tmp_path / "code.json"
    code.write_text(json.dumps({"code": [
        ["v_mfma_f32_16x16x16_f16"], ["ds_read_b128"],
    ]}))
    wave.write_text(json.dumps({"wave": {"begin": 0, "instructions": [
        [0, 0, 0, 0, 0], [4, 0, 0, 0, 1], [cadence, 0, 0, 0, 0],
    ]}}))
    result = run(sys.executable, KT / "mfma_efficiency.py", wave, code,
                 "--ideal-cadence", "16")
    assert result.returncode == 0, result.stderr
    assert f"median={cadence}" in result.stdout
    assert f"vs-ideal={100 * 16 / cadence:.0f}%" in result.stdout
    assert "lds" in result.stdout and "agent analysis" in result.stdout
    for decision in ("attack", "issue-bound", "bubble-diluted", "-> cards:", "catalogue gap"):
        assert decision not in result.stdout

PY_TOOLS = ["asm_loop_audit", "asm_schedule_viz", "mfma_efficiency", "deep_mfma_analysis",
            "gfx950_isa", "layout_facts", "amd_occupancy", "probe"]
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


@pytest.mark.parametrize('arch', ['gfx942', 'gfx950', 'gfx1201', None], ids=['cdna3', 'cdna4', 'rdna4', 'unspecified'])
@pytest.mark.parametrize('scripts', [KT, REPO / 'perf_knowledge/expert_skills/skills/gluon_authoring/scripts'],
                         ids=['shared', 'gluon-shim'])
def test_s_nop_reports_raw_encoding_without_duration_or_cause(tmp_path, arch, scripts):
    asm = tmp_path / 'nop.s'
    target = f'.amdgcn_target "amdgcn-amd-amdhsa--{arch}"\n' if arch else ''
    asm.write_text(target + '.type k,@function\nk:\n.L_loop:\n'
                   ' s_nop 0\n s_nop 3\n s_cbranch_scc0 .L_loop\n s_endpgm\n.size k, .-k\n')
    result = run(sys.executable, scripts / 'asm_loop_audit.py', asm, '--kernel', 'k')
    _ok(result)
    assert 'count: 2   raw immediate sum: 3' in result.stdout
    assert 'raw immediate counts: 0=1, 3=1' in result.stdout
    assert 'not measured stall time' in result.stdout
    for claim in ('requested stall cycles', 'exposed fixed-latency hazard',
                  'unhidden fixed-latency hazard', 'high s_nop =>', 'the fix is more'):
        assert claim not in result.stdout


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
    code = f"import sys; sys.path.insert(0, {str(KT)!r}); import amd_occupancy as o; print(o._find_hw_constants())"
    result = run(sys.executable, "-c", code)
    _ok(result)
    assert Path(result.stdout.strip()).resolve() == (HW_DATA / "hw_constants.json").resolve()


def test_hwdata_override_wins(tmp_path):
    (tmp_path / "hw_constants.json").write_text(json.dumps({"arch": {}}))
    code = f"import sys; sys.path.insert(0, {str(KT)!r}); import amd_occupancy as o; print(o._find_hw_constants())"
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
