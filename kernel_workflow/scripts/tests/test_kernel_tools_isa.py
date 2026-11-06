# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the asm / ISA / occupancy kernel tools in kernel_workflow/scripts/kernel_tools.

These tools moved out of the Gluon pack (perf_knowledge/expert_skills/skills/gluon_authoring/scripts)
into GEAK's shared kernel tools; the pack keeps a same-named shim at the old path. Every tool's
offline --selftest must pass both from its new home and THROUGH the shim, its data must resolve via
_hwdata (perf_knowledge/hardware/data), and a tool that needs an arch must refuse rather than default.
The pack's bench/harness scripts that now time through e2e_workflow/scripts/harness_lib.py are
covered here too. No GPU, no network, no torch.
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
PACK_SCRIPTS = REPO / "perf_knowledge" / "expert_skills" / "skills" / "gluon_authoring" / "scripts"
HW_DATA = REPO / "perf_knowledge" / "hardware" / "data"

PY_TOOLS = ["asm_loop_audit", "asm_schedule_viz", "mfma_efficiency", "deep_mfma_analysis",
            "gfx950_isa", "layout_facts", "amd_occupancy", "probe"]
SH_TOOLS = ["hw_sources.sh", "dump_ir.sh"]
PACK_BENCH = ["ab_bench", "create_harness", "parse_correctness", "parity_gate", "champion_gate",
              "lever_index", "check_term_index", "ttgir_bridge", "probe_levers"]


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


@pytest.mark.parametrize("where", ["kt", "shim"])
@pytest.mark.parametrize("tool", PY_TOOLS)
def test_python_tool_selftest(tool, where):
    path = (KT if where == "kt" else PACK_SCRIPTS) / f"{tool}.py"
    _ok(run(sys.executable, path, "--selftest"))


@pytest.mark.parametrize("where", ["kt", "shim"])
@pytest.mark.parametrize("tool", SH_TOOLS)
def test_shell_tool_selftest(tool, where):
    path = (KT if where == "kt" else PACK_SCRIPTS) / tool
    result = run("bash", path, "--selftest")
    _ok(result)
    assert "SELFTEST PASS" in result.stdout


def test_hwdata_selftest():
    result = run(sys.executable, KT / "_hwdata.py", "--selftest")
    _ok(result)
    assert "SELFTEST PASS" in result.stdout


@pytest.mark.parametrize("tool", PY_TOOLS)
def test_pack_path_is_a_shim(tool):
    """The pack file is a thin shim; the implementation lives only in kernel_tools."""
    body = (PACK_SCRIPTS / f"{tool}.py").read_text()
    assert "kernel_workflow" in body and "kernel_tools" in body
    assert len(body.splitlines()) < 40, f"{tool}.py in the pack is not a shim"
    assert (KT / f"{tool}.py").is_file()


@pytest.mark.parametrize("tool", SH_TOOLS)
def test_pack_shell_path_is_a_shim(tool):
    body = (PACK_SCRIPTS / tool).read_text()
    assert "exec bash" in body and f"kernel_tools/{tool}" in body


def test_shim_import_resolves_to_kernel_tools():
    code = ("import sys; sys.path.insert(0, %r); import amd_occupancy, asm_loop_audit; "
            "print(amd_occupancy.__file__); print(asm_loop_audit._OCC is not None)" % str(PACK_SCRIPTS))
    result = run(sys.executable, "-c", code)
    _ok(result)
    lines = result.stdout.split()
    assert Path(lines[0]).resolve() == (KT / "amd_occupancy.py").resolve()
    assert lines[1] == "True"


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


def test_lever_cards_resolve_from_pack():
    code = ("import sys; sys.path.insert(0, %r); import mfma_efficiency as m; "
            "print(len(m._cards_by_bucket()))" % str(KT))
    result = run(sys.executable, "-c", code)
    _ok(result)
    assert int(result.stdout.strip()) > 0


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


# ---- the pack's bench / harness / gate scripts (search instruments over GEAK's harness_lib) -------
@pytest.mark.parametrize("script", PACK_BENCH)
def test_pack_script_selftest(script):
    _ok(run(sys.executable, PACK_SCRIPTS / f"{script}.py", "--selftest"))


def test_check_term_index_resolves_pack():
    _ok(run(sys.executable, PACK_SCRIPTS / "check_term_index.py", "--pack", PACK_SCRIPTS.parent))


def test_ab_bench_times_through_harness_lib():
    spec = importlib.util.spec_from_file_location("ab_bench_t", PACK_SCRIPTS / "ab_bench.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    # The pack path is a shim onto kernel_tools/ab_bench.py: loading it binds the moved module.
    ab = sys.modules.get("ab_bench_t", ab)
    assert Path(ab.harness_lib_path()).resolve() == \
        (REPO / "e2e_workflow" / "scripts" / "harness_lib.py").resolve()
    hl = ab.load_harness_lib()
    assert hl.cache_policy()["mode"] == "read-evict"


def test_ab_bench_refuses_missing_harness_lib(tmp_path):
    adapter = tmp_path / "bm.py"
    adapter.write_text("def variants():\n    return {'a': lambda: None}\n")
    result = run(sys.executable, PACK_SCRIPTS / "ab_bench.py", "--module", adapter,
                 env={"GEAK_HARNESS_LIB": str(tmp_path / "missing")})
    assert result.returncode != 0 and "harness_lib" in result.stderr


def test_parse_correctness_geak_exit_codes(tmp_path):
    log = tmp_path / "ut.out"
    for rc, text, want in ((0, '{"case": "a", "correct": true, "max_rel_err": 0.001}\n', 0),
                           (1, '{"case": "a", "correct": false, "max_rel_err": 0.4}\n', 1),
                           (2, "ImportError\n", 2),
                           (3, "UT_HARNESS_INCOMPLETE: no replay bundle\n", 3)):
        log.write_text(text)
        result = run(sys.executable, PACK_SCRIPTS / "parse_correctness.py", "--log", log,
                     "--parse", "geak", "--exit-code", str(rc))
        assert result.returncode == want, (rc, result.stdout, result.stderr)
