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


# Source-derived regressions formerly embedded in the production ISA query module.
# The runtime data has no test-only grids; all quotes and known-bad figures remain here.
sys.path.insert(0, str(KT))
import _amd_isa_db as isa

EVIDENCE = json.loads((Path(__file__).parent / 'fixtures/gfx950-isa-evidence.json').read_text())
LAYOUT = json.loads((HW_DATA / 'gfx950-mfma-layout.json').read_text())


@pytest.mark.parametrize('figure', EVIDENCE['figure_layouts'], ids=lambda f: f['id'])
def test_source_figure_against_hardware_rules(figure):
    assert figure['_src'] and figure['_conf']
    good = bad = 0
    for row in figure['cells']:
        assert len(row) == len(figure['cell_columns'])
        cell = dict(zip(figure['cell_columns'], row))
        if cell['k'] is None or cell['lanes'] is None:
            continue
        for i in range(cell['mn'][0], cell['mn'][1] + 1):
            for k in range(cell['k'][0], cell['k'][1] + 1):
                item, lane = isa.a_item_lane(figure['shape']['m'], 1, figure['shape']['k'], i, k)
                regs = [r for r, _, _ in isa.item_to_bits(item, figure['elem_bits'])]
                valid = (cell['lanes'][0] <= lane <= cell['lanes'][1]
                         and all(int(cell['vgprs'][0][1:]) <= r <= int(cell['vgprs'][-1][1:]) for r in regs))
                good += valid
                bad += not valid
    assert good + bad > 0
    if figure['id'] in LAYOUT['verification']['unusable_ids']:
        assert bad > 0, 'Known-bad source figure now agrees: reopen the recorded finding'
    else:
        assert bad == 0


@pytest.mark.parametrize('bits,item,expected', [
    (16, 0, [(0, 15, 0)]), (16, 3, [(1, 31, 16)]), (32, 5, [(5, 31, 0)]),
    (8, 6, [(1, 23, 16)]), (4, 9, [(1, 7, 4)]), (6, 5, [(0, 31, 30), (1, 3, 0)]),
])
def test_documented_packing_witnesses(bits, item, expected):
    assert isa.item_to_bits(item, bits) == expected


def test_sparse_operands_never_use_dense_formula():
    for record in LAYOUT['instructions']:
        if record.get('sparse'):
            for operand in 'AB':
                with pytest.raises(SystemExit) as exc:
                    isa.guard_dense(LAYOUT, record, operand)
                assert exc.value.code == 3


def test_encoding_source_ranges_and_citations():
    data = isa.load_enc_db()
    for encoding in data['encodings']:
        for field in encoding['fields']:
            for span in field['ranges']:
                assert 0 <= span['bits'][1] <= span['bits'][0] < (encoding['bit_count'] or 64)
    for record in data['instructions']:
        if 'operation' in record:
            assert record['_src']['pdf_page']
    for key in ('chapter12_rules', 'tables', 'instruction_limitations'):
        assert data[key]


def test_prose_sources_offsets_and_unique_ids():
    data = isa.load_facts_db()
    records = [r for key in ('registers', 'resources', 'counters', 'wait_states',
                            'memory_ops', 'lds_facts', 'tables', 'concepts') for r in data[key]]
    for record in records:
        src = record['_src']
        assert src['quote'] and src['pdf_page'] and record['_conf']
        if src.get('printed_page') is not None:
            assert src['printed_page'] == src['pdf_page'] - 8
    ids = [r['id'] for r in records if r.get('id')]
    assert len(ids) == len(set(ids))


def test_runtime_evidence_separation():
    assert 'figure_layouts' not in LAYOUT and 'conversions' not in LAYOUT
    assert (REPO / LAYOUT['reference_evidence']).resolve() == (Path(__file__).parent / 'fixtures/gfx950-isa-evidence.json').resolve()
    assert len(EVIDENCE['figure_layouts']) == 10 and len(EVIDENCE['conversions']) == 62
    assert EVIDENCE['source'] == LAYOUT['source']
    for record in EVIDENCE['conversions']:
        assert record['id'] and record['_src'] and record['_conf']
