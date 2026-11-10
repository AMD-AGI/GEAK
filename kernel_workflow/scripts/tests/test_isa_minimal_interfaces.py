"""Regression checks for shared hardware queries and format-specific facts."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
KT = ROOT / 'kernel_workflow/scripts/kernel_tools'
OP = 'v_mfma_scale_f32_16x16x128_f8f6f4'


def run(*args, **env):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True,
                          env=dict(os.environ, **env), timeout=30)


def test_unspecified_fp4_width_is_not_inferred_from_xml():
    r = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'facts', OP)
    assert r.returncode == 0, r.stderr
    assert 'vgprs       A=? B=? C/D=4' in r.stdout
    assert 'xml_operand_vgprs A=8 B=8' in r.stdout
    assert 'not format-specific usage' in r.stdout


@pytest.mark.parametrize('fmt,expected', [('fp4', 4), ('fp6', 6), ('fp8', 8)])
def test_selected_format_uses_existing_operand_map(fmt, expected):
    r = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'facts', OP, '--as', fmt)
    assert r.returncode == 0, r.stderr
    assert f'vgprs       A={expected} B={expected} C/D=4' in r.stdout
    assert 'not compiled allocation' in r.stdout
    assert 'xml_operand_vgprs A=8 B=8' in r.stdout
    legacy = run(sys.executable, KT/'gfx950_isa.py', 'facts', OP, '--as', fmt)
    assert legacy.returncode == 0 and legacy.stdout == r.stdout


def test_format_override_cannot_reinterpret_fixed_fp16_instruction():
    r = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'facts',
            'v_mfma_f32_16x16x32_f16', '--as', 'fp4')
    assert r.returncode == 3 and 'format-selectable' in r.stderr


@pytest.mark.parametrize('rule,stride', [('scale.16x16x128',16), ('scale.32x32x64',32)])
def test_scale_source_query_needs_no_optional_dependencies(rule, stride):
    r = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'tables', rule,
            TILE_HW_SEED='', TILE_HW_TOOLS='', TILE_HW_CACHE='')
    assert r.returncode == 0, r.stderr
    assert f'lane = M + {stride} * (k / 32)' in r.stdout
    assert 'not a generated Gluon layout' in r.stdout
    assert 'DERIVED' in r.stdout and 'CDNA4 ISA p57' in r.stdout


def test_calculator_facts_report_inputs_without_tuning_verdict(tmp_path):
    calc = tmp_path/'amd_matrix_instruction_calculator'
    calc.mkdir()
    (calc/'matrix_calculator.py').write_text("print('''Matrix Dimensions\nM: 16\nN: 16\nK: 16\nGPRs required for A: 2\nGPRs required for B: 2\nSrc0: FP16\nSrc1: FP16\n''')")
    r = run(sys.executable, KT/'amd_isa.py', '--arch','gfx942','facts','v_mfma_f32_16x16x16_f16',
            TILE_HW_SEED=str(tmp_path), TILE_HW_TOOLS='', TILE_HW_CACHE='')
    assert r.returncode == 0, r.stderr
    assert 'elems/lane  A=4  B=4' in r.stdout
    assert 'disqualified' not in r.stdout and 'ds_read_tr' not in r.stdout


@pytest.mark.parametrize('opcode', ['v_smfmac_f32_16x16x64_f16', 'v_smfmac_i32_16x16x128_i8'])
def test_sparse_metadata_width_is_not_accumulator_width(opcode):
    r = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'facts', opcode)
    assert r.returncode == 0, r.stderr
    assert 'C/D=4*' in r.stdout  # VDST is 128 bits; SRC2 is the 32-bit sparse index.


@pytest.mark.parametrize('query', ['encoding', 'facts'])
def test_seed_xml_does_not_override_committed_queries(tmp_path, query):
    # A malformed stale seed must not be parsed or silently change hardware answers.
    (tmp_path/'amdgpu_isa_cdna4.xml').write_text('<stale-and-invalid')
    command = [sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', query,
               'v_smfmac_f32_16x16x64_f16']
    baseline = run(*command, TILE_HW_SEED='', TILE_HW_TOOLS='', TILE_HW_CACHE='')
    seeded = run(*command, TILE_HW_SEED=str(tmp_path), TILE_HW_TOOLS='', TILE_HW_CACHE='')
    assert baseline.returncode == seeded.returncode == 0
    assert (seeded.stdout, seeded.stderr) == (baseline.stdout, baseline.stderr)


def test_explicit_isa_data_directory_does_not_mix_sources(tmp_path):
    r = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'encoding',
            'v_smfmac_f32_16x16x64_f16', GEAK_HW_DATA_DIR=str(tmp_path))
    assert r.returncode == 3 and 'not found' in r.stderr
    caps = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'capabilities', '--json',
               GEAK_HW_DATA_DIR=str(tmp_path))
    import json
    assert json.loads(caps.stdout)['queries']['encoding']['status'] == 'missing_asset'


def test_sparse_a_gap_does_not_offer_metadata_as_complete_layout():
    r = run(sys.executable, KT/'amd_isa.py', '--arch', 'gfx950', 'layout',
            'v_smfmac_f32_16x16x64_f16', '-A')
    assert r.returncode == 3
    assert 'not a complete A layout' in r.stderr
