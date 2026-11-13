# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the matrix-stream readers in kernel_workflow/scripts/kernel_tools.

deep_mfma_analysis.py reads the static issue order of a hot loop; mfma_efficiency.py reads
already decoded ATT wave/code JSON. Both report observations only. No GPU, no decoder.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

KT = Path(__file__).resolve().parents[1] / "kernel_tools"
sys.path.insert(0, str(KT))
import deep_mfma_analysis as deep  # noqa: E402
import mfma_efficiency as wave  # noqa: E402


def run(*args, env=None, timeout=300):
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run([str(a) for a in args], capture_output=True, text=True,
                          env=full_env, timeout=timeout, check=False)


@pytest.mark.parametrize("tool", ["deep_mfma_analysis", "mfma_efficiency"])
def test_python_tool_selftest(tool):
    r = run(sys.executable, KT / f"{tool}.py", "--selftest")
    assert r.returncode == 0, r.stdout + r.stderr


def multi_kernel_text():
    text = '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n'
    for n, count in [('first', 1), ('second', 2)]:
        text += f'.type {n},@function\n{n}:\n.L_{n}:\n'
        text += ' v_mfma_f32_16x16x32_f16 v[0:3], v0, v1, v[0:3]\n' * count
        text += f' s_cbranch_scc0 .L_{n}\n s_endpgm\n.size {n}, .-{n}\n'
    for n, vgpr in [('first', 64), ('second', 128)]:
        text += f'.amdhsa_kernel {n}\n.amdhsa_next_free_vgpr {vgpr}\n.end_amdhsa_kernel\n'
    return text


def test_deep_analysis_selects_one_kernel(tmp_path):
    asm = tmp_path / 'two.s'
    asm.write_text(multi_kernel_text())
    for n, count in [('first', 1), ('second', 2)]:
        assert sum(sc == 'mfma' for _, sc in deep.hot_ops(asm, None, n)[1]) == count
    with pytest.raises(SystemExit, match='select one kernel'):
        deep.hot_ops(asm, None)


@pytest.mark.parametrize("cadence", [16, 32])
def test_mfma_report_preserves_measurements_without_decision_dispatch(tmp_path, cadence):
    wave_json = tmp_path / "wave.json"
    code = tmp_path / "code.json"
    code.write_text(json.dumps({"code": [
        ["v_mfma_f32_16x16x16_f16"], ["ds_read_b128"],
    ]}))
    wave_json.write_text(json.dumps({"wave": {"begin": 0, "instructions": [
        [0, 0, 0, 0, 0], [4, 0, 0, 0, 1], [cadence, 0, 0, 0, 0],
    ]}}))
    result = run(sys.executable, KT / "mfma_efficiency.py", wave_json, code,
                 "--ideal-cadence", "16")
    assert result.returncode == 0, result.stderr
    assert f"median={cadence}" in result.stdout
    assert f"vs-ideal={100 * 16 / cadence:.0f}%" in result.stdout
    assert "lds" in result.stdout and "agent analysis" in result.stdout
    for decision in ("attack", "issue-bound", "bubble-diluted", "-> cards:", "catalogue gap"):
        assert decision not in result.stdout



def test_static_gaps_exclude_prologue_and_epilogue():
    ops = [('ds_read_b128', 'lds')] * 3 + [('m', 'mfma'), ('v', 'valu'), ('m', 'mfma')]
    ops += [('ds_read_b128', 'lds')] * 5
    runs, hist, _, gaps = deep.analyze(ops)
    assert runs == [1, 1] and gaps == [1] and dict(hist) == {'valu': 1}
    assert sum(deep.analyze([('m', 'mfma')])[1].values()) == 0


@pytest.mark.parametrize('mnemonic,expected', [('s_wait_alu', 'aluwait'), ('s_delay_alu', 'aluwait'),
    ('ds_atomic_add', 'atomic'), ('v_sin_f32', 'exp')])
def test_shared_base_classification(mnemonic, expected):
    assert deep.sub_class(mnemonic) == expected


def test_unknown_dtype_or_instruction_shape_has_no_automatic_cadence():
    assert wave.mfma_dtype(['v_mfma_f32_16x16x16_f16', 'v_mfma_f32_16x16x32_f8f6f4']) is None
    assert wave.mfma_dtype(['v_mfma_f32_32x32x16_fp8_bf8']) is None
    assert wave.resolve_ideal_cadence(None, 'gfx950', 'fp16')[0] is None


@pytest.mark.parametrize('index', [-1, 2])
def test_wave_unresolved_code_index_is_an_error(tmp_path, index):
    w, c = tmp_path / 'wave.json', tmp_path / 'code.json'
    w.write_text(json.dumps({'wave': {'begin': 0, 'instructions': [[1, 0, 0, 0, index]]}}))
    c.write_text(json.dumps({'code': [['v_mfma_f32_16x16x16_f16']]}))
    with pytest.raises(ValueError, match='code index'):
        wave.load(w, c)


def test_single_matrix_instruction_has_no_cadence_or_trailing_interval():
    result = wave.analyze([(0, 'mfma', 'm'), (4, 'lds', 'ds_read_b128')])
    assert result['cadence_median'] is None and result['span'] is None
    assert not result['cyc_by_cls']

