# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for public screening and AMD artifact contracts (no GPU)."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'
GLUON_SCRIPTS = KT.parents[2] / 'perf_knowledge/expert_skills/skills/gluon_authoring/scripts'
sys.path.insert(0, str(KT))
import ab_bench as ab
import amd_artifacts as artifacts
import amd_resources as resources


def test_reference_hook_uses_geak_comparator(monkeypatch):
    seen = []
    def compare(out, ref, tol):
        seen.append((out, ref, tol))
        return True, 0.0
    monkeypatch.setattr(ab, 'load_harness_lib', lambda: SimpleNamespace(correct=compare))
    mod = SimpleNamespace(outputs=lambda n: float('-inf'), reference=lambda n: float('-inf'))
    passed, report = ab.gate_correctness(mod, ['a'], 0.001)
    assert passed == ['a'] and seen == [(float('-inf'), float('-inf'), 0.001)]
    assert report['a']['_verification'] == 'pass'


@pytest.mark.parametrize('result', [{}, {'err': float('nan')}, {'err': '0'}, {'err': -1}, {'correct': False}])
def test_invalid_oracle_is_not_timed(result):
    passed, _ = ab.gate_correctness(SimpleNamespace(oracle=lambda n: result), ['a'], 0.01)
    assert not passed


def test_control_cannot_overwrite_a_real_variant():
    base, candidate = lambda: 10, lambda: 8
    fns = {'base': base, '__control__base': candidate}
    names = list(fns)
    with pytest.raises(ValueError, match='reserved prefix'):
        ab.with_control(fns, names, 'base')
    assert fns['__control__base'] is candidate
    assert names == ['base', '__control__base']


@pytest.mark.parametrize('scripts', [KT, GLUON_SCRIPTS], ids=['shared', 'gluon-shim'])
@pytest.mark.parametrize('control', [[], ['--control', 'auto']], ids=['no-control', 'auto-control'])
@pytest.mark.parametrize('reserved', ['__control__base', '__control__other'])
def test_reserved_variant_rejected_before_hooks(tmp_path, scripts, control, reserved):
    adapter, output = tmp_path / 'adapter.py', tmp_path / 'result.json'
    adapter.write_text(f"def variants(): return {{'base': lambda: None, {reserved!r}: lambda: None}}\n"
                       "def oracle(name): raise RuntimeError('oracle must not run')\n"
                       "def fingerprint(name): raise RuntimeError('fingerprint must not run')\n")
    output.write_text('{"validity":{"valid":true},"results":{"cand":{"screen":"pass"}}}')
    proc = subprocess.run([sys.executable, str(scripts / 'ab_bench.py'), '--module', str(adapter),
                           '--json', str(output), *control], capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert 'reserved prefix' in proc.stderr and 'must not run' not in proc.stderr
    result = json.loads(output.read_text())
    assert result['schema_version'] == 2
    assert result['validity']['valid'] is False and result['results'] == {}


def ab_cli(tmp_path, *, oracle=True, timer='cuda_event', reverse=False, cells=3, tiny=False, scripts=KT):
    adapter = tmp_path / 'adapter.py'
    adapter.write_text("def base(): return 'base'\ndef cand(): return 'cand'\n"
                       "def variants(): return {'base': base, 'cand': cand}\n"
                       "def sync(): pass\n" +
                       ("def oracle(n): return {'err': 0.0}\n" if oracle else ''))
    hl = tmp_path / 'harness_lib.py'
    hl.write_text("count = 0\ndef time_op(call, **kwargs):\n"
                  "    global count\n    count += 1\n    name = call()\n"
                  f"    ms = (12 if {reverse!r} and count > 9 else 8) if name == 'cand' else 10\n"
                  f"    ms *= {1e-6 if tiny else 1}\n"
                  f"    return {{'ms': ms, 'timer': {timer!r}, 'primed': True, "
                  "'cache_condition': {'mode': 'read-evict'}}\n")
    output = tmp_path / 'result.json'
    args = [sys.executable, str(scripts / 'ab_bench.py'), '--module', str(adapter), '--json', str(output),
            '--cells', str(cells), '--iters', '1', '--warmup', '0', '--control', 'auto', '--permute']
    p = subprocess.run(args, text=True, capture_output=True, check=False,
                       env=dict(os.environ, GEAK_HARNESS_LIB=str(hl)))
    assert output.exists(), p.stderr
    return p, json.loads(output.read_text())


@pytest.mark.parametrize('scripts', [KT, GLUON_SCRIPTS], ids=['shared', 'gluon-shim'])
def test_ab_shared_and_legacy_json_contract(tmp_path, scripts):
    proc, data = ab_cli(tmp_path, scripts=scripts)
    assert proc.returncode == 0, proc.stderr
    assert data['schema_version'] == 2 and data['validity']['valid'] is True
    assert data['results']['cand']['screen'] == 'pass'
    assert data['results']['cand']['median'] == 8 and data['results']['cand']['min'] == 8
    assert len(data['windows']) == 2 and data['rotation']['complete'] is True
    assert 'frozen COMMANDMENT' in data['role']
    assert data['excluded_failing_oracle'] == []


@pytest.mark.parametrize('kwargs,valid', [({}, True), ({'reverse': True}, False),
    ({'timer': 'wall'}, False), ({'oracle': False}, False), ({'cells': 2}, False), ({'tiny': True}, True)])
def test_ab_validity_covers_both_windows(tmp_path, kwargs, valid):
    p, data = ab_cli(tmp_path, **kwargs)
    assert p.returncode == (0 if valid else 2), p.stderr
    assert data['validity']['valid'] is valid
    assert data['results']['cand']['screen'] == ('pass' if valid else 'invalid')
    assert len(data['windows']) == 2
    for window in data['windows']:
        assert window['control']['control'] in window['receipts']
        assert len(window['receipts']['cand']) == kwargs.get('cells', 3)
    if not kwargs.get('oracle', True):
        assert 'oracle UNVERIFIED' in p.stdout and 'oracle PASS' not in p.stdout


@pytest.mark.parametrize('ms', [0, -1, float('nan'), float('inf')])
def test_bad_timing_refused(ms):
    hl = SimpleNamespace(time_op=lambda *a, **k: {'ms': ms})
    with pytest.raises(RuntimeError, match='timing'):
        ab._time_cell(lambda: None, hl, 1, 0, True, lambda: None)


def multi_kernel_text():
    text = '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n'
    for n, count in [('first', 1), ('second', 2)]:
        text += f'.type {n},@function\n{n}:\n.L_{n}:\n'
        text += ' v_mfma_f32_16x16x32_f16 v[0:3], v0, v1, v[0:3]\n' * count
        text += f' s_cbranch_scc0 .L_{n}\n s_endpgm\n.size {n}, .-{n}\n'
    for n, vgpr in [('first', 64), ('second', 128)]:
        text += f'.amdhsa_kernel {n}\n.amdhsa_next_free_vgpr {vgpr}\n.end_amdhsa_kernel\n'
    text += '.amdgpu_metadata\namdhsa.kernels:\n'
    for n, vgpr in [('first', 64), ('second', 128)]:
        text += (f'  - .args:\n      - .name: arg_{n}\n'
                 f'    .name: {n}\n    .vgpr_count: {vgpr}\n')
    return text + '.end_amdgpu_metadata\n'


def test_inconsistent_module_is_refused_without_traceback(tmp_path):
    # Two complete modules concatenated: code names the second kernel, metadata blocks disagree.
    one = multi_kernel_text().replace('second', 'other')
    asm = tmp_path / 'cat.s'
    asm.write_text(multi_kernel_text() + one)
    for extra in ([], ['--kernel', 'first']):
        p = subprocess.run([sys.executable, str(KT / 'asm_loop_audit.py'), str(asm), *extra],
                           capture_output=True, text=True, check=False)
        assert p.returncode == 2, (p.stdout, p.stderr)
        assert 'Traceback' not in p.stderr and 'cannot identify one kernel' in p.stderr


def test_multi_kernel_identity_and_cli(tmp_path):
    asm = tmp_path / 'two.s'
    asm.write_text(multi_kernel_text())
    for n, vgpr in [('first', 64), ('second', 128)]:
        lines, name = artifacts.select_kernel(asm.read_text().splitlines(), n)
        assert name == n
        assert resources.analyze('\n'.join(lines))['vgpr_combined'] == vgpr
        assert artifacts.assembly_kernel_name('\n'.join(lines)) == n
        p = subprocess.run([sys.executable, str(KT / 'asm_loop_audit.py'), str(asm), '--kernel', n],
                           capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr
        assert f'[kernel: {n}]' in p.stdout
    (tmp_path / 'two.json').write_text(json.dumps({'name': 'second', 'shared': 1024}))
    with pytest.raises(ValueError, match='identity mismatch'):
        artifacts.inspect_artifact(asm, kernel='first')


def test_failed_rerun_clears_previous_actionable_json(tmp_path):
    output = tmp_path / 'result.json'
    output.write_text('{"validity":{"valid":true},"results":{"cand":{"screen":"pass"}}}')
    adapter = tmp_path / 'bad.py'
    adapter.write_text("def variants(): raise RuntimeError('adapter failed')\n")
    proc = subprocess.run([sys.executable, str(KT / 'ab_bench.py'), '--module', str(adapter),
                           '--json', str(output)], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    data = json.loads(output.read_text())
    assert data['validity']['valid'] is False and data['results'] == {}
