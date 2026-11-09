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
import create_harness as generator
import deep_mfma_analysis as deep
import mfma_efficiency as wave
import parse_correctness as correctness


@pytest.mark.parametrize('mode,text', [
    ('rel', 'rel=nan\nrel=0.001'), ('rel', 'rel=oops rel=0.001'),
    ('rel', 'rel=1e+ rel=0.001'), ('maxdiff', 'max|diff|=inf max|diff|=0.001'),
    ('dqdkdv', 'dQ=0.001 dK=NaN dV=0.001'),
    ('dqdkdv', 'dQ=0.001 dV=0.001'),
])
def test_bad_or_incomplete_metric_never_passes(mode, text):
    assert correctness.parse_correctness(text, mode, 0.01)['status'] != 'pass'


@pytest.mark.parametrize('metric', [float('nan'), float('inf'), -0.01, 'bad', True])
def test_geak_bad_metric_cannot_be_hidden_by_rc0(metric):
    text = json.dumps({'case': 'a', 'correct': True, 'max_rel_err': metric})
    assert correctness.parse_geak(text, 0)['status'] == 'fail'
    assert correctness.parse_geak(text, 2)['status'] == 'unknown'
    assert correctness.parse_geak(text, 3)['status'] == 'harness_incomplete'


def test_geak_null_metric_and_missing_verdict():
    assert correctness.parse_geak('{"correct":true,"max_rel_err":null}', 0)['status'] == 'pass'
    assert correctness.parse_geak('{"max_rel_err":0.001}', None)['status'] == 'unknown'
    assert correctness.parse_geak('{"correct":true}', None)['status'] == 'pass'


@pytest.mark.parametrize('summary', [{'correct': True}, {'max_rel_err': 0.0}])
@pytest.mark.parametrize('child', [
    {'case': 'bad', 'correct': False, 'max_rel_err': 0.4},
    {'case': 'invalid', 'correct': True, 'max_rel_err': 'bad'},
])
def test_geak_summary_cannot_hide_nested_failure(summary, child):
    text = json.dumps(dict(summary, report={'cases': [child]}))
    result = correctness.parse_geak(text, 0)
    assert result['status'] == 'fail'
    assert child in result['cases']
    assert correctness.parse_geak(text, 2)['status'] == 'unknown'
    assert correctness.parse_geak(text, 3)['status'] == 'harness_incomplete'


@pytest.mark.parametrize('scripts', [KT, GLUON_SCRIPTS], ids=['shared', 'gluon-shim'])
@pytest.mark.parametrize('document,want_rc,want_status', [
    ({'eager': [{'case': 'a', 'correct': True, 'max_rel_err': 0.001},
                {'case': 'output_independence', 'correct': True, 'max_rel_err': None}],
      'random': [{'case': 'r0', 'correct': True, 'max_rel_err': 0.002}]}, 0, 'pass'),
    ({'correct': True, 'cases': [{'case': 'bad', 'correct': False}]}, 1, 'fail'),
    ({'correct': False, 'cases': [{'case': 'good', 'correct': True}]}, 1, 'fail'),
])
def test_geak_parser_shared_and_legacy_cli(tmp_path, scripts, document, want_rc, want_status):
    log, output = tmp_path / 'ut.log', tmp_path / 'correctness.json'
    log.write_text(json.dumps(document))
    proc = subprocess.run([sys.executable, str(scripts / 'parse_correctness.py'), '--log', str(log),
                           '--parse', 'geak', '--exit-code', '0', '--out', str(output)],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == want_rc, proc.stderr
    result = json.loads(output.read_text())
    assert result['schema'] == 'correctness' and result['status'] == want_status
    assert result['exit_code'] == 0


def test_correctness_cli_emits_strict_json(tmp_path):
    log = tmp_path / 'check.log'
    log.write_text('{"correct": true, "max_rel_err": NaN}')
    p = subprocess.run([sys.executable, str(KT / 'parse_correctness.py'), '--log', str(log),
                        '--parse', 'geak', '--exit-code', '0'], capture_output=True, text=True, check=False)
    assert p.returncode == 1
    def reject(value):
        pytest.fail(f'invalid JSON numeric constant: {value}')
    result = json.loads(p.stdout, parse_constant=reject)
    assert result['status'] == 'fail'


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


@pytest.mark.parametrize('mode,quick,full', [('--benchmark', [], [1]), ('--full-benchmark', [1], [])])
def test_harness_selected_mode_must_have_shapes(tmp_path, mode, quick, full):
    from harness_stub_env import run_skeleton
    body = generator.render('k', 'triton', 'k.py')
    body = body.replace('SHAPES_QUICK = [', f'SHAPES_QUICK = {quick}\n_UNUSED_QUICK = [')
    body = body.replace('SHAPES_FULL = [', f'SHAPES_FULL = {full}\n_UNUSED_FULL = [')
    harness = tmp_path / 'harness.py'
    harness.write_text(body)
    rc, text = run_skeleton(harness, [mode])
    assert rc == 3 and 'UT_HARNESS_INCOMPLETE' in text
    assert 'GEAK_RESULT_LATENCY_MS' not in text


def test_layout_factory_only_in_gluon():
    assert 'def make_blocked_layout' not in generator.render('k', 'triton', 'k.py')
    assert 'def make_blocked_layout' in generator.render('k', 'gluon', 'k.py')


@pytest.mark.parametrize('scripts', [KT, GLUON_SCRIPTS], ids=['shared', 'gluon-shim'])
@pytest.mark.parametrize('kind', ['file', 'symlink', 'dangling-symlink'])
def test_generator_preserves_existing_runner(tmp_path, scripts, kind):
    kernel, output, target = tmp_path / 'kernel.py', tmp_path / 'unittest.py', tmp_path / 'oracle.py'
    kernel.write_text('@triton.jit\ndef k(x): pass\n')
    original = '# immutable oracle; do not replace\n'
    if kind == 'file':
        output.write_text(original)
    else:
        if kind == 'symlink':
            target.write_text(original)
        output.symlink_to(target)
    proc = subprocess.run([sys.executable, str(scripts / 'create_harness.py'), '--kernel-path', str(kernel),
                           '--output', str(output)], capture_output=True, text=True, check=False)
    assert proc.returncode == 2 and 'refusing to overwrite' in proc.stderr
    if kind == 'dangling-symlink':
        assert output.is_symlink() and not target.exists()
    else:
        assert output.read_text() == original
    if kind == 'symlink':
        assert output.is_symlink() and target.read_text() == original


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


def test_multi_kernel_identity_and_cli(tmp_path):
    asm = tmp_path / 'two.s'
    asm.write_text(multi_kernel_text())
    for n, count, vgpr in [('first', 1, 64), ('second', 2, 128)]:
        lines, name = artifacts.select_kernel(asm.read_text().splitlines(), n)
        assert name == n
        assert resources.analyze('\n'.join(lines))['vgpr_combined'] == vgpr
        assert artifacts.assembly_kernel_name('\n'.join(lines)) == n
        assert sum(sc == 'mfma' for _, sc in deep.hot_ops(asm, None, n)[1]) == count
        p = subprocess.run([sys.executable, str(KT / 'asm_loop_audit.py'), str(asm), '--kernel', n],
                           capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr
        assert f'[kernel: {n}]' in p.stdout
    with pytest.raises(SystemExit, match='select one kernel'):
        deep.hot_ops(asm, None)
    (tmp_path / 'two.json').write_text(json.dumps({'name': 'second', 'shared': 1024}))
    with pytest.raises(ValueError, match='identity mismatch'):
        artifacts.inspect_artifact(asm, kernel='first')


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
