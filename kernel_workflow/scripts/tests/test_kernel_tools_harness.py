# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the search-harness tools in kernel_workflow/scripts/kernel_tools.

create_harness.py generates an editable search skeleton (harness_stub_env.py lets its selftest
run a skeleton without torch); parse_correctness.py turns a GEAK unittest or legacy oracle output
into a machine verdict. Timing and comparison come from e2e_workflow/scripts/harness_lib.py.
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
GLUON_SCRIPTS = REPO / "perf_knowledge/expert_skills/skills/gluon_authoring/scripts"
HARNESS_LIB = REPO / "e2e_workflow" / "scripts" / "harness_lib.py"
sys.path.insert(0, str(KT))
import create_harness as generator  # noqa: E402
import parse_correctness as correctness  # noqa: E402


def run(*args, env=None, cwd=None, timeout=300):
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, check=False,
                          env=full_env, cwd=cwd or "/tmp", timeout=timeout)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("tool", ["create_harness", "parse_correctness"])
def test_python_tool_selftest(tool):
    result = run(sys.executable, KT / f"{tool}.py", "--selftest")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert "SELFTEST PASS" in out, out[-3000:]


def test_create_harness_bakes_in_the_repo_harness_lib(tmp_path):
    ch = _load("create_harness_kt", KT / "create_harness.py")
    assert Path(ch.default_harness_lib()).resolve() == HARNESS_LIB.resolve()
    out = tmp_path / "harness.py"
    (tmp_path / "k.py").write_text("import triton\n\n@triton.jit\ndef k(x):\n    pass\n")
    result = run(sys.executable, KT / "create_harness.py", "--kernel-path", tmp_path / "k.py",
                 "--kernel-name", "k", "--output", out)
    assert result.returncode == 0, result.stderr[-2000:]
    assert str(HARNESS_LIB.resolve()) in out.read_text()


def test_parse_correctness_geak_exit_codes(tmp_path):
    log = tmp_path / "ut.out"
    for rc, text, want in ((0, '{"case": "a", "correct": true, "max_rel_err": 0.001}\n', 0),
                           (1, '{"case": "a", "correct": false, "max_rel_err": 0.4}\n', 1),
                           (2, "ImportError\n", 2),
                           (3, "UT_HARNESS_INCOMPLETE: no replay bundle\n", 3)):
        log.write_text(text)
        result = run(sys.executable, KT / "parse_correctness.py", "--log", log,
                     "--parse", "geak", "--exit-code", str(rc))
        assert result.returncode == want, (rc, result.stdout, result.stderr)



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

