"""Exercise GEAK's execution boundary without needing TraceLens or a GPU installation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'
PACK = KT.parents[2] / 'perf_knowledge/expert_skills/skills/gluon_authoring'
sys.path.insert(0, str(KT.parents[2] / 'scripts'))

def load(name):
    spec = importlib.util.spec_from_file_location(name, KT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(KT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_raw_counts_do_not_become_absolute_utilization():
    parser = load('parse_pmc')
    d = parser.derived({'vals': {'SQ_VALU_MFMA_BUSY_CYCLES': [10.0], 'SQ_ACTIVE_INST_VALU': [1.0]}})
    for field in ('MfmaUtil', 'VALUBusy', 'VALUUtilization', 'LDSBankConflict', 'bubble_mfma_idle'):
        assert d[field] is None
    assert d['raw_activity']['values']['SQ_VALU_MFMA_BUSY_CYCLES'] == 10
    breakdown = load('kernel_breakdown')
    rows = breakdown.ranked_stack({'pct': {}, 'full_drain_pct': 0, 'nop_cycles': 0}, d)
    assert not any(source == 'PMC' for _, _, source in rows)


def test_real_zero_utilization_and_occupancy_are_preserved():
    d = load('parse_pmc').derived({'vals': {'MfmaUtil': [0], 'VALUBusy': [0],
        'MeanOccupancyPerActiveCU': [0], 'OccupancyPercent': [50], 'SQ_ACTIVE_INST_VALU': [10]}})
    assert (d['MfmaUtil'], d['VALUBusy'], d['occupancy'], d['bubble_mfma_idle']) == (0, 0, 0, 100)


def native_env(tmp_path):
    native = tmp_path / 'native'; (native / 'scripts').mkdir(parents=True)
    code = "import json,os,sys; print(json.dumps({'argv':sys.argv[1:], 'hip':os.getenv('HIP_VISIBLE_DEVICES'), 'cache':os.getenv('ROOFLINE_MIBENCH_CACHE')}))"
    for name in ('tl_only.py', 'run_roofline.py', 'case_runner_template.py'):
        (native / 'scripts' / name).write_text(code)
    env = dict(os.environ, GEAK_KERNEL_ROOFLINE_DIR=str(native), GEAK_TL_PYTHON=sys.executable,
               GEAK_GPU_PYTHON=sys.executable, KT_NO_PACK_LOCUS='1')
    for key in ('TILE_KERNEL_CONTAINER', 'GEAK_GPU_LOCK_HELD', 'GEAK_GPU_LEASE_ACTIVE', 'KT_ALLOW_UNLOCKED', 'GEAK_HARDWARE_CONTEXT'):
        env.pop(key, None)
    return env


def call(env, *args, cwd=None):
    return subprocess.run(['bash', str(KT / 'roofline.sh'), *args], env=env, cwd=cwd, capture_output=True, text=True, timeout=20)


def test_offline_delegation_preserves_argv_without_lock(tmp_path):
    result = call(native_env(tmp_path), '--', '--measurement', 'a path/measurement.json', '--output', 'out')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['argv'] == ['--measurement', 'a path/measurement.json', '--output', 'out']


def test_native_missing_does_not_fall_back(tmp_path):
    env = native_env(tmp_path); env['GEAK_KERNEL_ROOFLINE_DIR'] = str(tmp_path / 'missing')
    result = call(env, '--', '--measurement', 'x')
    assert result.returncode == 3 and 'no local calculator fallback' in result.stderr


def test_gpu_work_requires_lease_and_preserves_existing_selection(tmp_path):
    env = native_env(tmp_path)
    result = call(env, '--engine', 'measure', '--', '--output', 'out')
    assert result.returncode == 2 and 'gpu_lock' in result.stderr
    env.update(GEAK_GPU_LEASE_ACTIVE='1', HIP_VISIBLE_DEVICES='2,3')
    result = call(env, '--engine', 'measure', '--', '--output', 'out')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['hip'] == '2,3'
    result = call(env, '--engine', 'measure', '--', '--gpu', '7')
    assert result.returncode == 2


def test_frozen_reference_required_and_passed_to_native(tmp_path):
    env = native_env(tmp_path); env.update(GEAK_GPU_LOCK_HELD='2', HIP_VISIBLE_DEVICES='2')
    env.pop('ROOFLINE_MIBENCH_CACHE', None)
    result = call(env, '--engine', 'rocprof', '--', '--mode', 'full')
    assert result.returncode == 2 and 'frozen' in result.stderr
    cache = tmp_path / 'frozen.csv'; cache.write_text('reference')
    env['ROOFLINE_MIBENCH_CACHE'] = str(cache)
    result = call(env, '--engine', 'rocprof', '--', '--mode', 'full')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['cache'] == str(cache)


def test_legacy_entry_delegates_full_collection(tmp_path):
    env = native_env(tmp_path); env.update(GEAK_GPU_LOCK_HELD='0', HIP_VISIBLE_DEVICES='0')
    cache = tmp_path / 'frozen.csv'; cache.write_text('reference'); env['ROOFLINE_MIBENCH_CACHE'] = str(cache)
    result = subprocess.run(['bash', str(KT / 'rocprof_compute_probe.sh'), 'case', str(tmp_path / 'out'),
        '--full', '--kernel', 'a kernel', '--', 'python3', 'a script.py', '--config', 'x y'],
        env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)['argv']
    assert argv[argv.index('--mode')+1] == 'full'
    assert argv[argv.index('--kernel')+1] == 'a kernel'
    import shlex
    assert shlex.split(argv[argv.index('--cmd')+1]) == ['python3', 'a script.py', '--config', 'x y']


def scoped_env(tmp_path):
    import hardware_context as hc
    env = native_env(tmp_path)
    c = hc.build({'gfx':'gfx942', 'marketing_name':'AMD Instinct MI308X', 'physical_cu_count':80})
    context = tmp_path / 'hardware_context.json'
    context.write_text(json.dumps(c))
    env['GEAK_HARDWARE_CONTEXT'] = str(context)
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    rocminfo = bin_dir / 'rocminfo'
    rocminfo.write_text('#!/bin/sh\ncat <<EOF\nAgent 1\n Name: gfx942\n Marketing Name: AMD Instinct MI308X\n Compute Unit: 80\nEOF\n')
    rocminfo.chmod(0o755)
    env['PATH'] = str(bin_dir) + os.pathsep + env['PATH']
    return env, context, c


def bind(path, c):
    import hashlib
    Path(str(path) + '.hardware-context.json').write_text(json.dumps({
        'hardware_context':c, 'reference_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


@pytest.mark.parametrize('bad', [None, 'arch', 'scope', 'digest', 'missing'])
def test_offline_hardware_binding_before_native_reporting(tmp_path, bad):
    env, context, c = scoped_env(tmp_path)
    arch = tmp_path / 'arch.json'
    arch.write_text(json.dumps({'name':'MI300X' if bad == 'arch' else 'MI308X',
                               'max_achievable_tflops':{'matrix_bf16':123.456}}))
    measurement = tmp_path / 'measurement.json'; measurement.write_text('{}')
    if bad == 'scope':
        import hardware_context as hc
        c = hc.build({'gfx':'gfx942','marketing_name':'AMD Instinct MI300X','physical_cu_count':304})
    if bad != 'missing':
        bind(measurement, c)
    if bad == 'digest':
        measurement.write_text('{"changed":true}')
    original = arch.read_bytes()
    argv = ['--arch=' + str(arch), '--measurement', str(measurement), '--output', 'a path']
    result = call(env, '--', *argv)
    if bad:
        assert result.returncode == 2 and 'E_HARDWARE_CONTEXT' in result.stderr
        assert not result.stdout, 'native reporting must not execute on rejected scope'
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['argv'] == argv
    assert arch.read_bytes() == original


def test_measure_live_scope_and_relative_context_survive_workspace_change(tmp_path):
    env, context, c = scoped_env(tmp_path)
    env.update(GEAK_GPU_LEASE_ACTIVE='1', GEAK_HARDWARE_CONTEXT=context.name)
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    script = Path(env['GEAK_KERNEL_ROOFLINE_DIR']) / 'scripts/case_runner_template.py'
    script.write_text("import json,pathlib,sys\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); p.mkdir()\n"
                      "(p/'measurement.json').write_text(json.dumps({'status':'OK','environment':{'gpu_name':'AMD Instinct MI308X'}}))\n")
    result = call(env, '--engine', 'measure', '--workspace', str(workspace), '--', '--output', 'out', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    import hardware_context as hc
    hc.check_reference(c, workspace / 'out/measurement.json')
    raw = tmp_path / 'bin/rocminfo'
    raw.write_text(raw.read_text().replace('80', '38'))
    result = call(env, '--engine', 'measure', '--workspace', str(workspace), '--', '--output', 'refused', cwd=tmp_path)
    assert result.returncode == 2 and 'live GPU differs' in result.stderr
    assert not (workspace / 'refused').exists()


def test_scoped_rocprof_requires_bound_original_calibration(tmp_path):
    env, context, c = scoped_env(tmp_path)
    env['GEAK_GPU_LEASE_ACTIVE'] = '1'
    cache = tmp_path / 'frozen.csv'; cache.write_text('HBMBw,MFMABF16Flops\n4200,230000\n')
    env['ROOFLINE_MIBENCH_CACHE'] = str(cache)
    result = call(env, '--engine', 'rocprof', '--', '--mode', 'full')
    assert result.returncode == 2 and not result.stdout
    producer = subprocess.run([sys.executable, str(KT.parents[2] / 'scripts/hardware_context.py'),
        '--check', str(context), '--live', '--bind-reference', str(cache)], env=env, capture_output=True, text=True)
    assert producer.returncode == 0, producer.stderr
    result = call(env, '--engine', 'rocprof', '--', '--mode', 'full')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['cache'] == str(cache)
