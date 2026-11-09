"""CPU fixtures exercise collection failures, shared execution and evidence semantics."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_profile_kernel_layers import PROFILE, SCRIPTS, _decoder, _env

KT = SCRIPTS / 'kernel_tools'


def profile(tmp_path, mode, flags, **extra):
    env = _env(tmp_path, PROFILE_PMC_GROUPS='g:SQ_WAVES GRBM_COUNT',
               PROFILE_SPI_COUNTERS='SQ_WAVES', **extra)
    fake = tmp_path / 'bin/rocprofv3'
    code = fake.read_text()
    if mode == 'error':
        code = code.replace('pmc) for c', 'pmc) exit 1; for c')
    elif mode == 'empty':
        code = code.replace('pmc) for c', 'pmc) exit 0; for c')
    elif mode == 'wrong_kernel':
        code = code.replace('1,_gemm_fwd_kernel,$c', '1,other_kernel,$c')
    elif mode == 'partial':
        code = code.replace('for c in "${pmc[@]}"; do echo',
                            'for c in "${pmc[@]}"; do [ "$c" = GRBM_COUNT ] && continue; echo')
    elif mode == 'abort':
        code = code.replace('pmc) for c', 'pmc) exit 134; for c')
    fake.write_text(code)
    out = tmp_path / 'out'
    p = subprocess.run(['bash', str(PROFILE), '0', 'true', str(out),
                        '--kernel', '_gemm', *flags], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    return out, json.loads((out / 'profile_layers.json').read_text())


@pytest.mark.parametrize('mode,state', [
    ('error', 'failed:pmc'), ('empty', 'unavailable:pmc'),
    ('wrong_kernel', 'unavailable:pmc'), ('partial', 'partial:pmc'), ('abort', 'failed:pmc'),
])
def test_failed_or_partial_pmc_never_becomes_collected(tmp_path, mode, state):
    out, layers = profile(tmp_path, mode, ['--pmc'])
    assert layers['pmc'] == state
    record = json.loads((out / 'pmc/pmc_collection.json').read_text())
    assert not record['groups']['g']['passes']


@pytest.mark.parametrize('mode,state', [('error', 'failed:rocprofv3(rc=1)'),
                                       ('empty', 'unavailable:spi_metrics'),
                                       ('wrong_kernel', 'unavailable:spi_metrics')])
def test_spi_failure_is_not_a_successful_optional_layer(tmp_path, mode, state):
    _out, layers = profile(tmp_path, mode, ['--spi'])
    assert layers['spi'] == state


@pytest.mark.parametrize('markers', [
    {'GEAK_GPU_LOCK_HELD': '0', 'HIP_VISIBLE_DEVICES': '0'},
    {'GEAK_GPU_LEASE_ACTIVE': '1', 'GEAK_GPU_LEASE_IDS': '0', 'HIP_VISIBLE_DEVICES': '0'},
    {'GEAK_GPU_LOCKED_SET': '0', 'GEAK_GPU_LOCK_HELD': '0', 'HIP_VISIBLE_DEVICES': '0'},
])
def test_main_and_optional_profiles_reuse_the_same_outer_lease(tmp_path, markers):
    _out, layers = profile(tmp_path, 'ok', ['--pmc'], **markers)
    assert layers['profiler'] == 'rocprofv3' and layers['pmc'] == 'collected:pmc'
    assert not (tmp_path / 'locks').exists(), 'an inherited lease must never acquire a new lock'


@pytest.mark.parametrize('owner', ['lease', 'locked_set'])
def test_group_profile_preserves_order_and_clears_nested_cuda_mask(tmp_path, owner):
    markers = {'HIP_VISIBLE_DEVICES': '1,0', 'CUDA_VISIBLE_DEVICES': '1,0', 'ROCR_VISIBLE_DEVICES': '4,6'}
    if owner == 'lease':
        markers.update(GEAK_GPU_LEASE_ACTIVE='1', GEAK_GPU_LEASE_IDS='1,0')
    else:
        markers.update(GEAK_GPU_LOCKED_SET='1,0', GEAK_GPU_LOCK_HELD='0')
    env = _env(tmp_path, PROFILE_PMC_GROUPS='g:SQ_WAVES', **markers)
    fake = tmp_path / 'bin/rocprofv3'
    fake.write_text(fake.read_text().replace('case "$mode" in', '''echo "${HIP_VISIBLE_DEVICES-unset}|${ROCR_VISIBLE_DEVICES-unset}|${CUDA_VISIBLE_DEVICES-unset}" > "$out/masks"
case "$mode" in'''))
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(PROFILE), 'group:1,0', 'true', str(out), '--pmc'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((out / 'profile_layers.json').read_text())['pmc'] == 'collected:pmc'
    assert (out / 'pmc/pmc_g/masks').read_text().strip() == 'unset|6,4|unset'
    assert (out / 'rocprofv3/masks').read_text().strip() == 'unset|6,4|unset'
    assert not (tmp_path / 'locks').exists()


def test_conflicting_outer_lease_refuses_before_any_profile(tmp_path):
    env = _env(tmp_path, GEAK_GPU_LEASE_ACTIVE='1', GEAK_GPU_LEASE_IDS='1',
               HIP_VISIBLE_DEVICES='1')
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(PROFILE), '0', 'true', str(out), '--pmc'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'does not match' in result.stderr
    assert not out.exists()


@pytest.mark.parametrize('analyzer_rc,body,state', [
    (1, 'analyze failed', 'failed:rocprof-compute(rc=1)'),
    (0, 'no SPI metrics', 'unavailable:spi_metrics'),
    (0, '| 6.2.1 | Insufficient VGPRs | 0 | Percent |', 'collected:rocprof-compute'),
])
def test_compute_spi_requires_success_and_numeric_metrics(tmp_path, analyzer_rc, body, state):
    env = _env(tmp_path, PROFILER_PRIORITY='rocprof-compute')
    tool = tmp_path / 'bin/rocprof-compute'
    tool.write_text(f'''#!{sys.executable}
import pathlib,sys
if sys.argv[1] == 'profile':
    out = pathlib.Path(sys.argv[sys.argv.index('-n')+1]); out.mkdir(parents=True, exist_ok=True)
    (out / 'pmc_perf.csv').write_text('Kernel_Name,Counter\\nk,1\\n')
elif '--block' in sys.argv:
    print({body!r}); sys.exit({analyzer_rc})
else:
    print('main analyze')
''')
    tool.chmod(0o755)
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(PROFILE), '0', 'true', str(out), '--spi'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads((out / 'profile_layers.json').read_text())['spi'] == state


def container_fixture(tmp_path, env):
    """No Docker/GPU: emulate exec argv/env and a differently named shared mount."""
    engine = tmp_path / 'bin/docker'
    engine.write_text(f'''#!{sys.executable}
import json,os,subprocess,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['MOCK_DOCKER_LOG'], 'a') as f: f.write(json.dumps(args)+'\\n')
if args[0]=='inspect': print('true'); sys.exit(0)
if args[0]=='cp': sys.exit(0)
assert args[0]=='exec', args
args=args[1:]; env=dict(os.environ); cwd=None
def host(v): return v.replace('/fixture_mount', os.environ['MOCK_HOST_ROOT'])
while args[0].startswith('-'):
    flag,value=args[:2]; args=args[2:]
    if flag=='-e': k,v=value.split('=',1); env[k]=host(v)
    elif flag=='-w': cwd=host(value)
    else: raise AssertionError(flag)
assert args.pop(0)=='fixture'
sys.exit(subprocess.call([host(x) for x in args], env=env, cwd=cwd))
''')
    engine.chmod(0o755)
    env.update(TILE_KERNEL_CONTAINER='fixture', TILE_KERNEL_HOST_WORKDIR=str(tmp_path),
               TILE_KERNEL_CONTAINER_WORKDIR='/fixture_mount', MOCK_HOST_ROOT=str(tmp_path),
               MOCK_DOCKER_LOG=str(tmp_path / 'docker.jsonl'))


@pytest.mark.parametrize('container', [False, True])
def test_capture_and_safe_share_att_retry_and_unknown_cadence(tmp_path, container):
    env = _env(tmp_path, KT_ALLOW_UNLOCKED='1')
    decoder = tmp_path / 'decoder'; decoder.mkdir(); _decoder(decoder)
    env['ROCPROF_ATT_LIBRARY_PATH'] = str(decoder)
    source = tmp_path / 'kernel.py'; source.write_text('# fixture source\n')
    # No GPU: the application is true; static dump may be unavailable and must not invent it.
    fake = tmp_path / 'bin/rocprofv3'
    code = fake.read_text().replace('att) mkdir', 'att) mkdir')
    code = code.replace('"$KT_DIR_FOR_FAKE" "$out/ui_output_agent_1_dispatch_1";;', '''"$KT_DIR_FOR_FAKE" "$out/ui_output_agent_1_dispatch_1"
       if [ ! -f "$ATT_COUNT_FILE" ]; then
         echo first > "$ATT_COUNT_FILE"
         echo '{"code":null}' > "$out/ui_output_agent_1_dispatch_1/code.json"
       else echo second >> "$ATT_COUNT_FILE"; fi;;''')
    fake.write_text(code)
    env.update(ATT_COUNT_FILE=str(tmp_path / 'attempts'), LOCUS_ENV='ATT_COUNT_FILE')
    if container:
        container_fixture(tmp_path, env)
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(KT / 'capture.sh'), 'case', str(out),
                             '--source', str(source), '--kernel', '_gemm', '--', 'true'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    cap = json.loads((out / 'capture.json').read_text())
    status = json.loads((out / 'att/rocprofv3_safe.json').read_text())
    assert cap['branch'] == 'trace-live' and cap['pmc_state'] == 'not_attempted'
    assert cap['att_state'] == 'collected' and status['attempts'] == 2
    assert 'vs-ideal=UNKNOWN' in (out / 'att/mfma_eff.txt').read_text()
    assert (tmp_path / 'attempts').read_text().splitlines() == ['first', 'second']
    if container:
        calls = [json.loads(line) for line in (tmp_path / 'docker.jsonl').read_text().splitlines()]
        assert any('timeout' in row and 'rocprofv3' in row for row in calls)
        assert any('/fixture_mount/out/att' in row for row in calls)


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, KT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['directory', 'null', 'code_only', 'unknown_identity'])
def test_att_directory_or_incomplete_decode_is_not_collected(tmp_path, kind):
    tool = load_tool('profiler_artifacts')
    dispatch = tmp_path / 'ui_output_agent_1_dispatch_1'; dispatch.mkdir()
    if kind == 'null':
        (dispatch / 'code.json').write_text('{"code":null}')
    elif kind in ('code_only', 'unknown_identity'):
        (dispatch / 'code.json').write_text('{"code":[["s_endpgm"]]}')
        if kind == 'unknown_identity':
            (dispatch / 'se0_wv0.json').write_text('{"wave":{"instructions":[[1,0,1,0,0]]}}')
    result = tool.inspect(tmp_path, 'att', 'kernel')
    assert result['state'] in ('partial', 'unavailable')


def test_breakdown_preserves_evidence_without_attack_ranking(tmp_path):
    sys.path.insert(0, str(KT))
    try:
        import kernel_breakdown
        import _att_fixture
        asm = tmp_path / 'kernel.s'; asm.write_text(kernel_breakdown._SELFTEST_ASM)
        _att_fixture.write(str(tmp_path))
        result = subprocess.run([sys.executable, str(KT / 'kernel_breakdown.py'), str(asm),
                                 '--att', str(tmp_path)], capture_output=True, text=True, check=True)
    finally:
        sys.path.pop(0)
    assert 'full-drain=' in result.stdout and 'vs-ideal=UNKNOWN' in result.stdout
    for forbidden in ('attack #', 'ranked bubble', 'INFERRED', 'bubble:'):
        assert forbidden not in result.stdout


def test_main_profile_refuses_mixed_host_container_layers(tmp_path):
    env = _env(tmp_path, TILE_KERNEL_CONTAINER='external')
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(PROFILE), '0', 'true', str(out), '--pmc'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'inside the workload environment' in result.stderr
    assert not out.exists()


def test_baseline_trace_metadata_does_not_count_as_dispatch_evidence(tmp_path):
    env = _env(tmp_path)
    fake = tmp_path / 'bin/rocprofv3'
    fake.write_text(fake.read_text().replace(
        "trace) printf", "trace) echo '{}' > \"$out/metadata.json\"; exit 0; printf"))
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(PROFILE), '0', 'true', str(out)],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((out / 'profile_layers.json').read_text())['profiler'] == 'benchmark-only'
    assert json.loads((out / 'baseline_trace_artifacts.json').read_text())['state'] == 'unavailable'
