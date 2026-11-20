"""CPU fixtures exercise collection failures, shared execution and evidence semantics."""
import importlib.util
import json
import subprocess
import sys

import pytest
from _fake_rocprofv3 import SCRIPTS, _decoder, _env

KT = SCRIPTS / 'kernel_tools'


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
                            env=env, cwd=tmp_path, check=False, capture_output=True, text=True, timeout=60)
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
        import _att_fixture
        import kernel_breakdown
        asm = tmp_path / 'kernel.s'; asm.write_text(kernel_breakdown._SELFTEST_ASM)
        _att_fixture.write(str(tmp_path))
        result = subprocess.run([sys.executable, str(KT / 'kernel_breakdown.py'), str(asm),
                                 '--att', str(tmp_path)], capture_output=True, text=True, check=True)
    finally:
        sys.path.pop(0)
    assert 'full-drain=' in result.stdout and 'vs-ideal=UNKNOWN' in result.stdout
    for forbidden in ('attack #', 'ranked bubble', 'INFERRED', 'bubble:'):
        assert forbidden not in result.stdout


def test_breakdown_reports_nop_operands_and_requires_a_back_edge(tmp_path):
    sys.path.insert(0, str(KT))
    try:
        import kernel_breakdown
        asm = tmp_path / 'loop.s'
        asm.write_text('.LBB0_1:\n\ts_nop 0\n\ts_nop 0x3\n\ts_cbranch_scc1 .LBB0_1\n')
        result = kernel_breakdown.static_analyze(str(asm))
        assert result['nop_count'] == 2 and result['nop_immediate_sum'] == 3
        assert 'nop_cycles' not in result
        asm.write_text('.LBB0_1:\n\ts_nop 3\n\ts_endpgm\n')
        with pytest.raises(SystemExit, match='loop-only'):
            kernel_breakdown.static_analyze(str(asm))
        with pytest.raises(SystemExit, match='not found among back-edges'):
            kernel_breakdown.static_analyze(str(asm), '.LBB0_1')
    finally:
        sys.path.pop(0)


