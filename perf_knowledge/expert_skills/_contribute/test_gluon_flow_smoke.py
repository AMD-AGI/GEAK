"""Run real stage/tool CLIs with synthetic artifacts; no GPU or model required."""
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1] / 'skills/gluon_authoring'
SCRIPTS = PACK / 'scripts'


def cli(tool, *args, expected=0):
    result = subprocess.run([sys.executable, str(SCRIPTS / tool), *map(str, args)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, result.stdout + result.stderr
    return result


@pytest.mark.parametrize('stage', ['entry', 'transcribe', 'recover', 'evidence', 'climb', 'close'])
def test_stage_reference_to_real_shared_tool_receipt(stage, tmp_path):
    # A registry selects the role's current record view; it is not a measured
    # stage transition or evidence of a successful optimization.
    (tmp_path / 'run_registry.json').write_text(json.dumps({'current_stage': stage}))
    status = json.loads(cli('toolctl.py', 'stage', '--work', tmp_path, '--role', 'deep', '--json').stdout)
    assert status['current'] == stage and 'amd_isa' in status['allowed']
    ref, heading = status['reference'].split('#', 1)
    cli('toolctl.py', 'context', 'acquire', '--work', tmp_path, '--role', 'deep',
        '--artifact', f'stage_ref=pack:/{ref}', '--json')
    cli('toolctl.py', 'context', 'query', '--work', tmp_path, '--role', 'deep',
        '--artifact', 'stage_ref', '--heading', heading, '--json')
    receipt = tmp_path / 'isa-receipt.json'
    cli('toolctl.py', 'exec', '--work', tmp_path, '--role', 'deep', '--receipt', receipt,
        'amd_isa', '--', '--arch', 'gfx942', 'encoding', 'v_mfma_f32_16x16x16_f16')
    data = json.loads(receipt.read_text())
    assert data['stage'] == stage and data['result'] == {'status': 'ok', 'exit_code': 0}
    assert data['context_hash'] and data['lease_id']
    cli('toolctl.py', 'exec', '--work', tmp_path, '--role', 'deep', '--receipt', receipt,
        'amd_isa', '--', '--arch', 'gfx9999', 'capabilities', expected=3)
    assert json.loads(receipt.read_text())['result'] == {'status': 'failed', 'exit_code': 3}


def test_state_transition_requires_entry_artifact_and_never_uses_failed_receipt(tmp_path):
    cli('run_state.py', 'init', '--state', tmp_path / 'run_state.json', '--run-id', 'smoke')
    result = cli('toolctl.py', 'transition', '--work', tmp_path, '--role', 'deep',
                 '--to', 'transcribe', expected=2)
    assert 'missing required artifacts' in result.stderr
    assert json.loads((tmp_path / 'run_state.json').read_text())['stage'] == 'entry'
    result = cli('toolctl.py', 'transition', '--work', tmp_path, '--role', 'deep',
                 '--to', 'climb', expected=2)
    assert 'invalid transition' in result.stderr
    failed = tmp_path / 'failed.json'
    failed.write_text(json.dumps({'schema':'toolctl.command-receipt/1',
                                 'result':{'status':'failed','exit_code':3}}))
    result = cli('toolctl.py', 'transition', '--work', tmp_path, '--role', 'deep',
                 '--to', 'transcribe', '--artifact', f'plain_champion.json={failed}', expected=2)
    assert 'failed command receipt' in result.stderr
    assert json.loads((tmp_path / 'run_state.json').read_text())['stage'] == 'entry'


def payload_fixture(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    source = work / 'compile.py'
    source.write_text('''import os
from pathlib import Path
assert Path.cwd().name == 'work'
assert os.environ['SMOKE_VARIANT'] == 'variant with spaces'
Path('executed').write_text('CPU-only fixture')
p = Path(os.environ['TRITON_CACHE_DIR']) / 'hash'
p.mkdir()
for ext, text in {'ttgir': 'module {}', 'llir': 'define void @kernel()',
    'amdgcn': '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\\n.amdhsa_kernel kernel\\ns_endpgm\\n',
    'json': '{"name":"kernel","shared":0,"num_warps":4}'}.items():
    (p / ('kernel.' + ext)).write_text(text)
''')
    return {
        'schema': 'tile.profile-payload/1', 'policy': {'host_allowed_roots': [str(work)]},
        'locus': {'kind': 'host', 'image_inventory_sha256': 'synthetic-cpu-fixture'},
        'device': {'logical': '0', 'pci_bdf': '0000:01:00.0'},
        'source': {'path': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
        'execution': {'cwd': {'host': str(work), 'container': '/work'},
                      'argv': [sys.executable, 'compile.py'],
                      'env_allowlist': {'SMOKE_VARIANT': 'variant with spaces'}},
        'outputs': {'root': str(work / 'capture')}, 'capture': {'mode': 'isa-only'},
        'kernel_selector': {'regex': 'kernel'},
    }


@pytest.mark.parametrize('mutation', ['none', 'source_changed', 'locus_mismatch', 'lease_override'])
def test_payload_to_capture_preserves_execution_contract(tmp_path, mutation):
    data = payload_fixture(tmp_path)
    if mutation == 'source_changed':
        (tmp_path / 'work' / data['source']['path']).write_text('raise RuntimeError("must not execute")')
    if mutation == 'locus_mismatch':
        data['locus']['kind'] = 'container'
    if mutation == 'lease_override':
        data['execution']['env_allowlist']['HIP_VISIBLE_DEVICES'] = '7'
    payload = tmp_path / 'payload.json'
    payload.write_text(json.dumps(data))
    root = PACK.parents[3]
    env = dict(os.environ)
    env.pop('TILE_KERNEL_CONTAINER', None)
    result = subprocess.run(['bash', str(root / 'kernel_workflow/scripts/kernel_tools/capture.sh'),
        '--payload', str(payload)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    if mutation != 'none':
        assert result.returncode == 2, result.stdout + result.stderr
        assert not (tmp_path / 'work/executed').exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    capture = json.loads((tmp_path / 'work/capture/capture.json').read_text())
    assert capture['source_identity']['sha256'] == data['source']['sha256']
    assert capture['execution_identity']['payload_sha256'] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert capture['asm_audit'] and capture['sol_state'] == 'not_attempted'
