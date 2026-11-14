"""Public payload execution contract, independent of any authoring pack."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, GEAK_GLUON_PACK_DIR=str(tmp_path / 'absent-pack'))
    env.pop('TILE_KERNEL_CONTAINER', None)
    result = subprocess.run(['bash', str(root / 'kernel_workflow/scripts/kernel_tools/capture.sh'),
        '--payload', str(payload)], cwd=tmp_path, env=env, check=False, capture_output=True, text=True, timeout=30)
    if mutation != 'none':
        assert result.returncode == 2, result.stdout + result.stderr
        assert not (tmp_path / 'work/executed').exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    capture = json.loads((tmp_path / 'work/capture/capture.json').read_text())
    assert capture['source_identity']['sha256'] == data['source']['sha256']
    assert capture['execution_identity']['payload_sha256'] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert capture['asm_audit'] and capture['sol_state'] == 'not_attempted'


@pytest.mark.parametrize('key', ['CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL',
                                'GEAK_GPU_LEASE_ACTIVE', 'TILE_KERNEL_CONTAINER', 'LOCUS_ENV',
                                'KT_ALLOW_UNLOCKED', 'KT_NO_PACK_LOCUS'])
def test_payload_cannot_override_execution_controls(tmp_path, key):
    data = payload_fixture(tmp_path)
    data['execution']['env_allowlist'][key] = '1'
    payload = tmp_path / 'payload.json'; payload.write_text(json.dumps(data))
    capture = Path(__file__).resolve().parents[1] / 'kernel_tools/capture.sh'
    result = subprocess.run(['bash', str(capture), '--payload', str(payload)],
                            check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'may not override' in result.stderr
    assert not (tmp_path / 'work/executed').exists()


@pytest.mark.parametrize('mutation', ['absolute_shell', 'parent_escape'])
def test_payload_refuses_shell_and_path_prefix_escape(tmp_path, mutation):
    data = payload_fixture(tmp_path)
    if mutation == 'absolute_shell':
        data['execution']['argv'] = ['/bin/bash', '-c', 'true']
    else:
        data['outputs']['root'] = str(tmp_path / 'work/../outside')
    tool = Path(__file__).resolve().parents[1] / 'kernel_tools/profile_payload.py'
    payload = tmp_path / 'payload.json'; payload.write_text(json.dumps(data))
    result = subprocess.run([sys.executable, str(tool), 'validate', '--payload', str(payload)],
                            check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and not json.loads(result.stdout)['ok']
