"""Capture uses shared IR tools and cannot recycle another run's evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'


def test_capture_preserves_application_args_and_archives_old_evidence(tmp_path):
    source = tmp_path / 'compile.py'
    source.write_text('''import json, os, sys
from pathlib import Path
assert sys.argv[1:] == ['--out', 'app output', '--variant', 'application']
p = Path(os.environ['TRITON_CACHE_DIR']) / 'hash'
p.mkdir()
for ext, text in {'ttgir': 'module {}', 'llir': 'define void @kernel()',
                  'amdgcn': '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\\n.amdhsa_kernel kernel\\n.amdhsa_next_free_vgpr 32\\ns_endpgm\\n',
                  'json': json.dumps({'name': 'kernel', 'shared': 0, 'num_warps': 4})}.items():
    (p / ('kernel.' + ext)).write_text(text)
''')
    out = tmp_path / 'capture'
    (out / 'rc').mkdir(parents=True)
    (out / 'rc/rc_metrics.json').write_text('{"old":true}')
    (out / 'ir').mkdir()
    (out / 'ir/asm_audit.txt').write_text('old source')
    env = dict(os.environ, GEAK_GLUON_PACK_DIR=str(tmp_path / 'no-pack'))
    env.pop('TILE_KERNEL_CONTAINER', None)
    result = subprocess.run(['bash', str(KT / 'capture.sh'), 'case', str(out),
        '--kernel', 'kernel', '--source', str(source), '--isa-only', '--',
        sys.executable, str(source), '--out', 'app output', '--variant', 'application'],
        env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / 'ir/case/ir_manifest.json').read_text())
    assert manifest['kernel_name'] == 'kernel'
    capture = json.loads((out / 'capture.json').read_text())
    assert capture['sol_state'] == 'not_attempted'
    assert capture['asm_audit'] and Path(capture['asm_audit']).read_text() != 'old source'
    assert list(out.glob('rc.old_*/rc_metrics.json'))


@pytest.mark.parametrize('other_column', ['', 'requested_kernel'])
def test_capture_never_retargets_a_missing_kernel(tmp_path, other_column):
    source = tmp_path / 'compile.py'
    source.write_text('raise RuntimeError("must not reach IR capture")')
    profiler = tmp_path / 'rocprofv3'
    profiler.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('-d') + 1])
out.mkdir(parents=True, exist_ok=True)
(out / 'test_kernel_trace.csv').write_text(
    'Kernel_Name,Other\\nother_kernel,' + os.environ['SMOKE_OTHER_COLUMN'] + '\\n')
''')
    profiler.chmod(0o755)
    env = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
               KT_ALLOW_UNLOCKED='1', SMOKE_OTHER_COLUMN=other_column,
               GEAK_GLUON_PACK_DIR=str(tmp_path / 'no-pack'))
    env.pop('TILE_KERNEL_CONTAINER', None)
    out = tmp_path / 'capture'
    result = subprocess.run(['bash', str(KT / 'capture.sh'), 'case', str(out),
        '--kernel', 'requested_kernel', '--source', str(source), '--lite', '--',
        sys.executable, str(source)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'requested kernel selector matched no dispatch' in result.stderr
    assert 'other_kernel' in (out / 'preflight.txt').read_text()
    assert not (out / 'capture.json').exists()
    assert not list((out / 'ir').rglob('ir_manifest.json'))
