"""Exercise legacy entrypoints against the shared implementations and packaging contract."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
KT = ROOT / 'kernel_workflow/scripts/kernel_tools'
PACK = ROOT / 'perf_knowledge/expert_skills/skills/gluon_authoring/scripts'


def run(*args, cwd=None, **env):
    clean = {k: v for k, v in os.environ.items() if k not in ('GEAK_ROOT', 'GEAK_HARDWARE_CONTEXT')}
    return subprocess.run([str(a) for a in args], cwd=cwd, env={**clean, **env},
                          text=True, capture_output=True)


def test_legacy_default_preserved_shared_requires_arch(tmp_path):
    args = ['plan', '--warps', '4', '--acc', 'acc=128x128']
    legacy = run(sys.executable, PACK/'probe.py', *args, cwd=tmp_path)
    explicit = run(sys.executable, KT/'probe.py', *args, '--arch', 'gfx950')
    assert legacy.returncode == explicit.returncode == 0
    assert legacy.stdout == explicit.stdout
    assert 'gfx950 reference' in legacy.stderr
    assert run(sys.executable, KT/'probe.py', *args).returncode == 2
    rdna = run(sys.executable, PACK/'probe.py', *args, '--arch=gfx1201')
    shared = run(sys.executable, KT/'probe.py', *args, '--arch=gfx1201')
    assert rdna.returncode == shared.returncode == 0 and rdna.stdout == shared.stdout
    assert 'gfx950 reference' not in rdna.stderr


def test_context_supplies_legacy_arch_without_silent_fallback(tmp_path):
    output = run(sys.executable, '-c',
        'import sys,json;sys.path.insert(0,"scripts");import hardware_context as h;'
        'print(json.dumps(h.build({"gfx":"gfx942","marketing_name":"AMD Instinct MI300X","physical_cu_count":304})))',
        cwd=ROOT)
    assert output.returncode == 0, output.stderr
    context = tmp_path/'context.json'; context.write_text(output.stdout)
    args = ['plan', '--warps', '4', '--acc', 'a=128x128']
    actual = run(sys.executable, PACK/'probe.py', *args, GEAK_HARDWARE_CONTEXT=str(context))
    expected = run(sys.executable, KT/'probe.py', *args, '--arch', 'gfx942')
    assert actual.returncode == expected.returncode == 0 and actual.stdout == expected.stdout
    context.write_text('{}')
    bad = run(sys.executable, PACK/'probe.py', *args, GEAK_HARDWARE_CONTEXT=str(context))
    assert bad.returncode != 0 and 'Invalid GEAK_HARDWARE_CONTEXT' in bad.stderr
    assert 'gfx950 reference' not in bad.stderr


@pytest.mark.parametrize('name,command', [('probe.py', sys.executable), ('dump_ir.sh', 'bash')])
def test_relocated_entry_and_missing_dependency(tmp_path, name, command):
    entry = tmp_path/'skill with spaces'/name; entry.parent.mkdir(); shutil.copy(PACK/name, entry)
    missing = run(command, entry, '--help', cwd=tmp_path)
    assert missing.returncode != 0 and 'GEAK_ROOT=/path/to/GEAK' in missing.stderr
    installed = run(command, entry, '--help', cwd=tmp_path, GEAK_ROOT=str(ROOT))
    assert installed.returncode == 0, installed.stderr
    invalid = run(command, PACK/name, '--help', GEAK_ROOT=str(tmp_path/'bad root'))
    assert invalid.returncode != 0 and 'GEAK_ROOT=/path/to/GEAK' in invalid.stderr


def test_python_import_exposes_shared_explicit_arch_api():
    spec = importlib.util.spec_from_file_location('legacy_probe_test', PACK/'probe.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.waves_by_vgpr(168, 'gfx942') == 3
    assert module.waves_by_vgpr(176, 'gfx942') == 2
    assert module.waves_by_vgpr(128, 'gfx1201') == 10
    assert module.waves_by_vgpr(128, 'unknown') is None


def test_legacy_dump_measure_matches_shared(tmp_path):
    compiler = tmp_path/'compile fixture.py'
    compiler.write_text('''import os,json
from pathlib import Path
p=Path(os.environ['TRITON_CACHE_DIR'])/'one'
p.mkdir()
for ext in ('ttgir','llir','amdgcn'):
 (p/('kernel.'+ext)).write_text('.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\\n.amdhsa_kernel kernel\\n.vgpr_count: 168\\n.agpr_count: 8\\n')
(p/'kernel.json').write_text(json.dumps({'name':'kernel','shared':32768,'num_warps':4}))
''')
    records = []
    for tools, name in [(PACK, 'legacy'), (KT, 'shared')]:
        dest = tmp_path/name
        result = run('bash', tools/'dump_ir.sh', '--out', dest, '--variant', 'v', '--',
                     sys.executable, compiler, DUMP_IR_NO_ROCMINFO='1', cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        result = run(sys.executable, tools/'probe.py', 'measure', '--dir', dest/'v', '--json', dest/'probe.json')
        assert result.returncode == 0, result.stderr
        k = json.loads((dest/'probe.json').read_text())['kernels'][0]
        records.append({key:k[key] for key in ['arch','vgpr_combined','waves_by_vgpr_term','waves_per_simd','wg_size']})
    assert records[0] == records[1]
    assert records[0]['vgpr_combined'] == 168 and records[0]['waves_by_vgpr_term'] == 3
    assert records[0]['wg_size'] == 256 and records[0]['waves_per_simd'] == 2
