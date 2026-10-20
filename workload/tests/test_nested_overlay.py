"""Execute relocated, combined overlays whose computation lives beside a dispatcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

GEAK = Path(__file__).resolve().parents[2]
BUILDER = GEAK / 'workload/stages/applyback/build_overlay.py'


def test_combined_overlay_carries_each_nested_source_tree(tmp_path):
    tasks = tmp_path / 'tasks'
    rows = []
    for i, name in enumerate(['fixture_a', 'fixture_b'], 1):
        task = tasks / name
        ksrc = task / 'kernel_src'
        (ksrc / 'sub').mkdir(parents=True)
        (ksrc / 'sub/compute.py').write_text('VALUE = %d\n' % (i * 7))
        (ksrc / 'dispatcher.py').write_text(
            'from pathlib import Path\nimport importlib.util\n'
            'leaf = Path(__file__).parent / "sub/compute.py"\n'
            'spec = importlib.util.spec_from_file_location(__name__ + "_leaf", leaf)\n'
            'impl = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(impl)\n'
            'VALUE = impl.VALUE\nLEAF = str(leaf)\n'
        )
        (task / 'meta.json').write_text(json.dumps({
            'candidate_bind': {'kind': 'module', 'module': name, 'file': 'kernel_src/dispatcher.py'},
            'kernel_extraction': {'editable_leaves': [{'source_file': 'kernel_src/sub/compute.py'}]},
        }))
        rows.append({'kernel': name, 'verdict': 'pass', 'verified_speedup': 1.5})
    optimized = tmp_path / 'optimized.json'
    optimized.write_text(json.dumps({'kernels': rows}))
    overlay = tmp_path / 'overlay'
    manifest = tmp_path / 'applied.json'
    result = subprocess.run([sys.executable, str(BUILDER), '--optimized', str(optimized),
                             '--tasks-dir', str(tasks), '--overlay', str(overlay),
                             '--manifest', str(manifest), '--geak-root', str(GEAK)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    applied = json.loads(manifest.read_text())
    assert applied['all_shadowed'], applied
    for entry in applied['applied']:
        assert 'sub/compute.py' in entry['kernel_src_files']
        assert entry['kernel_src_hashes']['sub/compute.py']
    relocated = tmp_path / 'harness_overlay'
    shutil.copytree(overlay, relocated)
    shutil.rmtree(tasks)
    shutil.rmtree(overlay)
    env = dict(os.environ, PYTHONPATH=str(relocated))
    result = subprocess.run([sys.executable, '-c',
                             'import json, fixture_a, fixture_b; '
                             'print(json.dumps([[m.VALUE, m.LEAF] for m in [fixture_a, fixture_b]]))'],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert [row[0] for row in loaded] == [7, 14]
    assert all(Path(row[1]).is_relative_to(relocated) for row in loaded)
