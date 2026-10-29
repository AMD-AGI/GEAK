"""Cold verification must hand the serving stack the exact selected computation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_applyback_contracts import make_rebind_tasks, run_builder, json_write


@pytest.mark.parametrize('field', ['kernel_src', 'best_kernel_src'])
def test_selected_snapshot_survives_dirty_working_sources_and_relocation(tmp_path, field):
    tasks, installed = make_rebind_tasks(tmp_path)
    report = json.loads((tmp_path/'optimized.json').read_text())
    for row in report['kernels']:
        working = tasks/row['kernel']/'kernel_src'
        selected = tmp_path/'cold'/row['kernel']/'kernel_src'
        shutil.copytree(working, selected)
        row[field] = str(selected.relative_to(tmp_path))
        (working/'compute.py').write_text('VALUE = 999\n')
    # A task's metadata remains usable even when its working source was moved away.
    (tasks/'second/kernel_src').rename(tasks/'second/old-working-source')
    json_write(tmp_path/'optimized.json', report)
    result = run_builder(tmp_path, tasks, installed)
    assert result.returncode == 0, result.stdout + result.stderr
    applied = json.loads((tmp_path/'applied.json').read_text())
    for row in applied['applied']:
        assert Path(row['verified_kernel_src']).is_relative_to(tmp_path/'cold')
    relocated = tmp_path/'relocated'
    shutil.copytree(tmp_path/'overlay', relocated)
    (tmp_path/'cold').rename(tmp_path/'retained-cold')
    tasks.rename(tmp_path/'retained-tasks')
    proc = subprocess.run([sys.executable, '-c',
        'import json, installed_first, installed_second, installed_jit; '
        'print(json.dumps([installed_first.Backend().op(), installed_second.op(), installed_jit.op.fn()]))'],
        env=dict(os.environ, PYTHONPATH=str(relocated)+os.pathsep+str(installed)),
        cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [11, 22, 33]


@pytest.mark.parametrize('problem', ['missing_path', 'missing_hashes', 'changed_source', 'changed_binding', 'absent_snapshot'])
def test_unproven_source_cannot_replace_an_existing_overlay(tmp_path, problem):
    tasks, installed = make_rebind_tasks(tmp_path)
    report = json.loads((tmp_path/'optimized.json').read_text())
    row = report['kernels'][0]
    if problem == 'missing_path':
        row.pop('best_kernel_src')
    elif problem == 'missing_hashes':
        row.pop('kernel_src_hashes')
    elif problem == 'changed_source':
        (tasks/'first/kernel_src/compute.py').write_text('VALUE = 999\n')
    elif problem == 'changed_binding':
        (tasks/'first/meta.json').write_text('{"candidate_bind": {"module": "wrong_target"}}')
    else:
        row['best_kernel_src'] = 'missing/snapshot'
    json_write(tmp_path/'optimized.json', report)
    overlay = tmp_path/'overlay'
    overlay.mkdir()
    (overlay/'retained.txt').write_text('previous evidence')
    result = run_builder(tmp_path, tasks, installed)
    assert result.returncode == 2
    assert 'cold source handoff' in result.stderr
    assert (overlay/'retained.txt').read_text() == 'previous evidence'
    assert not list(tmp_path.glob('overlay.previous-*'))


def test_task_local_binding_adapter_has_its_own_cold_identity(tmp_path):
    tasks, installed = make_rebind_tasks(tmp_path)
    adapter = tasks/'first/adapter.py'
    shutil.copy2(tasks/'first/kernel_src/first_entry.py', adapter)
    meta = json.loads((tasks/'first/meta.json').read_text())
    meta['candidate_bind'].update(file='adapter.py', impl_module='adapter')
    json_write(tasks/'first/meta.json', meta)
    report = json.loads((tmp_path/'optimized.json').read_text())
    report['kernels'][0]['binding_file_sha256'] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    report['kernels'][0]['task_meta_sha256'] = hashlib.sha256((tasks/'first/meta.json').read_bytes()).hexdigest()
    json_write(tmp_path/'optimized.json', report)
    passed = run_builder(tmp_path, tasks, installed)
    assert passed.returncode == 0, passed.stderr
    adapter.write_text(adapter.read_text() + '\n# changed after cold verification\n')
    failed = run_builder(tmp_path, tasks, installed)
    assert failed.returncode == 2
    applied = json.loads((tmp_path/'applied.json').read_text())
    assert 'binding_file_sha256' in applied['skipped'][0]['reason']
