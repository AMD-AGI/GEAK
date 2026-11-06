"""Dependency failures must precede queries/install mutation; no network or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'
sys.path.insert(0, str(KT))
import matrix_dependency as matrix
import prepare_matrix


def test_offline_requires_inputs_before_creating_prefix(tmp_path):
    target = tmp_path / 'new'
    with pytest.raises(ValueError, match='requires'):
        prepare_matrix.prepare(target, offline=True)
    assert not target.exists()


def test_existing_prefix_is_preserved(tmp_path):
    marker = tmp_path / 'user-data'
    marker.write_text('keep')
    with pytest.raises(FileExistsError):
        prepare_matrix.prepare(tmp_path)
    assert marker.read_text() == 'keep'
    assert not (tmp_path / 'matrix-install.json').exists()


def test_target_probe_detects_old_or_broken_calculator(tmp_path):
    calc = tmp_path / 'matrix_calculator.py'
    calc.write_text("import sys\nif 'rdna4' in sys.argv: sys.exit(9)\nprint('Matrix Dimensions')\n")
    result = matrix.inspect(calc)
    assert result['status'] == 'unusable'
    assert result['checks'][-1]['exit_code'] == 9
    assert result['pinned_source'] is False


def test_managed_interpreter_and_explicit_override(tmp_path, monkeypatch):
    monkeypatch.delenv('GEAK_MATRIX_PYTHON', raising=False)
    calc = tmp_path / 'amd_matrix_instruction_calculator/matrix_calculator.py'
    python = tmp_path / 'matrix-venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assert matrix.interpreter(calc) == str(python)
    monkeypatch.setenv('GEAK_MATRIX_PYTHON', '/explicit/python')
    assert matrix.interpreter(calc) == '/explicit/python'


def test_unrecognized_source_commit_cannot_execute_installer_payload(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source / 'matrix_calculator.py').write_text('raise RuntimeError("must not execute")')
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', 'commit', '-qm', 'untrusted revision'], check=True)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_matrix.prepare(tmp_path / 'out', source, tmp_path, offline=True)
    receipt = json.loads((tmp_path / 'out/matrix-install.json').read_text())
    assert receipt['status'] == 'failed'
    assert not (tmp_path / 'out/matrix-venv').exists()
