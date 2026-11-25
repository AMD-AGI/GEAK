"""The Matrix Calculator adapter picks its interpreter without installing anything; no network or GPU."""
import sys
from pathlib import Path

KT = Path(__file__).resolve().parents[1] / 'kernel_tools'
sys.path.insert(0, str(KT))
import _amd_matrix as matrix  # noqa: E402


def test_managed_interpreter_and_explicit_override(tmp_path, monkeypatch):
    monkeypatch.delenv('GEAK_MATRIX_PYTHON', raising=False)
    calc = tmp_path / 'amd_matrix_instruction_calculator/matrix_calculator.py'
    assert matrix.interpreter(calc) == sys.executable
    python = tmp_path / 'matrix-venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assert matrix.interpreter(calc) == str(python)
    monkeypatch.setenv('GEAK_MATRIX_PYTHON', '/explicit/python')
    assert matrix.interpreter(calc) == '/explicit/python'
