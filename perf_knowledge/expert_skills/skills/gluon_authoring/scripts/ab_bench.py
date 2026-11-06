#!/usr/bin/env python3
"""Compatibility entry point for the GEAK shared kernel tools."""
import importlib.util
from pathlib import Path
import runpy
import sys

_shared = Path(__file__).resolve().parents[5] / 'kernel_workflow/scripts/kernel_tools'
sys.path.insert(0, str(_shared))
_target = _shared / (Path(__file__).stem + '.py')
if __name__ == '__main__':
    runpy.run_path(str(_target), run_name='__main__')
else:
    _spec = importlib.util.spec_from_file_location(__name__, _target)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
