#!/usr/bin/env python3
"""Shim: kernel_breakdown now lives in kernel_workflow/scripts/kernel_tools/kernel_breakdown.py (a GEAK shared kernel tool).

Kept so the pack's tool paths (runtime/tool-catalog.json, docs, `import kernel_breakdown`) keep resolving."""
import os as _os
import sys as _sys

_KT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                      "..", "..", "..", "..", "..", "kernel_workflow", "scripts", "kernel_tools"))
if _KT not in _sys.path:
    _sys.path.insert(0, _KT)
if __name__ == "__main__":
    import runpy
    _sys.argv[0] = _os.path.join(_KT, "kernel_breakdown.py")
    runpy.run_path(_sys.argv[0], run_name="__main__")
else:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(__name__, _os.path.join(_KT, "kernel_breakdown.py"))
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules[__name__] = _mod
    _spec.loader.exec_module(_mod)
