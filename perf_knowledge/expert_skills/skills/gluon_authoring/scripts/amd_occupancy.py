#!/usr/bin/env python3
"""Shim: amd_occupancy now lives in kernel_workflow/scripts/kernel_tools/amd_occupancy.py (a GEAK shared kernel tool).

Kept so the pack's tool paths (runtime/tool-catalog.json, docs, `import amd_occupancy`) keep resolving."""
import os as _os
import sys as _sys

_KT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                      "..", "..", "..", "..", "..", "kernel_workflow", "scripts", "kernel_tools"))
if _KT not in _sys.path:
    _sys.path.insert(0, _KT)
if __name__ == "__main__":
    import runpy
    _sys.argv[0] = _os.path.join(_KT, "amd_occupancy.py")
    runpy.run_path(_sys.argv[0], run_name="__main__")
else:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(__name__, _os.path.join(_KT, "amd_occupancy.py"))
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules[__name__] = _mod
    _spec.loader.exec_module(_mod)
