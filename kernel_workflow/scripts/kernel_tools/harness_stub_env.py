#!/usr/bin/env python3
"""Run a GENERATED harness skeleton on a machine that has none of its dependencies.

A GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/, beside create_harness.py; the
Gluon pack keeps a shim at scripts/harness_stub_env.py).

Every pack's `create_harness.py` writes a skeleton that imports torch and the DSL's own stack at
module scope, so nothing could execute one outside a configured GPU box. That is why all four
skeletons shipped for so long with the same defect: an unfilled shape stream turned every mode
into a no-op that printed nothing and exited 0, which reads exactly like a clean benchmark. The
`--selftest` in each generator uses this module to run the skeleton for real and assert it
REFUSES, which is the only way that property stays true after an edit.

`STUB_SITE` is written as `sitecustomize.py` next to the harness; Python imports it automatically
when its directory is on PYTHONPATH, before the harness runs. It fabricates any module rooted in
`_STUB_ROOTS`, so `import torch` / `from cutlass.cute import testing` resolve to inert objects.
Attribute access and calls return more stubs, which is enough to reach the refusal.
"""
import os
import subprocess
import sys
from pathlib import Path

STUB_SITE = '''
import importlib.abc, importlib.machinery, sys, types

_STUB_ROOTS = {"torch", "triton", "cutlass", "flydsl", "tilelang"}


class _Stub(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        sub = _Stub(f"{self.__name__}.{name}")
        sys.modules[sub.__name__] = sub
        setattr(self, name, sub)
        return sub

    def __call__(self, *a, **k):
        return self


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Finder())
'''

REFUSAL = "HARNESS_UNFILLED"
REFUSAL_RC = 2


def run_skeleton(harness, args, timeout=60):
    """Run `harness` with `args` under the stub importer. Returns (returncode, stdout+stderr)."""
    harness = Path(harness)
    (harness.parent / "sitecustomize.py").write_text(STUB_SITE)
    env = dict(os.environ, PYTHONPATH=str(harness.parent))
    r = subprocess.run([sys.executable, str(harness)] + list(args),
                       capture_output=True, text=True, timeout=timeout, env=env,
                       cwd=str(harness.parent))
    return r.returncode, r.stdout + r.stderr


def assert_refuses(harness, modes):
    """Every listed mode must refuse the unfilled skeleton, loudly and with a non-zero rc."""
    for mode in modes:
        args = [mode] if isinstance(mode, str) else list(mode)
        rc, out = run_skeleton(harness, args)
        assert rc == REFUSAL_RC, f"{args}: unfilled skeleton exited {rc}, expected {REFUSAL_RC}\n{out}"
        assert REFUSAL in out, f"{args}: ran without the {REFUSAL} refusal\n{out}"
        for green in ("CORRECTNESS PASS", "CORRECTNESS_OK", "GEAK_RESULT_LATENCY_MS"):
            assert green not in out, f"{args}: unfilled skeleton reported {green!r}\n{out}"
