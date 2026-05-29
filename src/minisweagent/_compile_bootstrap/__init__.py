"""Compile-mode bootstrap injected into harness subprocesses via PYTHONPATH.

This package's directory is appended to ``PYTHONPATH`` by
``save_and_test._build_test_env`` and ``run_harness._build_env`` so
Python's ``site`` module auto-loads :mod:`sitecustomize` *before* the
harness imports anything.

The bootstrap forces JIT-managed kernel build systems (currently aiter)
to rebuild from worktree source so agent edits to ``.cu`` kernels
actually enter the runtime binary.  It is dependency-free (stdlib
only) so it cannot fail to load even in minimal subprocess envs.
"""

from __future__ import annotations

from pathlib import Path


def bootstrap_dir() -> str:
    """Return the absolute path of the directory holding ``sitecustomize.py``.

    Append this to a subprocess's ``PYTHONPATH`` to enable the bootstrap.
    """
    return str(Path(__file__).parent.resolve())


__all__ = ["bootstrap_dir"]
