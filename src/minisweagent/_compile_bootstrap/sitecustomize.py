"""Auto-loaded by Python's ``site`` module in every GEAK-spawned harness
subprocess that includes our bootstrap directory in PYTHONPATH.

Forces JIT-managed kernel build systems to rebuild from worktree source
so agent edits to ``.cu`` / ``.cuh`` kernels under the worktree actually
enter the runtime binary.

Two layers:

  1. **Env-level**: ``AITER_REBUILD`` is set by save_and_test/run_harness
     before subprocess launch.  This file applies a defensive
     ``setdefault`` for cases where a subprocess is spawned outside the
     normal GEAK env-injection path.

  2. **Import-hook**: clear ``aiter.jit.core.rebuilded_list`` after
     ``aiter.jit.core`` finishes loading.  The shipped allowlist
     contains ``"module_aiter_core"`` which would otherwise be skipped
     even with ``AITER_REBUILD`` set, so any agent edit landing in
     ``module_aiter_core``'s sources (e.g. ``aiter_core_pybind.cu``)
     would silently fall through to the baseline ``.so``.

The hook is registered unconditionally.  When ``aiter.jit.core`` is
never imported (non-aiter runs), it costs literally one ``find_spec``
miss and stays dormant.

This file MUST stay dependency-free (stdlib only) — it loads before
any user import chain has a chance to run.
"""

from __future__ import annotations

import os
import sys

# ──────────────────────────────────────────────────────────────────────
# Layer 1: env-level rebuild flags
# ──────────────────────────────────────────────────────────────────────

# ``setdefault`` so callers that explicitly set =0 (debug) or =1
# (full from-scratch rebuild) are respected.  =2 = incremental rebuild,
# the default we want for every GEAK harness run.
os.environ.setdefault("AITER_REBUILD", "2")


# ──────────────────────────────────────────────────────────────────────
# Layer 2: post-import hook for aiter.jit.core
# ──────────────────────────────────────────────────────────────────────

_HOOKED_MODULE = "aiter.jit.core"


def _clear_aiter_rebuilded_list(module: object) -> None:
    """Empty ``rebuilded_list`` so ``module_aiter_core`` (the shipped
    default allowlist entry) is also rebuildable under AITER_REBUILD.

    Best-effort: any failure is swallowed because the bootstrap MUST
    NOT break the test harness even if aiter's internals change.
    """
    try:
        rl = getattr(module, "rebuilded_list", None)
        if isinstance(rl, list):
            rl.clear()
    except Exception:
        pass


class _PostExecLoader:
    """Wrap a real loader; call ``hook(module)`` after ``exec_module``."""

    __slots__ = ("_inner", "_hook")

    def __init__(self, inner: object, hook):
        self._inner = inner
        self._hook = hook

    def create_module(self, spec):  # noqa: D401 — loader protocol
        fn = getattr(self._inner, "create_module", None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            self._hook(module)
        except Exception:
            pass


class _AiterJitCoreFinder:
    """``meta_path`` finder that retargets the loader for
    :mod:`aiter.jit.core` so we can run :func:`_clear_aiter_rebuilded_list`
    immediately after the module finishes executing.
    """

    __slots__ = ("_fired",)

    def __init__(self) -> None:
        self._fired = False

    def find_spec(self, name: str, path, target=None):  # noqa: D401
        if self._fired or name != _HOOKED_MODULE:
            return None
        self._fired = True
        # Delegate spec resolution to the rest of the meta_path chain
        # by temporarily removing ourselves.
        import importlib.util  # local import — sitecustomize stays cheap to load

        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PostExecLoader(spec.loader, _clear_aiter_rebuilded_list)
        return spec


# Idempotent registration (sitecustomize can be loaded twice if the user
# also has another sitecustomize earlier in the path; defensive).
if not any(isinstance(h, _AiterJitCoreFinder) for h in sys.meta_path):
    sys.meta_path.insert(0, _AiterJitCoreFinder())
