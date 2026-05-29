"""Per-package profile registry.

A :class:`PackageProfile` tells GEAK how to special-case a particular
kernel package whose default install/worktree pipeline doesn't apply:

  * Wheel-only installs (e.g. vLLM in ``vllm/vllm-openai-rocm``) have
    no ``setup.py`` to ``pip install -e`` against and live inside
    ``site-packages``.  Their worktree must be a hardlink/copy
    "shadow tree" rather than a ``git worktree add``.

  * Future: in-tree JIT systems whose default ``rebuild_list`` skips
    a kernel of interest (handled today via the global sitecustomize
    bootstrap, but a profile is the natural extension point).

Default behaviour (no profile match) is preserved — packages that
already work end-to-end with ``git worktree add`` +
``ensure_worktree_installed`` (sgl-kernel, sglang, aiter monorepo)
do NOT need a profile.

Public API:

  * :func:`detect_packages` — return all matching profiles for a path.
  * :class:`PackageProfile` — the dataclass describing each profile.
  * :data:`REGISTRY` — the live list of registered profiles.

The vLLM profile is registered side-effectfully on import.
"""

from __future__ import annotations

# Side-effect imports register their profiles into REGISTRY.
from . import vllm_profile  # noqa: F401
from .profile import REGISTRY, PackageProfile, detect_packages

__all__ = ["PackageProfile", "REGISTRY", "detect_packages"]
