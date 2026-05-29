"""``PackageProfile`` — minimal profile descriptor used by worktree
creation, install gating, and harness-runtime env injection.

A profile is identified by its ``name`` and a ``detect(path)``
predicate.  Profiles can override the default git-worktree path
(``make_worktree``) and short-circuit the default
``ensure_worktree_installed`` flow (``skip_install``).

Profiles are registered side-effectfully on import (see
:mod:`minisweagent.kernel_packages.vllm_profile`).  Detection on a
path returns *every* matching profile in registration order — the
caller decides what to do with overlap (typically: pick the first
profile with a custom ``make_worktree``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackageProfile:
    """Special-case handler for one kernel package layout.

    Attributes
    ----------
    name : str
        Stable identifier (matches ``pip`` distribution name where
        possible).
    detect : Callable[[Path], bool]
        Predicate: does the given source / worktree path match this
        profile?  Must be cheap (filesystem checks only) — called
        once per worktree on every test invocation.
    make_worktree : Optional[Callable[[Path, Path], Path]]
        ``(src, dst) -> Path`` builder.  ``None`` means "use the
        default git-worktree-add path" (see
        :func:`minisweagent.run.task_file.create_worktree`).
    runtime_env : dict[str, str]
        Subprocess env vars injected into harness test runs (applied
        with ``setdefault`` semantics so callers can override).
    skip_install : bool
        When True, :func:`ensure_worktree_installed` skips its
        recursive sub-project install for worktrees matching this
        profile.  Use for wheel-only packages whose worktree is a
        shadow tree (no ``setup.py`` to install against).
    """

    name: str
    detect: Callable[[Path], bool]
    make_worktree: Callable[[Path, Path], Path] | None = None
    runtime_env: dict[str, str] = field(default_factory=dict)
    skip_install: bool = False


REGISTRY: list[PackageProfile] = []


def register(profile: PackageProfile) -> None:
    """Append ``profile`` to the registry (no dedup — caller must check)."""
    REGISTRY.append(profile)


def detect_packages(path: Path | str) -> list[PackageProfile]:
    """Return every registered profile whose ``detect`` matches ``path``.

    Order = registration order.  Errors raised by individual ``detect``
    callables are swallowed (best-effort); a malformed profile must not
    block detection of well-formed peers.
    """
    p = Path(path)
    matched: list[PackageProfile] = []
    for profile in REGISTRY:
        try:
            if profile.detect(p):
                matched.append(profile)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.debug("PackageProfile %s.detect raised: %s", profile.name, exc)
    return matched


__all__ = ["PackageProfile", "REGISTRY", "detect_packages", "register"]
