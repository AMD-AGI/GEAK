"""``shadow_worktree`` — make a writable worktree for a wheel-installed
site-packages Python package without copying multi-GB ``.so`` binaries.

Strategy
========

For wheel-only packages (e.g. vLLM ships as a 850MB ``_C.abi3.so`` +
several smaller ``.abi3.so`` files plus ``.py`` source under
``/usr/local/lib/python3.X/dist-packages/vllm/``) we cannot:

  * ``git init`` the source directory — it lives under a shared system
    path with hundreds of unrelated packages, and the index would have
    to track ~1GB of compiled binaries.
  * ``git worktree add`` — same reason; also the source isn't a git
    repo to begin with.
  * ``pip install -e`` — there's no ``setup.py`` / ``pyproject.toml``
    in an installed wheel.

Instead, we build a "shadow tree" at ``dst``:

  * ``.py`` / ``.json`` / ``.yaml`` / ``.txt`` / data files → **physical
    copy** (so the agent's editor cannot accidentally clobber the
    baseline through a hardlink — see ``editor_tool.write_file``
    using ``Path.write_text`` with ``O_TRUNC``, which would rewrite
    the shared inode in place).
  * ``.so`` / ``.dylib`` / ``.pyd`` / ``.dll`` → **symlink** to the
    baseline file.  Read-only by intent: any agent edit to a binary
    is meaningless because we can't recompile (no source available).
    The symlink also keeps the worktree small (~100 MB instead of
    multi-GB).
  * The shadow is initialised as a git repo (``.gitignore *.so`` first
    so binaries never enter the index) so subsequent ``git diff``
    based patch capture works for ``.py`` modifications.

The result is a fully-importable Python package directory at
``dst/<src_name>/`` that imports identical native code (via symlink)
but allows the agent to edit any ``.py`` independently of the baseline
install.

PYTHONPATH semantics
====================

Callers should add ``dst`` (NOT ``dst/<src_name>``) to ``PYTHONPATH``
so ``import <src_name>`` resolves to ``dst/<src_name>/__init__.py``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger(__name__)


# Filename glob patterns that should be SYMLINKED rather than copied.
# Matched against bare filenames (``fnmatch``); covers the full
# ``foo.cpython-312-x86_64-linux-gnu.so`` form via the ``*.so`` glob.
_BINARY_PATTERNS: tuple[str, ...] = (
    "*.so",
    "*.so.*",
    "*.dylib",
    "*.pyd",
    "*.dll",
)

# Directory names skipped during the shadow walk.  ``__pycache__`` is
# regenerated on first import; dot-dirs are typically VCS / cache.
_SKIP_DIRS: frozenset[str] = frozenset({"__pycache__"})

# Marker file written at the shadow root so downstream consumers can
# unambiguously detect "this is a vllm-style shadow worktree".
SHADOW_MARKER = ".geak_shadow_worktree.json"


def shadow_worktree(src: Path | str, dst: Path | str) -> Path:
    """Build a writable shadow of ``src`` at ``dst``.

    Parameters
    ----------
    src : Path
        Source Python package directory (e.g.
        ``/usr/local/lib/python3.12/dist-packages/vllm``).  Must contain
        ``__init__.py``.
    dst : Path
        Worktree root.  After this call, ``dst/<src.name>/`` will hold
        the shadow contents and ``dst/`` will be a freshly-initialised
        git repo with ``.so`` files ``.gitignore``'d.

    Returns
    -------
    Path
        ``dst`` (so callers can chain).
    """
    src = Path(src).resolve()
    dst = Path(dst)
    if not (src / "__init__.py").is_file():
        raise ValueError(
            f"shadow_worktree: src {src} is not a Python package "
            f"(missing __init__.py)"
        )

    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)

    pkg_dst = dst / src.name
    n_copied, n_symlinked = _populate(src, pkg_dst)

    _write_marker(dst, src)
    _init_git_repo(dst)
    logger.info(
        "shadow_worktree: %s -> %s (copied=%d, symlinked=%d)",
        src,
        pkg_dst,
        n_copied,
        n_symlinked,
    )
    return dst


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _populate(src: Path, pkg_dst: Path) -> tuple[int, int]:
    """Walk ``src`` and materialise it under ``pkg_dst``.

    Returns ``(copied_count, symlinked_count)``.
    """
    pkg_dst.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    n_symlinked = 0
    for root, dirs, files in os.walk(src, followlinks=False):
        # Filter dirs in-place so os.walk doesn't descend into them.
        dirs[:] = [
            d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        rel_root = Path(root).resolve().relative_to(src)
        out_root = pkg_dst / rel_root
        out_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            sp = Path(root) / f
            dp = out_root / f
            if _is_binary(f):
                try:
                    os.symlink(sp.resolve(), dp)
                    n_symlinked += 1
                except OSError as exc:
                    logger.warning(
                        "shadow_worktree: symlink failed for %s: %s; falling back to copy",
                        sp,
                        exc,
                    )
                    try:
                        shutil.copy2(sp, dp)
                        n_copied += 1
                    except Exception as copy_exc:  # noqa: BLE001
                        logger.warning(
                            "shadow_worktree: copy fallback failed for %s: %s; skipping",
                            sp,
                            copy_exc,
                        )
            else:
                try:
                    shutil.copy2(sp, dp)
                    n_copied += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "shadow_worktree: copy failed for %s: %s; skipping",
                        sp,
                        exc,
                    )
    return n_copied, n_symlinked


def _is_binary(filename: str) -> bool:
    return any(fnmatch(filename, pat) for pat in _BINARY_PATTERNS)


def _write_marker(dst: Path, src: Path) -> None:
    """Plant the shadow marker so detection works post-creation."""
    marker = dst / SHADOW_MARKER
    payload = {
        "package_name": src.name,
        "source_path": str(src),
        "binary_patterns": list(_BINARY_PATTERNS),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _init_git_repo(dst: Path) -> None:
    """Initialise ``dst`` as a git repo with binaries ``.gitignore``'d.

    Order matters: write ``.gitignore`` BEFORE ``git add`` so the
    binary symlinks never enter the index.
    """
    gitignore = dst / ".gitignore"
    gitignore.write_text(
        "# auto-generated by GEAK shadow_worktree\n"
        "*.so\n"
        "*.so.*\n"
        "*.dylib\n"
        "*.pyd\n"
        "*.dll\n"
        "__pycache__/\n"
    )
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        subprocess.run(
            ["git", "init", "-q"],
            cwd=dst,
            check=True,
            env=env,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", "-A"],
            cwd=dst,
            check=True,
            env=env,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=geak@local",
                "-c",
                "user.name=geak",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "GEAK shadow_worktree baseline",
            ],
            cwd=dst,
            check=True,
            env=env,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "shadow_worktree: git init/commit failed (rc=%s); "
            "patches via git diff may not work correctly. stderr: %s",
            exc.returncode,
            (exc.stderr or b"").decode(errors="replace")[-400:],
        )


def is_shadow_worktree(path: Path | str) -> bool:
    """Return True if ``path`` was produced by :func:`shadow_worktree`."""
    return (Path(path) / SHADOW_MARKER).is_file()


__all__ = ["shadow_worktree", "is_shadow_worktree", "SHADOW_MARKER"]
