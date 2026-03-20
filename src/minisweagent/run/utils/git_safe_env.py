"""Safe environment helpers for git subprocesses.

When launchers isolate global git config with ``GIT_CONFIG_GLOBAL=/dev/null``,
some git commands (notably ``git worktree add``) can fail because ``/dev/null``
is treated as an invalid config file. This module swaps that setting for a
real empty gitconfig file while preserving all other environment variables.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def get_git_safe_env(base_path: Path | None = None) -> dict[str, str]:
    """Return an environment mapping safe for git subprocesses.

    If ``GIT_CONFIG_GLOBAL`` points at ``/dev/null``, replace it with an empty
    gitconfig file under ``base_path`` (or a temp file when ``base_path`` is
    omitted). Otherwise return a copy of the current process environment.
    """

    env = os.environ.copy()
    current = env.get("GIT_CONFIG_GLOBAL")
    if current != "/dev/null":
        return env

    if base_path is not None:
        base_path = Path(base_path).resolve()
        base_path.mkdir(parents=True, exist_ok=True)
        cfg = base_path / ".geak_empty.gitconfig"
        cfg.write_text("", encoding="utf-8")
        env["GIT_CONFIG_GLOBAL"] = str(cfg)
        return env

    fd, path = tempfile.mkstemp(prefix="geak_", suffix=".gitconfig")
    os.close(fd)
    cfg = Path(path)
    cfg.write_text("", encoding="utf-8")
    env["GIT_CONFIG_GLOBAL"] = str(cfg)
    return env
