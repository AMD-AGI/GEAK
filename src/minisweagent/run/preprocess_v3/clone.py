"""URL resolution, repo cloning, and baseline/eval split (pre-step 0a).

This module is the deterministic, offline part of the v3 preprocess
pipeline that runs before any LLM subagent gets dispatched:

1. :func:`resolve_repo_url` canonicalises a raw repo identifier
   (``org/repo``, full https URL, or ``git@github.com:org/repo.git``)
   into the canonical ``https://github.com/org/repo.git`` form. No
   network access.

2. :func:`clone_repo` shells out to ``git clone`` (and optionally
   ``git checkout``) into a caller-chosen destination, raising a
   typed :class:`CloneError` on any failure.

3. :func:`split_repo_for_baseline_and_eval` produces two sibling
   working trees — ``baseline/`` and ``eval/`` — under a caller-chosen
   work root. The baseline tree is the stable measurement copy used by
   the baseline+profile step (PR 2); the eval tree is the mutation
   surface used by the optimization loop later in the pipeline. They
   start byte-identical (``shutil.copytree`` from the same clone) so
   downstream diffs are well-defined.

These three primitives are **new** to ``preprocess_v3``. The existing
``src/minisweagent/run/preprocess/resolve_kernel_url.py`` solves an
adjacent but different problem (resolving a single *file* URL such as a
GitHub blob link, optionally with ``#L42`` line fragments, into a local
path + clone) and is intentionally not depended on here — the v3
contract takes a *repository* identifier as the input, not a file URL.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class CloneError(RuntimeError):
    """Raised when ``git clone`` or the follow-up ``git checkout`` fails.

    Carries the underlying ``CompletedProcess`` (when available) so callers
    can inspect ``stderr`` for richer diagnostics without re-running git.
    """

    def __init__(
        self,
        message: str,
        *,
        completed: subprocess.CompletedProcess | None = None,
    ) -> None:
        super().__init__(message)
        self.completed = completed


_ORG_REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_SSH_PATTERN = re.compile(r"^git@github\.com:(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+?)(?:\.git)?$")


def resolve_repo_url(raw: str) -> str:
    """Canonicalise a raw repo identifier to ``https://github.com/org/repo.git``.

    Accepts (and normalises) three forms:

    * ``org/repo`` — implicit GitHub shorthand.
    * Full https URL: ``https://github.com/org/repo``,
      ``https://github.com/org/repo.git``, optionally with a trailing
      slash. Sub-paths like ``/tree/main`` are stripped down to the
      repo root.
    * SSH URL: ``git@github.com:org/repo.git`` (``.git`` optional).

    The function is purely string-level — no DNS lookup, no
    ``ls-remote``. It's safe to call from offline environments and
    deterministic across machines.

    Args:
        raw: Raw repo identifier, whitespace-trimmed.

    Returns:
        Canonical ``https://github.com/<owner>/<repo>.git`` URL.

    Raises:
        ValueError: When ``raw`` is empty or matches none of the
            supported forms. Other hosts (gitlab, bitbucket, …) are
            currently out of scope; extend the pattern set when
            needed.
    """
    if raw is None:
        raise ValueError("resolve_repo_url: raw URL must not be None")
    spec = str(raw).strip()
    if not spec:
        raise ValueError("resolve_repo_url: raw URL must not be empty")

    ssh_match = _SSH_PATTERN.match(spec)
    if ssh_match:
        return f"https://github.com/{ssh_match.group('owner')}/{ssh_match.group('repo')}.git"

    if _ORG_REPO_PATTERN.match(spec):
        return f"https://github.com/{spec}.git"

    if spec.startswith(("http://", "https://")):
        parsed = urlparse(spec)
        if parsed.netloc not in {"github.com", "www.github.com"}:
            raise ValueError(f"resolve_repo_url: only github.com is supported, got netloc={parsed.netloc!r}")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"resolve_repo_url: URL missing org/repo: {spec!r}")
        owner, repo = parts[0], parts[1]
        repo = repo.removesuffix(".git")
        return f"https://github.com/{owner}/{repo}.git"

    raise ValueError(f"resolve_repo_url: unsupported URL form: {spec!r}")


def _run_git(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run a git command, capturing output. Thin wrapper for testability."""
    return subprocess.run(  # noqa: S603 — args are validated upstream by the caller
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def clone_repo(
    url: str,
    dest: Path,
    ref: str | None = None,
    *,
    timeout: int = 600,
) -> Path:
    """Clone ``url`` into ``dest`` and optionally check out ``ref``.

    Designed to be called by pre-step-0a after :func:`resolve_repo_url`
    has produced a canonical URL.

    Args:
        url: A git-cloneable URL. Caller is responsible for any
            normalisation; pass the output of :func:`resolve_repo_url`
            when starting from a user-supplied identifier.
        dest: Destination directory. **Must not already exist** —
            cloning into an existing directory makes the rollback
            semantics fuzzy. ``dest.parent`` is created on demand.
        ref: Optional ref (branch, tag, or commit sha) to check out
            after cloning. ``None`` keeps git's default HEAD.
        timeout: Per-subprocess timeout in seconds. Applies to both
            the clone and the checkout independently.

    Returns:
        The resolved ``Path`` to the cloned tree (== ``dest`` resolved).

    Raises:
        CloneError: When ``git clone`` or ``git checkout`` fails, or
            when ``dest`` already exists.
        FileNotFoundError: If ``git`` is not on PATH.
    """
    dest = Path(dest)
    if dest.exists():
        raise CloneError(f"clone_repo: destination already exists: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    clone_proc = _run_git(["git", "clone", url, str(dest)], timeout=timeout)
    if clone_proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneError(
            f"git clone failed for {url!r} -> {dest}: "
            f"{clone_proc.stderr.strip() or clone_proc.stdout.strip() or 'no output'}",
            completed=clone_proc,
        )

    if ref:
        checkout_proc = _run_git(
            ["git", "-C", str(dest), "checkout", ref],
            timeout=timeout,
        )
        if checkout_proc.returncode != 0:
            raise CloneError(
                f"git checkout {ref!r} failed in {dest}: "
                f"{checkout_proc.stderr.strip() or checkout_proc.stdout.strip() or 'no output'}",
                completed=checkout_proc,
            )

    return dest.resolve()


def _ignore_git_dir(_dir: str, names: list[str]) -> list[str]:
    """``shutil.copytree`` ignore filter: skip the ``.git`` metadata dir.

    The baseline and eval trees are working copies — they don't need
    git history, and dropping ``.git`` keeps the copy quick on large
    repos. If a future caller needs git in the eval tree (e.g. to
    re-init for ``git apply`` in the optimization loop), that should
    be done explicitly inside the eval tree rather than copied here.
    """
    return [n for n in names if n == ".git"]


def split_repo_for_baseline_and_eval(
    clone_path: Path,
    work_root: Path,
) -> tuple[Path, Path]:
    """Split a cloned repo into ``baseline/`` and ``eval/`` working trees.

    Both targets are produced as direct ``shutil.copytree`` snapshots of
    ``clone_path`` (sans the ``.git`` directory — see
    :func:`_ignore_git_dir`). They start byte-identical so the diff
    between baseline and eval after the optimization loop is exactly
    the candidate patch.

    Args:
        clone_path: Path to the source clone (typically the output of
            :func:`clone_repo`).
        work_root: Parent directory under which ``baseline/`` and
            ``eval/`` are created. Created on demand. **Existing
            ``baseline/`` or ``eval/`` directories under ``work_root``
            are removed first** so this function is idempotent across
            repeated pipeline runs against the same work root.

    Returns:
        ``(baseline_path, eval_path)`` — both resolved absolute paths.

    Raises:
        FileNotFoundError: If ``clone_path`` does not exist or is not a
            directory.
    """
    clone_path = Path(clone_path)
    if not clone_path.is_dir():
        raise FileNotFoundError(f"split_repo_for_baseline_and_eval: clone not found: {clone_path}")

    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    baseline_dir = work_root / "baseline"
    eval_dir = work_root / "eval"

    for target in (baseline_dir, eval_dir):
        if target.exists():
            logger.debug("split_repo_for_baseline_and_eval: removing stale %s", target)
            shutil.rmtree(target)

    shutil.copytree(clone_path, baseline_dir, ignore=_ignore_git_dir)
    shutil.copytree(clone_path, eval_dir, ignore=_ignore_git_dir)

    return baseline_dir.resolve(), eval_dir.resolve()


__all__ = [
    "CloneError",
    "clone_repo",
    "resolve_repo_url",
    "split_repo_for_baseline_and_eval",
]
