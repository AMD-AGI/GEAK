"""Post-run hooks: apply best patch + commit, and clean up run artifacts.

Exposes two independent operations and a coordinator:

- ``apply_and_commit_best_patch(result, repo)`` -- applies the winning
  ``.diff`` (from ``BestPatchResult.best_patch_file``) to ``repo`` on the
  current branch using the same ``git apply`` fallback helper as the
  per-round eval worktree logic, then commits with a message that points
  at the run's ``final_report.json``.
- ``cleanup_run_artifacts(result, output_dir)`` -- prunes per-run artifacts
  while preserving ``final_report.json`` and the winning ``.diff`` inside
  the original ``output_dir``. Independent of apply.
- ``finalize_run(result, repo, output_dir, *, apply_best_patch, cleanup)``
  -- CLI-level entry point that runs either/both according to the boolean
  flags.

Failure semantics:

- Apply fails  -> no commit. Cleanup still runs if requested (the winning
  patch file is preserved in the summary, so the user can re-apply it by
  hand).
- Commit fails -> apply stays in the working tree. Cleanup still runs if
  requested.
- Cleanup fails -> the two retained files have already been copied to a
  safe temp location before ``rmtree`` runs, so we surface the error but
  never lose both summary and artifacts.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.run.utils.generated_artifacts import apply_patch_with_generated_helper_fallback
from minisweagent.run.utils.git_safe_env import get_git_safe_env

logger = logging.getLogger(__name__)


_FINAL_REPORT_NAME = "final_report.json"

# Defaults used when the container / host has no configured git identity.
# Override via the ``GEAK_GIT_AUTHOR_NAME`` / ``GEAK_GIT_AUTHOR_EMAIL`` env
# vars (set at container run time or by the developer).
_DEFAULT_GIT_AUTHOR_NAME = "GEAK Agent"
_DEFAULT_GIT_AUTHOR_EMAIL = "geak@amd.com"


def apply_and_commit_best_patch(
    result: BestPatchResult | None,
    repo: Path | None,
) -> str | None:
    """Apply ``result.best_patch_file`` to ``repo`` and commit on the current branch.

    Returns the commit SHA on success, or ``None`` when a precondition fails
    or any git step fails. Never raises; every failure is logged with a
    clear reason so the caller can decide whether to proceed.
    """
    if not _validate_apply_preconditions(result, repo):
        return None

    assert result is not None and repo is not None  # for type narrowing
    repo = Path(repo).resolve()
    patch_path = Path(result.best_patch_file).resolve()  # type: ignore[arg-type]

    if not _repo_is_clean(repo):
        logger.warning(
            "[geak apply] Skipping: repo %s has uncommitted tracked changes. "
            "Commit or stash them first, then re-run apply manually.",
            repo,
        )
        return None

    if not _apply_patch_to_repo(patch_path, repo):
        return None

    commit_sha = _commit_applied_patch(result, repo)
    if commit_sha is None:
        logger.warning(
            "[geak apply] Commit failed; leaving applied changes in the working tree of %s.",
            repo,
        )
        return None

    logger.info("[geak apply] Committed %s on %s.", commit_sha, repo)
    return commit_sha


def cleanup_run_artifacts(
    result: BestPatchResult | None,
    output_dir: Path | None,
) -> None:
    """Prune ``output_dir`` to just ``final_report.json`` + winning ``.diff``.

    Independent of apply: it is safe to call this whether or not the patch
    was applied/committed. If the patch file lives under ``output_dir`` it is
    preserved across the rmtree so the user can always re-apply it by hand.
    """
    if not _validate_cleanup_preconditions(result, output_dir):
        return

    assert result is not None and output_dir is not None  # for type narrowing
    output_dir = Path(output_dir).resolve()
    patch_path = Path(result.best_patch_file).resolve() if result.best_patch_file else None

    _cleanup_artifacts(output_dir, patch_path)


def finalize_run(
    result: BestPatchResult | None,
    repo: Path | None,
    output_dir: Path | None,
    *,
    apply_best_patch: bool = True,
    cleanup: bool = True,
) -> None:
    """CLI-level coordinator invoked from ``mini.py``.

    ``apply_best_patch`` and ``cleanup`` are fully independent; either can
    be toggled without affecting the other. If both are False, this is a
    no-op (the caller shouldn't have called in the first place, but it's
    safe).
    """
    if not apply_best_patch and not cleanup:
        return

    if apply_best_patch:
        apply_and_commit_best_patch(result, repo)

    if cleanup:
        cleanup_run_artifacts(result, output_dir)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_apply_preconditions(  # pylint: disable=too-many-return-statements
    result: BestPatchResult | None,
    repo: Path | None,
) -> bool:
    if result is None:
        logger.warning("[geak apply] Skipping: no BestPatchResult produced by the run.")
        return False
    if not result.best_patch_file:
        logger.warning("[geak apply] Skipping: BestPatchResult has no best_patch_file.")
        return False
    patch_path = Path(result.best_patch_file)
    if not patch_path.is_file():
        logger.warning("[geak apply] Skipping: best patch file does not exist: %s", patch_path)
        return False
    if patch_path.stat().st_size == 0:
        logger.warning("[geak apply] Skipping: best patch file is empty: %s", patch_path)
        return False
    if repo is None:
        logger.warning("[geak apply] Skipping: no --repo resolved; cannot apply/commit.")
        return False
    repo = Path(repo)
    if not repo.exists():
        logger.warning("[geak apply] Skipping: repo path does not exist: %s", repo)
        return False
    if not (repo / ".git").exists():
        logger.warning("[geak apply] Skipping: %s is not a git repo (no .git). Cannot commit.", repo)
        return False
    return True


def _validate_cleanup_preconditions(
    result: BestPatchResult | None,
    output_dir: Path | None,
) -> bool:
    if output_dir is None:
        logger.warning("[geak --cleanup] Skipping: no output_dir provided.")
        return False
    if not Path(output_dir).exists():
        logger.warning("[geak --cleanup] Skipping: output_dir does not exist: %s", output_dir)
        return False
    # ``result`` is optional for cleanup -- if present we use it to locate
    # the winning patch file to preserve. Missing/empty result just means
    # we fall back to preserving only ``final_report.json``.
    del result
    return True


def _has_git_identity(repo: Path, env: dict[str, str]) -> bool:
    """Return True if ``user.name`` AND ``user.email`` are resolvable for ``repo``.

    ``git config --get`` walks the full precedence chain (env -> local -> global -> system),
    so this correctly mirrors what ``git commit`` would see.
    """
    for key in ("user.name", "user.email"):
        proc = subprocess.run(
            ["git", "config", "--get", key],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return False
    return True


def _ensure_git_identity(repo: Path, env: dict[str, str]) -> dict[str, str]:
    """Return an env dict guaranteed to carry a git author/committer identity.

    Precedence (first non-empty wins):

    1. Existing ``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*`` env vars (respected as-is).
    2. An already-configured ``user.name`` + ``user.email`` (local/global/system).
    3. ``GEAK_GIT_AUTHOR_NAME`` / ``GEAK_GIT_AUTHOR_EMAIL`` env vars.
    4. Hard-coded defaults (``GEAK Agent <geak@amd.com>``).

    The returned env is only applied to the commit subprocess; global and repo
    git config are never mutated, so this is safe on the user's machine.
    """
    commit_env = dict(env)

    author_name = commit_env.get("GIT_AUTHOR_NAME")
    author_email = commit_env.get("GIT_AUTHOR_EMAIL")
    committer_name = commit_env.get("GIT_COMMITTER_NAME")
    committer_email = commit_env.get("GIT_COMMITTER_EMAIL")

    if author_name and author_email and committer_name and committer_email:
        return commit_env

    if _has_git_identity(repo, commit_env):
        return commit_env

    name = author_name or committer_name or os.environ.get("GEAK_GIT_AUTHOR_NAME") or _DEFAULT_GIT_AUTHOR_NAME
    email = author_email or committer_email or os.environ.get("GEAK_GIT_AUTHOR_EMAIL") or _DEFAULT_GIT_AUTHOR_EMAIL

    commit_env.setdefault("GIT_AUTHOR_NAME", name)
    commit_env.setdefault("GIT_AUTHOR_EMAIL", email)
    commit_env.setdefault("GIT_COMMITTER_NAME", name)
    commit_env.setdefault("GIT_COMMITTER_EMAIL", email)

    logger.info(
        "[geak --cleanup] No git identity configured; committing as %s <%s> "
        "(override via GEAK_GIT_AUTHOR_NAME / GEAK_GIT_AUTHOR_EMAIL).",
        name,
        email,
    )
    return commit_env


def _repo_is_clean(repo: Path) -> bool:
    """Return True if the tracked working tree + index are clean."""
    env = get_git_safe_env(repo)
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        logger.warning(
            "[geak --cleanup] git status failed (rc=%s): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return not result.stdout.strip()


def _apply_patch_to_repo(patch_path: Path, repo: Path) -> bool:
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    env = get_git_safe_env(repo)

    apply_result, removed_paths = apply_patch_with_generated_helper_fallback(
        patch_text=patch_text,
        cwd=repo,
        env=env,
    )
    if removed_paths:
        logger.info(
            "[geak --cleanup] Stripped generated helper artifacts during apply: %s",
            ", ".join(removed_paths),
        )
    if apply_result.returncode != 0:
        logger.warning(
            "[geak --cleanup] git apply failed (rc=%s); leaving repo and artifacts untouched.\nstderr: %s",
            apply_result.returncode,
            apply_result.stderr.strip()[:1000],
        )
        return False

    logger.info("[geak --cleanup] Applied %s to %s", patch_path, repo)
    return True


def _commit_applied_patch(
    result: BestPatchResult,
    repo: Path,
) -> str | None:
    """Stage + commit the applied changes. Return commit SHA or None on failure."""
    env = get_git_safe_env(repo)

    add = subprocess.run(
        ["git", "add", "-A"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if add.returncode != 0:
        logger.warning(
            "[geak apply] git add -A failed (rc=%s): %s",
            add.returncode,
            add.stderr.strip(),
        )
        return None

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if staged.returncode == 0 and not staged.stdout.strip():
        logger.warning("[geak apply] Patch applied cleanly but resulted in no staged changes. Skipping empty commit.")
        return None

    message = _build_commit_message(result)
    commit_env = _ensure_git_identity(repo, env)
    commit = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=commit_env,
    )
    if commit.returncode != 0:
        logger.warning(
            "[geak apply] git commit failed (rc=%s): %s",
            commit.returncode,
            commit.stderr.strip(),
        )
        return None

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return sha.stdout.strip() if sha.returncode == 0 else "HEAD"


def _build_commit_message(result: BestPatchResult) -> str:
    speedup = result.best_speedup
    speedup_str = f"{speedup:.4f}x" if isinstance(speedup, (int, float)) else "unknown"
    patch_id = result.patch_id or "unknown"
    title = f"geak: apply best patch ({patch_id}, {speedup_str})"

    body_lines: list[str] = []
    if result.best_patch_file:
        patch_path = Path(result.best_patch_file)
        report_path = patch_path.parent / _FINAL_REPORT_NAME
        body_lines.append(f"Report: {report_path}")
        body_lines.append(f"Patch:  {result.best_patch_file}")
    summary = (result.llm_conclusion or "").strip()
    if summary:
        body_lines.append("")
        body_lines.append(summary[:1000])

    return title + "\n\n" + "\n".join(body_lines) + "\n"


def _cleanup_artifacts(
    output_dir: Path,
    patch_path: Path | None,
) -> None:
    """Prune per-run artifacts, keeping only ``final_report.json`` and the winning ``.diff``.

    Lingering ``git worktree`` slots registered under ``output_dir`` are pruned
    first via ``git worktree remove --force`` on each slot's owning repo (found
    by reading the slot's ``.git`` gitfile) so the subsequent ``rmtree`` doesn't
    leave dangling admin files behind.
    """
    _prune_worktrees_under(output_dir)

    final_report = output_dir / _FINAL_REPORT_NAME

    with tempfile.TemporaryDirectory(prefix="geak_cleanup_") as tmp_str:
        tmp = Path(tmp_str)
        saved_report: Path | None = None
        saved_patch: Path | None = None

        if final_report.is_file():
            saved_report = tmp / _FINAL_REPORT_NAME
            shutil.copy2(final_report, saved_report)

        if patch_path is not None and patch_path.is_file():
            saved_patch = tmp / patch_path.name
            shutil.copy2(patch_path, saved_patch)

        try:
            shutil.rmtree(output_dir)
        except OSError as exc:
            logger.warning(
                "[geak --cleanup] Failed to remove %s: %s. Retained summary files are preserved in %s.",
                output_dir,
                exc,
                tmp,
            )
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        if saved_report is not None:
            shutil.copy2(saved_report, output_dir / _FINAL_REPORT_NAME)
        if saved_patch is not None:
            shutil.copy2(saved_patch, output_dir / saved_patch.name)

    logger.info("[geak --cleanup] Pruned %s; kept final_report.json + best patch.", output_dir)


def _iter_worktree_slots(output_dir: Path) -> list[Path]:
    """Return directories under ``output_dir`` that look like git worktrees.

    A worktree slot has a ``.git`` *file* (not directory) whose contents point
    at the owning repo's ``worktrees/<slug>`` admin directory.
    """
    slots: list[Path] = []
    if not output_dir.is_dir():
        return slots
    for candidate in output_dir.rglob(".git"):
        try:
            if candidate.is_file():
                slots.append(candidate.parent)
        except OSError:
            continue
    return slots


def _owning_repo_for_worktree(slot: Path) -> Path | None:
    """Read the slot's ``.git`` gitfile to locate the owning main repo.

    Returns the main repo root (two levels up from ``<repo>/.git/worktrees/<slug>``)
    or ``None`` when the gitfile is malformed / the admin path is gone.
    """
    gitfile = slot / ".git"
    try:
        text = gitfile.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not text.startswith(prefix):
        return None
    admin_dir = Path(text[len(prefix) :].strip())
    if not admin_dir.is_absolute():
        admin_dir = (slot / admin_dir).resolve()
    # admin_dir is <repo>/.git/worktrees/<slug>; walk up to <repo>.
    if admin_dir.parent.name != "worktrees" or admin_dir.parent.parent.name != ".git":
        return None
    return admin_dir.parent.parent.parent


def _prune_worktrees_under(output_dir: Path) -> None:
    """Remove git worktrees whose working directory is inside ``output_dir``."""
    output_dir_resolved = output_dir.resolve()
    for slot in _iter_worktree_slots(output_dir_resolved):
        repo = _owning_repo_for_worktree(slot)
        if repo is None:
            continue
        env = get_git_safe_env(repo)
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(slot)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
