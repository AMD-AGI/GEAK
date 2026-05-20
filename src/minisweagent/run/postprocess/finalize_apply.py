"""Post-run hooks: apply best patch + commit, and clean up worktrees.

Exposes two independent operations and a coordinator:

- ``apply_and_commit_best_patch(result, repo)`` -- applies the winning
  ``.diff`` (from ``BestPatchResult.best_patch_file``) to ``repo`` on the
  current branch using the same ``git apply`` fallback helper as the
  per-round eval worktree logic, then commits with a message that points
  at the run's ``final_report.json``. Returns the commit SHA or None
  (back-compat). Sibling ``apply_and_commit_best_patch_detailed`` returns
  a structured outcome dict for callers that want to render UX from it.
- ``cleanup_run_artifacts(result, output_dir)`` -- removes lingering git
  worktrees under ``output_dir``. All other run artifacts are preserved.
- ``finalize_apply_and_cleanup(result, repo, output_dir, *, apply_best_patch, cleanup)``
  -- CLI-level entry point that runs either/both according to the boolean
  flags. Returns ``{apply_status, cleanup_status, commit_sha, reason}``.

Failure semantics:

- Apply fails  -> no commit. Cleanup still runs if requested.
- Commit fails -> apply stays in the working tree. Cleanup still runs if
  requested.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
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
    or any git step fails. Never raises. Thin back-compat wrapper around
    :func:`apply_and_commit_best_patch_detailed`; new callers that want to
    render UX from the failure reason should call the detailed function
    directly.
    """
    return apply_and_commit_best_patch_detailed(result, repo).get("commit_sha")


def apply_and_commit_best_patch_detailed(
    result: BestPatchResult | None,
    repo: Path | None,
) -> dict:
    """Apply + commit and return a structured outcome dict.

    Return shape:

    .. code-block:: python

        {"status": "committed" | "skipped_precondition"
                  | "apply_failed" | "commit_failed",
         "commit_sha": str | None,
         "reason": str | None}

    Never raises. Every non-success branch logs a clear reason so the
    caller can decide whether to render an additional user-facing message.
    """
    if not _validate_apply_preconditions(result, repo):
        return {"status": "skipped_precondition", "commit_sha": None, "reason": None}

    assert result is not None and repo is not None  # for type narrowing
    repo = Path(repo).resolve()
    patch_path = Path(result.best_patch_file).resolve()  # type: ignore[arg-type]

    applied, apply_reason = _apply_patch_to_repo(patch_path, repo)
    if not applied:
        return {"status": "apply_failed", "commit_sha": None, "reason": apply_reason}

    patch_files = _extract_patch_files(patch_path)
    commit_sha, commit_reason = _commit_applied_patch(result, repo, patch_files)
    if commit_sha is None:
        logger.warning(
            "[geak apply] Commit failed; leaving applied changes in the working tree of %s.",
            repo,
        )
        return {"status": "commit_failed", "commit_sha": None, "reason": commit_reason}

    logger.info("[geak apply] Committed %s on %s.", commit_sha, repo)
    return {"status": "committed", "commit_sha": commit_sha, "reason": None}


def cleanup_run_artifacts(
    result: BestPatchResult | None,
    output_dir: Path | None,
) -> str:
    """Remove lingering git worktrees under ``output_dir``.

    All other run artifacts (logs, patches, intermediate files) are kept
    intact so the full run directory remains available for inspection.

    Returns one of:

    - ``"ran"`` -- worktree cleanup completed successfully.
    - ``"skipped_empty"`` -- ``output_dir`` is None, missing, or not a
      directory.
    """
    if output_dir is None:
        logger.warning("[geak --cleanup] Skipping: no output_dir provided.")
        return "skipped_empty"
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        logger.warning(
            "[geak --cleanup] Skipping: output_dir does not exist or is not a dir: %s",
            output_dir,
        )
        return "skipped_empty"

    _prune_worktrees_under(output_dir)
    logger.info("[geak --cleanup] Pruned worktrees under %s.", output_dir)
    return "ran"


def finalize_apply_and_cleanup(
    result: BestPatchResult | None,
    repo: Path | None,
    output_dir: Path | None,
    *,
    apply_best_patch: bool = True,
    cleanup: bool = True,
) -> dict:
    """CLI-level coordinator invoked from ``mini.py``.

    Returns a structured outcome dict for the CLI caller to render messages
    from:

    .. code-block:: python

        {"apply_status": <apply status, or "skipped_disabled">,
         "cleanup_status": <cleanup status, or "skipped_disabled">,
         "commit_sha": str | None,
         "reason": str | None}

    ``apply_best_patch`` and ``cleanup`` are fully independent; either can
    be toggled without affecting the other. With both False the function
    returns a no-op outcome (both statuses ``"skipped_disabled"``).
    """
    outcome: dict = {
        "apply_status": "skipped_disabled",
        "cleanup_status": "skipped_disabled",
        "commit_sha": None,
        "reason": None,
    }

    if apply_best_patch:
        apply_outcome = apply_and_commit_best_patch_detailed(result, repo)
        outcome["apply_status"] = apply_outcome["status"]
        outcome["commit_sha"] = apply_outcome.get("commit_sha")
        outcome["reason"] = apply_outcome.get("reason")

    if cleanup:
        outcome["cleanup_status"] = cleanup_run_artifacts(result, output_dir)

    return outcome


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
        "[geak apply] No git identity configured; committing as %s <%s> "
        "(override via GEAK_GIT_AUTHOR_NAME / GEAK_GIT_AUTHOR_EMAIL).",
        name,
        email,
    )
    return commit_env


_DIFF_GIT_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def _extract_patch_files(patch_path: Path) -> list[str]:
    """Parse ``diff --git a/... b/...`` headers to get the list of files touched by the patch."""
    files: list[str] = []
    header_re = _DIFF_GIT_HEADER_RE
    for line in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = header_re.match(line)
        if m:
            files.append(m.group(2))
    return files


def _filter_patch_to_existing_files(patch_text: str, repo: Path) -> str | None:
    """Keep only diff sections whose target file already exists in *repo*.

    Returns the filtered patch text, or ``None`` if nothing remains.
    """
    lines = patch_text.splitlines(keepends=True)
    preamble: list[str] = []
    sections: list[list[str]] = []
    current: list[str] | None = None

    for line in lines:
        if line.startswith("diff --git "):
            if current is not None:
                sections.append(current)
            current = [line]
        elif current is None:
            preamble.append(line)
        else:
            current.append(line)
    if current is not None:
        sections.append(current)

    if not sections:
        return None

    kept: list[str] = list(preamble)
    skipped: list[str] = []
    for section in sections:
        m = _DIFF_GIT_HEADER_RE.match(section[0].rstrip("\n"))
        if m is None:
            kept.extend(section)
            continue
        b_path = m.group(2)
        if (repo / b_path).exists():
            kept.extend(section)
        else:
            skipped.append(b_path)

    if skipped:
        logger.info(
            "[geak apply] Filtered out %d patch section(s) for files not in repo: %s",
            len(skipped),
            ", ".join(skipped),
        )

    result = "".join(kept)
    return result if result.strip() else None


def _try_stash_apply_pop(
    patch_text: str,
    repo: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str] | None:
    """Stash dirty changes, apply the patch, then pop the stash.

    Returns the successful ``CompletedProcess`` from ``git apply``, or
    ``None`` if any step fails (stash state is restored on failure).
    """
    # Check if there is anything to stash
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo), capture_output=True, text=True, check=False, env=env,
    )
    if status.returncode != 0 or not status.stdout.strip():
        # Nothing to stash or git status failed — skip this strategy
        return None

    stash = subprocess.run(
        ["git", "stash", "--quiet"],
        cwd=str(repo), capture_output=True, text=True, check=False, env=env,
    )
    if stash.returncode != 0:
        logger.debug("[geak apply] git stash failed (rc=%s): %s", stash.returncode, stash.stderr.strip())
        return None

    apply_result, _ = apply_patch_with_generated_helper_fallback(
        patch_text=patch_text, cwd=repo, env=env,
    )

    # Always pop the stash regardless of apply outcome
    pop = subprocess.run(
        ["git", "stash", "pop", "--quiet"],
        cwd=str(repo), capture_output=True, text=True, check=False, env=env,
    )
    if pop.returncode != 0:
        logger.warning(
            "[geak apply] git stash pop failed (rc=%s): %s. "
            "Run 'git stash list' in %s to recover.",
            pop.returncode, pop.stderr.strip(), repo,
        )

    if apply_result.returncode == 0:
        return apply_result
    return None


def _apply_patch_to_repo(patch_path: Path, repo: Path) -> tuple[bool, str | None]:
    """Apply ``patch_path`` to ``repo`` with a 3-step fallback chain.

    1. Direct ``git apply`` (works even on dirty repos if no conflicts).
    2. ``git stash`` -> apply -> ``git stash pop`` (isolates dirty changes).
    3. Filter patch to only files existing in repo, then apply.

    Returns ``(success, reason-on-failure)``.
    """
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    env = get_git_safe_env(repo)

    # --- Attempt 1: direct apply ---
    apply_result, removed_paths = apply_patch_with_generated_helper_fallback(
        patch_text=patch_text,
        cwd=repo,
        env=env,
    )
    if removed_paths:
        logger.info(
            "[geak apply] Stripped generated helper artifacts: %s",
            ", ".join(removed_paths),
        )
    if apply_result.returncode == 0:
        logger.info("[geak apply] Applied %s to %s", patch_path.name, repo)
        return True, None

    first_stderr = apply_result.stderr.strip()[:1000]
    logger.info("[geak apply] Direct apply failed; trying stash fallback...")

    # --- Attempt 2: stash → apply → pop ---
    stash_result = _try_stash_apply_pop(patch_text, repo, env)
    if stash_result is not None:
        logger.info("[geak apply] Applied %s to %s (after stash/pop)", patch_path.name, repo)
        return True, None

    logger.info("[geak apply] Stash fallback failed; trying existing-files-only filter...")

    # --- Attempt 3: filter to existing files only ---
    filtered = _filter_patch_to_existing_files(patch_text, repo)
    if filtered and filtered != patch_text:
        filtered_result, _ = apply_patch_with_generated_helper_fallback(
            patch_text=filtered, cwd=repo, env=env,
        )
        if filtered_result.returncode == 0:
            logger.info(
                "[geak apply] Applied %s to %s (existing-files-only filter)",
                patch_path.name, repo,
            )
            return True, None

    # --- All attempts failed ---
    logger.warning(
        "[geak apply] All apply strategies failed for %s.\nFirst stderr: %s",
        patch_path.name, first_stderr,
    )
    reason = (
        f"git apply rc={apply_result.returncode}: {first_stderr}"
        if first_stderr
        else f"git apply rc={apply_result.returncode}"
    )
    return False, reason


def _commit_applied_patch(
    result: BestPatchResult,
    repo: Path,
    patch_files: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """Stage + commit the applied changes. Return (commit SHA, reason-on-failure).

    When *patch_files* is provided, only those paths are staged so that
    pre-existing dirty tracked changes in the repo are left untouched.
    Falls back to ``git add -A`` when *patch_files* is empty/None.
    """
    env = get_git_safe_env(repo)

    if patch_files:
        add = subprocess.run(
            ["git", "add", "--"] + patch_files,
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    else:
        add = subprocess.run(
            ["git", "add", "-A"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    if add.returncode != 0:
        stderr = add.stderr.strip()
        logger.warning("[geak apply] git add failed (rc=%s): %s", add.returncode, stderr)
        return None, f"git add rc={add.returncode}: {stderr}" if stderr else f"git add rc={add.returncode}"

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
        return None, "patch applied cleanly but produced no staged changes"

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
        stderr = commit.stderr.strip()
        logger.warning("[geak apply] git commit failed (rc=%s): %s", commit.returncode, stderr)
        return None, f"git commit rc={commit.returncode}: {stderr}" if stderr else f"git commit rc={commit.returncode}"

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return (sha.stdout.strip() if sha.returncode == 0 else "HEAD"), None


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
