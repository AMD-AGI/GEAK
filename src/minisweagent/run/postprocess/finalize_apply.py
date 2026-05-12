"""Post-run hooks: apply best patch + commit, and clean up run artifacts.

Exposes two independent operations and a coordinator:

- ``apply_and_commit_best_patch(result, repo)`` -- applies the winning
  ``.diff`` (from ``BestPatchResult.best_patch_file``) to ``repo`` on the
  current branch using the same ``git apply`` fallback helper as the
  per-round eval worktree logic, then commits with a message that points
  at the run's ``final_report.json``. Returns the commit SHA or None
  (back-compat). Sibling ``apply_and_commit_best_patch_detailed`` returns
  a structured outcome dict for callers that want to render UX from it.
- ``cleanup_run_artifacts(result, output_dir)`` -- iterates ``output_dir``
  and deletes every top-level entry not in the keep-set
  (``final_report.json``, the agent log, ``COMMANDMENT.md``, and the
  winning ``.diff`` when present). Preserves the ``output_dir`` inode so
  the live ``FileHandler`` on the agent log keeps writing through cleanup
  and any later traceback. Returns a ``cleanup_status`` string:
  ``"ran"`` / ``"failed"`` / ``"skipped_empty"`` / ``"skipped_disabled"``.
- ``finalize_apply_and_cleanup(result, repo, output_dir, *, apply_best_patch, cleanup)``
  -- CLI-level entry point that runs either/both according to the boolean
  flags. Returns ``{apply_status, cleanup_status, commit_sha, reason}``.

Failure semantics:

- Apply fails  -> no commit. Cleanup still runs if requested.
- Commit fails -> apply stays in the working tree. Cleanup still runs if
  requested.
- Cleanup per-entry failure -> logged and counted; the loop continues;
  ``cleanup_status`` lands on ``"failed"``. The keep-set is still on disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.run.utils.generated_artifacts import apply_patch_with_generated_helper_fallback
from minisweagent.run.utils.git_safe_env import get_git_safe_env
from minisweagent.utils.log import DEFAULT_LOG_FILENAME

logger = logging.getLogger(__name__)


_FINAL_REPORT_NAME = "final_report.json"
_COMMANDMENT_NAME = "COMMANDMENT.md"

# Files at the top level of output_dir that ``cleanup_run_artifacts`` always
# preserves. The per-call effective keep-set adds the winning patch basename
# when ``result.best_patch_file`` exists. ``DEFAULT_LOG_FILENAME`` is imported
# from minisweagent.utils.log so a rename there cannot silently desync.
_PRESERVED_FILES: tuple[str, ...] = (
    _FINAL_REPORT_NAME,
    DEFAULT_LOG_FILENAME,
    _COMMANDMENT_NAME,
)

# Maximum string length that ``_rewrite_kept_report`` will attempt to
# normalize as a path. No legitimate filesystem path is anywhere near 4 KiB;
# this guards against rewriting accidentally large fields that slipped past
# the key allow-list.
_PATH_REWRITE_MAX_LEN = 4096

# Scalar keys in ``final_report.json`` whose value is a single path string.
# These are the only fields ``_rewrite_kept_report`` will normalize besides
# any key whose name ends in ``_path`` / ``_dir`` / ``_file`` / ``_patch``.
_SCALAR_PATH_KEYS: frozenset[str] = frozenset({"best_patch", "kernel_path", "repo_root"})

# Keys whose values are LLM-authored documents (free-text). Anything nested
# under these is skipped entirely by ``_rewrite_kept_report``, even if a
# nested key would otherwise match the path-key heuristic.
_DOCUMENT_KEYS: frozenset[str] = frozenset(
    {"summary", "agent_summary", "verification_note", "round_evaluation", "round_summaries"}
)

_PATH_KEY_SUFFIXES: tuple[str, ...] = ("_path", "_dir", "_file", "_patch")

# Defaults used when the container / host has no configured git identity.
# Override via the ``GEAK_GIT_AUTHOR_NAME`` / ``GEAK_GIT_AUTHOR_EMAIL`` env
# vars (set at container run time or by the developer).
_DEFAULT_GIT_AUTHOR_NAME = "GEAK Agent"
_DEFAULT_GIT_AUTHOR_EMAIL = "geak@amd.com"

# Auto-generated run dir name pattern emitted by ``generate_patch_output_dir``.
# Anchored only on the ``_YYYYmmdd_HHMMSS`` suffix so real kernel names
# (which may contain ``.``, ``-``, uppercase) all match.
_AUTO_RUN_DIR_RE = re.compile(r"^.+_\d{8}_\d{6}$")


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

        {"status": "committed" | "skipped_dirty" | "skipped_precondition"
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

    if not _repo_is_clean(repo):
        reason = (
            f"repo {repo} has uncommitted tracked changes. "
            "Commit or stash them first, then re-run apply manually."
        )
        logger.warning("[geak apply] Skipping: %s", reason)
        return {"status": "skipped_dirty", "commit_sha": None, "reason": reason}

    applied, apply_reason = _apply_patch_to_repo(patch_path, repo)
    if not applied:
        return {"status": "apply_failed", "commit_sha": None, "reason": apply_reason}

    commit_sha, commit_reason = _commit_applied_patch(result, repo)
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
    """Prune ``output_dir`` to the keep-set in place.

    Top-level entries surviving cleanup: ``final_report.json``,
    ``geak_agent.log`` (canonical agent log), ``COMMANDMENT.md``, and the
    winning ``.diff`` when present. Independent of apply: safe to call whether
    or not the patch was applied/committed.

    Iterates over ``output_dir.iterdir()`` and deletes top-level entries not
    in the keep-set. The directory's own inode is never unlinked, so the
    ``FileHandler`` already writing to ``output_dir / DEFAULT_LOG_FILENAME``
    keeps its FD live through cleanup, the post-cleanup log line, and any
    later traceback.

    Returns one of:

    - ``"ran"`` -- the iterate-and-delete loop finished with zero per-entry
      exceptions.
    - ``"failed"`` -- the loop completed but at least one entry's deletion
      raised and was logged. The keep-set is still on disk.
    - ``"skipped_empty"`` -- precondition check failed. Two sub-cases: (a)
      ``output_dir`` is missing/not a dir; (b) neither ``final_report.json``
      nor a winning patch file exists. Nothing is touched.
    """
    if not _validate_cleanup_preconditions(result, output_dir):
        return "skipped_empty"

    assert output_dir is not None  # for type narrowing
    output_dir = Path(output_dir).resolve()
    patch_path = (
        Path(result.best_patch_file).resolve()
        if (result is not None and result.best_patch_file)
        else None
    )

    return _cleanup_artifacts(output_dir, patch_path)


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


def _validate_cleanup_preconditions(
    result: BestPatchResult | None,
    output_dir: Path | None,
) -> bool:
    """Return True iff there's a non-empty run worth pruning.

    False in two cases (both map to ``cleanup_status="skipped_empty"`` in
    :func:`cleanup_run_artifacts`):

    1. ``output_dir`` is None / missing / not a directory.
    2. ``output_dir`` exists but neither ``final_report.json`` nor a
       readable winning patch file at ``result.best_patch_file`` is there.
       Without either of those there's nothing the cleanup would preserve,
       so we leave whatever's on disk untouched for post-mortem.
    """
    if output_dir is None:
        logger.warning("[geak --cleanup] Skipping: no output_dir provided.")
        return False
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_dir():
        logger.warning(
            "[geak --cleanup] Skipping: output_dir does not exist or is not a dir: %s",
            output_dir,
        )
        return False

    has_report = (output_dir_path / _FINAL_REPORT_NAME).is_file()
    has_patch = bool(
        result is not None
        and result.best_patch_file
        and Path(result.best_patch_file).is_file()
    )
    if not (has_report or has_patch):
        logger.warning(
            "[geak --cleanup] Skipping: %s has neither %s nor a winning patch; "
            "leaving the directory intact for post-mortem.",
            output_dir_path,
            _FINAL_REPORT_NAME,
        )
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


def _apply_patch_to_repo(patch_path: Path, repo: Path) -> tuple[bool, str | None]:
    """Apply ``patch_path`` to ``repo``. Return (success, reason-on-failure)."""
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
        stderr_tail = apply_result.stderr.strip()[:1000]
        logger.warning(
            "[geak --cleanup] git apply failed (rc=%s); leaving repo and artifacts untouched.\nstderr: %s",
            apply_result.returncode,
            stderr_tail,
        )
        reason = f"git apply rc={apply_result.returncode}: {stderr_tail}" if stderr_tail else f"git apply rc={apply_result.returncode}"
        return False, reason

    logger.info("[geak --cleanup] Applied %s to %s", patch_path, repo)
    return True, None


def _commit_applied_patch(
    result: BestPatchResult,
    repo: Path,
) -> tuple[str | None, str | None]:
    """Stage + commit the applied changes. Return (commit SHA, reason-on-failure)."""
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
        stderr = add.stderr.strip()
        logger.warning("[geak apply] git add -A failed (rc=%s): %s", add.returncode, stderr)
        return None, f"git add -A rc={add.returncode}: {stderr}" if stderr else f"git add -A rc={add.returncode}"

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


def _cleanup_artifacts(
    output_dir: Path,
    patch_path: Path | None,
) -> str:
    """Iterate-and-delete cleanup. Returns "ran" or "failed".

    Approach: move the winning patch (if nested) up to ``output_dir`` root,
    prune lingering ``git worktree`` admin entries, then loop over
    ``output_dir.iterdir()`` and delete every entry whose name is not in
    the effective keep-set. The ``output_dir`` inode is never unlinked, so
    the agent log's open ``FileHandler`` FD survives.

    Per-entry exceptions are caught, logged, and counted. The loop never
    aborts: if a worktree slot is locked or an NFS mount glitches, we still
    finish removing everything else. The return value flips from ``"ran"``
    to ``"failed"`` if any entry raised.
    """
    _prune_worktrees_under(output_dir)

    # Move the winning patch up to output_dir root so the keep-set check
    # operates uniformly on top-level entries.
    kept_patch_name: str | None = None
    if patch_path is not None and patch_path.is_file():
        kept_patch_name = patch_path.name
        target = output_dir / kept_patch_name
        if patch_path != target:
            try:
                shutil.copy2(patch_path, target)
            except OSError as exc:
                logger.warning(
                    "[geak --cleanup] Failed to copy winning patch %s to %s: %s",
                    patch_path,
                    target,
                    exc,
                )
                kept_patch_name = None

    keep_set: set[str] = set(_PRESERVED_FILES)
    if kept_patch_name is not None:
        keep_set.add(kept_patch_name)

    failed = 0
    for entry in list(output_dir.iterdir()):
        if entry.name in keep_set:
            continue
        try:
            if entry.is_symlink():
                # Never recurse through a symlink-to-dir: rmtree would error
                # or worse, follow into the target. unlink() removes just
                # the symlink itself.
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry, ignore_errors=False)
            else:
                entry.unlink()
        except OSError as exc:
            failed += 1
            logger.warning(
                "[geak --cleanup] Failed to remove %s: %s (continuing).",
                entry,
                exc,
            )

    _rewrite_kept_report(
        output_dir,
        (output_dir / kept_patch_name) if kept_patch_name is not None else None,
    )

    if failed:
        logger.warning(
            "[geak --cleanup] Completed with %d per-entry failure(s); kept %s.",
            failed,
            ", ".join(sorted(keep_set)),
        )
        return "failed"

    logger.info(
        "[geak --cleanup] Pruned %s; kept %s.",
        output_dir,
        ", ".join(sorted(keep_set)),
    )
    return "ran"


def _rewrite_kept_report(output_dir: Path, kept_patch_path: Path | None) -> None:
    """Rewrite paths in the kept ``final_report.json`` to match post-cleanup reality.

    Uses a KEY ALLOW-LIST rather than a value-shape walk. ``final_report.json``
    contains LLM-authored fields (``summary``, ``agent_summary``, etc.) that
    routinely paste paths into free-text prose; a value-walk would mangle
    them and break downstream consumers (e.g. ``record_optimization_outcome``
    reads ``summary[:100]`` as a strategy name).

    Rewrites:

    - ``best_patch``: absolute path of the surviving file, or ``None`` when
      no patch survived. Key is always present after this function returns.
    - Other scalar keys in ``_SCALAR_PATH_KEYS`` and any key whose name ends
      in ``_path`` / ``_dir`` / ``_file`` / ``_patch``: if the value resolves
      under ``output_dir`` and the path no longer exists, replace with
      ``{"path": <original>, "pruned": True}``.

    Hard-skips ``_DOCUMENT_KEYS`` (and anything nested under them).
    """
    report_path = output_dir / _FINAL_REPORT_NAME
    if not report_path.is_file():
        return

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("[geak --cleanup] Could not read %s for normalization: %s", report_path, exc)
        return

    if not isinstance(report, dict):
        return

    if kept_patch_path is not None:
        report["best_patch"] = str(kept_patch_path.resolve())
    else:
        report["best_patch"] = None

    output_dir_resolved = output_dir.resolve()

    def _is_path_key(name: str) -> bool:
        return name in _SCALAR_PATH_KEYS or name.endswith(_PATH_KEY_SUFFIXES)

    def _rewrite_string(value: str) -> object:
        if len(value) > _PATH_REWRITE_MAX_LEN:
            return value
        try:
            resolved = Path(value).resolve()
        except (OSError, ValueError):
            return value
        try:
            resolved.relative_to(output_dir_resolved)
        except ValueError:
            # Outside output_dir -- external paths stay untouched.
            return value
        if resolved.exists():
            return value
        return {"path": value, "pruned": True}

    def _walk(node: object, key_name: str | None) -> object:
        # Hard-skip document keys: do not descend into them.
        if key_name in _DOCUMENT_KEYS:
            return node
        if isinstance(node, dict):
            return {k: _walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            # Lists inherit eligibility from the enclosing key. Only walk
            # them if the key name itself looks like a path key (so e.g.
            # `kernel_paths` would be walked, but `agent_summaries` would
            # not). Practically every list-of-paths in final_report.json
            # lives behind such a key.
            if key_name is not None and _is_path_key(key_name):
                return [_walk(item, key_name) for item in node]
            return node
        if isinstance(node, str) and key_name is not None and _is_path_key(key_name):
            # Skip best_patch -- already set above by direct assignment.
            if key_name == "best_patch":
                return node
            return _rewrite_string(node)
        return node

    rewritten = _walk(report, None)
    assert isinstance(rewritten, dict)

    try:
        report_path.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("[geak --cleanup] Could not write normalized %s: %s", report_path, exc)


def prune_old_runs(
    parent: Path,
    keep: int,
    *,
    exclude: Path | None = None,
    stale_after_s: float = 600.0,
) -> int:
    """Keep the N most recent auto-generated run dirs under ``parent``.

    Returns the number of directories actually removed.

    Behavior:

    - Returns 0 cleanly when ``parent`` is missing or not a directory.
    - Only touches directories whose name matches ``^.+_\\d{8}_\\d{6}$``
      (the suffix ``generate_patch_output_dir`` emits). Anything else under
      ``parent`` -- notes, unrelated dirs, files -- is left alone regardless
      of mtime.
    - Skips ``exclude`` even if it would otherwise be a delete candidate.
    - Skips any dir whose effective freshness is within the last
      ``stale_after_s`` seconds. Freshness key, first hit wins:

      1. ``(dir / DEFAULT_LOG_FILENAME).stat().st_mtime`` if the log exists
         (a long-running but actively-logging sibling looks fresh because
         ``FileHandler`` bumps this on every line).
      2. ``max(child.stat().st_mtime for child in dir.iterdir())`` if any
         children remain.
      3. ``dir.stat().st_mtime`` as a last-resort fallback (only bumped on
         child create/rename/unlink in the dir, not on appends).

    Never raises; per-dir failures are logged and counted as not-removed.
    """
    if keep < 0:
        keep = 0
    parent = Path(parent)
    if not parent.is_dir():
        return 0

    exclude_resolved: Path | None = None
    if exclude is not None:
        try:
            exclude_resolved = Path(exclude).resolve()
        except (OSError, ValueError):
            exclude_resolved = None

    now = time.time()

    candidates: list[tuple[float, Path]] = []
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        if not _AUTO_RUN_DIR_RE.match(entry.name):
            continue
        if exclude_resolved is not None:
            try:
                if entry.resolve() == exclude_resolved:
                    continue
            except (OSError, ValueError):
                pass
        freshness = _freshness_mtime(entry)
        if freshness is None:
            # Unreadable dir; skip rather than risk deletion.
            continue
        if (now - freshness) < stale_after_s:
            # Actively-fresh: protect against concurrent runs sharing
            # ``parent``. The most-common case is another geak invocation
            # currently writing to ``geak_agent.log``.
            continue
        candidates.append((freshness, entry))

    if len(candidates) <= keep:
        return 0

    # Newest first; the tail past `keep` is the delete-list.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    to_delete = [path for _, path in candidates[keep:]]

    removed = 0
    for path in to_delete:
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            logger.warning("[geak --keep-runs] Failed to remove %s: %s", path, exc)
    return removed


def _freshness_mtime(run_dir: Path) -> float | None:
    """Best-effort 'how recently was this dir touched' for the stale filter."""
    log_path = run_dir / DEFAULT_LOG_FILENAME
    try:
        if log_path.is_file():
            return log_path.stat().st_mtime
    except OSError:
        pass

    try:
        child_mtimes = []
        for child in run_dir.iterdir():
            try:
                child_mtimes.append(child.stat().st_mtime)
            except OSError:
                continue
        if child_mtimes:
            return max(child_mtimes)
    except OSError:
        pass

    try:
        return run_dir.stat().st_mtime
    except OSError:
        return None


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
