"""Task file utilities -- read/write Markdown task files with YAML frontmatter.

Task files are the intermediate format between task-generator and downstream
tools (openevolve-worker, geak). Each file has YAML frontmatter with metadata
and a Markdown body with the full task prompt.

Also provides git worktree helpers extracted from ParallelAgent so that
CLI tools can create isolated work directories.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from minisweagent.run.utils.generated_artifacts import apply_patch_with_generated_helper_fallback
from minisweagent.run.utils.git_safe_env import get_git_safe_env

# ============================================================================
# Task file I/O
# ============================================================================

# Path keys in frontmatter that should be stored as relative and resolved on read
_PATH_KEYS = ("kernel_path", "repo_root", "commandment", "baseline_metrics", "profiling", "codebase_context", "starting_patch")


def write_task_file(
    path: Path,
    metadata: dict[str, Any],
    body: str,
    *,
    relative_to: Path | None = None,
) -> None:
    """Write a task file with YAML frontmatter and Markdown body.

    Args:
        path: Output file path.
        metadata: Dict of frontmatter fields. Path-valued keys in _PATH_KEYS
                  are converted to relative paths if *relative_to* is set.
        body: Markdown body (the full task prompt).
        relative_to: If set, path-valued frontmatter fields are made relative
                     to this directory.
    """
    fm = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if relative_to and k in _PATH_KEYS and isinstance(v, (str, Path)):
            abs_path = Path(v).resolve()
            try:
                fm[k] = os.path.relpath(abs_path, relative_to.resolve())
            except ValueError:
                fm[k] = str(abs_path)
        else:
            fm[k] = v

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(yaml.dump(fm, default_flow_style=False, sort_keys=False))
        f.write("---\n\n")
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")


def read_task_file(path: Path) -> tuple[dict[str, Any], str]:
    """Read a task file and return (metadata, body).

    Path-valued fields in metadata are resolved to absolute paths relative
    to the task file's directory.
    """
    text = Path(path).read_text(encoding="utf-8")

    # Split on --- delimiters
    parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
    if len(parts) < 3:
        raise ValueError(f"Task file {path} does not have valid YAML frontmatter (need --- delimiters)")

    fm_text = parts[1]
    body = parts[2].lstrip("\n")

    metadata = yaml.safe_load(fm_text) or {}
    task_dir = Path(path).resolve().parent

    # Resolve relative paths to absolute
    for key in _PATH_KEYS:
        if metadata.get(key):
            rel = metadata[key]
            resolved = (task_dir / rel).resolve()
            metadata[key] = str(resolved)

    return metadata, body


# ============================================================================
# Git worktree helpers (extracted from ParallelAgent)
# ============================================================================


def _ensure_safe_directory(repo_path: Path, env: dict[str, str] | None = None) -> None:
    """Ensure repository is in git's safe.directory list."""
    repo_path_str = str(repo_path.resolve())
    run_env = env if env is not None else None
    try:
        result = subprocess.run(
            ["git", "config", "--global", "--get-all", "safe.directory"],
            capture_output=True,
            text=True,
            env=run_env,
        )
        safe_dirs = result.stdout.strip().split("\n") if result.stdout.strip() else []
        if repo_path_str not in safe_dirs:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", repo_path_str],
                check=True,
                capture_output=True,
                text=True,
                env=run_env,
            )
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", repo_path_str],
                check=True,
                capture_output=True,
                text=True,
                env=run_env,
            )
        except subprocess.CalledProcessError:
            pass


def _copy_untracked_files(
    repo_path: Path, worktree_path: Path, env: dict[str, str] | None = None
) -> None:
    """Copy untracked files from repo to worktree."""
    run_env = env if env is not None else None
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            env=run_env,
        )
        for rel_path in (f.strip() for f in result.stdout.splitlines() if f.strip()):
            src = repo_path / rel_path
            dst = worktree_path / rel_path
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    except subprocess.CalledProcessError:
        pass


def _apply_dirty_tracked_changes(
    repo_path: Path, worktree_path: Path, env: dict[str, str] | None = None
) -> None:
    """Apply tracked-but-uncommitted repo changes to the fresh worktree.

    `git worktree add` checks out `HEAD`, which omits local tracked edits in the
    source repo. Apply that dirty diff so worker slots run the exact live code.
    """
    run_env = env if env is not None else None
    try:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            env=run_env,
        )
    except subprocess.CalledProcessError:
        return

    patch_text = result.stdout
    if not patch_text.strip():
        return

    apply_result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "--binary", "-"],
        cwd=worktree_path,
        input=patch_text,
        capture_output=True,
        text=True,
        env=run_env,
    )
    if apply_result.returncode != 0:
        error_text = apply_result.stderr or apply_result.stdout or "unknown error"
        raise RuntimeError(
            "Failed to sync dirty tracked files into worktree: "
            f"{error_text[:500]}"
        )


def create_worktree(repo_path: Path, worktree_path: Path) -> Path:
    """Create a git worktree, cleaning up any existing one first.

    Extracted from ParallelAgent._create_worktree() for reuse by CLI tools.

    Args:
        repo_path: Path to the git repository.
        worktree_path: Desired path for the new worktree.

    Returns:
        The worktree path (same as input, for chaining).
    """
    worktree_str = str(worktree_path.resolve())
    git_env = get_git_safe_env(worktree_path.parent)

    # Clean up any existing worktree at this path
    try:
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            env=git_env,
        )
        worktree_exists = any(worktree_str in line or str(worktree_path) in line for line in result.stdout.splitlines())
        if worktree_exists:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", str(worktree_path), "--force"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=git_env,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=repo_path,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=git_env,
                )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
            env=git_env,
        )
    except Exception:
        pass

    # Remove directory if it still exists
    if worktree_path.exists():
        try:
            shutil.rmtree(worktree_path)
        except Exception:
            pass

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_safe_directory(repo_path, git_env)

    # Create new worktree with detached HEAD
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_path)],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            env=git_env,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        if "missing but already registered worktree" in error_msg.lower():
            subprocess.run(["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True, env=git_env)
            subprocess.run(
                ["git", "worktree", "add", "--detach", "-f", str(worktree_path)],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                env=git_env,
            )
        elif "dubious ownership" in error_msg.lower():
            _ensure_safe_directory(repo_path, git_env)
            _ensure_safe_directory(worktree_path, git_env)
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_path)],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                env=git_env,
            )
        elif "already used by worktree" in error_msg.lower():
            subprocess.run(["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True, env=git_env)
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=repo_path,
                check=False,
                capture_output=True,
                text=True,
                env=git_env,
            )
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_path)],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                env=git_env,
            )
        else:
            raise RuntimeError(f"Failed to create worktree: {error_msg}") from e

    _ensure_safe_directory(worktree_path, git_env)
    _apply_dirty_tracked_changes(repo_path, worktree_path, git_env)
    _copy_untracked_files(repo_path, worktree_path, git_env)
    return worktree_path


def create_worktree_with_patch(
    repo_path: Path, worktree_path: Path, patch_file: str | Path,
) -> Path:
    """Create a git worktree and apply a starting patch.

    Used to make round N+1 agents start from round N's best kernel.
    """
    wt = create_worktree(repo_path, worktree_path)
    patch_path = Path(patch_file)
    if not patch_path.exists() or patch_path.stat().st_size == 0:
        return wt
    git_env = get_git_safe_env(worktree_path.parent)
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    result, removed_paths = apply_patch_with_generated_helper_fallback(
        patch_text=patch_text,
        cwd=wt,
        env=git_env,
    )
    if result.returncode != 0:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to apply starting patch %s: %s", patch_file, result.stderr[:500],
        )
    elif removed_paths:
        import logging
        logging.getLogger(__name__).warning(
            "Applied starting patch after stripping generated helper artifacts: %s",
            ", ".join(removed_paths[:5]),
        )
    return wt


def replace_paths(text: str, repo_path: Path, worktree_path: Path) -> str:
    """Replace repo paths with worktree paths in text."""
    repo_str = str(repo_path.resolve())
    wt_str = str(worktree_path.resolve())
    return text.replace(repo_str, wt_str)


def is_git_repo(path: Path) -> bool:
    """Check if a path is inside a git repository."""
    try:
        base = path if path.is_dir() else path.parent
        git_env = get_git_safe_env(base)
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            env=git_env,
        )
        return result.stdout.strip() == "true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
