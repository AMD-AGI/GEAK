"""Task file utilities -- read/write Markdown task files with YAML frontmatter.

Task files are the intermediate format between task-generator and downstream
tools (openevolve-worker, geak). Each file has YAML frontmatter with metadata
and a Markdown body with the full task prompt.

Also provides git worktree helpers extracted from ParallelAgent so that
CLI tools can create isolated work directories.
"""

from __future__ import annotations

import logging
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
_PATH_KEYS = (
    "kernel_path",
    "repo_root",
    "commandment",
    "baseline_metrics",
    "profiling",
    "codebase_context",
    "starting_patch",
)


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
        if k in _PATH_KEYS and isinstance(v, (str, Path)) and v:
            # Resolve path-valued fields against the writer's CWD before
            # storing. Without this, a caller passing a CWD-relative string
            # (e.g. "outputs/foo/kernel.py") would be written verbatim, and
            # ``read_task_file`` would later resolve it against the task
            # file's own directory — producing a nonsense path like
            # ``<task_dir>/outputs/foo/kernel.py``. By normalising to
            # absolute on write, the read-side resolution becomes a no-op
            # for absolute paths, regardless of who later opens the file.
            abs_path = Path(v).resolve()
            if relative_to:
                try:
                    fm[k] = os.path.relpath(abs_path, relative_to.resolve())
                except ValueError:
                    fm[k] = str(abs_path)
            else:
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


def _neutralize_nested_git_repos(root: Path) -> list[Path]:
    """Rename ``.git`` dirs in nested repos to ``.git.bak``.

    This turns nested git repos / submodules into plain directories so that
    git treats their content as regular files.
    """
    root = root.resolve()
    renamed: list[Path] = []
    for git_dir in root.rglob(".git"):
        if git_dir.parent == root:
            continue
        if git_dir.is_dir():
            backup = git_dir.parent / ".git.bak"
            try:
                if backup.exists():
                    shutil.rmtree(backup)
                git_dir.rename(backup)
                renamed.append(backup)
            except Exception:
                pass
    return renamed


def _resolve_output_root(repo_path: Path, worktree_path: Path) -> Path | None:
    """Return the top-level GEAK output directory inside repo_path, if any.

    When the worktree is created inside the repo (e.g.
    ``<repo>/optimization_logs/<run>/results/.../slot_N``), returns the first
    directory component relative to repo_path (e.g. ``<repo>/optimization_logs``).
    Returns None if the worktree is not inside the repo.
    """
    try:
        relative = worktree_path.relative_to(repo_path)
    except ValueError:
        return None
    if not relative.parts:
        return None
    return repo_path / relative.parts[0]


def _copy_nested_git_repos(repo_path: Path, worktree_path: Path) -> None:
    """Copy nested git repositories that are invisible to the top-level repo.

    `git worktree add` and `git ls-files` skip directories containing their own
    `.git`. This function finds all such nested repos under *repo_path* and
    copies them into *worktree_path* so the worktree has a complete snapshot.
    """
    repo_path = repo_path.resolve()
    worktree_path = worktree_path.resolve()
    top_git = repo_path / ".git"
    output_root = _resolve_output_root(repo_path, worktree_path)

    for dirpath, dirnames, _filenames in os.walk(repo_path):
        current = Path(dirpath)
        # Skip the top-level .git directory itself and anything inside worktrees
        if current == top_git or ".git" in current.parts[len(repo_path.parts) :]:
            dirnames.clear()
            continue

        # Skip the worktree directory itself to avoid infinite recursion when
        # the worktree is created inside the repo (e.g. optimization_logs/).
        if current == worktree_path or worktree_path in current.parents:
            dirnames.clear()
            continue

        # Skip the entire GEAK output directory to prevent recursive copying
        # of artifacts from previous runs (fixes #189, #181).
        if output_root is not None:
            try:
                current.relative_to(output_root)
                dirnames.clear()
                continue
            except ValueError:
                pass

        if current != repo_path and (".git" in dirnames or (current / ".git").is_file()):
            # current is a nested git repo root — skip descending further
            # via os.walk (copytree will handle the full subtree).
            dirnames.clear()

            rel = current.relative_to(repo_path)
            dst = worktree_path / rel
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(current, dst, symlinks=True)


def _copy_untracked_files(repo_path: Path, worktree_path: Path, env: dict[str, str] | None = None) -> None:
    """Copy untracked files from repo to worktree."""
    run_env = env if env is not None else None
    resolved_repo = repo_path.resolve()
    resolved_wt = worktree_path.resolve()
    output_root = _resolve_output_root(resolved_repo, resolved_wt)
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
            # Skip files inside the worktree to avoid infinite recursion when
            # the worktree is created inside the repo.
            try:
                src.resolve().relative_to(resolved_wt)
                continue
            except ValueError:
                pass
            # Skip files inside the GEAK output directory to prevent recursive
            # copying of artifacts from previous runs (fixes #189, #181).
            if output_root is not None:
                try:
                    src.resolve().relative_to(output_root)
                    continue
                except ValueError:
                    pass
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    except subprocess.CalledProcessError:
        pass


def _apply_dirty_tracked_changes(repo_path: Path, worktree_path: Path, env: dict[str, str] | None = None) -> None:
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
        raise RuntimeError(f"Failed to sync dirty tracked files into worktree: {error_text[:500]}")


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
        pass  # best-effort worktree prune; failure is harmless

    # Remove directory if it still exists
    if worktree_path.exists():
        try:
            shutil.rmtree(worktree_path)
        except Exception:
            pass  # best-effort directory cleanup

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
            subprocess.run(
                ["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True, env=git_env
            )
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
            subprocess.run(
                ["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True, env=git_env
            )
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
    _copy_nested_git_repos(repo_path, worktree_path)
    # Neutralize .git dirs in copied nested repos so the worktree's git
    # treats their content as regular files (clean diffs, no gitlink noise).
    _neutralize_nested_git_repos(worktree_path)
    return worktree_path


def create_worktree_with_patch(
    repo_path: Path,
    worktree_path: Path,
    patch_file: str | Path,
) -> Path:
    """Create a git worktree and apply a cumulative starting patch.

    Used to make round N+1 agents start from round N's best kernel.

    The starting patch is *cumulative* -- it encodes the full delta from
    HEAD to the desired base state (including all prior round changes).
    We therefore skip the dirty-tracked / untracked sync that
    ``create_worktree`` normally performs; those synced changes would
    conflict with the patch's own context lines, causing ``git apply``
    to fail with "patch does not apply" or "already exists".
    """
    log = logging.getLogger(__name__)

    patch_path = Path(patch_file)
    if not patch_path.exists() or patch_path.stat().st_size == 0:
        log.info(
            "No valid starting patch (%s); creating worktree with dirty/untracked sync",
            patch_file,
        )
        return create_worktree(repo_path, worktree_path)

    log.info(
        "Creating clean-HEAD worktree (no dirty sync) for cumulative starting patch: %s",
        Path(patch_file).name,
    )
    wt = _create_worktree_clean(repo_path, worktree_path)
    git_env = get_git_safe_env(worktree_path.parent)
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    result, removed_paths = apply_patch_with_generated_helper_fallback(
        patch_text=patch_text,
        cwd=wt,
        env=git_env,
    )
    if result.returncode != 0:
        log.warning(
            "Failed to apply starting patch %s on clean HEAD: %s",
            patch_file,
            result.stderr[:500],
        )
    elif removed_paths:
        log.warning(
            "Applied starting patch after stripping generated helper artifacts: %s",
            ", ".join(removed_paths[:5]),
        )
    else:
        log.info("Starting patch applied successfully on clean HEAD worktree")
    return wt


def _create_worktree_clean(repo_path: Path, worktree_path: Path) -> Path:
    """Create a git worktree at HEAD without syncing dirty or untracked files.

    This is used by ``create_worktree_with_patch`` where a cumulative patch
    will be applied on top of a pristine HEAD checkout.  Skipping the
    dirty/untracked sync avoids conflicts between the main repo's working
    tree state and the patch's context lines.
    """
    log = logging.getLogger(__name__)
    worktree_str = str(worktree_path.resolve())
    git_env = get_git_safe_env(worktree_path.parent)

    _cleanup_existing_worktree(repo_path, worktree_path, worktree_str, git_env)

    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_safe_directory(repo_path, git_env)

    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_path)],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    )
    _ensure_safe_directory(worktree_path, git_env)
    log.info("Clean-HEAD worktree created at %s (no dirty/untracked sync)", worktree_path)
    return worktree_path


def _cleanup_existing_worktree(
    repo_path: Path,
    worktree_path: Path,
    worktree_str: str,
    git_env: dict[str, str],
) -> None:
    """Remove a previously registered worktree, falling back to prune."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            env=git_env,
        )
        if not any(worktree_str in line or str(worktree_path) in line for line in result.stdout.splitlines()):
            return
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            env=git_env,
        )
    except (subprocess.CalledProcessError, OSError):
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
            env=git_env,
        )


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
