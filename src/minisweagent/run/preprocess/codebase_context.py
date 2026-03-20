"""Codebase context generator -- builds a CODEBASE_CONTEXT.md briefing file.

Runs as Step 2 of the preprocessor pipeline, right after resolve-kernel-url
and before test-discovery. The generated file captures the repository layout
and the kernel's transitive dependency tree so that downstream components
(orchestrator, task generator, sub-agents) can start with full situational
awareness instead of re-exploring the directory structure.

The entire generation is deterministic -- no LLM calls.
"""

from __future__ import annotations

import ast
import logging
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SKIP_DIRS: set[str] = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    "_build",
    ".eggs",
    "*.egg-info",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
}

_MAX_TREE_DEPTH = 4       # Max directory nesting in the repo layout tree (kernel ancestors always expand)
_MAX_TREE_ENTRIES = 300   # Max total entries in the repo layout tree before truncation
_MAX_DEP_FILES = 30       # Max resolved in-repo files in the dependency BFS


# ── Directory tree ────────────────────────────────────────────────────


def _should_skip_dir(name: str) -> bool:
    """Check whether a directory name should be pruned from the tree."""
    if name.startswith("."):
        return True
    if name in _SKIP_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _build_directory_tree(
    root: Path,
    kernel_path: Path,
    *,
    max_depth: int = _MAX_TREE_DEPTH,
    max_entries: int = _MAX_TREE_ENTRIES,
) -> str:
    """Build a pruned directory tree string with annotations.

    Directories along the kernel's ancestor path are always expanded
    regardless of depth limits so the target file is never hidden.
    """
    lines: list[str] = []
    counter = [0]  # To enable passing by reference.
    kernel_abs = kernel_path.resolve()
    root_abs = root.resolve()

    # Pre-compute directories on the path to the kernel so they're always expanded
    kernel_ancestors: set[Path] = set()
    p = kernel_abs.parent
    while p != p.parent:
        kernel_ancestors.add(p)
        if p == root_abs:
            break
        p = p.parent

    def _walk(current: Path, prefix: str, depth: int) -> None:
        if counter[0] >= max_entries:
            return

        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        dirs = [e for e in entries if e.is_dir() and not _should_skip_dir(e.name)]
        files = [e for e in entries if e.is_file()]
        items = dirs + files

        for i, item in enumerate(items):
            if counter[0] >= max_entries:
                lines.append(f"{prefix}└── ... ({len(items) - i} more)")
                counter[0] += 1
                return

            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            annotation = ""
            if item.is_file() and item.resolve() == kernel_abs:
                annotation = "    ← TARGET KERNEL"

            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                counter[0] += 1
                is_kernel_ancestor = item.resolve() in kernel_ancestors
                if depth + 1 < max_depth or is_kernel_ancestor:
                    _walk(item, child_prefix, depth + 1)
                else:
                    try:
                        child_count = sum(1 for _ in item.iterdir())
                    except PermissionError:
                        child_count = 0
                    if child_count > 0:
                        lines.append(f"{child_prefix}... ({child_count} items)")
                        counter[0] += 1
            else:
                lines.append(f"{prefix}{connector}{item.name}{annotation}")
                counter[0] += 1

    root_name = root.name or str(root)
    lines.append(f"{root_name}/")
    counter[0] += 1
    _walk(root, "", 0)

    return "\n".join(lines)


# ── Import extraction ─────────────────────────────────────────────────

_CPP_INCLUDE_RE = re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


@dataclass
class _ImportEntry:
    """A single import statement with the module path and imported names."""
    module: str
    names: list[str]


def _extract_py_imports(source: str) -> list[_ImportEntry]:
    """Use ``ast.parse`` to extract structured import info from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    entries: list[_ImportEntry] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = ("." * (node.level or 0)) + node.module
            names = [a.name for a in node.names if a.name != "*"]

            for name in names:
                qual = f"{module}.{name}"
                if qual not in seen:
                    seen.add(qual)
                    entries.append(_ImportEntry(module=qual, names=[name]))

            if module not in seen:
                seen.add(module)
                entries.append(_ImportEntry(module=module, names=names))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in seen:
                    seen.add(alias.name)
                    entries.append(_ImportEntry(module=alias.name, names=[]))

    return entries


def _extract_imports(kernel_path: Path) -> list[_ImportEntry]:
    """Extract structured import info from a source file.

    For Python files, uses ``ast.parse`` to handle all import styles
    (multi-line, parenthesized, conditional, aliased).  For C/C++ files,
    falls back to a regex for ``#include`` directives.
    """
    try:
        source = kernel_path.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    suffix = kernel_path.suffix.lower()
    if suffix == ".py":
        return _extract_py_imports(source)
    if suffix in (".cpp", ".cc", ".cu", ".hip", ".cuh", ".h", ".hpp"):
        return [_ImportEntry(module=m.group(1), names=[])
                for m in _CPP_INCLUDE_RE.finditer(source)]
    return []


# ── Dependency tree ───────────────────────────────────────────────────


def _resolve_import_to_path(
    module_name: str,
    repo_root: Path,
    source_file: Path,
) -> Path | None:
    """Resolve a Python module name or C++ include to a file under *repo_root*.

    Returns None for stdlib / third-party modules that don't exist in the repo.
    """
    suffix = source_file.suffix.lower()

    if suffix == ".py":
        # Handle relative imports (leading dots)
        stripped = module_name.lstrip(".")
        num_dots = len(module_name) - len(stripped)
        if num_dots > 0:
            base = source_file.parent
            for _ in range(num_dots - 1):
                base = base.parent
            parts = stripped.replace(".", "/") if stripped else ""
            search_base = base / parts if parts else base
        else:
            parts = module_name.replace(".", "/")
            search_base = repo_root / parts

        for candidate in (
            search_base.with_suffix(".py") if search_base.suffix != ".py" else search_base,
            search_base / "__init__.py",
        ):
            if candidate.is_file():
                return candidate.resolve()

        # Fallback: same-directory import (e.g. `from chip_info import ...`
        # when sys.path includes the source file's directory)
        if num_dots == 0 and "/" not in parts:
            for candidate in (
                source_file.parent / f"{parts}.py",
                source_file.parent / parts / "__init__.py",
            ):
                if candidate.is_file():
                    return candidate.resolve()

        return None

    # C/C++ includes: try relative to source dir, include/ subdir, then repo root
    if suffix in (".cpp", ".cc", ".cu", ".hip", ".cuh", ".h", ".hpp"):
        search_dirs = [
            source_file.parent,
            source_file.parent / "include",
            repo_root,
        ]
        for base in search_dirs:
            candidate = base / module_name
            if candidate.is_file():
                return candidate.resolve()
        return None

    return None


# Patterns used by _describe_file to auto-detect file descriptions
_DOCSTRING_RE = re.compile(r'^(?:[ \t]*#[^\n]*\n)*[ \t]*(?:\'\'\'|""")(.+?)(?:\'\'\'|""")', re.DOTALL)  # Python module docstring
_TRITON_JIT_RE = re.compile(r"@triton\.(?:jit|autotune)")  # Triton kernel decorator
_GLOBAL_RE = re.compile(r"__global__\s+void\s+(\w+)")  # HIP/CUDA kernel function


def _describe_file(path: Path) -> str:
    """Auto-detect a brief description of a source file from its content."""
    try:
        source = path.read_text(errors="replace")[:4096]
    except (OSError, UnicodeDecodeError):
        return "Source file"

    # Try module docstring first line
    m = _DOCSTRING_RE.match(source)
    if m:
        first_line = m.group(1).strip().split("\n")[0].strip()
        if len(first_line) > 10:
            return first_line[:120]

    suffix = path.suffix.lower()
    if suffix == ".py":
        if _TRITON_JIT_RE.search(source):
            return "Triton kernel definitions (@triton.jit)"
        if re.search(r"^class\s+\w+", source, re.MULTILINE):
            return "Class definitions"
        return "Python module"

    if suffix in (".cu", ".hip"):
        gm = _GLOBAL_RE.search(source)
        if gm:
            return f"GPU kernel (defines {gm.group(1)})"
        return "GPU kernel source"

    if suffix in (".h", ".hpp", ".cuh"):
        return "Header file"

    return "Source file"


def _build_dependency_tree(
    repo_root: Path,
    kernel_path: Path,
    *,
    max_files: int = _MAX_DEP_FILES,
) -> list[dict]:
    """BFS over imports starting from *kernel_path*.

    Returns a list of resolved in-repo dependencies.  Each entry:
    ``{file, names, imported_by, depth, description}`` where *names*
    lists the specific symbols imported from that file.
    """
    resolved: list[dict] = []
    visited: set[Path] = {kernel_path.resolve()}
    # Track names per resolved file so multiple import lines merge
    file_entry: dict[Path, dict] = {}

    queue: deque[tuple[Path, int]] = deque()
    queue.append((kernel_path, 0))

    while queue and len(resolved) < max_files:
        current, depth = queue.popleft()

        for entry in _extract_imports(current):
            target = _resolve_import_to_path(entry.module, repo_root, current)
            if target is None:
                continue

            # If this file was already resolved, just merge imported names
            if target in file_entry:
                existing = file_entry[target]
                for n in entry.names:
                    if n not in existing["names"]:
                        existing["names"].append(n)
                continue

            if target in visited:
                continue
            visited.add(target)

            try:
                rel_file = target.relative_to(repo_root)
            except ValueError:
                rel_file = target
            try:
                rel_parent = current.relative_to(repo_root)
            except ValueError:
                rel_parent = current

            rec = {
                "file": str(rel_file),
                "names": list(entry.names),
                "imported_by": str(rel_parent),
                "depth": depth + 1,
                "description": _describe_file(target),
            }
            resolved.append(rec)
            file_entry[target] = rec

            if len(resolved) >= max_files:
                break

            queue.append((target, depth + 1))

    return resolved

# ── Main entry point ──────────────────────────────────────────────────


def generate_codebase_context(
    repo_root: Path,
    kernel_path: Path,
    output_dir: Path,
) -> Path:
    """Generate CODEBASE_CONTEXT.md and write it to *output_dir*.

    Parameters
    ----------
    repo_root:
        Root directory of the repository.
    kernel_path:
        Path to the target kernel file.
    output_dir:
        Directory to write CODEBASE_CONTEXT.md into.

    Returns
    -------
    Path to the written CODEBASE_CONTEXT.md file.
    """
    repo_root = Path(repo_root).resolve()
    kernel_path = Path(kernel_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rel_kernel = kernel_path.relative_to(repo_root)
    except ValueError:
        rel_kernel = kernel_path

    sections: list[str] = ["# Codebase Context\n"]

    # 1. Repository layout
    tree = _build_directory_tree(repo_root, kernel_path)
    sections.append("## Repository Layout\n")
    sections.append(f"```\n{tree}\n```\n")

    # 2. Kernel dependency tree
    deps = _build_dependency_tree(repo_root, kernel_path)

    sections.append("## Kernel Dependency Tree\n")
    sections.append(f"Target kernel: `{rel_kernel}`\n")

    # Group by depth
    by_depth: dict[int, list[dict]] = {}
    for dep in deps:
        by_depth.setdefault(dep["depth"], []).append(dep)

    for depth in sorted(by_depth):
        if depth == 1:
            sections.append("### Direct dependencies\n")
            sections.append("| File | Imports | Description |")
            sections.append("|------|---------|-------------|")
            for dep in by_depth[depth]:
                names = ", ".join(f"`{n}`" for n in dep["names"]) if dep["names"] else "*module*"
                sections.append(
                    f"| `{dep['file']}` | {names} | {dep['description']} |"
                )
        else:
            sections.append(f"\n### Transitive dependencies (depth {depth})\n")
            sections.append(
                "Improving these may improve the target kernel's performance.\n"
            )
            sections.append("| File | Imports | Used by | Description |")
            sections.append("|------|---------|---------|-------------|")
            for dep in by_depth[depth]:
                names = ", ".join(f"`{n}`" for n in dep["names"]) if dep["names"] else "*module*"
                sections.append(
                    f"| `{dep['file']}` | {names} | `{dep['imported_by']}` | {dep['description']} |"
                )
        sections.append("")

    if not deps:
        sections.append("No in-repo dependencies found.\n")

    out_path = output_dir / "CODEBASE_CONTEXT.md"
    content = "\n".join(sections)
    out_path.write_text(content)
    logger.info("Wrote codebase context to %s (%d bytes)", out_path, len(content))

    return out_path


# ── CLI entry point ───────────────────────────────────────────────────


def main() -> None:
    """CLI: ``codebase-context --repo-root <dir> --kernel-path <file> -o <dir>``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate CODEBASE_CONTEXT.md from repo layout and kernel file",
    )
    parser.add_argument("--repo-root", required=True, help="Root directory of the repository")
    parser.add_argument("--kernel-path", required=True, help="Path to the target kernel file")
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Output directory for CODEBASE_CONTEXT.md (default: cwd)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    kernel_path = Path(args.kernel_path).resolve()
    output_dir = Path(args.output).resolve()

    if not repo_root.is_dir():
        print(f"ERROR: repo root not found: {repo_root}", file=sys.stderr)
        sys.exit(1)
    if not kernel_path.is_file():
        print(f"ERROR: kernel file not found: {kernel_path}", file=sys.stderr)
        sys.exit(1)

    out_path = generate_codebase_context(repo_root, kernel_path, output_dir)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()