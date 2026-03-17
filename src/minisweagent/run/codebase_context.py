"""Codebase context generator -- builds a CODEBASE_CONTEXT.md briefing file.

Runs as Step 2 of the preprocessor pipeline, right after resolve-kernel-url
and before test-discovery. The generated file captures the repository layout,
key files, and the kernel's import chain so that downstream components
(discovery, orchestrator, task generator, sub-agents) can start with full
situational awareness instead of re-exploring the directory structure.

The entire generation is deterministic -- no LLM calls.
"""

from __future__ import annotations

import logging
import re
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

_CONFIG_FILES: set[str] = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "CMakeLists.txt",
    "Makefile",
    "meson.build",
    "Cargo.toml",
    "go.mod",
    "package.json",
    "requirements.txt",
    "environment.yml",
    ".gitignore",
}

_MAX_TREE_DEPTH = 4
_MAX_TREE_ENTRIES = 300
_MAX_KEY_FILES = 40
_MAX_ADJACENT_CONTEXT_FILES = 12
_HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".cuh"}
_CODE_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".cuh", ".cu", ".hip", ".cpp", ".cc", ".cxx"}


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

    Returns an ASCII tree like::

        repo/
        ├── ops/
        │   ├── kernel.py    ← TARGET KERNEL
        │   └── utils.py
        └── tests/
            └── test_kernel.py
    """
    lines: list[str] = []
    counter = [0]
    kernel_abs = kernel_path.resolve()

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
                if depth + 1 < max_depth:
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


# ── Key files identification ──────────────────────────────────────────


def _classify_file(path: Path, kernel_path: Path, repo_root: Path) -> str | None:
    """Return a role string for a file based on filename heuristics, or None."""
    name = path.name.lower()
    rel = path.resolve()

    if rel == kernel_path.resolve():
        return "TARGET KERNEL - edit this"

    if name in _CONFIG_FILES or name in {n.lower() for n in _CONFIG_FILES}:
        return "Project config"

    if name.startswith("test_") or name.endswith("_test.py") or name.startswith("test."):
        return "Test file"
    if "test" in name and path.suffix in (".py", ".cpp", ".cc"):
        return "Test file (likely)"

    if name.startswith("bench_") or name.startswith("benchmark_"):
        return "Benchmark"
    if "benchmark" in name or "bench" in name:
        if path.suffix in (".py", ".cpp", ".cc"):
            return "Benchmark (likely)"

    if name in ("readme.md", "readme.rst", "readme.txt", "readme"):
        return "Documentation"
    if name in ("license", "license.md", "license.txt"):
        return "License"

    if name.endswith((".cu", ".hip", ".cl")):
        return "GPU kernel source"
    if name.endswith(".cuh"):
        return "GPU kernel header"

    return None


def _normalize_related_name(path: Path | str) -> str:
    name = Path(str(path)).stem.lower()
    for prefix in ("benchmark_", "bench_", "test_", "example_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name


def _identify_key_files(
    repo_root: Path,
    kernel_path: Path,
    *,
    max_files: int = _MAX_KEY_FILES,
) -> list[dict[str, str]]:
    """Walk the repo and return a list of key files with roles."""
    results: list[dict[str, str]] = []
    kernel_abs = kernel_path.resolve()
    root_abs = repo_root.resolve()

    # Always include the kernel itself first
    try:
        rel = kernel_abs.relative_to(root_abs)
    except ValueError:
        rel = kernel_abs
    results.append({"file": str(rel), "role": "TARGET KERNEL - edit this"})

    seen = {kernel_abs}

    for item in _iter_repo_files(repo_root):
        if len(results) >= max_files:
            break
        abs_item = item.resolve()
        if abs_item in seen:
            continue

        role = _classify_file(item, kernel_path, repo_root)
        if role:
            seen.add(abs_item)
            try:
                rel = abs_item.relative_to(root_abs)
            except ValueError:
                rel = abs_item
            results.append({"file": str(rel), "role": role})

    return results


def _related_name_score(candidate: Path, kernel_path: Path) -> tuple[int, int]:
    kernel_name = _normalize_related_name(kernel_path)
    candidate_name = _normalize_related_name(candidate)
    if not kernel_name or not candidate_name:
        return (3, 0)
    if candidate_name == kernel_name:
        return (0, len(kernel_name))
    if kernel_name in candidate_name or candidate_name in kernel_name:
        return (1, min(len(kernel_name), len(candidate_name)))
    kernel_tokens = {tok for tok in kernel_name.split("_") if tok}
    candidate_tokens = {tok for tok in candidate_name.split("_") if tok}
    overlap = len(kernel_tokens & candidate_tokens)
    if overlap > 0:
        return (2, overlap)
    return (3, 0)


def _adjacent_kernel_context_files(
    repo_root: Path,
    kernel_path: Path,
    *,
    max_files: int = _MAX_ADJACENT_CONTEXT_FILES,
) -> list[dict[str, str]]:
    """Return nearby implementation/config/benchmark files for header kernels."""

    if kernel_path.suffix.lower() not in _HEADER_SUFFIXES:
        return []

    root_abs = repo_root.resolve()
    kernel_abs = kernel_path.resolve()
    seen: set[Path] = {kernel_abs}
    candidates: list[tuple[tuple[int, int, int, str], Path, str]] = []

    search_roots: list[tuple[Path, str]] = [
        (kernel_path.parent, "Adjacent implementation"),
        (kernel_path.parent / "detail", "Adjacent detail implementation"),
        (kernel_path.parent.parent / "detail", "Adjacent detail implementation"),
    ]
    for base, default_role in search_roots:
        if not base.is_dir():
            continue
        for item in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if not item.is_file() or item.resolve() in seen:
                continue
            if item.suffix.lower() not in _CODE_SUFFIXES:
                continue
            score_tier, overlap = _related_name_score(item, kernel_path)
            if score_tier >= 3:
                continue
            role = "Adjacent config" if "config" in item.name.lower() else default_role
            rel = item.resolve().relative_to(root_abs)
            candidates.append(((score_tier, 0, -overlap, rel.as_posix()), item.resolve(), role))
            seen.add(item.resolve())

    for item in _iter_repo_files(repo_root):
        abs_item = item.resolve()
        if abs_item in seen or not item.is_file():
            continue
        rel = abs_item.relative_to(root_abs)
        rel_text = rel.as_posix().lower()
        if "/benchmark/" not in rel_text and "/test/" not in rel_text:
            continue
        score_tier, overlap = _related_name_score(item, kernel_path)
        if score_tier >= 3:
            continue
        role = "Matched benchmark" if "/benchmark/" in rel_text else "Matched test file"
        candidates.append(((score_tier, 1, -overlap, rel.as_posix()), abs_item, role))
        seen.add(abs_item)

    results: list[dict[str, str]] = []
    for _key, abs_item, role in sorted(candidates, key=lambda item: item[0])[:max_files]:
        try:
            rel = abs_item.relative_to(root_abs)
        except ValueError:
            rel = abs_item
        results.append({"file": str(rel), "role": role})
    return results


def _iter_repo_files(repo_root: Path):
    """Yield files from the repo, skipping ignored directories."""
    try:
        for item in sorted(repo_root.iterdir(), key=lambda p: p.name.lower()):
            if item.is_dir():
                if _should_skip_dir(item.name):
                    continue
                yield from _iter_repo_files(item)
            elif item.is_file():
                yield item
    except PermissionError:
        return


# ── Import chain extraction ───────────────────────────────────────────

_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
)
_CPP_INCLUDE_RE = re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def _extract_imports(kernel_path: Path) -> list[str]:
    """Extract import/include statements from the kernel file."""
    try:
        source = kernel_path.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    suffix = kernel_path.suffix.lower()

    if suffix == ".py":
        imports: list[str] = []
        for m in _PY_IMPORT_RE.finditer(source):
            mod = m.group(1) or m.group(2)
            if mod and not mod.startswith("__"):
                imports.append(mod)
        return imports

    if suffix in (".cpp", ".cc", ".cu", ".hip", ".cuh", ".h", ".hpp"):
        includes: list[str] = []
        for m in _CPP_INCLUDE_RE.finditer(source):
            includes.append(m.group(1))
        return includes

    return []


# ── Ignore list ───────────────────────────────────────────────────────


def _find_skip_dirs_present(repo_root: Path) -> list[str]:
    """Return which skip-listed directories actually exist in the repo root."""
    present: list[str] = []
    try:
        for item in repo_root.iterdir():
            if item.is_dir() and _should_skip_dir(item.name):
                present.append(item.name)
    except PermissionError:
        pass
    return sorted(present)


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

    sections: list[str] = ["# Codebase Context\n"]

    # 1. Repository layout
    tree = _build_directory_tree(repo_root, kernel_path)
    sections.append("## Repository Layout\n")
    sections.append(f"```\n{tree}\n```\n")

    # 2. Key files
    key_files = _identify_key_files(repo_root, kernel_path)
    if key_files:
        sections.append("## Key Files\n")
        sections.append("| File | Role |")
        sections.append("|------|------|")
        for kf in key_files:
            sections.append(f"| `{kf['file']}` | {kf['role']} |")
        sections.append("")

    adjacent_files = _adjacent_kernel_context_files(repo_root, kernel_path)
    if adjacent_files:
        sections.append("## Adjacent Kernel Context\n")
        sections.append(
            "For header / template kernels, the hot implementation often lives in nearby "
            "`detail/`, config, benchmark, or test files. Start by reading these:\n"
        )
        sections.append("| File | Role |")
        sections.append("|------|------|")
        for item in adjacent_files:
            sections.append(f"| `{item['file']}` | {item['role']} |")
        sections.append("")

    # 3. Import / dependency chain
    imports = _extract_imports(kernel_path)
    if imports:
        try:
            rel_kernel = kernel_path.relative_to(repo_root)
        except ValueError:
            rel_kernel = kernel_path
        sections.append("## Kernel Import Chain\n")
        sections.append(f"Imports found in `{rel_kernel}`:\n")
        for imp in imports:
            sections.append(f"- `{imp}`")
        sections.append("")

    # 4. Directories to ignore
    skip_present = _find_skip_dirs_present(repo_root)
    if skip_present:
        sections.append("## Directories to Ignore\n")
        sections.append(
            "These directories exist in the repo but should not be explored or modified:\n"
        )
        for d in skip_present:
            sections.append(f"- `{d}/`")
        sections.append("")

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
