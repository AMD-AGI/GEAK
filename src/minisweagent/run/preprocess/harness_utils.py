"""Bootstrap local helpers for the GEAK preprocessing pipeline.

This file begins as a copy of ``run/pipeline_helpers.py`` so preprocess can
own its harness-generation runtime without depending on sibling ``run/``
modules. It may still contain broader helpers initially and can be pruned
down once the preprocess boundary is stable.
"""

from __future__ import annotations

import argparse
import ast
import copy
import logging
import os
import re
import shlex
import shutil
import textwrap
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.repo_paths import ensure_preprocess_mcp_importable
from minisweagent.run.utils.pipeline_context import (
    build_bottleneck_guidance as _shared_bottleneck_guidance,
)
from minisweagent.run.utils.pipeline_context import (
    format_gpu_info as _shared_format_gpu_info,
)
from minisweagent.run.utils.pipeline_context import (
    gpu_arch_context as _shared_gpu_arch_context,
)
from minisweagent.run.utils.pipeline_context import (
    inject_pipeline_context as _shared_inject_pipeline_context,
)
from minisweagent.run.utils.pipeline_context import (
    run_baseline_profile as _shared_run_baseline_profile,
)

logger = logging.getLogger(__name__)

REQUIRED_HARNESS_FLAGS = ("--profile", "--correctness", "--benchmark", "--full-benchmark")

MAX_HARNESS_RETRIES = 2

# Use one canonical benchmark definition everywhere. The legacy
# GEAK_AGENT_BENCHMARK_ITERATIONS split is intentionally ignored so
# agent-time patch testing and final verification stay apples-to-apples.
DEFAULT_EVAL_BENCHMARK_ITERATIONS = int(os.getenv("GEAK_EVAL_BENCHMARK_ITERATIONS", "30"))
DEFAULT_AGENT_BENCHMARK_ITERATIONS = DEFAULT_EVAL_BENCHMARK_ITERATIONS
DEFAULT_PIPELINE_OUTPUT_DIR = "geak_output"
DEFAULT_HETEROGENEOUS = False


# ── agent filtering ──────────────────────────────────────────────────


def add_agent_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--allowed-agents`` and ``--excluded-agents`` to *parser*."""
    parser.add_argument(
        "--allowed-agents",
        default=None,
        help=("Comma-separated list of allowed agent types (e.g. strategy_agent). Sets GEAK_ALLOWED_AGENTS."),
    )
    parser.add_argument(
        "--excluded-agents",
        default=None,
        help=("Comma-separated list of excluded agent types (e.g. openevolve). Sets GEAK_EXCLUDED_AGENTS."),
    )


def apply_agent_filter_env(args: argparse.Namespace) -> None:
    """Propagate ``--allowed-agents`` / ``--excluded-agents`` to env vars."""
    configure_agent_filter_env(
        getattr(args, "allowed_agents", None),
        getattr(args, "excluded_agents", None),
    )


def configure_agent_filter_env(
    allowed_agents: str | None,
    excluded_agents: str | None,
) -> None:
    """Apply generic default agent filters.

    Default behavior excludes ``openevolve`` unless the user explicitly
    supplies an allowlist/excludelist or pre-sets ``GEAK_EXCLUDED_AGENTS``.
    This keeps the default pipeline focused on the lighter-weight agents while
    still allowing users to opt in deliberately.
    """

    if allowed_agents:
        os.environ["GEAK_ALLOWED_AGENTS"] = allowed_agents
        if excluded_agents is not None:
            os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents
        return

    if excluded_agents:
        os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents
        return

    os.environ.setdefault("GEAK_EXCLUDED_AGENTS", "openevolve")


# ── model loading ────────────────────────────────────────────────────


def load_geak_model(
    model_name: str | None,
    *,
    config_spec: str = "geak",
) -> Any:
    """Load an LLM model using the standard GEAK config-resolution pattern.

    Reads the YAML config for *config_spec*, extracts the ``model`` section,
    and delegates to ``get_model``.  Falls back to the ``GEAK_MODEL``
    environment variable when *model_name* is ``None``.
    """
    import yaml

    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    resolved_name = model_name or os.environ.get("GEAK_MODEL") or "claude-opus-4.6"
    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    return get_model(resolved_name, config=model_config)


def geak_model_factory(
    model_name: str | None,
    *,
    config_spec: str = "geak",
):
    """Return a zero-arg callable that creates a fresh model each time."""
    import yaml

    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    resolved_name = model_name or os.environ.get("GEAK_MODEL") or "claude-opus-4.6"
    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    def _factory():
        return get_model(resolved_name, config=copy.deepcopy(model_config))

    return _factory


def _ensure_mcp_importable() -> None:
    """Add MCP tool source directories to sys.path if not already present."""
    ensure_preprocess_mcp_importable(
        "mcp_tools/profiler-mcp/src",
        "mcp_tools/metrix-mcp/src",
        "mcp_tools/automated-test-discovery/src",
    )


# ── harness path extraction ──────────────────────────────────────────


def extract_harness_path(test_command: str) -> str:
    """Extract the harness script path from a test command string.

    Handles patterns like::

        'pytest /path/to/test.py -v'                -> '/path/to/test.py'
        'python /path/to/harness.py --correctness'  -> '/path/to/harness.py'
        '/path/to/harness.py'                       -> '/path/to/harness.py'
    """
    try:
        tokens = shlex.split(test_command)
    except ValueError:
        tokens = test_command.split()

    for token in tokens:
        if token.endswith(".py") and "/" in token:
            return token

    for token in tokens:
        if token.endswith(".py"):
            return token

    return tokens[-1] if tokens else test_command


def _preferred_harness_path(log_dir: Path, kernel_path: Path | None) -> Path:
    if kernel_path is not None:
        stem = kernel_path.stem or "kernel"
        return log_dir / f"test_{stem}_harness.py"
    return log_dir / "geak_test_harness.py"


def _materialized_harness_bootstrap(
    *,
    repo_root: Path,
    kernel_path: Path | None,
) -> str:
    kernel_dir = kernel_path.resolve().parent if kernel_path is not None else None
    rel_kernel_dir: Path | None = None
    if kernel_dir is not None:
        try:
            rel_kernel_dir = kernel_dir.relative_to(repo_root.resolve())
        except ValueError:
            rel_kernel_dir = None

    rel_kernel_dir_text = str(rel_kernel_dir).replace("\\", "/") if rel_kernel_dir is not None else ""
    original_kernel_dir = str(kernel_dir) if kernel_dir is not None else ""
    return (
        "# GEAK materialized harness bootstrap\n"
        "def _resolve_geak_kernel_dir():\n"
        "    candidates = []\n"
        '    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()\n'
        "    if work_dir:\n"
        "        candidates.append(work_dir)\n"
        '    repo_root = os.environ.get("GEAK_REPO_ROOT", "").strip()\n'
        f"    rel_kernel_dir = {rel_kernel_dir_text!r}\n"
        "    if repo_root and rel_kernel_dir:\n"
        "        candidates.append(os.path.join(repo_root, rel_kernel_dir))\n"
        f"    original_kernel_dir = {original_kernel_dir!r}\n"
        "    if original_kernel_dir:\n"
        "        candidates.append(original_kernel_dir)\n"
        "    for candidate in candidates:\n"
        '        if candidate and os.path.isfile(os.path.join(candidate, "kernel.py")):\n'
        "            return candidate\n"
        "    return original_kernel_dir or os.getcwd()\n"
        "\n"
        "_KERNEL_DIR = _resolve_geak_kernel_dir()\n"
        "if _KERNEL_DIR not in sys.path:\n"
        "    sys.path.insert(0, _KERNEL_DIR)\n"
    )


def _rewrite_relative_imports(source: str, harness_path: Path, repo_root: Path) -> str:
    """Convert relative imports in *source* to absolute imports anchored at *repo_root*.

    Example::

        # harness at: repo_root/aiter/ops/triton/test_harness.py
        from ..utils import foo
        # becomes:
        from aiter.ops.utils import foo

    If the harness is not inside *repo_root* (can't compute package path), the
    source is returned unchanged with a warning.
    """
    try:
        harness_abs = harness_path.resolve()
        repo_abs = repo_root.resolve()
        rel = harness_abs.relative_to(repo_abs)
    except ValueError:
        logger.warning(
            "Harness %s is outside repo_root %s — cannot rewrite relative imports",
            harness_path,
            repo_root,
        )
        return source

    # Package parts of the harness (excluding the file itself)
    # e.g. aiter/ops/triton/test_harness.py -> ["aiter", "ops", "triton"]
    harness_package_parts = list(rel.parts[:-1])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Collect relative imports to rewrite: (lineno, col_offset, end_lineno, original_text, new_text)
    rewrites: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            continue  # absolute import, leave alone

        # Walk up `level` packages from harness_package_parts
        if node.level > len(harness_package_parts):
            logger.warning(
                "Relative import %r in %s exceeds package depth; skipping",
                ast.unparse(node),
                harness_path,
            )
            continue

        base_parts = harness_package_parts[: len(harness_package_parts) - (node.level - 1)]
        if node.module:
            abs_module = ".".join(base_parts + [node.module])
        else:
            abs_module = ".".join(base_parts)

        names_str = ", ".join(
            (f"{alias.name} as {alias.asname}" if alias.asname else alias.name) for alias in node.names
        )
        new_import = f"from {abs_module} import {names_str}"
        old_import = ast.unparse(node)
        rewrites.append((node.lineno, old_import, new_import))

    if not rewrites:
        return source

    # Apply rewrites line-by-line (ast.unparse gives canonical form; use line numbers)
    lines = source.splitlines(keepends=True)
    rewrite_map: dict[int, str] = {lineno: new_text for lineno, _, new_text in rewrites}
    # Build set of lines to skip (multi-line imports span lineno..end_lineno)
    # For simplicity parse again to get end_lineno
    skip_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.lineno in rewrite_map:
            for ln in range(node.lineno, node.end_lineno + 1):
                skip_lines.add(ln)

    new_lines: list[str] = []
    for i, line in enumerate(lines, 1):
        if i in rewrite_map:
            new_lines.append(rewrite_map[i] + "\n")
        elif i in skip_lines:
            pass  # continuation lines of the rewritten import
        else:
            new_lines.append(line)

    logger.info("Rewrote %d relative import(s) in %s to absolute", len(rewrites), harness_path)
    return "".join(new_lines)


def _rewrite_materialized_harness_source(
    source_text: str,
    *,
    repo_root: Path,
    kernel_path: Path | None,
    harness_path: Path | None = None,
) -> str:
    bootstrap = _materialized_harness_bootstrap(
        repo_root=repo_root,
        kernel_path=kernel_path,
    )
    legacy_patterns = [
        re.compile(
            r"(?ms)^# Ensure the kernel directory is importable\n"
            r"_KERNEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n"
            r"if _KERNEL_DIR not in sys\.path:\n"
            r"\s+sys\.path\.insert\(0, _KERNEL_DIR\)\n"
        ),
        re.compile(
            r"(?ms)^_KERNEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n"
            r"if _KERNEL_DIR not in sys\.path:\n"
            r"\s+sys\.path\.insert\(0, _KERNEL_DIR\)\n"
        ),
    ]
    for pattern in legacy_patterns:
        if pattern.search(source_text):
            source_text = pattern.sub(bootstrap, source_text, count=1)
            break
    else:
        import_block = re.compile(
            r"(?ms)\A((?:from __future__ import annotations\n)?(?:import .+\n|from .+ import .+\n)+)"
        )
        match = import_block.match(source_text)
        if match:
            source_text = source_text[: match.end()] + "\n" + bootstrap + source_text[match.end() :]
        else:
            source_text = bootstrap + "\n" + source_text

    # Rewrite relative imports to absolute so PYTHONPATH ordering is respected
    if harness_path is not None:
        source_text = _rewrite_relative_imports(source_text, harness_path, repo_root)

    return source_text


def _materialize_validated_harness(
    *,
    test_command: str,
    harness_path: str,
    repo_root: Path,
    log_dir: Path | None,
    kernel_path: Path | None,
    gpu_id: int,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    if log_dir is None:
        return None

    source_harness = Path(harness_path).resolve()
    target_dir = log_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_harness = _preferred_harness_path(target_dir, kernel_path)
    if source_harness == target_harness:
        return None

    rewritten_text = _rewrite_materialized_harness_source(
        source_harness.read_text(),
        repo_root=repo_root,
        kernel_path=kernel_path,
        harness_path=source_harness,
    )
    target_harness.write_text(rewritten_text)
    shutil.copymode(source_harness, target_harness)

    materialized_command = test_command.replace(str(source_harness), str(target_harness))
    valid, static_errors = validate_harness(str(target_harness))
    if not valid:
        raise RuntimeError("Materialized harness static validation failed: " + "; ".join(static_errors))

    exec_ok, exec_errors, harness_results = execute_harness_validation(
        str(target_harness),
        repo_root=str(repo_root),
        gpu_id=gpu_id,
    )
    if not exec_ok:
        raise RuntimeError(
            "Materialized harness runtime validation failed: " + "; ".join(e.splitlines()[0] for e in exec_errors)
        )
    return materialized_command, str(target_harness), harness_results


# ── kernel-in-harness detection and splitting ────────────────────────

# Markers that indicate a Python file contains kernel definitions (not just test logic)
_TRITON_KERNEL_MARKERS = ("@triton.jit", "@triton.autotune", "triton.jit", "tl.constexpr")
# HIP/CUDA markers for non-Python kernels (detected via regex in source text)
_HIP_KERNEL_RE = re.compile(r"\b(__global__|__kernel__|__device__)\b")
_C_LIKE_KERNEL_SOURCE_SUFFIXES = frozenset({".hip", ".cu", ".cpp", ".cc", ".cxx"})
_C_LIKE_TEST_ENTRY_RE = re.compile(
    r"(?m)^(?P<signature>[^\n{;#]*\b(?P<name>main|run_[A-Za-z_]\w*|test_[A-Za-z_]\w*)\s*\([^;{]*\)\s*)\{"
)

_TRITON_DECORATOR_NAMES = frozenset({"triton.jit", "triton.autotune", "jit", "autotune"})


def _is_triton_decorator(decorator: ast.expr) -> bool:
    """Return True if decorator node is @triton.jit or @triton.autotune."""
    if isinstance(decorator, ast.Attribute):
        return (
            isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"
            and decorator.attr in ("jit", "autotune")
        )
    if isinstance(decorator, ast.Name):
        return decorator.id in ("jit", "autotune")
    # @triton.autotune(...) call
    if isinstance(decorator, ast.Call):
        return _is_triton_decorator(decorator.func)
    return False


def _harness_has_kernel_definitions(source: str) -> bool:
    """Return True if the Python source contains embedded kernel definitions."""
    if not any(marker in source for marker in _TRITON_KERNEL_MARKERS):
        return False
    # Confirm with AST to avoid false positives (e.g. comments mentioning @triton.jit)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to text match if AST fails
        return any(marker in source for marker in _TRITON_KERNEL_MARKERS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_is_triton_decorator(d) for d in node.decorator_list):
                return True
    return False


def _source_has_c_like_kernel_definitions(source: str) -> bool:
    """Return True if a C-like source file looks like HIP/CUDA kernel code."""
    return _HIP_KERNEL_RE.search(source) is not None and (
        "__global__" in source or "hipLaunchKernelGGL" in source or "<<<" in source
    )


def _infer_repo_root_for_split(path: Path) -> Path | None:
    """Infer a repo root for split/materialization helpers."""
    candidate = path.resolve().parent
    for ancestor in [candidate, *candidate.parents]:
        if any((ancestor / marker).exists() for marker in (".git", "pyproject.toml", "setup.py", "setup.cfg")):
            return ancestor
    return None


def _find_matching_brace(source: str, open_brace_index: int) -> int | None:
    """Return the exclusive end index of the brace block starting at *open_brace_index*."""
    depth = 0
    i = open_brace_index
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False

    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_single_quote:
            if ch == "\\" and nxt:
                i += 2
                continue
            if ch == "'":
                in_single_quote = False
            i += 1
            continue
        if in_double_quote:
            if ch == "\\" and nxt:
                i += 2
                continue
            if ch == '"':
                in_double_quote = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1

    return None


def _find_c_like_test_blocks(source: str) -> list[tuple[str, int, int]]:
    """Return ``(name, start, end)`` ranges for host-side test entry functions."""
    blocks: list[tuple[str, int, int]] = []
    seen_ranges: set[tuple[int, int]] = set()

    for match in _C_LIKE_TEST_ENTRY_RE.finditer(source):
        brace_index = match.end() - 1
        end_index = _find_matching_brace(source, brace_index)
        if end_index is None:
            logger.warning("Could not find matching brace for %s in merged source split", match.group("name"))
            continue
        key = (match.start(), end_index)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        blocks.append((match.group("name"), match.start(), end_index))

    return sorted(blocks, key=lambda item: item[1])


def _generate_c_like_python_harness(
    *,
    wrapper_path: Path,
    c_harness_path: Path,
    clean_kernel_path: Path,
    original_source_dir: Path,
    repo_root: Path | None,
) -> None:
    """Generate a Python GEAK harness wrapper for merged HIP/CUDA split outputs."""
    include_dirs = [original_source_dir]
    if repo_root is not None:
        include_dirs.append(repo_root)
    for extra_name in ("Common", "common", "include", "includes", "inc", "src"):
        candidate = original_source_dir / extra_name
        if candidate.exists():
            include_dirs.append(candidate)
    parent_common = original_source_dir.parent / "Common"
    if parent_common.exists():
        include_dirs.append(parent_common)

    deduped_include_dirs: list[str] = []
    seen_dirs: set[str] = set()
    for path in include_dirs:
        resolved = str(path.resolve())
        if resolved not in seen_dirs:
            seen_dirs.add(resolved)
            deduped_include_dirs.append(resolved)

    repo_root_str = str(repo_root.resolve()) if repo_root is not None else str(original_source_dir.resolve())
    wrapper_source = textwrap.dedent(
        f"""\
        import argparse
        import os
        import re
        import statistics
        import subprocess
        import sys
        import time
        from pathlib import Path

        C_HARNESS = Path({str(c_harness_path.resolve())!r})
        CLEAN_KERNEL = Path({str(clean_kernel_path.resolve())!r})
        SOURCE_DIR = Path({str(original_source_dir.resolve())!r})
        REPO_ROOT = Path({repo_root_str!r})
        BINARY = C_HARNESS.with_suffix("")
        INCLUDE_DIRS = {deduped_include_dirs!r}


        def _compile_binary() -> Path:
            needs_rebuild = (not BINARY.exists()) or any(
                src.exists() and src.stat().st_mtime > BINARY.stat().st_mtime
                for src in (C_HARNESS, CLEAN_KERNEL)
            )
            if not needs_rebuild:
                return BINARY

            cmd = ["hipcc", "-O3", "-std=c++17", str(C_HARNESS), "-o", str(BINARY)]
            for inc in INCLUDE_DIRS:
                cmd.extend(["-I", inc])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(SOURCE_DIR),
            )
            if proc.returncode != 0:
                if proc.stdout:
                    sys.stdout.write(proc.stdout)
                if proc.stderr:
                    sys.stderr.write(proc.stderr)
                raise SystemExit(proc.returncode)
            return BINARY


        def _extract_latency_ms(output: str, fallback_ms: float) -> float:
            patterns = (
                (r"GEAK_RESULT_LATENCY_MS=([\\d.]+(?:e[+-]?\\d+)?)", 1.0),
                (r"Perf:\\s*([\\d.]+(?:e[+-]?\\d+)?)\\s*us/launch", 1.0 / 1000.0),
                (r"Perf:\\s*([\\d.]+(?:e[+-]?\\d+)?)\\s*ms/launch", 1.0),
                (r"TOTAL_KERNEL_TIME_MS:\\s*([\\d.]+(?:e[+-]?\\d+)?)", 1.0),
                (r"BENCHMARK_LATENCY_MS:\\s*([\\d.]+(?:e[+-]?\\d+)?)", 1.0),
            )
            for pattern, scale in patterns:
                match = re.search(pattern, output)
                if match:
                    return float(match.group(1)) * scale
            return fallback_ms


        def _run_once() -> tuple[subprocess.CompletedProcess[str], float]:
            binary = _compile_binary()
            t0 = time.perf_counter()
            proc = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                cwd=str(SOURCE_DIR),
                env=os.environ.copy(),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return proc, elapsed_ms


        def _run_mode(measure: bool) -> int:
            proc, elapsed_ms = _run_once()
            if proc.stdout:
                sys.stdout.write(proc.stdout)
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            if measure and proc.returncode == 0:
                latency_ms = _extract_latency_ms(proc.stdout + "\\n" + proc.stderr, elapsed_ms)
                print(f"GEAK_RESULT_LATENCY_MS={{latency_ms:.6f}}")
            return proc.returncode


        def run_correctness() -> int:
            return _run_mode(measure=False)


        def run_profile() -> int:
            return _run_mode(measure=False)


        def run_benchmark() -> int:
            return _run_mode(measure=True)


        def run_full_benchmark() -> int:
            return _run_mode(measure=True)


        def main() -> int:
            parser = argparse.ArgumentParser()
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument("--correctness", action="store_true")
            group.add_argument("--profile", action="store_true")
            group.add_argument("--benchmark", action="store_true")
            group.add_argument("--full-benchmark", action="store_true")
            parser.parse_known_args()

            if parser.parse_known_args()[0].correctness:
                return run_correctness()
            if parser.parse_known_args()[0].profile:
                return run_profile()
            if parser.parse_known_args()[0].benchmark:
                return run_benchmark()
            return run_full_benchmark()


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )
    wrapper_path.write_text(wrapper_source)


def _detect_and_split_c_like_kernel_from_harness(
    harness_path: Path,
    output_dir: Path,
    source: str,
) -> tuple[str, str] | None:
    """Split host-side test logic out of merged HIP/CUDA source files.

    The conservative C-like path handles merged sources where the executable
    host-side validation entrypoint lives in the same file as the HIP/CUDA
    kernel. We move ``main()`` plus any ``run_*`` / ``test_*`` helper
    functions into a generated harness file, and keep the clean kernel copy in
    ``output_dir`` for agent patching.
    """
    if harness_path.suffix not in _C_LIKE_KERNEL_SOURCE_SUFFIXES:
        return None
    if not _source_has_c_like_kernel_definitions(source):
        return None

    test_blocks = _find_c_like_test_blocks(source)
    if not test_blocks:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_kernel_path = output_dir / harness_path.name
    c_harness_path = output_dir / f"test_{harness_path.stem}_harness{harness_path.suffix}"
    wrapper_harness_path = output_dir / f"test_{harness_path.stem}_harness.py"

    clean_chunks: list[str] = []
    harness_chunks: list[str] = []
    cursor = 0
    for _name, start, end in test_blocks:
        clean_chunks.append(source[cursor:start])
        harness_chunks.append(source[start:end].strip())
        cursor = end
    clean_chunks.append(source[cursor:])

    clean_source = "".join(clean_chunks).rstrip() + "\n"
    clean_kernel_path.write_text(clean_source)

    harness_source = (
        f'#include "{clean_kernel_path.name}"\n\n' + "\n\n".join(chunk for chunk in harness_chunks if chunk) + "\n"
    )
    c_harness_path.write_text(harness_source)

    _generate_c_like_python_harness(
        wrapper_path=wrapper_harness_path,
        c_harness_path=c_harness_path,
        clean_kernel_path=clean_kernel_path,
        original_source_dir=harness_path.parent,
        repo_root=_infer_repo_root_for_split(harness_path),
    )
    logger.info(
        "Split merged C-like source: kernel=%s, harness_source=%s, harness_wrapper=%s",
        clean_kernel_path,
        c_harness_path,
        wrapper_harness_path,
    )
    return str(wrapper_harness_path), str(clean_kernel_path)


def detect_and_split_kernel_from_harness(
    harness_path: str | Path,
    output_dir: Path,
) -> tuple[str, str] | None:
    """Split test logic out of a merged kernel+harness file.

    When a single file contains both kernel definitions (``@triton.jit`` /
    ``@triton.autotune``) and test/harness logic, agents would see mixed
    content and patch evaluation would be unreliable.  This function:

    1. Detects whether the file contains both kernel defs and test roots.
    2. Finds all *test-root* functions: names starting with ``test_`` or
       ``run_`` (GEAK convention), functions decorated with pytest/unittest
       markers, functions that reference GEAK CLI flags (``--correctness``
       etc.), and functions called from an ``if __name__ == "__main__"``
       block.
    3. Uses BFS from those seeds to collect all same-file functions they
       call—**except** functions decorated with ``@triton.jit`` /
       ``@triton.autotune``, which belong to the kernel and must stay.
    4. Strips the collected test functions (and the ``__main__`` block)
       from the original file so the original becomes a clean kernel file.
    5. Writes a new ``test_<stem>_harness.py`` in ``output_dir`` containing:
       - All import statements (cheap copy of everything in the original)
       - A sys.path bootstrap so the harness can import the kernel by stem
       - ``from <stem> import *``
       - All collected test functions
       - The ``__main__`` block (if present)

    Returns ``(new_harness_path, kernel_path)`` where *kernel_path* is the
    (now stripped) original file.  Returns ``None`` when no split is needed
    (no kernel defs, or no test roots found).
    """
    harness_path = Path(harness_path).resolve()
    source = harness_path.read_text()

    c_like_result = _detect_and_split_c_like_kernel_from_harness(harness_path, output_dir, source)
    if c_like_result is not None:
        return c_like_result

    # Fast guard: only proceed if the file has kernel definitions at all
    if not _harness_has_kernel_definitions(source):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning("Could not parse harness for splitting (SyntaxError: %s); skipping split", exc)
        return None

    source_lines = source.splitlines(keepends=True)

    # ── helpers ────────────────────────────────────────────────────────
    _GEAK_FLAGS = {"--correctness", "--profile", "--benchmark", "--full-benchmark"}

    def _is_triton_fn(node: ast.AST) -> bool:
        """True if node is a function decorated with @triton.jit / @triton.autotune."""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        return any(_is_triton_decorator(d) for d in node.decorator_list)

    def _is_test_decorator(decorator: ast.expr) -> bool:
        s = ast.unparse(decorator)
        return any(s.startswith(pfx) for pfx in ("pytest", "unittest", "mock", "fixture"))

    def _node_source(node: ast.AST) -> str:
        start = (
            node.decorator_list[0].lineno if hasattr(node, "decorator_list") and node.decorator_list else node.lineno
        ) - 1
        end = node.end_lineno
        return "".join(source_lines[start:end])

    def _has_geak_flags(node: ast.AST) -> bool:
        src = _node_source(node)
        return any(flag in src for flag in _GEAK_FLAGS) or "ArgumentParser" in src

    def _is_main_block(node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
        )

    # Map of fn_name -> list of ast nodes (handles duplicate function names)
    fn_map: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_map.setdefault(node.name, []).append(node)

    def _collect_called_names(node: ast.AST) -> set[str]:
        """Return names of same-file functions called within node."""
        names: set[str] = set()
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id in fn_map:
                    names.add(call.func.id)
        return names

    # ── identify __main__ block ────────────────────────────────────────
    main_block: ast.If | None = None
    for node in tree.body:
        if _is_main_block(node):
            main_block = node
            break

    # ── find test-root seeds ───────────────────────────────────────────
    seeds: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_triton_fn(node):
            continue  # never seed from kernel functions
        name = node.name
        # GEAK naming conventions
        if name.startswith("test_") or name.startswith("run_"):
            seeds.add(name)
            continue
        # pytest / unittest decorators
        if any(_is_test_decorator(d) for d in node.decorator_list):
            seeds.add(name)
            continue
        # Contains GEAK CLI flags or ArgumentParser usage
        if _has_geak_flags(node):
            seeds.add(name)

    # Add functions called directly from __main__ block as seeds
    if main_block is not None:
        seeds.update(
            _collect_called_names(main_block)
            - {name for name, nodes in fn_map.items() if all(_is_triton_fn(n) for n in nodes)}
        )

    if not seeds:
        logger.debug("No test roots found in %s; skipping split", harness_path)
        return None

    # ── BFS to collect all test-reachable functions ────────────────────
    test_fns: set[str] = set()
    queue = list(seeds)
    while queue:
        name = queue.pop()
        if name in test_fns:
            continue
        nodes = fn_map.get(name)
        if nodes is None:
            continue
        # Skip if ALL definitions for this name are triton kernel fns
        if all(_is_triton_fn(n) for n in nodes):
            continue
        test_fns.add(name)
        for n in nodes:
            for callee in _collect_called_names(n):
                if callee not in test_fns:
                    queue.append(callee)

    if not test_fns:
        return None

    # ── compute line ranges for nodes to strip from original ───────────
    def _node_line_range(node: ast.AST) -> tuple[int, int]:
        start = (
            node.decorator_list[0].lineno if hasattr(node, "decorator_list") and node.decorator_list else node.lineno
        ) - 1
        return start, node.end_lineno  # start 0-indexed, end 1-indexed (exclusive)

    strip_line_set: set[int] = set()
    test_fn_chunks: list[tuple[int, str]] = []  # (start_line, source_chunk)

    # Scan tree.body directly (not fn_map) to catch duplicate function names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in test_fns:
            start, end = _node_line_range(node)
            strip_line_set.update(range(start, end))
            test_fn_chunks.append((start, "".join(source_lines[start:end])))

    # Also strip __main__ block
    main_chunk: str | None = None
    if main_block is not None:
        start, end = _node_line_range(main_block)
        strip_line_set.update(range(start, end))
        main_chunk = "".join(source_lines[start:end])

    # ── collect all import statements (cheap copy, relative→absolute) ──
    # Relative imports must be rewritten here using harness_path (the original
    # file's location inside the repo) as the reference.  The new harness file
    # lives in output_dir (outside the repo), so if we waited until after
    # writing it _rewrite_relative_imports would refuse to rewrite because it
    # couldn't determine the package path from outside the repo root.
    raw_imports = "".join(
        "".join(source_lines[node.lineno - 1 : node.end_lineno])
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    # Try to find repo_root for this file so relative imports can be resolved.
    # Walk up from harness_path looking for standard repo markers.
    _candidate = harness_path.resolve().parent
    _repo_root_for_split: Path | None = None
    for _ancestor in [_candidate, *_candidate.parents]:
        if any((_ancestor / m).exists() for m in (".git", "pyproject.toml", "setup.py", "setup.cfg")):
            _repo_root_for_split = _ancestor
            break
    if _repo_root_for_split is not None:
        raw_imports = _rewrite_relative_imports(raw_imports, harness_path, _repo_root_for_split)
    import_chunks = [raw_imports] if raw_imports.strip() else []

    # ── write new harness file ─────────────────────────────────────────
    stem = harness_path.stem  # e.g. "kernel" or "naive_softmax"
    output_dir.mkdir(parents=True, exist_ok=True)
    new_harness_path = output_dir / f"test_{stem}_harness.py"

    harness_parts: list[str] = []
    # 1. All imports
    harness_parts.extend(import_chunks)
    harness_parts.append("\n")
    # 2. sys.path bootstrap so the kernel module is importable
    harness_parts.append(
        "import sys as _geak_sys\n"
        f"_KERNEL_DIR = {repr(str(harness_path.parent))}\n"
        "if _KERNEL_DIR not in _geak_sys.path:\n"
        "    _geak_sys.path.insert(0, _KERNEL_DIR)\n"
        "\n"
    )
    # 3. Star-import kernel module
    harness_parts.append(f"from {stem} import *\n\n")
    # 4. Test functions (sorted by original line order for determinism)
    test_fn_chunks.sort(key=lambda t: t[0])
    for _, chunk in test_fn_chunks:
        harness_parts.append(chunk)
        if not chunk.endswith("\n\n"):
            harness_parts.append("\n")
    # 5. __main__ block
    if main_chunk is not None:
        harness_parts.append("\n")
        harness_parts.append(main_chunk)

    new_harness_path.write_text("".join(harness_parts))
    logger.info("Wrote test harness to %s", new_harness_path)

    # ── write clean kernel copy to output_dir (leave original untouched) ─
    # We must NOT modify the original file: the geak framework uses git to
    # apply/revert patches against the repo, so mutating the original breaks
    # those operations.  Instead, write the cleaned copy to output_dir where
    # GEAK_WORK_DIR (which is prepended to PYTHONPATH) will find it first.
    kernel_lines = [line for i, line in enumerate(source_lines) if i not in strip_line_set]
    while kernel_lines and kernel_lines[-1].strip() == "":
        kernel_lines.pop()
    kernel_lines.append("\n")
    clean_kernel_path = output_dir / harness_path.name
    clean_kernel_path.write_text("".join(kernel_lines))
    logger.info(
        "Wrote clean kernel (test logic stripped) to %s; original %s untouched",
        clean_kernel_path,
        harness_path,
    )

    return str(new_harness_path), str(clean_kernel_path)


# ── harness validation ───────────────────────────────────────────────


_GPU_ALLOC_IN_PROFILE_RE = re.compile(
    r"""torch\.(?:randn?|empty|zeros|ones|full)\s*\("""
    r"""[^)]*device\s*=\s*["']cuda["']""",
)


def validate_harness(harness_path: str) -> tuple[bool, list[str]]:
    """Static-analyse a harness script to verify it supports required CLI flags.

    Checks that the harness uses an argument-parsing library (argparse, click,
    or typer) and defines all four required flags: ``--correctness``,
    ``--profile``, ``--benchmark``, and ``--full-benchmark``.  Also checks
    that the ``run_profile`` function (if present) does not allocate tensors
    directly on CUDA, which would pollute the profiler trace with GPU RNG /
    memset kernels.

    Returns ``(valid, errors)`` where *errors* is empty when *valid* is True.
    """
    harness = Path(harness_path)
    errors: list[str] = []

    if not harness.is_file():
        return False, [f"Harness file not found: {harness}"]

    source = harness.read_text()

    has_parser = "argparse" in source or "ArgumentParser" in source or "click" in source or "typer" in source
    if not has_parser:
        errors.append(
            "Harness does not use argparse/click/typer -- "
            "CLI flags like --profile and --correctness will be silently ignored"
        )

    for flag in REQUIRED_HARNESS_FLAGS:
        if flag not in source:
            errors.append(f"Harness source does not define '{flag}' flag")

    # Check for GPU-side tensor allocation inside the profile function.
    # rocprofv3 captures ALL GPU kernels, so torch.randn(..., device='cuda')
    # inside run_profile pollutes the trace with RNG kernels.
    _in_profile_fn = False
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("def ") and "profile" in stripped:
            _in_profile_fn = True
            continue
        if _in_profile_fn and stripped.startswith("def "):
            _in_profile_fn = False
        if _in_profile_fn and _GPU_ALLOC_IN_PROFILE_RE.search(line):
            errors.append(
                f"Line {lineno}: GPU tensor allocation inside profile function "
                f"(device='cuda'). Use device='cpu' then .to('cuda') to avoid "
                f"polluting the profiler trace with RNG/memset kernels. "
                f"See src/minisweagent/run/preprocess/INSTRUCTIONS.md point 8."
            )
            break  # one warning is enough

    return len(errors) == 0, errors


# ── harness runtime execution ─────────────────────────────────────────


def execute_harness_validation(
    harness_path: str,
    repo_root: str | None = None,
    gpu_id: int = 0,
    benchmark_extra_args: str | None = None,
    use_uta_timeouts: bool = False,
) -> tuple[bool, list[str], list[dict]]:
    """Run the harness across all modes and return ``(ok, errors, results)``.

    Delegates to :func:`minisweagent.tools.run_harness.run_harness` with
    ``mode="all"`` which executes correctness -> profile -> benchmark ->
    full-benchmark in sequence, short-circuiting on first failure.

    Parameters
    ----------
    benchmark_extra_args:
        Extra benchmark tuning args. ``--iterations N`` is normalized into
        ``GEAK_BENCHMARK_ITERATIONS`` so harnesses do not need to expose it
        as a CLI flag. Any remaining args are passed via
        ``GEAK_BENCHMARK_EXTRA_ARGS``.
    use_uta_timeouts:
        When True, use ``UTA_MODE_TIMEOUTS`` instead of the default
        ``MODE_TIMEOUTS``.  The UTA timeout for ``--correctness`` is more
        relaxed (default 900 s, overridable via ``GEAK_UTA_CORRECTNESS_TIMEOUT``)
        to handle kernels with expensive initialisation (e.g. physics sims)
        that exceed the normal 300 s limit and cause the agent to retry forever.

    Returns
    -------
    ok : bool
        True if every mode passed.
    errors : list[str]
        Human-readable error descriptions for failed modes (empty on success).
    results : list[dict]
        Per-mode result dicts from :func:`run_harness`.
    """
    from minisweagent.run.preprocess.run_harness import UTA_MODE_TIMEOUTS, results_errors, run_harness

    env_overrides: dict[str, str] = {}
    # Keep validation fast: override iterations to a small number unless
    # the caller explicitly provides benchmark_extra_args.
    if not benchmark_extra_args:
        env_overrides["GEAK_BENCHMARK_ITERATIONS"] = "5"
    else:
        import re as _re

        remaining_extra_args = benchmark_extra_args.strip()
        # Extract --iterations N into the env var so benchmark reruns work
        # even when the harness argparse only accepts mode flags.
        _iter_match = _re.search(r"(?:^|\s)--iterations\s+(\d+)(?=\s|$)", remaining_extra_args)
        if _iter_match:
            env_overrides["GEAK_BENCHMARK_ITERATIONS"] = _iter_match.group(1)
            remaining_extra_args = _re.sub(
                r"(?:^|\s)--iterations\s+\d+(?=\s|$)",
                " ",
                remaining_extra_args,
            ).strip()
        if remaining_extra_args:
            env_overrides["GEAK_BENCHMARK_EXTRA_ARGS"] = remaining_extra_args

    mode_timeouts = UTA_MODE_TIMEOUTS if use_uta_timeouts else None

    results = run_harness(
        harness_path,
        mode="all",
        repo_root=repo_root,
        gpu_id=gpu_id,
        env_overrides=env_overrides,
        mode_timeouts=mode_timeouts,
    )
    if not isinstance(results, list):
        results = [results]

    ok = all(r["success"] for r in results)
    errors = results_errors(results) if not ok else []
    return ok, errors, results


# ── validated harness creation (UnitTestAgent + retry) ───────────────


def create_validated_harness(
    *,
    model: Any,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None,
    kernel_path: Path | None,
    discovery_context: str,
    max_retries: int = MAX_HARNESS_RETRIES,
    gpu_id: int = 0,
) -> tuple[str, list[dict]]:
    """Run UnitTestAgent with static + runtime validation and retry loop.

    After the agent produces a harness:
      1. :func:`validate_harness` performs static analysis (argparse,
         ``--profile``, ``--correctness`` flags, GPU allocation patterns).
      2. :func:`execute_harness_validation` actually runs the harness in
         all four modes (correctness, profile, benchmark, full-benchmark)
         to catch import errors, shape mismatches, OOM, etc.

    If either step fails the errors are fed back into the discovery context
    and the agent is re-invoked, up to *max_retries* additional attempts.

    Returns ``(test_command, harness_results)`` on success where
    *harness_results* is the list of per-mode result dicts.

    Raises
    ------
    RuntimeError
        If validation still fails after all retries.
    """
    from minisweagent.run.preprocess.unit_test_agent import run_unit_test_agent

    max_attempts = max_retries + 1
    harness_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        ctx = discovery_context
        if harness_errors:
            ctx += (
                f"\n\nHARNESS VALIDATION FAILED (attempt {attempt}/{max_attempts}):\n"
                + "\n".join(f"- {e}" for e in harness_errors)
                + "\n\nYou MUST fix the harness so that ALL modes work: "
                "--correctness, --profile, --benchmark, --full-benchmark. "
                "See src/minisweagent/run/preprocess/INSTRUCTIONS.md sections 1a and 1b."
            )

        test_command = run_unit_test_agent(
            model=model,
            repo=repo,
            kernel_name=kernel_name,
            log_dir=log_dir,
            preferred_harness_path=_preferred_harness_path(log_dir, kernel_path) if log_dir else None,
            kernel_path=kernel_path,
            discovery_context=ctx,
        )
        logger.info("UnitTestAgent test_command (attempt %d): %s", attempt, test_command)

        harness = extract_harness_path(test_command)

        # Phase 1: static analysis
        valid, harness_errors = validate_harness(harness)
        if not valid:
            logger.warning(
                "Harness static validation failed (attempt %d/%d): %s",
                attempt,
                max_attempts,
                harness_errors,
            )
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Harness validation failed after {max_attempts} attempts: " + "; ".join(harness_errors)
                )
            continue

        logger.info("Harness static validation: OK")

        # Phase 2: runtime execution of all modes
        # Use relaxed UTA timeouts here: complex kernels (e.g. physics sims)
        # can exceed the normal 300 s correctness limit during init, causing
        # the agent to retry in an infinite loop (issue #123).
        repo_root = str(repo) if repo else None
        exec_ok, exec_errors, harness_results = execute_harness_validation(
            harness,
            repo_root=repo_root,
            gpu_id=gpu_id,
            use_uta_timeouts=True,
        )
        if exec_ok:
            try:
                materialized = _materialize_validated_harness(
                    test_command=test_command,
                    harness_path=harness,
                    repo_root=repo,
                    log_dir=log_dir,
                    kernel_path=kernel_path,
                    gpu_id=gpu_id,
                )
            except Exception as exc:
                harness_errors = [str(exc)]
                logger.warning(
                    "Harness materialization failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    harness_errors,
                )
                if attempt == max_attempts:
                    raise RuntimeError(f"Harness materialization failed after {max_attempts} attempts: {exc}")
                continue
            if materialized is not None:
                test_command, harness, harness_results = materialized
                logger.info("Materialized harness to %s", harness)
            logger.info("Harness runtime validation: ALL MODES PASSED")
            return test_command, harness_results

        harness_errors = exec_errors
        logger.warning(
            "Harness runtime validation failed (attempt %d/%d): %s",
            attempt,
            max_attempts,
            [e.splitlines()[0] for e in exec_errors],
        )

        if attempt == max_attempts:
            raise RuntimeError(
                f"Harness runtime validation failed after {max_attempts} attempts: "
                + "; ".join(e.splitlines()[0] for e in exec_errors)
            )

    raise AssertionError("unreachable")  # pragma: no cover


# ── shared pipeline context and profiling guidance wrappers ──────────


def _bottleneck_guidance(metrics: dict) -> list[str]:
    return _shared_bottleneck_guidance(metrics)


def _format_gpu_info(gpu_info: dict) -> list[str]:
    return _shared_format_gpu_info(gpu_info)


def _gpu_arch_context(profiling_path: str) -> list[str]:
    return _shared_gpu_arch_context(profiling_path)


def inject_pipeline_context(
    task_body: str,
    config: dict,
    *,
    commandment_text: str | None = None,
    baseline_metrics: dict | None = None,
    profiling_path: str | None = None,
    kernel_path: str | None = None,
    repo_root: str | None = None,
    test_command: str | None = None,
    codebase_context: str | None = None,
    benchmark_baseline: str | None = None,
) -> tuple[str, dict]:
    return _shared_inject_pipeline_context(
        task_body,
        config,
        commandment_text=commandment_text,
        baseline_metrics=baseline_metrics,
        profiling_path=profiling_path,
        kernel_path=kernel_path,
        repo_root=repo_root,
        test_command=test_command,
        codebase_context=codebase_context,
        benchmark_baseline=benchmark_baseline,
    )


def run_baseline_profile(test_command: str, gpu_id: int = 0) -> dict:
    return _shared_run_baseline_profile(
        test_command,
        gpu_id=gpu_id,
        ensure_mcp_importable=_ensure_mcp_importable,
        extract_harness_path=extract_harness_path,
    )
