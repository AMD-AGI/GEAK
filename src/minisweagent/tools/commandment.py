"""Generate and validate COMMANDMENT.md files for OpenEvolve.

A COMMANDMENT.md is the evaluation contract between the agent and OpenEvolve.
It has exactly three sections:
  ## SETUP        -- prepare the evaluation environment
  ## CORRECTNESS  -- verify the optimized kernel is correct
  ## PROFILE      -- benchmark the optimized kernel

This module generates a valid COMMANDMENT.md deterministically from:
  - kernel_path: the kernel file to optimize
  - harness_path: the test harness (must support --correctness and --profile)
  - repo_root: the repository root (for PYTHONPATH)
  - inner_kernel: whether the kernel is an inner kernel imported by a wrapper

Generation includes a built-in validation loop: after generating, the content
is validated using ``validate_commandment()``.  If validation fails, known
fixable issues are auto-corrected and re-validated, up to ``max_retries``
times.  The generator returns only fully-valid content.

Usage (Python):
    from minisweagent.tools.commandment import generate_commandment
    content = generate_commandment(
        kernel_path="/path/to/kernel.py",
        harness_path="/path/to/test_harness.py",
        repo_root="/path/to/repo",
    )

Usage (CLI):
    python -m minisweagent.tools.commandment \\
        --kernel-path /path/to/kernel.py \\
        --harness /path/to/test_harness.py \\
        --repo-root /path/to/repo \\
        --output COMMANDMENT.md
"""

from __future__ import annotations

# Import validate_commandment directly from the sibling module to avoid
# pulling in the full minisweagent.tools package (whose __init__.py imports
# heavy dependencies like typer via strategy_manager).
import importlib.util as _ilu
import re
from pathlib import Path

_vc_path = Path(__file__).with_name("validate_commandment.py")
_vc_spec = _ilu.spec_from_file_location("validate_commandment", _vc_path)
_vc_mod = _ilu.module_from_spec(_vc_spec)
_vc_spec.loader.exec_module(_vc_mod)
validate_commandment = _vc_mod.validate_commandment
format_validation_message = _vc_mod.format_validation_message

_MAX_FIX_RETRIES = 3


def generate_commandment(
    kernel_path: str | Path,
    harness_path: str | Path,
    repo_root: str | Path | None = None,
    *,
    inner_kernel: bool = False,
    inner_kernel_relpath: str | None = None,
    warmup_runs: int = 2,
    profile_replays: int = 5,
) -> str:
    """Generate a valid COMMANDMENT.md and return its content.

    Args:
        kernel_path: Absolute path to the kernel file being optimized.
        harness_path: Absolute path to the test harness script.  Must accept
            ``--correctness`` and ``--profile`` flags.
        repo_root: Repository root for PYTHONPATH.  If *None*, uses the
            parent of *kernel_path*.
        inner_kernel: If *True*, the kernel is an inner file imported by a
            wrapper.  The SETUP section will create package directory
            structure inside ``${GEAK_WORK_DIR}`` and shadow the original.
        inner_kernel_relpath: Relative path from *repo_root* to the inner
            kernel file (e.g. ``aiter/ops/triton/_triton_kernels/rope/rope.py``).
            Required when *inner_kernel* is True.
        warmup_runs: Number of warm-up invocations before profiling.  Kept
            in sync with profiler-mcp's default so that agent-side and
            preprocessor-side profiling see identical warm-up conditions.
        profile_replays: Number of replay passes for ``kernel-profile``.

    Returns:
        The content of a valid COMMANDMENT.md as a string.

    Raises:
        ValueError: If validation still fails after auto-fix retries.
    """
    kernel_path = Path(kernel_path).resolve()
    harness_path = Path(harness_path).resolve()

    if repo_root is not None:
        repo_root = Path(repo_root).resolve()
    else:
        repo_root = kernel_path.parent

    if inner_kernel and not inner_kernel_relpath:
        try:
            inner_kernel_relpath = str(kernel_path.relative_to(repo_root))
        except ValueError:
            inner_kernel_relpath = kernel_path.name

    if inner_kernel:
        content = _generate_inner_kernel(
            kernel_path=kernel_path,
            harness_path=harness_path,
            repo_root=repo_root,
            inner_kernel_relpath=inner_kernel_relpath,
            warmup_runs=warmup_runs,
            profile_replays=profile_replays,
        )
    else:
        content = _generate_simple(
            kernel_path=kernel_path,
            harness_path=harness_path,
            repo_root=repo_root,
            warmup_runs=warmup_runs,
            profile_replays=profile_replays,
        )

    return _validate_and_fix(content, harness_path=str(harness_path))


def _warmup_block(command: str, warmup_runs: int) -> str:
    """Build the warmup section for the PROFILE block.

    Returns a single command for 1 run, or a bash for-loop for multiple runs
    (avoids emitting duplicate identical lines).
    """
    if warmup_runs <= 0:
        return ""
    if warmup_runs == 1:
        return command
    return f"for _i in $(seq 1 {warmup_runs}); do {command}; done"


def _generate_simple(
    kernel_path: Path,
    harness_path: Path,
    repo_root: Path,
    warmup_runs: int,
    profile_replays: int,
) -> str:
    """Generate COMMANDMENT for a simple (non-inner) kernel.

    All paths are expressed via environment variables so that the
    COMMANDMENT can be executed verbatim in any worktree:

      * ``GEAK_WORK_DIR``  -- the agent's working copy / worktree root
      * ``GEAK_REPO_ROOT`` -- the original repository root
      * ``GEAK_GPU_DEVICE`` -- GPU device ID
      * ``GEAK_HARNESS``   -- absolute path to the test harness script
    """
    warmup_block = _warmup_block(
        "${GEAK_WORK_DIR}/run.sh ${GEAK_HARNESS} --profile > /dev/null 2>&1 || true",
        warmup_runs,
    )

    return f"""\
## SETUP
printf '#!/bin/bash\\nexport PYTHONPATH=%s:%s:${{PYTHONPATH}}\\nexport HIP_VISIBLE_DEVICES=%s\\nexec python3 "$@"\\n' "${{GEAK_WORK_DIR}}" "${{GEAK_REPO_ROOT}}" "${{GEAK_GPU_DEVICE}}" > ${{GEAK_WORK_DIR}}/run.sh && chmod +x ${{GEAK_WORK_DIR}}/run.sh

## CORRECTNESS
${{GEAK_WORK_DIR}}/run.sh ${{GEAK_HARNESS}} --correctness

## PROFILE
{warmup_block}
kernel-profile "${{GEAK_WORK_DIR}}/run.sh ${{GEAK_HARNESS}} --profile" --gpu-devices ${{GEAK_GPU_DEVICE}} --replays {profile_replays}
"""


def _generate_inner_kernel(
    kernel_path: Path,
    harness_path: Path,
    repo_root: Path,
    inner_kernel_relpath: str,
    warmup_runs: int,
    profile_replays: int,
) -> str:
    """Generate COMMANDMENT for an inner kernel (imported by a wrapper).

    The SETUP section creates the package directory structure inside
    ``${GEAK_WORK_DIR}`` so the mutated candidate shadows the original.
    """
    rel_path = Path(inner_kernel_relpath)
    rel_dir = rel_path.parent
    basename = rel_path.name

    # Build mkdir for the package directory
    mkdir_path = f"${{GEAK_WORK_DIR}}/{rel_dir}"

    # Build touch commands for __init__.py at each package level
    init_paths = []
    current = rel_dir
    while str(current) not in (".", ""):
        init_paths.append(f"${{GEAK_WORK_DIR}}/{current}/__init__.py")
        current = current.parent
    init_touch = " ".join(init_paths) if init_paths else ""

    # Copy candidate to the correct import path
    copy_cmd = f"cp ${{GEAK_WORK_DIR}}/{kernel_path.name} ${{GEAK_WORK_DIR}}/{rel_dir}/{basename}"

    warmup_block = _warmup_block(
        "${GEAK_WORK_DIR}/run_harness.sh --profile > /dev/null 2>&1 || true",
        warmup_runs,
    )

    setup_lines = [
        f"mkdir -p {mkdir_path}",
        copy_cmd,
    ]
    if init_touch:
        setup_lines.append(f"touch {init_touch}")

    setup_lines.append(
        "printf '#!/bin/bash\\nexport PYTHONPATH=%s:%s:${PYTHONPATH}\\n"
        'export HIP_VISIBLE_DEVICES=%s\\nexec python3 ${GEAK_HARNESS} "$@"\\n\' '
        '"${GEAK_WORK_DIR}" "${GEAK_REPO_ROOT}" "${GEAK_GPU_DEVICE}" > ${GEAK_WORK_DIR}/run_harness.sh '
        "&& chmod +x ${GEAK_WORK_DIR}/run_harness.sh"
    )

    setup_block = "\n".join(setup_lines)

    return f"""\
## SETUP
{setup_block}

## CORRECTNESS
${{GEAK_WORK_DIR}}/run_harness.sh --correctness

## PROFILE
{warmup_block}
kernel-profile "${{GEAK_WORK_DIR}}/run_harness.sh --profile" --gpu-devices ${{GEAK_GPU_DEVICE}} --replays {profile_replays}
"""


def _validate_and_fix(content: str, *, harness_path: str | None = None) -> str:
    """Validate content and attempt to auto-fix known issues.

    Raises ValueError if the content cannot be made valid after retries.
    """
    for attempt in range(_MAX_FIX_RETRIES + 1):
        result = validate_commandment(content, harness_path=harness_path)
        if result["valid"]:
            return content

        if attempt == _MAX_FIX_RETRIES:
            break

        content = _auto_fix(content, result["errors"])

    msg = format_validation_message(validate_commandment(content, harness_path=harness_path))
    raise ValueError(f"COMMANDMENT.md generation failed validation after {_MAX_FIX_RETRIES} retries:\n{msg}")


def _auto_fix(content: str, errors: list[str]) -> str:
    """Attempt to fix known validation errors in-place."""
    for error in errors:
        # Fix: unknown section headers -> remove them
        m = re.search(r"Unknown section\(s\): (.*?)\.", error)
        if m:
            unknown_names = re.findall(r"## (\w+)", m.group(1))
            for name in unknown_names:
                content = re.sub(rf"^## {re.escape(name)}\b.*$", "", content, flags=re.MULTILINE)

        # Fix: shell built-in as command prefix -> wrap in bash -c
        m = re.search(r"Command starts with shell built-in '(\w+)': '(.*?)'", error)
        if m:
            original_cmd = m.group(2)
            fixed_cmd = f'bash -c "{original_cmd}"'
            content = content.replace(original_cmd, fixed_cmd)

        # Fix: inline env var prefix -> wrap in bash -c
        m = re.search(r"Command uses inline env var prefix '(\w+=\S+)' .* '(.*?)'", error)
        if m:
            original_cmd = m.group(2)
            fixed_cmd = f'bash -c "{original_cmd}"'
            content = content.replace(original_cmd, fixed_cmd)

    return content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _extract_harness_from_command(command: str) -> str | None:
    """Extract the .py script path from a test command like 'python /path/to/test.py -v'."""
    import shlex

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        if token.endswith(".py"):
            return token
    return None


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Generate a valid COMMANDMENT.md for OpenEvolve",
    )
    parser.add_argument("--kernel-path", default=None, help="Path to the kernel file")
    parser.add_argument("--harness", default=None, help="Path to the test harness script")
    parser.add_argument("--repo-root", default=None, help="Repository root for PYTHONPATH (default: kernel parent dir)")
    parser.add_argument(
        "--from-discovery",
        default=None,
        metavar="FILE",
        help="Read discovery.json and extract kernel-path, harness, and repo-root",
    )
    parser.add_argument("--inner-kernel", action="store_true", help="Kernel is an inner file imported by a wrapper")
    parser.add_argument("--inner-kernel-relpath", default=None, help="Relative path from repo-root to inner kernel")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warm-up runs before profiling (default: 2)")
    parser.add_argument("--profile-replays", type=int, default=5, help="Profiling replay count (default: 5)")
    parser.add_argument("-o", "--output", default=None, help="Output file path (default: stdout)")

    args = parser.parse_args()

    # Populate from discovery JSON if provided (explicit flags override)
    if args.from_discovery:
        disc = json.loads(Path(args.from_discovery).read_text())
        if not args.kernel_path:
            args.kernel_path = (disc.get("kernel") or {}).get("file")
        if not args.repo_root:
            args.repo_root = disc.get("workspace")
        if not args.harness:
            focused = disc.get("focused_test") or {}
            args.harness = focused.get("focused_test_file")
            if not args.harness:
                tests = disc.get("tests") or []
                if tests:
                    args.harness = _extract_harness_from_command(tests[0].get("command", ""))

    if not args.kernel_path:
        parser.error("--kernel-path is required (or provide --from-discovery)")
    if not args.harness:
        parser.error("--harness is required (or provide --from-discovery)")

    try:
        content = generate_commandment(
            kernel_path=args.kernel_path,
            harness_path=args.harness,
            repo_root=args.repo_root,
            inner_kernel=args.inner_kernel,
            inner_kernel_relpath=args.inner_kernel_relpath,
            warmup_runs=args.warmup_runs,
            profile_replays=args.profile_replays,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(content)
        print(f"Wrote {out_path}", file=sys.stderr)

        # Print validation summary to stderr
        result = validate_commandment(content)
        print(format_validation_message(result), file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
