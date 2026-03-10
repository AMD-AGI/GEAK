"""Save-and-test tool: saves patches and runs correctness + benchmark tests."""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent.benchmark_parsing import (
    compute_shape_speedups,
    extract_latency_ms,
    parse_shape_latencies_ms,
)
from minisweagent.debug_runtime import emit_debug_log


@dataclass
class SaveAndTestContext:
    """Context required for save_and_test tool execution."""

    cwd: str
    test_command: str | None
    timeout: int
    patch_output_dir: str | None
    env_vars: dict | None = None
    base_repo_path: Path | None = None
    log_fn: Callable[[str], None] | None = None
    patch_counter: int = 0
    helper_harness_logged: bool = False


class SaveAndTestTool:
    """Tool to save patch and run performance test."""

    def __init__(self):
        self.context: SaveAndTestContext | None = None

    def set_context(self, context: SaveAndTestContext):
        """Set execution context from agent."""
        self.context = context

    def __call__(self, *, description: str = "", **kwargs) -> dict[str, Any]:
        if not self.context:
            return {"output": "SaveAndTestTool: context not configured", "returncode": 1}

        ctx = self.context
        patch_name = f"patch_{ctx.patch_counter}"
        ctx.patch_counter += 1

        desc_str = f" ({description})" if description else ""
        self._log(f"\n[SaveAndTest] Saving patch and running test{desc_str}...")

        try:
            # Get patch content
            patch_content = self._get_patch_content()

            if not patch_content.strip():
                self._log("[SaveAndTest] No changes detected, baseline running.")
            else:
                self._log(f"[SaveAndTest] Patch {patch_name} captured, running test...")

            # Run test
            test_output, test_passed, test_returncode = self._run_test()

            status = "✓ PASSED" if test_passed else "✗ FAILED"
            self._log(f"[SaveAndTest] Test result for {patch_name}: {status}")

            # Save files
            if ctx.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)

            output = self._format_output(patch_name, patch_content, test_output, test_passed, test_returncode)
            return {"output": output, "returncode": 0 if test_passed else 1}

        except subprocess.TimeoutExpired:
            return self._handle_error(patch_name, patch_content, "Test command timed out", "TIMEOUT")
        except Exception as e:
            return self._handle_error(patch_name, "", str(e), f"ERROR - {e}")

    def _log(self, message: str):
        if self.context and self.context.log_fn:
            self.context.log_fn(message)

    def _generated_harness_helper_path(self) -> Path | None:
        """Return the generated worktree-root harness helper path, if any.

        During optimization runs, ``GEAK_HARNESS`` may point at a convenience
        helper like ``<worktree>/test_harness_<kernel>.py``. That file is
        infrastructure, not part of the candidate patch, and can be deleted by
        agent-side git operations. We treat only these root-level helper files
        specially; tracked harness files inside the task directory remain normal
        repo content.
        """
        ctx = self.context
        if not ctx:
            return None

        harness = (ctx.env_vars or {}).get("GEAK_HARNESS")
        if not isinstance(harness, str) or not harness.strip():
            return None

        cwd = Path(os.path.abspath(ctx.cwd))
        harness_path = Path(harness)
        if not harness_path.is_absolute():
            harness_path = cwd / harness_path
        harness_path = Path(os.path.abspath(harness_path))

        if harness_path.parent != cwd:
            return None
        if harness_path.suffix != ".py" or not harness_path.name.startswith("test_harness_"):
            return None
        return harness_path

    def _generated_helper_excludes(self) -> list[str]:
        """Return generated helper files that should never appear in patches."""
        ctx = self.context
        if not ctx:
            return []

        cwd = Path(os.path.abspath(ctx.cwd))
        excludes = [
            "run.sh",
            "run_harness.sh",
            "build",
            "build_harness",
            "test_harness.py",
            "test_harness_*.py",
            "test_harness_*.cpp",
            "rocprim_version.hpp",
        ]
        harness_helper = self._generated_harness_helper_path()
        if harness_helper is not None:
            try:
                excludes.append(harness_helper.relative_to(cwd).as_posix())
            except ValueError:
                pass
            if not ctx.helper_harness_logged:
                # region agent log
                emit_debug_log(
                    "save_and_test.py:_generated_helper_excludes",
                    "Ignoring generated worktree harness helper during patch capture",
                    {"cwd": str(cwd), "harness_helper": str(harness_helper)},
                    hypothesis_id="H10",
                )
                # endregion
                ctx.helper_harness_logged = True
        return excludes

    def _base_repo_counterpart(self, helper_path: Path) -> Path | None:
        ctx = self.context
        if not ctx or not ctx.base_repo_path:
            return None

        cwd = Path(os.path.abspath(ctx.cwd))
        try:
            rel = helper_path.relative_to(cwd)
        except ValueError:
            return None

        candidate = ctx.base_repo_path / rel
        if candidate.exists() or candidate.is_symlink():
            return candidate
        return None

    @staticmethod
    def _materialize_helper_file(source: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if source.is_symlink():
            dest.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, dest)

    def _fallback_task_harness(self, helper_path: Path) -> Path | None:
        ctx = self.context
        if not ctx:
            return None

        cwd = Path(os.path.abspath(ctx.cwd))
        candidates = sorted(cwd.glob("tasks/**/test_harness.py"))
        return candidates[0] if candidates else None

    def _restore_missing_harness_helper(self) -> None:
        """Recreate a missing worktree-root harness helper before testing."""
        helper_path = self._generated_harness_helper_path()
        if helper_path is None:
            return
        if helper_path.exists():
            return

        source = self._base_repo_counterpart(helper_path)
        restore_mode = None
        if source is not None:
            self._materialize_helper_file(source, helper_path)
            restore_mode = "copied_from_base_repo"
        else:
            fallback = self._fallback_task_harness(helper_path)
            if fallback is None:
                return
            helper_path.parent.mkdir(parents=True, exist_ok=True)
            if helper_path.exists() or helper_path.is_symlink():
                helper_path.unlink()
            helper_path.symlink_to(os.path.relpath(fallback, helper_path.parent))
            source = fallback
            restore_mode = "symlinked_to_task_harness"

        # region agent log
        emit_debug_log(
            "save_and_test.py:_restore_missing_harness_helper",
            "Restored missing worktree harness helper before save_and_test execution",
            {
                "helper_path": str(helper_path),
                "source_path": str(source) if source is not None else None,
                "restore_mode": restore_mode,
            },
            hypothesis_id="H10",
        )
        # endregion

    def _find_true_baseline_file(self) -> Path | None:
        """Walk upward from patch_output_dir to find the original benchmark baseline."""
        ctx = self.context
        if not ctx or not ctx.patch_output_dir:
            return None

        current = Path(ctx.patch_output_dir).resolve()
        for _ in range(8):
            candidate = current / "benchmark_baseline.txt"
            if candidate.is_file():
                return candidate
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None

    def _format_speedup_summary(self, test_output: str) -> list[str]:
        """Summarize overall and per-shape speedups against the true baseline."""
        baseline_file = self._find_true_baseline_file()
        if baseline_file is None:
            return []

        try:
            baseline_text = baseline_file.read_text()
        except OSError:
            return []

        baseline_ms = extract_latency_ms(baseline_text)
        candidate_ms = extract_latency_ms(test_output)
        if baseline_ms is None or candidate_ms is None or baseline_ms <= 0 or candidate_ms <= 0:
            return []

        lines = [
            "\nSpeedup vs true baseline:",
            (
                f"Overall: {baseline_ms / candidate_ms:.4f}x "
                f"({baseline_ms:.6f} ms -> {candidate_ms:.6f} ms)"
            ),
        ]

        baseline_shapes = parse_shape_latencies_ms(baseline_text)
        candidate_shapes = parse_shape_latencies_ms(test_output)
        per_shape = compute_shape_speedups(baseline_shapes, candidate_shapes)
        if per_shape:
            lines.append("Per-shape:")
            for shape, metrics in per_shape.items():
                lines.append(
                    "  "
                    f"{shape}: {metrics['speedup']:.4f}x "
                    f"({metrics['baseline_ms']:.6f} ms -> {metrics['candidate_ms']:.6f} ms)"
                )
        return lines

    def _get_patch_content(self) -> str:
        """Get current changes as patch content."""
        ctx = self.context
        cwd = ctx.cwd

        if self._is_git_repo(Path(cwd)):
            excludes = [
                "traj.json",
                "*.log",
                ".rocprofv3/",
                "__pycache__/",
                "*.pyc",
                ".pytest_cache/",
                "*.egg-info/",
                "*.so",
                ".geak_resolved/",
                *self._generated_helper_excludes(),
            ]
            exclude_args = " ".join(f"':(exclude){entry}'" for entry in excludes)
            result = subprocess.run(
                f"git add -N . && git diff -- . {exclude_args}",
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
                shell=True,
            )
            return result.stdout

        if ctx.base_repo_path and ctx.base_repo_path.exists():
            excludes = [".git", "__pycache__", *self._generated_helper_excludes()]
            if ctx.patch_output_dir:
                run_dir_name = Path(ctx.patch_output_dir).resolve().parent.name
                if run_dir_name:
                    excludes.append(run_dir_name)

            result = subprocess.run(
                [
                    "diff",
                    "-ruN",
                    "--exclude=.git",
                    "--exclude=__pycache__",
                    *[f"--exclude={p}" for p in excludes if p not in (".git", "__pycache__")],
                    str(ctx.base_repo_path),
                    str(cwd),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout

        return ""

    def _run_test(self) -> tuple[str, bool, int]:
        """Run test command and return (output, passed, returncode)."""
        ctx = self.context

        if not ctx.test_command:
            error_msg = "[SaveAndTest] ERROR: test_command is not configured."
            self._log(error_msg)
            return error_msg, False, -1

        test_env = os.environ.copy()
        if ctx.env_vars:
            test_env.update(ctx.env_vars)
        test_env["PYTHONUNBUFFERED"] = "1"
        self._restore_missing_harness_helper()

        # If test_command still contains the original repo root path, replace it with the
        # current working directory (worktree). Uses base_repo_path from context instead of
        # any hardcoded path. Skip replacement if cwd is already present (already rewritten).
        test_command = ctx.test_command
        if ctx.base_repo_path:
            repo_root = str(ctx.base_repo_path)
            if repo_root in test_command and ctx.cwd not in test_command:
                test_command = test_command.replace(repo_root, ctx.cwd)
        self._log(f"[SaveAndTest] Running: {test_command}")

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp:
            tmp_file = tmp.name

        try:
            wrapped_cmd = f"({test_command}) > {tmp_file} 2>&1; echo $? > {tmp_file}.exitcode"
            subprocess.run(
                wrapped_cmd,
                shell=True,
                cwd=ctx.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=ctx.timeout,
                env=test_env,
            )

            test_output = Path(tmp_file).read_text() if Path(tmp_file).exists() else ""
            exitcode_file = Path(f"{tmp_file}.exitcode")
            try:
                returncode = int(exitcode_file.read_text().strip()) if exitcode_file.exists() else 0
            except (ValueError, OSError):
                returncode = 0

            return test_output, returncode == 0, returncode
        finally:
            for f in [tmp_file, f"{tmp_file}.exitcode"]:
                try:
                    Path(f).unlink(missing_ok=True)
                except Exception:
                    pass

    def _format_output(
        self, patch_name: str, patch_content: str, test_output: str, test_passed: bool, returncode: int
    ) -> str:
        status = "PASSED ✓" if test_passed else "FAILED ✗"

        lines = [
            f"\n{'=' * 60}",
            f"Patch saved: {patch_name}",
            f"Test status: {status}",
            f"Return code: {returncode}",
        ]
        lines.extend(self._format_speedup_summary(test_output))

        # Add log file locations if patch_output_dir is configured
        if self.context.patch_output_dir:
            output_dir = Path(self.context.patch_output_dir).resolve()
            patch_file = output_dir / f"{patch_name}.patch"
            test_log_file = output_dir / f"{patch_name}_test.txt"
            lines.extend(
                [
                    "\nFiles saved to:",
                    f"  - Patch: {patch_file}",
                    f"  - Test log: {test_log_file}",
                ]
            )

        lines.extend(
            [
                f"{'=' * 60}",
                "\n## Test Output:",
                f"```\n{test_output}\n```",
                f"{'=' * 60}\n",
            ]
        )

        return "\n".join(lines)

    def _handle_error(self, patch_name: str, patch_content: str, error_msg: str, status: str) -> dict:
        ctx = self.context

        self._log(f"[SaveAndTest] Test for {patch_name}: ✗ {status}")

        if ctx.patch_output_dir:
            self._save_patch_file(patch_name, patch_content)
            self._save_test_output(patch_name, error_msg)

        output = self._format_output(patch_name, patch_content, error_msg, False, -1)
        return {"output": output, "returncode": 1}

    def _save_patch_file(self, patch_name: str, patch_content: str):
        output_dir = Path(self.context.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{patch_name}.patch").write_text(patch_content)

    def _save_test_output(self, patch_name: str, test_output: str):
        output_dir = Path(self.context.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{patch_name}_test.txt").write_text(test_output)

    @staticmethod
    def _is_git_repo(path: Path) -> bool:
        try:
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False
