"""Save-and-test tool: saves patches and runs correctness + benchmark tests."""

import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    def _get_patch_content(self) -> str:
        """Get current changes as patch content.

        Relies on .gitignore to exclude build artifacts. The parallel_agent ensures
        all workspaces are git repos with proper .gitignore before worktree creation.
        """
        ctx = self.context
        cwd = ctx.cwd

        if not self._is_git_repo(Path(cwd)):
            self._log("[SaveAndTest] Warning: not a git repo, cannot generate patch")
            return ""

        # git add -N stages "intent to add" for new files (respects .gitignore)
        # git diff then includes these new files in the diff output
        result = subprocess.run(
            "git add -N . && git diff",
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True,
        )
        return result.stdout

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
