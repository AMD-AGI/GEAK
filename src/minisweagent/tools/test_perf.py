"""Test performance tool for saving patches and running performance tests."""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class TestPerfContext:
    """Context required for test_perf tool execution."""
    cwd: str
    test_command: str | None
    timeout: int
    patch_output_dir: str | None
    metric: str | None
    env_vars: dict | None = None
    base_repo_path: Path | None = None
    log_fn: Callable[[str], None] | None = None
    model_query_fn: Callable[[list], dict] | None = None
    # Mutable state (will be updated by tool)
    patch_counter: int = 0
    patch_results: dict | None = None


class TestPerfTool:
    """Tool to save patch and run performance test."""
    
    def __init__(self):
        self.context: TestPerfContext | None = None
    
    def set_context(self, context: TestPerfContext):
        """Set execution context from agent."""
        self.context = context
    
    def __call__(self, *, description: str = "", **kwargs) -> dict[str, Any]:
        if not self.context:
            return {"output": "TestPerfTool: context not configured", "returncode": 1}
        
        ctx = self.context
        patch_name = f"patch_{ctx.patch_counter}"
        ctx.patch_counter += 1
        
        desc_str = f" ({description})" if description else ""
        self._log(f"\n[TestPerf] Saving patch and running test{desc_str}...")
        
        try:
            # Get patch content
            patch_content = self._get_patch_content()
            
            if not patch_content.strip():
                self._log(f"[TestPerf] No changes detected, baseline running.")
            else:
                self._log(f"[TestPerf] Patch {patch_name} captured, running test...")
            
            # Run test
            test_output, test_passed, test_returncode = self._run_test()
            
            # Save results
            if ctx.patch_results is not None:
                ctx.patch_results[patch_name] = {
                    "patch_file": f"{patch_name}.patch",
                    "test_output_file": f"{patch_name}_test.txt",
                    "test_passed": test_passed,
                    "returncode": test_returncode,
                }
            
            status = "✓ PASSED" if test_passed else "✗ FAILED"
            self._log(f"[TestPerf] Test result for {patch_name}: {status}")
            
            # Extract metric
            metric_result = self._extract_metric(patch_name, test_output)
            if ctx.patch_results is not None:
                ctx.patch_results[patch_name]["metric_result"] = metric_result
            
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
        """Get current changes as patch content."""
        ctx = self.context
        cwd = ctx.cwd
        
        if self._is_git_repo(Path(cwd)):
            result = subprocess.run(
                "git add -N . && git diff",
                cwd=cwd, capture_output=True, text=True, timeout=10, shell=True
            )
            return result.stdout
        
        if ctx.base_repo_path and ctx.base_repo_path.exists():
            excludes = [".git", "__pycache__"]
            if ctx.patch_output_dir:
                run_dir_name = Path(ctx.patch_output_dir).resolve().parent.name
                if run_dir_name:
                    excludes.append(run_dir_name)
            
            result = subprocess.run(
                ["diff", "-ruN", "--exclude=.git", "--exclude=__pycache__",
                 *[f"--exclude={p}" for p in excludes if p not in (".git", "__pycache__")],
                 str(ctx.base_repo_path), str(cwd)],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout
        
        return ""
    
    def _run_test(self) -> tuple[str, bool, int]:
        """Run test command and return (output, passed, returncode)."""
        ctx = self.context
        
        if not ctx.test_command:
            error_msg = "[TestPerf] ERROR: test_command is not configured."
            self._log(error_msg)
            return error_msg, False, -1
        
        test_env = os.environ.copy()
        if ctx.env_vars:
            test_env.update(ctx.env_vars)
        test_env["PYTHONUNBUFFERED"] = "1"
        
        test_command = ctx.test_command.replace("WORK_REPO", ctx.cwd)
        self._log(f"[TestPerf] Running: {test_command}")
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
            tmp_file = tmp.name
        
        try:
            wrapped_cmd = f"({test_command}) > {tmp_file} 2>&1; echo $? > {tmp_file}.exitcode"
            subprocess.run(
                wrapped_cmd, shell=True, cwd=ctx.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=ctx.timeout, env=test_env
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
    
    def _extract_metric(self, patch_name: str, test_output: str):
        """Extract performance metrics from test output using LLM."""
        ctx = self.context
        if not ctx.model_query_fn:
            return None
        
        self._log(f"[TestPerf] Extracting metric for {patch_name}...")
        
        metric_prompt = ctx.metric or "Extract the performance metrics from the test output."
        prompt = (
            f"Task: {metric_prompt}\n\nTest output:\n{test_output}\n\n"
            "Extract the requested metric. Return ONLY a valid JSON object. "
            "If not found, return null.\n"
            'Examples: {"value": 123.45}, {"bandwidth": 123, "latency": 0.5}'
        )
        
        response = ctx.model_query_fn([
            {"role": "system", "content": "Analyze kernel test output and extract metrics."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            result = json.loads(response.get("content", "").strip())
            self._log(f"[TestPerf] Metric extracted: {result}")
            return result
        except json.JSONDecodeError:
            self._log(f"[TestPerf] Failed to parse metric")
            return None
    
    def _format_output(self, patch_name: str, patch_content: str, test_output: str, 
                       test_passed: bool, returncode: int) -> str:
        status = "PASSED ✓" if test_passed else "FAILED ✗"
        return "\n".join([
            f"\n{'='*60}",
            f"Patch saved: {patch_name}",
            f"Test status: {status}",
            f"Return code: {returncode}",
            f"{'='*60}",
            "\n## Test Output:",
            f"```\n{test_output}\n```",
            f"{'='*60}\n",
        ])
    
    def _handle_error(self, patch_name: str, patch_content: str, error_msg: str, status: str) -> dict:
        ctx = self.context
        if ctx.patch_results is not None:
            ctx.patch_results[patch_name] = {
                "patch_file": f"{patch_name}.patch",
                "test_output_file": f"{patch_name}_test.txt",
                "test_passed": False,
                "returncode": -1,
                "metric_result": None,
            }
        self._log(f"[TestPerf] Test for {patch_name}: ✗ {status}")
        
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
                check=True, capture_output=True, text=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
