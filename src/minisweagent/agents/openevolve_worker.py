"""OpenEvolveWorker -- thin agent that runs openevolve-mcp on assigned GPUs.

Unlike strategy agents that run a full LLM optimization loop, this agent
just kicks off the OpenEvolve evolutionary optimizer and collects results.
It does NOT use the LLM for reasoning -- it delegates entirely to OpenEvolve.

Used by ParallelAgent as one of the heterogeneous sub-agent types.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent, Submitted

logger = logging.getLogger(__name__)


@dataclass
class OpenEvolveWorkerConfig(AgentConfig):
    """Config for OpenEvolveWorker."""

    kernel_path: str = ""
    max_iterations: int = 10
    output_dir: str | None = None
    commandment_path: str | None = None
    baseline_metrics_path: str | None = None


class OpenEvolveWorker(DefaultAgent):
    """Agent that runs OpenEvolve evolutionary optimization.

    Instead of an LLM loop, this agent:
    1. Calls the openevolve-mcp tool with the kernel path and config
    2. Waits for it to complete (can take hours)
    3. Saves the best result as a patch
    4. Returns the result

    GPU assignment is handled by the caller (ParallelAgent sets HIP_VISIBLE_DEVICES).
    """

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=OpenEvolveWorkerConfig, **kwargs)

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run OpenEvolve on the configured kernel.

        The task string is logged but not used for LLM reasoning.
        The actual work is done by the openevolve-mcp tool.
        """
        kernel_path = self.config.kernel_path
        if not kernel_path:
            return "Error", "No kernel_path configured for OpenEvolveWorker"

        self._log_message(
            f"[OpenEvolveWorker] Starting evolutionary optimization\n"
            f"  kernel: {kernel_path}\n"
            f"  max_iterations: {self.config.max_iterations}\n"
            f"  GPU: {self.env.config.env.get('HIP_VISIBLE_DEVICES', 'default')}\n"
        )

        # Build MCP tool arguments
        mcp_args: dict[str, Any] = {
            "kernel_path": str(Path(kernel_path).resolve()),
            "max_iterations": self.config.max_iterations,
        }
        if self.config.output_dir:
            mcp_args["output_dir"] = self.config.output_dir
        if self.config.commandment_path:
            mcp_args["commandment_path"] = self.config.commandment_path
        if self.config.baseline_metrics_path:
            mcp_args["baseline_metrics_path"] = self.config.baseline_metrics_path

        # Get GPU from env
        gpu = self.env.config.env.get("HIP_VISIBLE_DEVICES", "0") if hasattr(self.env, "config") else "0"
        mcp_args["gpu"] = int(gpu.split(",")[0]) if gpu else 0

        # Call openevolve via ToolRuntime (uses MCPToolBridge)
        try:
            result = self.toolruntime.dispatch({
                "name": "openevolve",
                "arguments": mcp_args,
            })
        except ValueError:
            # openevolve not registered in ToolRuntime -- fall back to optimizer
            self._log_message("[OpenEvolveWorker] openevolve tool not in ToolRuntime, using optimizer.core")
            try:
                from minisweagent.optimizer.core import OptimizerType, optimize_kernel

                opt_result = optimize_kernel(
                    kernel_path=kernel_path,
                    optimizer_type=OptimizerType.OPENEVOLVE,
                    max_iterations=self.config.max_iterations,
                    gpu=mcp_args.get("gpu", 0),
                    output_dir=self.config.output_dir,
                    commandment_path=self.config.commandment_path,
                    baseline_metrics_path=self.config.baseline_metrics_path,
                )
                result = {
                    "output": json.dumps({
                        "success": True,
                        "optimized_code": opt_result.optimized_code,
                        "metrics": opt_result.metrics,
                        "iterations": opt_result.iterations,
                    }),
                    "returncode": 0,
                }
            except Exception as e:
                result = {"output": f"OpenEvolve optimization failed: {e}", "returncode": 1}

        self._log_message(f"[OpenEvolveWorker] Result: returncode={result.get('returncode')}")

        if result.get("returncode", 1) == 0:
            # Try to save the result as a patch
            output_text = result.get("output", "")
            try:
                data = json.loads(output_text)
                speedup = data.get("metrics", {}).get("speedup", data.get("speedup", "unknown"))
                self._log_message(f"[OpenEvolveWorker] Speedup: {speedup}")
            except (json.JSONDecodeError, AttributeError):
                pass

            raise Submitted(f"OpenEvolve completed: {output_text[:500]}")

        return "Error", result.get("output", "OpenEvolve failed")

    def _log_message(self, message: str):
        """Log to file or stdout."""
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception:
                pass
        logger.info(message)


# ---------------------------------------------------------------------------
# Standalone CLI -- bypasses the agent runtime and calls optimizer.core
# directly, reproducing the same behavior ParallelAgent would trigger.
#
# Usage:
#   python -m minisweagent.agents.openevolve_worker \
#       --kernel-path /path/to/kernel.py \
#       --commandment COMMANDMENT.md \
#       --baseline-metrics baseline_metrics.json \
#       --iterations 10 --gpu 0
# ---------------------------------------------------------------------------

def main():
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Run OpenEvolve optimization on a kernel (standalone, no agent runtime)",
    )
    parser.add_argument("--kernel-path", default=None, help="Path to the kernel file to optimize")
    parser.add_argument(
        "--from-task", default=None, metavar="FILE",
        help="Read a task .md file (YAML frontmatter) to populate kernel-path, commandment, etc.",
    )
    parser.add_argument("--commandment", default=None, help="Path to COMMANDMENT.md")
    parser.add_argument("--baseline-metrics", default=None, help="Path to baseline_metrics.json")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations (default: 10)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")

    args = parser.parse_args()

    # Populate from task file if provided (explicit flags override)
    if args.from_task:
        from minisweagent.run.task_file import read_task_file
        meta, _body = read_task_file(Path(args.from_task))
        if not args.kernel_path:
            args.kernel_path = meta.get("kernel_path")
        if not args.commandment:
            args.commandment = meta.get("commandment")
        if not args.baseline_metrics:
            args.baseline_metrics = meta.get("baseline_metrics")

    if not args.kernel_path:
        parser.error("--kernel-path is required (or provide --from-task)")

    kernel_path = Path(args.kernel_path).resolve()
    if not kernel_path.is_file():
        print(f"ERROR: kernel file not found: {args.kernel_path}", file=sys.stderr)
        sys.exit(1)

    if args.commandment and not Path(args.commandment).is_file():
        print(f"ERROR: commandment file not found: {args.commandment}", file=sys.stderr)
        sys.exit(1)

    if args.baseline_metrics and not Path(args.baseline_metrics).is_file():
        print(f"ERROR: baseline metrics file not found: {args.baseline_metrics}", file=sys.stderr)
        sys.exit(1)

    # Auto-worktree isolation when using --from-task with a git repo
    worktree_path = None
    original_repo = None
    if args.from_task:
        from minisweagent.run.task_file import read_task_file, create_worktree, is_git_repo, replace_paths
        meta, _ = read_task_file(Path(args.from_task))
        repo_root = meta.get("repo_root")
        if repo_root:
            repo_path = Path(repo_root).resolve()
            if repo_path.is_dir() and is_git_repo(repo_path):
                output_dir_base = Path(args.output_dir).resolve() if args.output_dir else kernel_path.parent / "optimization_output"
                wt_dest = output_dir_base / "worktree"
                print(f"[OpenEvolveWorker CLI] Creating isolated worktree at {wt_dest}...", file=sys.stderr)
                worktree_path = create_worktree(repo_path, wt_dest)
                original_repo = repo_path
                # Rewrite kernel_path into worktree
                kernel_path = Path(replace_paths(str(kernel_path), repo_path, worktree_path))
                if args.commandment:
                    args.commandment = replace_paths(args.commandment, repo_path, worktree_path)
                if args.baseline_metrics:
                    args.baseline_metrics = replace_paths(args.baseline_metrics, repo_path, worktree_path)

    output_dir = args.output_dir or str(kernel_path.parent / "optimization_output")

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)

    print(
        f"[OpenEvolveWorker CLI] Starting optimization\n"
        f"  kernel:           {kernel_path}\n"
        f"  commandment:      {args.commandment or '(none)'}\n"
        f"  baseline_metrics: {args.baseline_metrics or '(none)'}\n"
        f"  iterations:       {args.iterations}\n"
        f"  gpu:              {args.gpu}\n"
        f"  output_dir:       {output_dir}"
        + (f"\n  worktree:         {worktree_path}" if worktree_path else ""),
        flush=True,
    )

    from minisweagent.optimizer.core import OptimizerType, optimize_kernel

    try:
        result = optimize_kernel(
            kernel_code="",
            kernel_path=str(kernel_path),
            optimizer=OptimizerType.OPENEVOLVE,
            max_iterations=args.iterations,
            gpu=args.gpu,
            output_dir=output_dir,
            commandment_path=args.commandment,
            baseline_metrics_path=args.baseline_metrics,
        )
    except Exception as e:
        print(f"ERROR: OpenEvolve failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "optimizer": result.optimizer_used,
        "iterations": result.iterations,
        "metrics": result.metrics,
        "optimized_code_length": len(result.optimized_code),
    }, indent=2))
    print("[OpenEvolveWorker CLI] Done.", flush=True)


if __name__ == "__main__":
    main()
