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
