# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Tool: baseline_metrics -- format profiler output into OpenEvolve JSON.

Wraps minisweagent.baseline_metrics for use as a ToolRuntime tool.
"""

from __future__ import annotations

import json
from typing import Any


class BaselineMetricsTool:
    """ToolRuntime-compatible callable for baseline_metrics."""

    def __call__(
        self,
        profiler_output: str,
        kernel_names: str | None = None,
        kernel_indices: str | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Build baseline metrics JSON from profiler output.

        Args:
            profiler_output: JSON string from MetrixTool/profiler-mcp output.
            kernel_names: Comma-separated kernel names to include.
            kernel_indices: Comma-separated kernel indices (0-based) to include.
            output_path: If set, write JSON to this file path.

        Returns:
            {output: str, returncode: int}
        """
        try:
            from minisweagent.baseline_metrics import build_baseline_metrics
        except ImportError as e:
            return {"output": f"baseline_metrics not available: {e}", "returncode": 1}

        try:
            profiler_data = json.loads(profiler_output)
        except json.JSONDecodeError as e:
            return {"output": f"Invalid JSON profiler output: {e}", "returncode": 1}

        names = [n.strip() for n in kernel_names.split(",")] if kernel_names else None
        indices = [int(i.strip()) for i in kernel_indices.split(",")] if kernel_indices else None

        try:
            baseline = build_baseline_metrics(
                profiler_data,
                kernel_names=names,
                kernel_indices=indices,
            )
        except Exception as e:
            return {"output": f"Failed to build baseline metrics: {e}", "returncode": 1}

        result_json = json.dumps(baseline, indent=2)

        if output_path:
            try:
                from pathlib import Path

                out = Path(output_path)
                out.write_text(result_json)
                # Post-write roundtrip validation: ensure the file is valid JSON
                json.loads(out.read_text())
            except json.JSONDecodeError as e:
                return {
                    "output": f"CRITICAL: Wrote {output_path} but it contains invalid JSON: {e}\n{result_json}",
                    "returncode": 1,
                }
            except Exception as e:
                return {"output": f"Built metrics but failed to write: {e}\n{result_json}", "returncode": 1}

        return {"output": result_json, "returncode": 0}
