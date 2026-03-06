# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Tests for baseline_metrics_tool."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from minisweagent.tools.baseline_metrics_tool import BaselineMetricsTool


class TestBaselineMetricsTool:
    def test_invalid_json(self):
        tool = BaselineMetricsTool()
        result = tool(profiler_output="not json at all")
        assert result["returncode"] == 1
        assert "Invalid JSON" in result["output"]

    @patch("minisweagent.baseline_metrics.build_baseline_metrics")
    def test_build_success(self, mock_build):
        mock_build.return_value = {"duration_us": 100, "bottleneck": "memory"}
        tool = BaselineMetricsTool()
        result = tool(profiler_output='{"results": []}')
        assert result["returncode"] == 0
        parsed = json.loads(result["output"])
        assert parsed["duration_us"] == 100

    @patch("minisweagent.baseline_metrics.build_baseline_metrics")
    def test_output_path(self, mock_build):
        mock_build.return_value = {"duration_us": 50}
        tool = BaselineMetricsTool()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        result = tool(profiler_output='{"results": []}', output_path=path)
        assert result["returncode"] == 0
        written = json.loads(Path(path).read_text())
        assert written["duration_us"] == 50
        Path(path).unlink()

    @patch("minisweagent.baseline_metrics.build_baseline_metrics")
    def test_kernel_names_parsing(self, mock_build):
        mock_build.return_value = {}
        tool = BaselineMetricsTool()
        tool(profiler_output="{}", kernel_names="rope_fwd, rope_bwd")
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["kernel_names"] == ["rope_fwd", "rope_bwd"]

    @patch("minisweagent.baseline_metrics.build_baseline_metrics")
    def test_kernel_indices_parsing(self, mock_build):
        mock_build.return_value = {}
        tool = BaselineMetricsTool()
        tool(profiler_output="{}", kernel_indices="0, 2, 5")
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["kernel_indices"] == [0, 2, 5]
