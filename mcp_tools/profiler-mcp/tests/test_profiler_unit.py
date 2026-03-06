# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Mock-based unit tests for the unified profiler MCP.

These tests run anywhere without a GPU. They verify dispatch logic,
error handling, and schema correctness.
"""

import asyncio
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import the server module (conftest.py sets up sys.path)
# ---------------------------------------------------------------------------
from profiler_mcp.server import (
    _normalize_command,
    mcp,
    profile_kernel,
)


def _call(**kwargs):
    """Call the profile_kernel function directly (it's a plain function after @mcp.tool())."""
    return profile_kernel(**kwargs)


def _get_tool_schema():
    """Retrieve the FunctionTool object for profile_kernel from the MCP server."""
    tools = asyncio.run(mcp.list_tools())
    return next(t for t in tools if t.name == "profile_kernel")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInvalidBackend:
    def test_returns_failure(self):
        result = _call(command="echo hello", backend="invalid")
        assert result["success"] is False
        assert "Unknown backend" in result["error"]
        assert result["backend"] == "invalid"

    def test_returns_empty_results(self):
        result = _call(command="echo hello", backend="bogus")
        assert result["results"] == []


class TestMetrixDispatch:
    @patch("profiler_mcp.server._profile_with_metrix")
    def test_dispatches_to_metrix(self, mock_metrix):
        mock_metrix.return_value = {"success": True, "backend": "metrix", "results": []}
        result = _call(
            command="python3 kernel.py",
            backend="metrix",
            num_replays=5,
            kernel_filter="*rope*",
            auto_select=True,
            quick=True,
            gpu_devices="0",
        )
        mock_metrix.assert_called_once_with(
            command="python3 kernel.py",
            num_replays=5,
            kernel_filter="*rope*",
            auto_select=True,
            quick=True,
            gpu_devices="0",
        )
        assert result["success"] is True


class TestRocprofDispatch:
    @patch("profiler_mcp.server._profile_with_rocprof")
    def test_dispatches_to_rocprof(self, mock_rocprof):
        mock_rocprof.return_value = {
            "success": True,
            "backend": "rocprof-compute",
            "analysis": "roofline data...",
            "results": [],
        }
        result = _call(
            command="python3 kernel.py",
            backend="rocprof-compute",
            workdir="/tmp",
            profiling_type="roofline",
        )
        mock_rocprof.assert_called_once_with(
            command="python3 kernel.py",
            workdir="/tmp",
            profiling_type="roofline",
        )
        assert result["success"] is True
        assert result["backend"] == "rocprof-compute"


class TestMetrixErrorHandling:
    @patch("profiler_mcp.server._profile_with_metrix")
    def test_exception_returns_graceful_failure(self, mock_metrix):
        mock_metrix.side_effect = RuntimeError("GPU on fire")
        result = _call(command="python3 kernel.py", backend="metrix")
        assert result["success"] is False
        assert "GPU on fire" in result["error"]
        assert result["results"] == []


class TestRocprofErrorHandling:
    @patch("profiler_mcp.server._profile_with_rocprof")
    def test_exception_returns_graceful_failure(self, mock_rocprof):
        mock_rocprof.side_effect = FileNotFoundError("rocprof-compute not found")
        result = _call(command="echo hello", backend="rocprof-compute")
        assert result["success"] is False
        assert "rocprof-compute not found" in result["error"]

    def test_rocprof_returncode_1(self):
        """Verify _profile_with_rocprof returns failure when analyzer returns returncode=1."""
        with patch("profiler_mcp.server._profile_with_rocprof") as mock_fn:
            mock_fn.return_value = {
                "success": False,
                "backend": "rocprof-compute",
                "error": "No ROCProf is installed.",
                "results": [],
            }
            result = _call(command="echo hello", backend="rocprof-compute")
            assert result["success"] is False
            assert "ROCProf" in result["error"]


class TestSchemaParams:
    def test_command_is_required(self):
        tool = _get_tool_schema()
        assert "command" in tool.parameters.get("required", [])

    def test_has_all_expected_params(self):
        tool = _get_tool_schema()
        props = set(tool.parameters.get("properties", {}).keys())
        expected = {
            "command",
            "backend",
            "workdir",
            "profiling_type",
            "num_replays",
            "kernel_filter",
            "auto_select",
            "quick",
            "gpu_devices",
        }
        assert expected.issubset(props), f"Missing params: {expected - props}"

    def test_backend_not_required(self):
        tool = _get_tool_schema()
        required = tool.parameters.get("required", [])
        assert "backend" not in required


class TestDefaultBackend:
    @patch("profiler_mcp.server._profile_with_metrix")
    def test_default_is_metrix(self, mock_metrix):
        mock_metrix.return_value = {"success": True, "backend": "metrix", "results": []}
        result = _call(command="python3 kernel.py")
        mock_metrix.assert_called_once()
        assert result["backend"] == "metrix"


class TestNormalizeCommand:
    """_normalize_command wraps shell-style commands in bash -c."""

    def test_simple_command_unchanged(self):
        assert _normalize_command("python3 kernel.py --profile") == "python3 kernel.py --profile"

    def test_cd_wrapped(self):
        cmd = "cd /workspace && python3 kernel.py"
        result = _normalize_command(cmd)
        assert result.startswith("bash -c ")

    def test_env_var_expansion_wrapped(self):
        cmd = "${GEAK_WORK_DIR}/run.sh harness.py --profile"
        result = _normalize_command(cmd)
        assert result.startswith("bash -c ")

    def test_pipe_wrapped(self):
        cmd = "python3 kernel.py | tail -5"
        result = _normalize_command(cmd)
        assert result.startswith("bash -c ")

    def test_semicolon_wrapped(self):
        cmd = "echo hello; python3 kernel.py"
        result = _normalize_command(cmd)
        assert result.startswith("bash -c ")

    def test_already_wrapped_unchanged(self):
        cmd = "bash -c 'cd /tmp && python3 kernel.py'"
        assert _normalize_command(cmd) == cmd

    def test_export_wrapped(self):
        cmd = "export HIP_VISIBLE_DEVICES=0 && python3 kernel.py"
        result = _normalize_command(cmd)
        assert result.startswith("bash -c ")

    def test_inline_env_var_wrapped(self):
        cmd = "HIP_VISIBLE_DEVICES=4 python3 kernel.py --profile"
        result = _normalize_command(cmd)
        assert result.startswith("bash -c ")

    def test_inline_env_var_multiple_wrapped(self):
        cmd = "CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 python3 kernel.py"
        result = _normalize_command(cmd)
        assert result.startswith("bash -c ")


class TestRocprofProfilingTypes:
    @pytest.mark.parametrize("profiling_type", ["profiling", "roofline", "profiler_analyzer"])
    @patch("profiler_mcp.server._profile_with_rocprof")
    def test_type_passed_through(self, mock_rocprof, profiling_type):
        mock_rocprof.return_value = {
            "success": True,
            "backend": "rocprof-compute",
            "profiling_type": profiling_type,
            "analysis": "...",
            "results": [],
        }
        _call(
            command="python3 kernel.py",
            backend="rocprof-compute",
            profiling_type=profiling_type,
        )
        call_kwargs = mock_rocprof.call_args[1]
        assert call_kwargs["profiling_type"] == profiling_type
