"""
Minimal unit tests for MetrixTool.
"""

from unittest.mock import MagicMock, Mock

import pytest

from mini_kernel.mcp_tools.metrix import MetrixTool


class TestMetrixTool:
    """Test MetrixTool basic functionality."""

    def test_init(self):
        """Test MetrixTool initialization."""
        # Test single GPU
        tool = MetrixTool(gpu_devices="3")
        assert tool.gpu_devices == ["3"]
        assert tool.profiler is not None
        assert "3" in tool.gpu_info_map

        # Test multiple GPUs
        tool_multi = MetrixTool(gpu_devices=["0", "1"])
        assert tool_multi.gpu_devices == ["0", "1"]
        assert "0" in tool_multi.gpu_info_map
        assert "1" in tool_multi.gpu_info_map

    def test_find_main_kernel(self):
        """Test _find_main_kernel filters framework kernels."""
        tool = MetrixTool()

        # Mock kernels with Statistics objects
        framework_kernel = Mock()
        framework_kernel.name = "at::native::vectorized_elementwise"
        framework_kernel.duration_us = Mock(avg=100)

        user_kernel = Mock()
        user_kernel.name = "topk_stage1"
        user_kernel.duration_us = Mock(avg=450)

        kernels = [framework_kernel, user_kernel]

        main = tool._find_main_kernel(kernels)
        assert main == user_kernel
        assert main.name == "topk_stage1"

    def test_classify_bottleneck(self):
        """Test bottleneck classification logic."""
        tool = MetrixTool()

        # Latency-bound
        metrics = {"duration_us": 5.0}
        assert tool._classify_bottleneck(metrics) == "latency"

        # Memory-bound
        metrics = {"duration_us": 500.0, "memory.hbm_bandwidth_utilization": 60.0}
        assert tool._classify_bottleneck(metrics) == "memory"

        # Compute-bound
        metrics = {
            "duration_us": 500.0,
            "memory.hbm_bandwidth_utilization": 2.0,
            "memory.l2_hit_rate": 85.0,
        }
        assert tool._classify_bottleneck(metrics) == "compute"

        # Balanced
        metrics = {"duration_us": 500.0}
        assert tool._classify_bottleneck(metrics) == "balanced"

    def test_classify_bottleneck_lds(self):
        """Test LDS bottleneck classification."""
        tool = MetrixTool()

        # LDS-bound (high bank conflicts in full profile mode)
        metrics = {
            "duration_us": 500.0,
            "memory.hbm_bandwidth_utilization": 20.0,
            "memory.lds_bank_conflicts": 0.5,  # > 0.1 threshold
            "memory.l2_hit_rate": 70.0,
        }
        assert tool._classify_bottleneck(metrics, quick=False) == "lds"

    def test_classify_bottleneck_full_profile(self):
        """Test classification with full profile metrics."""
        tool = MetrixTool()

        # Memory-bound with poor coalescing
        metrics = {
            "duration_us": 500.0,
            "memory.hbm_bandwidth_utilization": 40.0,
            "memory.coalescing_efficiency": 30.0,  # Poor coalescing
            "memory.l2_hit_rate": 60.0,
        }
        assert tool._classify_bottleneck(metrics, quick=False) == "memory"

        # Memory-bound with poor load efficiency
        metrics = {
            "duration_us": 500.0,
            "memory.hbm_bandwidth_utilization": 35.0,
            "memory.coalescing_efficiency": 80.0,
            "memory.global_load_efficiency": 40.0,  # Poor load efficiency
            "memory.l2_hit_rate": 60.0,
        }
        assert tool._classify_bottleneck(metrics, quick=False) == "memory"

        # Compute-bound with high cache hits
        metrics = {
            "duration_us": 500.0,
            "memory.hbm_bandwidth_utilization": 3.0,
            "memory.l1_hit_rate": 85.0,
            "memory.l2_hit_rate": 88.0,
        }
        assert tool._classify_bottleneck(metrics, quick=False) == "compute"

    def test_extract_all_metrics(self):
        """Test metric extraction from kernel object, including duration."""
        tool = MetrixTool()

        # Mock kernel with duration and metrics (all Statistics objects)
        kernel = Mock()
        kernel.duration_us = Mock(avg=123.45)
        kernel.metrics = {
            "HBM_BW": Mock(avg=67.3),
            "L2_Hit": Mock(avg=45.2),
        }

        metrics = tool._extract_all_metrics(kernel)
        assert metrics["duration_us"] == 123.45
        assert metrics["HBM_BW"] == 67.3
        assert metrics["L2_Hit"] == 45.2

    def test_validate_metrics(self):
        """Test metrics validation."""
        tool = MetrixTool()

        # Valid quick metrics
        quick_metrics = {
            "duration_us": 100.0,
            "memory.hbm_bandwidth_utilization": 50.0,
            "memory.l2_hit_rate": 75.0,
        }
        tool._validate_metrics(
            quick_metrics, "test_kernel", quick=True
        )  # Should not raise

        # Valid full metrics
        full_metrics = {
            "duration_us": 100.0,
            "memory.hbm_bandwidth_utilization": 50.0,
            "memory.hbm_read_bandwidth": 100.0,
            "memory.hbm_write_bandwidth": 50.0,
            "memory.bytes_transferred_hbm": 1000000,
            "memory.l1_hit_rate": 80.0,
            "memory.l2_hit_rate": 75.0,
            "memory.l2_bandwidth": 30.0,
            "memory.coalescing_efficiency": 85.0,
            "memory.global_load_efficiency": 90.0,
            "memory.global_store_efficiency": 88.0,
            "memory.lds_bank_conflicts": 0.05,
        }
        tool._validate_metrics(
            full_metrics, "test_kernel", quick=False
        )  # Should not raise

        # Missing metrics (quick mode)
        incomplete_metrics = {"duration_us": 100.0}
        try:
            tool._validate_metrics(incomplete_metrics, "test_kernel", quick=True)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Missing expected metrics" in str(e)
            assert "test_kernel" in str(e)

    def test_generate_observations_memory(self):
        """Test observation generation for memory-bound kernels."""
        tool = MetrixTool()
        metrics = {"memory.hbm_bandwidth_utilization": 67.3}

        observations = tool._generate_observations("memory", metrics)
        assert len(observations) > 0
        assert any("memory-bound" in obs.lower() for obs in observations)

    def test_generate_observations_compute(self):
        """Test observation generation for compute-bound kernels."""
        tool = MetrixTool()
        metrics = {"memory.l2_hit_rate": 92.0}

        observations = tool._generate_observations("compute", metrics)
        assert len(observations) > 0
        assert any("compute-bound" in obs.lower() for obs in observations)

    def test_generate_observations_latency(self):
        """Test observation generation for latency-bound kernels."""
        tool = MetrixTool()
        metrics = {"memory.hbm_bandwidth_utilization": 0.2}

        observations = tool._generate_observations("latency", metrics)
        assert len(observations) > 0
        assert any("latency-bound" in obs.lower() for obs in observations)

    def test_generate_observations_balanced(self):
        """Test observation generation for balanced kernels."""
        tool = MetrixTool()
        metrics = {
            "memory.hbm_bandwidth_utilization": 15.0,
            "memory.l2_hit_rate": 65.0,
        }

        observations = tool._generate_observations("balanced", metrics)
        assert len(observations) > 0
        assert any("balanced" in obs.lower() for obs in observations)

    def test_generate_observations_lds(self):
        """Test observation generation for LDS-bound kernels."""
        tool = MetrixTool()
        metrics = {"memory.lds_bank_conflicts": 0.5}

        observations = tool._generate_observations("lds", metrics, quick=False)
        assert len(observations) > 0
        assert any("lds" in obs.lower() for obs in observations)

    def test_observations_with_gpu_info(self):
        """Test observations with GPU context."""
        tool = MetrixTool()
        metrics = {
            "memory.hbm_bandwidth_utilization": 0.2,
            "memory.hbm_read_bandwidth": 10.0,
            "memory.hbm_write_bandwidth": 5.0,
        }
        gpu_info = {
            "detected": True,
            "model": "AMD Instinct MI300X",
            "architecture": "gfx942",
            "peak_hbm_bandwidth_gbs": 5300.0,
            "lds_size_per_cu_kb": 64.0,
        }

        observations = tool._generate_observations(
            "latency", metrics, quick=False, gpu_info=gpu_info
        )
        assert len(observations) > 0
        # Should mention achieved bandwidth vs peak
        assert any("5300" in obs for obs in observations)  # Peak BW mentioned

    def test_get_tool_definition(self):
        """Test MCP tool definition format."""
        tool = MetrixTool()
        definition = tool.get_tool_definition()

        assert definition["name"] == "metrix"
        assert "description" in definition
        assert "inputSchema" in definition
        assert "command" in definition["inputSchema"]["properties"]
        assert "auto_select" in definition["inputSchema"]["properties"]
        assert "command" in definition["inputSchema"]["required"]

    def test_profile_return_format(self):
        """Test profile always returns kernels list."""
        tool = MetrixTool()

        # Mock the profiler with full expected metrics
        mock_kernel = Mock()
        mock_kernel.name = "test_kernel"
        mock_kernel.duration_us = Mock(avg=100.0)
        mock_kernel.metrics = {
            "memory.hbm_bandwidth_utilization": Mock(avg=50.0),
            "memory.hbm_read_bandwidth": Mock(avg=100.0),
            "memory.hbm_write_bandwidth": Mock(avg=50.0),
            "memory.bytes_transferred_hbm": Mock(avg=1000000),
            "memory.l1_hit_rate": Mock(avg=80.0),
            "memory.l2_hit_rate": Mock(avg=75.0),
            "memory.l2_bandwidth": Mock(avg=30.0),
            "memory.coalescing_efficiency": Mock(avg=85.0),
            "memory.global_load_efficiency": Mock(avg=90.0),
            "memory.global_store_efficiency": Mock(avg=88.0),
            "memory.lds_bank_conflicts": Mock(avg=0.05),
        }

        tool.profiler.profile = Mock(return_value=Mock(kernels=[mock_kernel]))

        # Test auto_select=True (default)
        result = tool.profile(command="test", auto_select=True)
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 1  # Single GPU
        gpu_result = result["results"][0]
        assert "device_id" in gpu_result
        assert "gpu_info" in gpu_result
        assert "kernels" in gpu_result
        assert isinstance(gpu_result["kernels"], list)
        assert len(gpu_result["kernels"]) == 1
        assert "bottleneck" in gpu_result["kernels"][0]
        assert "observations" in gpu_result["kernels"][0]

        # Test auto_select=False
        result = tool.profile(command="test", auto_select=False)
        assert "results" in result
        gpu_result = result["results"][0]
        assert "gpu_info" in gpu_result
        assert "kernels" in gpu_result
        assert isinstance(gpu_result["kernels"], list)
        assert "name" in gpu_result["kernels"][0]
        assert "duration_us" in gpu_result["kernels"][0]
        assert "observations" in gpu_result["kernels"][0]

    def test_profile_multiple_gpus(self):
        """Test profiling with multiple GPUs."""
        tool = MetrixTool(gpu_devices=["0", "1"])

        # Mock kernel for profiling
        mock_kernel = Mock()
        mock_kernel.name = "test_kernel"
        mock_kernel.duration_us = Mock(avg=100.0)
        mock_kernel.metrics = {
            "memory.hbm_bandwidth_utilization": Mock(avg=50.0),
            "memory.hbm_read_bandwidth": Mock(avg=100.0),
            "memory.hbm_write_bandwidth": Mock(avg=50.0),
            "memory.bytes_transferred_hbm": Mock(avg=1000000),
            "memory.l1_hit_rate": Mock(avg=80.0),
            "memory.l2_hit_rate": Mock(avg=75.0),
            "memory.l2_bandwidth": Mock(avg=30.0),
            "memory.coalescing_efficiency": Mock(avg=85.0),
            "memory.global_load_efficiency": Mock(avg=90.0),
            "memory.global_store_efficiency": Mock(avg=88.0),
            "memory.lds_bank_conflicts": Mock(avg=0.05),
        }

        tool.profiler.profile = Mock(return_value=Mock(kernels=[mock_kernel]))

        result = tool.profile(command="test", auto_select=False)

        # Should have results for both GPUs
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 2

        # Check each GPU result
        for i, gpu_result in enumerate(result["results"]):
            assert "device_id" in gpu_result
            assert gpu_result["device_id"] in ["0", "1"]
            assert "gpu_info" in gpu_result
            assert "kernels" in gpu_result
            assert isinstance(gpu_result["kernels"], list)

    def test_execute_method(self):
        """Test MCP execute method."""
        tool = MetrixTool()

        # Mock profiler
        mock_kernel = Mock()
        mock_kernel.name = "test_kernel"
        mock_kernel.duration_us = Mock(avg=100.0)
        mock_kernel.metrics = {
            "memory.hbm_bandwidth_utilization": Mock(avg=50.0),
            "memory.hbm_read_bandwidth": Mock(avg=100.0),
            "memory.hbm_write_bandwidth": Mock(avg=50.0),
            "memory.bytes_transferred_hbm": Mock(avg=1000000),
            "memory.l1_hit_rate": Mock(avg=80.0),
            "memory.l2_hit_rate": Mock(avg=75.0),
            "memory.l2_bandwidth": Mock(avg=30.0),
            "memory.coalescing_efficiency": Mock(avg=85.0),
            "memory.global_load_efficiency": Mock(avg=90.0),
            "memory.global_store_efficiency": Mock(avg=88.0),
            "memory.lds_bank_conflicts": Mock(avg=0.05),
        }
        tool.profiler.profile = Mock(return_value=Mock(kernels=[mock_kernel]))

        # Test execute with various arguments
        result = tool.execute(
            {
                "command": "test_command",
                "num_replays": 5,
                "kernel_filter": "*topk*",
                "auto_select": True,
                "quick": True,
            }
        )

        assert "results" in result
        assert isinstance(result["results"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
