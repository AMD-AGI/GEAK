"""Tests for workload-aware prompt guidance in pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from minisweagent.run.pipeline_helpers import (
    _bottleneck_guidance,
    create_validated_harness,
)


def test_bottleneck_guidance_adds_search_specific_hip_hints() -> None:
    metrics = {
        "kernel_name": "rocprim::detail::binary_search lower_bound",
        "bottleneck": "latency",
        "metrics": {
            "memory.hbm_bandwidth_utilization": 0.3,
            "memory.l2_hit_rate": 70.6,
        },
        "top_kernels": [
            {
                "name": "transform_kernel<binary_search<lower_bound>>",
                "bottleneck": "latency",
            }
        ],
    }

    text = "\n".join(_bottleneck_guidance("latency", metrics))

    assert "Optimization Guidance (bottleneck: latency-bound)" in text
    assert "Workload Guidance (HIP search / pointer-chasing)" in text
    assert "branchless search logic" in text
    assert "Deprioritize generic vectorization" in text


def _demo_harness_results() -> list[dict[str, object]]:
    return [
        {"mode": "correctness", "success": True, "returncode": 0, "duration_s": 0.1, "stdout": "ALL PASS\n"},
        {"mode": "profile", "success": True, "returncode": 0, "duration_s": 0.1, "stdout": "PROFILE OK\n"},
        {"mode": "benchmark", "success": True, "returncode": 0, "duration_s": 0.1, "stdout": "GEAK_RESULT_LATENCY_MS=1.0\n"},
        {"mode": "full-benchmark", "success": True, "returncode": 0, "duration_s": 0.1, "stdout": "GEAK_RESULT_LATENCY_MS=1.0\n"},
    ]


def test_create_validated_harness_materializes_harness_into_log_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "tasks" / "demo"
    task_dir.mkdir(parents=True)
    kernel_path = task_dir / "kernel.py"
    kernel_path.write_text("def kernel():\n    return 1\n")

    source_harness = task_dir / "test_demo_harness.py"
    source_harness.write_text(
        "\n".join(
            [
                "import argparse",
                "import os",
                "import sys",
                "",
                "# Ensure the kernel directory is importable",
                "_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))",
                "if _KERNEL_DIR not in sys.path:",
                "    sys.path.insert(0, _KERNEL_DIR)",
                "",
                "from kernel import kernel",
                "",
                "def main():",
                "    parser = argparse.ArgumentParser()",
                "    parser.add_argument('--correctness', action='store_true')",
                "    parser.add_argument('--profile', action='store_true')",
                "    parser.add_argument('--benchmark', action='store_true')",
                "    parser.add_argument('--full-benchmark', action='store_true')",
                "    parser.add_argument('--iterations', type=int, default=1)",
                "    parser.parse_args()",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n"
    )

    output_dir = tmp_path / "output"
    seen_harnesses: list[str] = []

    def _fake_execute(harness_path: str, **kwargs):
        seen_harnesses.append(harness_path)
        return True, [], _demo_harness_results()

    with (
        patch(
            "minisweagent.agents.unit_test_agent.run_unit_test_agent",
            return_value=f"python {source_harness} --correctness && python {source_harness} --benchmark",
        ),
        patch(
            "minisweagent.run.pipeline_helpers.execute_harness_validation",
            side_effect=_fake_execute,
        ),
    ):
        test_command, harness_results = create_validated_harness(
            model=object(),
            repo=repo_root,
            kernel_name="kernel",
            log_dir=output_dir,
            kernel_path=kernel_path,
            discovery_context="",
            gpu_id=0,
        )

    materialized_harness = output_dir / "test_kernel_harness.py"
    assert harness_results == _demo_harness_results()
    assert materialized_harness.is_file()
    assert str(materialized_harness) in test_command
    assert str(source_harness) not in test_command
    assert str(source_harness) in seen_harnesses[0]
    assert str(materialized_harness) in seen_harnesses[-1]

    text = materialized_harness.read_text()
    assert "GEAK materialized harness bootstrap" in text
    assert "GEAK_WORK_DIR" in text
    assert "GEAK_REPO_ROOT" in text
    assert "os.path.dirname(os.path.abspath(__file__))" not in text
