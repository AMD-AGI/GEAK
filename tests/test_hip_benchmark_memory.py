from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.memory.working_memory import WorkingMemory
from minisweagent.memory.working_notebook import parse_speedup_report
from minisweagent.run.postprocess.benchmark_parsing import (
    extract_benchmark_config_lines,
    extract_latency_ms,
    parse_shape_count,
    parse_shape_latencies_ms,
)

HIP_BASELINE_OUTPUT = "\n".join(
    [
        "Perf: 0.0500 ms (shape_0_forward)",
        "Perf: 0.2000 ms (shape_0_forward_backward)",
    ]
)

HIP_CANDIDATE_OUTPUT = "\n".join(
    [
        "Perf: 0.0250 ms (shape_0_forward)",
        "Perf: 0.1000 ms (shape_0_forward_backward)",
    ]
)


def test_extract_latency_ms_uses_geomean_for_raw_hip_perf_output() -> None:
    assert extract_latency_ms(HIP_BASELINE_OUTPUT) == pytest.approx(0.1)
    assert extract_latency_ms(HIP_CANDIDATE_OUTPUT) == pytest.approx(0.05)


def test_parse_shape_latencies_ms_extracts_hip_shape_labels() -> None:
    assert parse_shape_latencies_ms(HIP_BASELINE_OUTPUT) == {
        "shape_0_forward": 0.05,
        "shape_0_forward_backward": 0.2,
    }
    assert parse_shape_count(HIP_BASELINE_OUTPUT) == 2
    assert extract_benchmark_config_lines(HIP_BASELINE_OUTPUT) == [
        "shape_0_forward",
        "shape_0_forward_backward",
    ]


def test_parse_speedup_report_computes_overall_and_per_shape_for_hip_perf_output() -> None:
    parsed = parse_speedup_report(
        HIP_CANDIDATE_OUTPUT,
        baseline_ms=0.1,
        baseline_shape_latencies_ms=parse_shape_latencies_ms(HIP_BASELINE_OUTPUT),
    )

    assert parsed["baseline_ms"] == pytest.approx(0.1)
    assert parsed["candidate_ms"] == pytest.approx(0.05)
    assert parsed["overall_speedup"] == pytest.approx(2.0)
    assert parsed["per_shape"] == {
        "shape_0_forward": {
            "baseline_ms": 0.05,
            "candidate_ms": 0.025,
            "speedup": 2.0,
        },
        "shape_0_forward_backward": {
            "baseline_ms": 0.2,
            "candidate_ms": 0.1,
            "speedup": 2.0,
        },
    }


def test_working_memory_uses_shared_hip_baseline_and_speedup_parsing(tmp_path: Path) -> None:
    baseline_path = tmp_path / "benchmark_baseline.txt"
    baseline_path.write_text(HIP_BASELINE_OUTPUT)

    wm = WorkingMemory()
    wm.load_baseline_from_artifacts(benchmark_baseline_path=str(baseline_path))

    assert wm.baseline_latency_ms == pytest.approx(0.1)
    assert wm.baseline_shape_latencies_ms == {
        "shape_0_forward": 0.05,
        "shape_0_forward_backward": 0.2,
    }

    wm.note_tool_result(HIP_CANDIDATE_OUTPUT, 0)

    assert wm.best_latency_ms == pytest.approx(0.05)
    assert wm.best_speedup == pytest.approx(2.0)
