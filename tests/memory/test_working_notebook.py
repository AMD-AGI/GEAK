"""Tests for the within-session working notebook."""

from __future__ import annotations

from pathlib import Path

from minisweagent.memory.working_notebook import WorkingNotebook, parse_speedup_report, summarize_working_notebook


def test_parse_speedup_report_extracts_overall_and_per_shape() -> None:
    text = """
Speedup vs true baseline:
Overall: 1.2500x (0.080000 ms -> 0.064000 ms)
Per-shape:
  (64,1024,32): 1.1000x (0.040000 ms -> 0.036364 ms)
  (64,4096,32): 0.9800x (0.050000 ms -> 0.051020 ms)
"""

    parsed = parse_speedup_report(text)

    assert parsed["overall_speedup"] == 1.25
    assert parsed["baseline_ms"] == 0.08
    assert parsed["candidate_ms"] == 0.064
    assert parsed["per_shape"]["(64,1024,32)"]["speedup"] == 1.1
    assert parsed["per_shape"]["(64,4096,32)"]["candidate_ms"] == 0.05102


def test_working_notebook_summary_tracks_best_dead_ends_and_shapes(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "_working_memory"
    nb = WorkingNotebook(notebook_dir, writer_id="agent-a")
    nb.record_baseline(
        baseline_latency_ms=0.08,
        bottleneck_type="latency",
        kernel_category="topk",
    )
    nb.record_attempt(strategy="ALGO(algorithm rewrite)", change_category="algorithmic", step=1)
    nb.record_result(
        output="""
Speedup vs true baseline:
Overall: 1.2500x (0.080000 ms -> 0.064000 ms)
Per-shape:
  (64,1024,32): 1.1000x (0.040000 ms -> 0.036364 ms)
  (64,4096,32): 1.3000x (0.040000 ms -> 0.030769 ms)
""",
        returncode=0,
        strategy="ALGO(algorithm rewrite)",
        change_category="algorithmic",
        tag="OK",
        message="Algorithmic rewrite benchmarked",
        step=2,
    )
    nb.record_attempt(strategy="PATH(dispatch/backend)", change_category="wrapper", step=3)
    nb.record_result(
        output="""
Speedup vs true baseline:
Overall: 0.9800x (0.080000 ms -> 0.081633 ms)
Per-shape:
  (64,1024,32): 0.9500x (0.040000 ms -> 0.042105 ms)
""",
        returncode=0,
        strategy="PATH(dispatch/backend)",
        change_category="wrapper",
        tag="OK",
        message="Dispatch path tweak benchmarked",
        step=4,
    )

    summary = summarize_working_notebook(notebook_dir)

    assert "Baseline: kernel=topk | bottleneck=latency | baseline=0.0800ms" in summary
    assert "Best so far: 1.2500x via ALGO(algorithm rewrite) [algorithmic]" in summary
    assert "Dead ends:" in summary
    assert "PATH(dispatch/backend) [wrapper]: 0.9800x" in summary
    assert "(64,4096,32) best 1.3000x via ALGO(algorithm rewrite)" in summary
