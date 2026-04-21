from __future__ import annotations

from minisweagent.memory import integration


def test_assemble_memory_context_derives_bottleneck_from_profiling_metrics(monkeypatch) -> None:
    captured: dict = {}

    def _fake_retrieve(**kwargs):
        captured.update(kwargs)
        return "memory-context"

    monkeypatch.setattr(integration, "is_retrieve_enabled", lambda: True)

    import minisweagent.memory.cross_session as cross_session

    monkeypatch.setattr(cross_session, "retrieve", _fake_retrieve)

    profiling_metrics = {
        "selected_kernel_summary": {
            "primary_bottleneck": "compute",
            "mode": "single",
            "bottleneck_mix": [],
        },
        "bottleneck": "memory",
    }

    result = integration.assemble_memory_context(
        kernel_path="/tmp/kernel.py",
        bottleneck_type="memory",
        profiling_metrics=profiling_metrics,
    )

    assert result == "memory-context"
    assert captured["bottleneck_type"] == "compute"


def test_record_optimization_outcome_derives_bottleneck_from_profiling_metrics(monkeypatch) -> None:
    captured: dict = {}

    def _fake_record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(integration, "is_record_enabled", lambda: True)

    import minisweagent.memory.cross_session as cross_session

    monkeypatch.setattr(cross_session, "record", _fake_record)

    profiling_metrics = {
        "selected_kernel_summary": {
            "primary_bottleneck": "latency",
            "mode": "single",
            "bottleneck_mix": [],
        },
        "bottleneck": "memory",
    }

    integration.record_optimization_outcome(
        kernel_path="/tmp/kernel.py",
        strategy_name="test strategy",
        speedup_achieved=1.2,
        bottleneck_type="memory",
        profiling_metrics=profiling_metrics,
    )

    assert captured["bottleneck_type"] == "latency"
