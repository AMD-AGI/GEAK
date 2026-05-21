"""Tests for ``run/compose.py`` — the single source of truth for task body composition."""

from __future__ import annotations

from unittest.mock import patch

from minisweagent.run.compose import ComposeInputs, compose_task_body


def test_compose_passes_prompt_through_when_memory_empty():
    with patch("minisweagent.memory.integration.assemble_memory_context", return_value=""):
        body = compose_task_body(
            ComposeInputs(
                user_prompt="Optimize kernel X",
                mode="fixed",
                preprocess_ctx={"kernel_path": "/tmp/x.py"},
            )
        )
    assert body == "Optimize kernel X"


def test_compose_appends_memory_when_retriever_returns_context():
    fake_memory = "### RECORD 1\nstrategy: tile=128"
    with patch("minisweagent.memory.integration.assemble_memory_context", return_value=fake_memory):
        body = compose_task_body(
            ComposeInputs(
                user_prompt="Optimize kernel X",
                mode="fixed",
                preprocess_ctx={"kernel_path": "/tmp/x.py"},
            )
        )
    assert "Optimize kernel X" in body
    assert "Optimization Memory" in body
    assert fake_memory in body


def test_compose_swallows_memory_errors_and_keeps_prompt():
    def _boom(**_: object) -> str:
        raise RuntimeError("KB unavailable")

    with patch("minisweagent.memory.integration.assemble_memory_context", side_effect=_boom):
        body = compose_task_body(
            ComposeInputs(
                user_prompt="Optimize kernel X",
                mode="fixed",
                preprocess_ctx={"kernel_path": "/tmp/x.py"},
            )
        )
    assert body == "Optimize kernel X"


def test_compose_appends_extra_addenda():
    with patch("minisweagent.memory.integration.assemble_memory_context", return_value=""):
        body = compose_task_body(
            ComposeInputs(
                user_prompt="Optimize kernel X",
                mode="planned",
                preprocess_ctx={"kernel_path": "/tmp/x.py"},
                extra_addenda=["  ## CONSTRAINTS\n- no fp64  ", "", "  ## DIRECTIVES\n- try ILP  "],
            )
        )
    assert body.startswith("Optimize kernel X")
    assert "## CONSTRAINTS" in body
    assert "## DIRECTIVES" in body
    # Empty strings are skipped
    assert body.count("\n\n") >= 2


def test_compose_parses_stringified_baseline_json():
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return ""

    with patch("minisweagent.memory.integration.assemble_memory_context", side_effect=_capture):
        compose_task_body(
            ComposeInputs(
                user_prompt="x",
                mode="fixed",
                preprocess_ctx={
                    "kernel_path": "/tmp/k.py",
                    "baseline_metrics": '{"bottleneck": "memory-bound", "peak_bw": 400}',
                },
            )
        )
    assert captured.get("bottleneck_type") == "memory-bound"
    assert isinstance(captured.get("profiling_metrics"), dict)
    assert captured["profiling_metrics"].get("peak_bw") == 400


def test_compose_tolerates_unparsable_baseline_string():
    with patch("minisweagent.memory.integration.assemble_memory_context", return_value=""):
        body = compose_task_body(
            ComposeInputs(
                user_prompt="x",
                mode="fixed",
                preprocess_ctx={
                    "kernel_path": "/tmp/k.py",
                    "baseline_metrics": "this is not json",
                },
            )
        )
    assert body == "x"
