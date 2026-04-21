from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

from minisweagent.agents.default import DefaultAgent
from minisweagent.tools.save_and_test import SaveAndTestContext, SaveAndTestTool


def _install_fake_patch_profiler(monkeypatch, calls: list[dict]) -> None:
    import minisweagent.run.pipeline_helpers as pipeline_helpers
    import minisweagent.run.preprocess.baseline as baseline

    monkeypatch.setattr(pipeline_helpers, "_ensure_mcp_importable", lambda: None)
    monkeypatch.setattr(baseline, "build_baseline_metrics", lambda *_args, **_kwargs: {"top_kernels": []})

    server = types.ModuleType("profiler_mcp.server")

    def _fake_profile_kernel(**kwargs):
        calls.append(kwargs)
        return {"results": []}

    server.profile_kernel = types.SimpleNamespace(fn=_fake_profile_kernel)
    package = types.ModuleType("profiler_mcp")
    package.server = server
    monkeypatch.setitem(sys.modules, "profiler_mcp", package)
    monkeypatch.setitem(sys.modules, "profiler_mcp.server", server)


def test_default_agent_threads_patch_profile_flags_into_save_and_test_context(tmp_path: Path) -> None:
    save_tool = SaveAndTestTool()
    agent = DefaultAgent.__new__(DefaultAgent)
    agent.env = SimpleNamespace(config=SimpleNamespace(cwd=str(tmp_path), timeout=123, env={}))
    agent.config = SimpleNamespace(
        test_command="python test_harness.py --correctness",
        patch_output_dir=str(tmp_path),
        profile_every_patch=True,
        patch_profile_quick=False,
        source_file_paths=None,
        source_file_path=None,
    )
    agent.base_repo_path = tmp_path
    agent._log_message = lambda _message: None
    agent.patch_counter = 7
    agent.toolruntime = SimpleNamespace(_tool_table={"save_and_test": save_tool})

    DefaultAgent._setup_save_and_test_context(agent)

    assert save_tool.context is not None
    assert save_tool.context.profile_every_patch is True
    assert save_tool.context.patch_profile_quick is False
    assert save_tool.context.patch_counter == 7


def test_save_and_test_env_overrides_context_flags(monkeypatch, tmp_path: Path) -> None:
    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="python test_harness.py --correctness",
            timeout=30,
            patch_output_dir=str(tmp_path),
            profile_every_patch=False,
            patch_profile_quick=False,
        )
    )

    monkeypatch.setenv("GEAK_PROFILE_EVERY_PATCH", "1")
    monkeypatch.setenv("GEAK_PATCH_PROFILE_QUICK", "1")

    assert tool._patch_profiling_enabled() is True
    assert tool._patch_profile_quick() is True


def test_run_patch_profile_uses_resolved_quick_flag(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_patch_profiler(monkeypatch, calls)

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="python test_harness.py --correctness",
            timeout=30,
            patch_output_dir=str(tmp_path),
            env_vars={},
            profile_every_patch=True,
            patch_profile_quick=False,
        )
    )

    monkeypatch.setenv("GEAK_PATCH_PROFILE_QUICK", "1")

    raw_result, metrics = tool._run_patch_profile(
        harness_path=tmp_path / "test_harness.py",
        profile_env={},
        gpu_devices="2",
    )

    assert raw_result == {"results": []}
    assert metrics == {"top_kernels": []}
    assert len(calls) == 1
    assert calls[0]["command"] == f"python {tmp_path / 'test_harness.py'} --profile"
    assert calls[0]["backend"] == "metrix"
    assert calls[0]["quick"] is True
    assert calls[0]["gpu_devices"] == "2"


def test_patch_profile_summary_uses_selected_kernel_summary() -> None:
    lines = SaveAndTestTool._format_patch_profile_summary(
        {
            "enabled": True,
            "status": "ok",
            "metrics": {
                "selected_kernel_summary": {
                    "mode": "mixed",
                    "primary_bottleneck": "memory",
                    "primary_bottleneck_pct_of_selected": 60.0,
                    "bottleneck_mix": [
                        {"bottleneck": "memory", "pct_of_selected": 60.0},
                        {"bottleneck": "compute", "pct_of_selected": 40.0},
                    ],
                },
                "top_kernels": [],
            },
        }
    )

    text = "\n".join(lines)
    assert "Mixed bottlenecks: memory 60.0%, compute 40.0% (primary: memory)" in text
