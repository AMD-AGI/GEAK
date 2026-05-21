"""Unit tests for ``minisweagent.run.orchestrator``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.orchestrator import _probe_preprocess_dir


class TestProbePreprocessDir:
    def test_empty_dir_uses_preprocess_as_repo_root(self, tmp_path: Path) -> None:
        pc = _probe_preprocess_dir(tmp_path)
        assert pc.kernel_path == ""
        assert pc.repo_root == str(tmp_path)
        assert pc.harness_path == ""
        assert pc.preprocess_dir == str(tmp_path)
        assert pc.discovery is None

    def test_resolved_json_sets_kernel_and_finds_git_root(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        sub = repo / "kernels"
        sub.mkdir(parents=True)
        k = sub / "k.hip"
        k.write_text("kernel")
        (repo / ".git").mkdir()

        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(k.resolve()),
                    "local_repo_path": None,
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert Path(pc.kernel_path).resolve() == k.resolve()
        assert Path(pc.repo_root).resolve() == repo.resolve()

    def test_resolved_json_no_git_uses_local_repo_path(self, tmp_path: Path) -> None:
        k = tmp_path / "solo.cu"
        k.write_text("x")
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(k.resolve()),
                    "local_repo_path": "/workspace/myproject",
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.repo_root == "/workspace/myproject"

    def test_testcase_selection_sets_harness_path(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "testcase_selection.json").write_text(
            json.dumps({"harness_path": "/h/run.py"})
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.harness_path == "/h/run.py"

    def test_discovery_json_loaded(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "discovery.json").write_text(json.dumps({"kernel": {"type": "triton"}}))

        pc = _probe_preprocess_dir(pp)
        assert pc.discovery == {"kernel": {"type": "triton"}}

    def test_optional_artifact_paths(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "COMMANDMENT.md").write_text("cmd")
        (pp / "CODEBASE_CONTEXT.md").write_text("ctx")
        (pp / "baseline_metrics.json").write_text("{}")
        (pp / "profile.json").write_text("{}")

        pc = _probe_preprocess_dir(pp)
        assert pc.commandment_path == str(pp / "COMMANDMENT.md")
        assert pc.codebase_context_path == str(pp / "CODEBASE_CONTEXT.md")
        assert pc.baseline_metrics_path == str(pp / "baseline_metrics.json")
        assert pc.profiling_result_path == str(pp / "profile.json")
