"""Unit tests for ``minisweagent.run.utils.task_parser``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent.run.utils import task_parser as tp


class TestResolvePathCase:
    def test_relative_path_returns_none(self, tmp_path: Path) -> None:
        assert tp._resolve_path_case(Path("relative/path")) is None

    def test_resolves_wrong_case_component(self, tmp_path: Path) -> None:
        sub = tmp_path / "MyRepo"
        sub.mkdir()
        (sub / "a.txt").write_text("x")
        wrong = tmp_path / "myrepo" / "a.txt"
        assert not wrong.exists()
        resolved = tp._resolve_path_case(wrong)
        assert resolved is not None
        assert resolved == (tmp_path / "MyRepo" / "a.txt").resolve()

    def test_missing_component_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "does_not_exist" / "file.txt"
        assert tp._resolve_path_case(p.resolve()) is None


class TestNormalizePath:
    def test_empty_returns_none(self) -> None:
        assert tp._normalize_path("") is None

    def test_existing_path_resolves(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("x")
        assert tp._normalize_path(str(f)) == str(f.resolve())

    def test_unknown_path_returns_original_string(self) -> None:
        assert tp._normalize_path("no/such/path/anywhere/xyz") == "no/such/path/anywhere/xyz"


class TestParseTaskInfo:
    class _Model:
        def __init__(self, content: str) -> None:
            self._content = content

        def query(self, messages: list) -> dict:
            return {"content": self._content}

    def test_parses_json_object(self) -> None:
        payload = {
            "kernel_name": "gemm",
            "kernel_url": "https://example.com/k.py",
            "kernel_type": "triton",
            "repo": None,
            "test_command": "pytest",
            "metric": "latency",
            "num_parallel": 2,
            "gpu_ids": "0,1",
            "output_dir": None,
            "model": "m",
            "config": None,
        }
        out = tp.parse_task_info("task", self._Model(json.dumps(payload)))
        assert out["kernel_name"] == "gemm"
        assert out["kernel_type"] == "triton"
        assert out["num_parallel"] == 2

    def test_strips_json_from_markdown_fence(self) -> None:
        inner = json.dumps(
            {
                "kernel_name": "k",
                "kernel_url": None,
                "kernel_type": "hip",
                "repo": None,
                "test_command": None,
                "metric": None,
                "num_parallel": None,
                "gpu_ids": None,
                "output_dir": None,
                "model": None,
                "config": None,
            }
        )
        content = f"Here:\n```json\n{inner}\n```"
        out = tp.parse_task_info("x", self._Model(content))
        assert out["kernel_name"] == "k"
        assert out["kernel_type"] == "hip"

    def test_invalid_kernel_type_becomes_other(self) -> None:
        payload = {
            "kernel_name": None,
            "kernel_url": None,
            "kernel_type": "cuda",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
        }
        out = tp.parse_task_info("t", self._Model(json.dumps(payload)))
        assert out["kernel_type"] == "other"

    def test_malformed_json_returns_fallback(self) -> None:
        out = tp.parse_task_info("t", self._Model("not json {{{"))
        assert out["kernel_name"] is None
        assert out["kernel_type"] == "other"

    def test_query_exception_returns_fallback(self) -> None:
        class Bad:
            def query(self, messages):
                raise RuntimeError("boom")

        out = tp.parse_task_info("t", Bad())
        assert out["kernel_name"] is None
        assert out["kernel_type"] == "other"

    def test_repo_resolves_when_path_exists(self, tmp_path: Path) -> None:
        payload = {
            "kernel_name": None,
            "kernel_url": None,
            "kernel_type": "other",
            "repo": str(tmp_path),
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "output_dir": None,
            "model": None,
            "config": None,
        }
        out = tp.parse_task_info("t", self._Model(json.dumps(payload)))
        assert out["repo"] == str(tmp_path.resolve())


class TestParsePipelineParams:
    class _Model:
        def __init__(self, content: str) -> None:
            self._content = content

        def query(self, messages: list) -> dict:
            return {"content": self._content}

    def test_parses_fields(self) -> None:
        payload = {
            "kernel_url": "/tmp/a.hip",
            "preprocess_dir": None,
            "heterogeneous": True,
            "max_rounds": 3,
            "start_round": 1,
            "pipeline_intent": True,
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["heterogeneous"] is True
        assert out["max_rounds"] == 3
        assert out["start_round"] == 1
        assert out["pipeline_intent"] is True

    def test_coerces_numeric_strings(self) -> None:
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": "10",
            "start_round": "2",
            "pipeline_intent": False,
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["max_rounds"] == 10
        assert out["start_round"] == 2

    def test_invalid_int_fields_become_none(self) -> None:
        payload = {
            "kernel_url": None,
            "preprocess_dir": None,
            "heterogeneous": None,
            "max_rounds": "nope",
            "start_round": None,
            "pipeline_intent": False,
        }
        out = tp.parse_pipeline_params("t", self._Model(json.dumps(payload)))
        assert out["max_rounds"] is None

    def test_exception_returns_fallback(self) -> None:
        class Bad:
            def query(self, messages):
                raise RuntimeError("x")

        out = tp.parse_pipeline_params("t", Bad())
        assert out["kernel_url"] is None
        assert out["pipeline_intent"] is False


class TestGeneratePatchOutputDir:
    def test_uses_kernel_name_and_timestamp(self) -> None:
        with patch("minisweagent.run.utils.task_parser.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20250101_120000"
            out = tp.generate_patch_output_dir("my/kernel")
        assert out.replace("\\", "/") == "optimization_logs/my_kernel_20250101_120000"

    def test_none_kernel_name_uses_optimization_prefix(self) -> None:
        with patch("minisweagent.run.utils.task_parser.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20250101_120000"
            out = tp.generate_patch_output_dir(None)
        assert "optimization_20250101_120000" in out.replace("\\", "/")

    def test_respects_base_dir(self) -> None:
        with patch("minisweagent.run.utils.task_parser.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "ts"
            out = tp.generate_patch_output_dir("k", base_dir="custom_logs")
        assert out.replace("\\", "/") == "custom_logs/k_ts"


class TestDisplayParsedConfig:
    def test_includes_patch_output_dir_and_defaults(self) -> None:
        info = {
            "kernel_type": "triton",
            "kernel_name": "n",
            "kernel_url": "u",
            "repo": None,
            "test_command": None,
            "metric": None,
            "num_parallel": None,
            "gpu_ids": None,
            "model": None,
            "config": None,
        }
        text = tp.display_parsed_config(info, "/tmp/out")
        assert "patch_output_dir" in text
        assert "/tmp/out" in text
        assert "triton" in text
        assert "Resolved Configuration" in text
