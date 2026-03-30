"""Tests for ``minisweagent.config`` path resolution and YAML helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.config import (
    builtin_config_dir,
    get_config_path,
    load_agent_config,
    load_config,
)


class TestGetConfigPath:
    def test_adds_yaml_suffix_when_missing(self) -> None:
        path = get_config_path("geak")
        assert path.suffix == ".yaml"
        assert path.name == "geak.yaml"

    def test_resolves_builtin_geak(self) -> None:
        path = get_config_path("geak.yaml")
        assert path.is_file()
        assert path.parent == builtin_config_dir

    def test_prefers_file_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd_file = tmp_path / "my_profile.yaml"
        cwd_file.write_text("model:\n  model_name: from-cwd\n")
        monkeypatch.chdir(tmp_path)

        path = get_config_path("my_profile.yaml")
        assert path.resolve() == cwd_file.resolve()

    def test_uses_mswea_config_dir_when_not_in_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alt = tmp_path / "alt"
        alt.mkdir()
        cfg = alt / "only_here.yaml"
        cfg.write_text("agent: {}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MSWEA_CONFIG_DIR", str(alt))

        path = get_config_path("only_here.yaml")
        assert path == cfg.resolve()

    def test_raises_file_not_found_with_tried_paths(self) -> None:
        with pytest.raises(FileNotFoundError) as exc_info:
            get_config_path("definitely_missing_config_xyz")
        msg = str(exc_info.value)
        assert "definitely_missing_config_xyz" in msg
        assert "tried:" in msg


class TestLoadConfig:
    def test_loads_geak_yaml(self) -> None:
        cfg = load_config("geak.yaml")
        assert "model" in cfg
        assert "agent" in cfg
        assert cfg["model"].get("model_class") == "amd_llm"

    def test_returns_empty_dict_for_empty_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        monkeypatch.chdir(tmp_path)
        assert load_config("empty.yaml") == {}


class TestLoadAgentConfig:
    def test_returns_agent_and_model_sections(self) -> None:
        agent, model = load_agent_config("geak.yaml")
        assert isinstance(agent, dict)
        assert isinstance(model, dict)
        assert "mode" in agent
        assert "model_class" in model

    def test_missing_sections_are_empty_dicts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bare = tmp_path / "bare.yaml"
        bare.write_text("other: 1\n")
        monkeypatch.chdir(tmp_path)
        agent, model = load_agent_config("bare.yaml")
        assert agent == {}
        assert model == {}
