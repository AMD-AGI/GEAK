import json
from pathlib import Path

import pytest

from minisweagent.skills.skill_runtime import SkillDescriptor, SkillRuntime


def _make_runtime(skills: dict) -> SkillRuntime:
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.skills = skills
    return rt


def _write_skill(skill_dir: Path, name: str, description: str, body: str = "# Body\n") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}",
        encoding="utf-8",
    )


class TestExtractYamlFrontmatter:
    def test_parses_name_and_description(self):
        rt = SkillRuntime.__new__(SkillRuntime)
        md = "---\nname: test-skill\ndescription: When testing.\n---\n\n# Hi\n"
        fm = rt._extract_yaml_frontmatter(md)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "When testing."

    def test_missing_frontmatter_raises(self):
        rt = SkillRuntime.__new__(SkillRuntime)
        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            rt._extract_yaml_frontmatter("# No frontmatter\n")


class TestParseMetadata:
    def test_builds_descriptor(self, tmp_path: Path):
        skill_dir = tmp_path / "my-folder"
        _write_skill(skill_dir, "alpha", "Do alpha tasks.")
        rt = SkillRuntime.__new__(SkillRuntime)
        desc = rt._parse_metadata(skill_dir)
        assert desc.name == "alpha"
        assert desc.description == "Do alpha tasks."
        assert desc.path == skill_dir
        assert desc.loaded is False


class TestDiscoverSkills:
    def test_finds_skills_in_subdirs(self, tmp_path: Path):
        _write_skill(tmp_path / "one", "skill-one", "First.")
        _write_skill(tmp_path / "two", "skill-two", "Second.")
        (tmp_path / "not-a-skill.txt").write_text("x", encoding="utf-8")
        nested = tmp_path / "parent" / "child"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: nested\n---\n", encoding="utf-8")
        # _discover_skills only iterates immediate children; nested SKILL.md is ignored
        rt = SkillRuntime.__new__(SkillRuntime)
        found = rt._discover_skills(tmp_path)
        assert set(found.keys()) == {"skill-one", "skill-two"}

    def test_skips_directory_without_skill_md(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        _write_skill(tmp_path / "ok", "ok-skill", "OK")
        rt = SkillRuntime.__new__(SkillRuntime)
        found = rt._discover_skills(tmp_path)
        assert set(found.keys()) == {"ok-skill"}

    def test_skips_invalid_skill_md(self, tmp_path: Path, capsys):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
        _write_skill(tmp_path / "good", "good-skill", "Good")
        rt = SkillRuntime.__new__(SkillRuntime)
        found = rt._discover_skills(tmp_path)
        assert set(found.keys()) == {"good-skill"}
        assert "Get skills fail" in capsys.readouterr().out

    def test_duplicate_name_collapses_to_one_entry(self, tmp_path: Path):
        _write_skill(tmp_path / "a", "same-name", "First description")
        _write_skill(tmp_path / "b", "same-name", "Second description")
        rt = SkillRuntime.__new__(SkillRuntime)
        found = rt._discover_skills(tmp_path)
        assert len(found) == 1
        assert found["same-name"].name == "same-name"
        assert found["same-name"].description in ("First description", "Second description")


class TestBuildSystemPrompt:
    def test_includes_skill_tags_and_instructions(self):
        skills = {
            "demo": SkillDescriptor(name="demo", description="Use for demos.", path=Path("/tmp/demo")),
        }
        rt = _make_runtime(skills)
        prompt = rt.build_system_prompt()
        assert "<available_skills>" in prompt
        assert "</available_skills>" in prompt
        assert "<name>demo</name>" in prompt
        assert "<description>Use for demos.</description>" in prompt
        assert "```skills" in prompt
        assert "use_skill" in prompt

    def test_empty_skills_still_has_wrapper(self):
        rt = _make_runtime({})
        prompt = rt.build_system_prompt()
        assert "<available_skills>" in prompt
        assert "</available_skills>" in prompt


class TestLoadSkill:
    def _skill_block(self, skill_name: str) -> str:
        payload = json.dumps({"action": "use_skill", "skill": skill_name})
        return f"prefix\n```skills\n{payload}\n```\n"

    def test_loads_skill_content_and_sets_loaded(self, tmp_path: Path):
        skill_dir = tmp_path / "s"
        _write_skill(skill_dir, "load-me", "Desc", body="# Extra\n")
        desc = SkillDescriptor(name="load-me", description="Desc", path=skill_dir)
        rt = _make_runtime({"load-me": desc})
        result = rt.load_skill({"content": self._skill_block("load-me")})
        assert result["returncode"] == 0
        assert "# Loaded skill: load-me" in result["output"]
        assert "name: load-me" in result["output"]
        assert desc.loaded is True

    def test_second_load_leaves_output_empty(self, tmp_path: Path):
        skill_dir = tmp_path / "s"
        _write_skill(skill_dir, "idempotent", "D", body="x")
        desc = SkillDescriptor(name="idempotent", description="D", path=skill_dir)
        rt = _make_runtime({"idempotent": desc})
        first = rt.load_skill({"content": self._skill_block("idempotent")})
        assert first["output"]
        second = rt.load_skill({"content": self._skill_block("idempotent")})
        assert second["output"] == ""

    def test_unknown_skill_message(self):
        rt = _make_runtime({})
        result = rt.load_skill({"content": self._skill_block("missing")})
        assert "not exist" in result["output"]
        assert "missing" in result["output"]

    def test_no_skills_block_returns_empty_output(self):
        rt = _make_runtime({})
        result = rt.load_skill({"content": "no code block here"})
        assert result["output"] == ""

    def test_empty_content_returns_empty_output(self, tmp_path: Path):
        skill_dir = tmp_path / "s"
        _write_skill(skill_dir, "x", "d")
        rt = _make_runtime({"x": SkillDescriptor(name="x", description="d", path=skill_dir)})
        assert rt.load_skill({"content": ""})["output"] == ""

    def test_invalid_json_reports_error(self):
        rt = _make_runtime({})
        result = rt.load_skill({"content": "```skills\n{not json}\n```"})
        assert result["output"].startswith("No skills. Error:")

    def test_non_use_skill_action_does_not_load(self, tmp_path: Path):
        skill_dir = tmp_path / "s"
        _write_skill(skill_dir, "x", "d")
        desc = SkillDescriptor(name="x", description="d", path=skill_dir)
        rt = _make_runtime({"x": desc})
        payload = json.dumps({"action": "other", "skill": "x"})
        result = rt.load_skill({"content": f"```skills\n{payload}\n```"})
        assert result["output"] == ""
        assert desc.loaded is False


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_SILU = _REPO_ROOT / "examples" / "skills" / "silu-optimization" / "SKILL.md"
_USER_SKILLS_SILU = _REPO_ROOT / "skills" / "silu-optimization" / "SKILL.md"


@pytest.mark.skipif(not _EXAMPLE_SILU.is_file(), reason="example skill examples/skills/silu-optimization not present")
class TestSkillRuntimeIntegration:
    def test_example_silu_skill_parse_metadata(self):
        rt = SkillRuntime.__new__(SkillRuntime)
        desc = rt._parse_metadata(_EXAMPLE_SILU.parent)
        assert desc.name == "silu-optimization"
        assert "AMD" in desc.description or "silu" in desc.description.lower()

    def test_init_discovers_user_skills_dir(self):
        runtime = SkillRuntime()
        assert isinstance(runtime.skills, dict)
        if _USER_SKILLS_SILU.is_file():
            assert "silu-optimization" in runtime.skills
            s = runtime.skills["silu-optimization"]
            assert s.path == _USER_SKILLS_SILU.parent
            assert "AMD" in s.description or "silu" in s.description.lower()
