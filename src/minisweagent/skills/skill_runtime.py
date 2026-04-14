import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from minisweagent import get_repo_root

logger = logging.getLogger(__name__)


@dataclass
class SkillDescriptor:
    name: str
    description: str
    path: Path
    loaded: bool = False  # runtime state


class SkillRuntime:
    def __init__(self):
        skills_dir = get_repo_root() / "skills"
        self.skills = self._discover_skills(skills_dir)

    def _extract_yaml_frontmatter(self, markdown: str) -> dict:
        FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = FRONTMATTER_RE.match(markdown)
        if not match:
            raise ValueError("SKILL.md missing YAML frontmatter")

        return yaml.safe_load(match.group(1))

    def _parse_metadata(self, skill_path: Path) -> SkillDescriptor:
        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")

        fm = self._extract_yaml_frontmatter(content)

        return SkillDescriptor(name=fm["name"], description=fm["description"], path=skill_path, loaded=False)

    def _discover_skills(self, skills_root: Path) -> dict:
        if not skills_root.is_dir():
            logger.debug("Skills directory not found: %s", skills_root)
            return {}
        skills = []
        for p in skills_root.iterdir():
            if p.is_dir() and (p / "SKILL.md").exists():
                try:
                    skills.append(self._parse_metadata(p))
                except Exception as e:
                    print(f"Get skills fail: {e}")
        return {s.name: s for s in skills}

    def build_system_prompt(self) -> str:
        target_lang = os.environ.get("GEAK_TARGET_LANGUAGE", "")

        auto_loaded: list[str] = []
        if target_lang:
            for _name, s in self.skills.items():
                if target_lang.lower() in s.description.lower() and not s.loaded:
                    skill_md = s.path / "SKILL.md"
                    content = skill_md.read_text(encoding="utf-8")
                    auto_loaded.append(f"\n# Auto-loaded skill: {s.name}\n{content}")
                    s.loaded = True

        blocks = ["\n<available_skills>"]

        for _name, s in self.skills.items():
            blocks.append(
                f"""  <skill>
        <name>{s.name}</name>
        <description>{s.description}</description>
    </skill>"""
            )

        blocks.append("</available_skills>")

        if auto_loaded:
            blocks.append(
                "\nThe following skills have been automatically loaded because they match "
                f"the target language '{target_lang}'. Follow their guidance closely."
            )
            blocks.extend(auto_loaded)
        else:
            blocks.append(
                """
You can use the above skills to optimize related kernels.
If a skill is relevant, respond with:

```skills
{
"action": "use_skill",
"skill": "<skill-name>"
}
```
Otherwise, respond normally.
    """
            )

        return "\n".join(blocks)

    def load_skill(self, response: dict) -> dict:
        results = {
            "output": "",
            "returncode": 0,
        }
        if response["content"]:
            match = re.search(r"```skills\s*(\{.*?\})\s*```", response["content"], re.DOTALL)
            if not match:
                return results
            try:
                kill_action = json.loads(match.group(1))
                if kill_action["action"] == "use_skill":
                    if kill_action["skill"] not in self.skills.keys():
                        results["output"] = f"The skill {kill_action['skill']} is not exist."
                        return results
                    skill = self.skills[kill_action["skill"]]
                    if skill.loaded:
                        return results
                    skill_md = skill.path / "SKILL.md"
                    content = skill_md.read_text(encoding="utf-8")
                    results["output"] = f"\n# Loaded skill: {skill.name}\n{content}"
                    skill.loaded = True
            except Exception as e:
                results["output"] = f"No skills. Error: {e}"
        return results


if __name__ == "__main__":
    skills_str = """
You are an agent that can use the above skills.
If a skill is relevant, respond with:

```skills
{
"action": "use_skill",
"skill": "fused_bucketized-optimization"
}
```
Otherwise, respond normally.
    """
    response = {
        "content": skills_str,
    }
    skill_runtime = SkillRuntime()
    prompt = skill_runtime.build_system_prompt()
    print(prompt)
    skill = skill_runtime.load_skill(response)
    print(skill)
