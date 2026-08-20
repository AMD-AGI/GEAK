from pathlib import Path

import pytest
import yaml


SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
MOE_SKILLS = [
    "flydsl_decode_moe_stage1_blkmap",
    "flydsl_prefill_moe_stage2_fp8partial",
]


def load_skill(skill_id):
    text = (SKILLS_ROOT / skill_id / "skill.md").read_text()
    _, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter), body


@pytest.mark.parametrize("skill_id", MOE_SKILLS)
def test_flydsl_skill_uses_portable_minimum_version_guidance(skill_id):
    frontmatter, body = load_skill(skill_id)
    normalized_body = " ".join(body.split())

    assert "runtime" not in frontmatter
    assert "FlyDSL `>=0.2.2`" in body
    assert "newer" in body.lower()
    assert "A version difference alone is not a reason to skip the skill" in normalized_body
    assert "compile/parity/A/B" in body
    assert "must not be reused as evidence" in normalized_body
    assert "runtime_compat.py" not in body
    assert "backend_incompatible" not in body

    artifact = SKILLS_ROOT.parent / frontmatter["validation"]["artifact"]
    evidence = yaml.safe_load(artifact.read_text())
    assert evidence["skill_id"] == skill_id
    assert evidence["flydsl_version"] == "0.2.2"
