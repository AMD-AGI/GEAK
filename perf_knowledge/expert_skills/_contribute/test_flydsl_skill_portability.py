from pathlib import Path
import importlib.util
import subprocess
import sys

import pytest
import yaml


SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"
EXPERT_SKILLS_ROOT = SKILLS_ROOT.parent
GEAK_ROOT = EXPERT_SKILLS_ROOT.parents[1]
MOE_SKILLS = [
    "flydsl_decode_moe_stage1_blkmap",
    "flydsl_prefill_moe_stage2_fp8partial",
]
DENSE_SKILL = "flydsl_fp8_blockscale_gemm"
SUBMITTED_SKILLS = [*MOE_SKILLS, DENSE_SKILL]


def load_scaffold_module():
    path = EXPERT_SKILLS_ROOT / "_contribute" / "scaffold.py"
    spec = importlib.util.spec_from_file_location("expert_skill_scaffold", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.mark.parametrize("skill_id", MOE_SKILLS)
def test_moe_skills_match_both_taxonomy_ids_on_existing_flydsl(skill_id):
    frontmatter, _ = load_skill(skill_id)
    match = frontmatter["match"]

    assert match["operator"] == ["grouped_gemm_moe", "fused_moe_grouped_gemm"]
    assert match["from_backend"] == "flydsl"
    assert match["to_backend"] == "flydsl"


def test_dense_skill_matches_dense_and_scaled_quant_taxonomy():
    frontmatter, _ = load_skill(DENSE_SKILL)

    assert frontmatter["match"]["operator"] == ["dense_gemm", "scaled_quant_gemm"]


@pytest.mark.parametrize(
    "fragment",
    [
        GEAK_ROOT / "kernel_workflow" / "roles" / "_fragments" / "expert_skills.md",
        GEAK_ROOT / "e2e_workflow" / "roles" / "_fragments" / "expert_skills.md",
    ],
)
def test_selector_prompt_defines_list_operator_membership(fragment):
    text = fragment.read_text()

    assert "a list containing" in text


@pytest.mark.parametrize("skill_id", MOE_SKILLS)
def test_emit_plan_uses_in_place_optimize_for_moe_skills(skill_id):
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERT_SKILLS_ROOT / "_contribute" / "validate_skill.py"),
            skill_id,
            "--emit-plan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=optimize" in result.stdout
    assert "target_language=triton" not in result.stdout
    assert "mode=author" not in result.stdout


def test_emit_plan_lists_author_and_optimize_for_mixed_source_skill():
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERT_SKILLS_ROOT / "_contribute" / "validate_skill.py"),
            "gluon_authoring",
            "--emit-plan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=author target_language=gluon" in result.stdout
    assert "mode=optimize" in result.stdout


@pytest.mark.parametrize("skill_id", SUBMITTED_SKILLS)
def test_kernel_skills_do_not_claim_an_unmeasured_e2e_gate(skill_id):
    frontmatter, _ = load_skill(skill_id)

    assert "e2e_delta_min_pct" not in frontmatter["expects"]
    assert frontmatter["validation"]["measured"]["e2e_pct"] == ""


@pytest.mark.parametrize(
    "document",
    [
        EXPERT_SKILLS_ROOT / "README.md",
        GEAK_ROOT / "kernel_workflow" / "roles" / "_fragments" / "expert_skills.md",
        GEAK_ROOT / "e2e_workflow" / "roles" / "_fragments" / "expert_skills.md",
    ],
)
def test_disabled_wording_is_limited_to_the_skill_fragment(document):
    text = document.read_text()
    normalized = " ".join(text.split())

    assert "base role prompt remains active" in normalized


def test_dependency_skill_reindexes_as_non_profile_matched():
    scaffold = load_scaffold_module()
    entry = scaffold.skill_index_entry(
        "ensure_flydsl",
        {
            "id": "ensure_flydsl",
            "scope": "dependency",
            "match": {"needs": "flydsl"},
        },
    )

    assert entry["scope"] == "dependency"
    assert "expects" not in entry
    assert entry["validation_status"] == "n/a"


def test_cross_backend_examples_use_canonical_backend_ids():
    text = (GEAK_ROOT / "kernel_workflow" / "roles" / "tech_lead.md").read_text()

    assert "ck→ck_tile" not in text


def test_dense_skill_frontmatter_matches_archived_down_proj_evidence():
    frontmatter, body = load_skill(DENSE_SKILL)
    artifact = EXPERT_SKILLS_ROOT / frontmatter["validation"]["artifact"]
    evidence = yaml.safe_load(artifact.read_text())

    assert "down_proj" in frontmatter["validation"]["measured"]["isolated"]
    assert "tile_n=256/tile_k=128" in frontmatter["validation"]["measured"]["isolated"]
    assert evidence["skill_id"] == DENSE_SKILL
    assert evidence["provider"] == "standalone_flydsl"
    assert evidence["method"] == (
        "CUDA-event same-session paired A/B, 3 archived on-box GEAK measurements"
    )
    assert evidence["warmup_per_run"] == 10
    assert evidence["iterations_per_run"] == 100
    assert evidence["inner_repetitions"] == 3
    assert len(evidence["baseline_latency_ms"]) == 3
    assert len(evidence["skill_latency_ms"]) == 3
    assert evidence["speedup"] == pytest.approx(
        evidence["baseline_median_ms"] / evidence["skill_median_ms"], rel=5e-5
    )
    assert evidence["parity"]["pass"] is True
    assert evidence["conclusions"]["decisive_lever"] == "tile_n=256/tile_k=128"
    assert evidence["conclusions"]["xcd_swizzle"] == "non-load-bearing for down_proj"
    assert evidence["conclusions"]["eight_wave"] == "measured dead-end for down_proj"
    assert evidence["raw_logs"].startswith("external archived GEAK artifacts")
    assert "artifacts" not in evidence
    assert "8-wave-blockscale port is also an open" not in body
