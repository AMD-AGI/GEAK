"""Generic Expert Skill Planner Extension envelope tests."""

import copy
import hashlib
import importlib.util
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = (
    ROOT / "perf_knowledge" / "expert_skills" / "_contribute" / "validate_skill.py"
)
SPEC = importlib.util.spec_from_file_location("validate_expert_skill", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

SKILL_DIR = (
    ROOT
    / "perf_knowledge"
    / "expert_skills"
    / "skills"
    / "megamoe_ep_mega_fusion"
)


def test_repository_planner_extension_passes_generic_envelope():
    skill_path, metadata, _, _ = MODULE.load("megamoe_ep_mega_fusion")
    assert MODULE.planner_extension_errors(skill_path, metadata) == []

    extension = yaml.safe_load(
        (SKILL_DIR / "planner_extension.yaml").read_text()
    )
    assert extension["plan_version"] == "mega-plan-v2"
    assert extension["candidate_templates"][0]["id"] == "full_persistent_pipeline"
    assert (
        extension["ir_bindings"]["schedule"]["policies"][
            "g1_completion"
        ]
        == "stripe_system_atomic_threshold"
    )
    first_queue = next(
        queue for queue in extension["ir_bindings"]["queues"]
        if queue["id"] == "first_compute_queue"
    )
    assert first_queue["claim_unit"] == "stripe"
    assert first_queue["work_domain"] == "g1_output_stripes"
    assert [route["id"] for route in extension["failure_routes"]] == [
        "repair_complete_persistent_pipeline"
    ]
    selection = extension["selection_policy"]
    assert selection["intermediate_wip"] == {
        "lane": "same_candidate_required",
        "commit_each_turn": True,
        "candidate_status": "authoring",
        "claim_complete_for_committed_turn_artifact": True,
        "runtime_flags": False,
        "benchmark_or_promote_before_required_checks_pass": False,
        "continue_head_lineage": True,
    }
    assert selection["required_failure_order"] == [
        "plan",
        "correctness",
        "abi",
        "lifecycle",
        "resource",
        "compiler",
        "schedule",
        "performance",
    ]
    route = extension["failure_routes"][0]
    assert route["topology"] == "full_persistent_pipeline"
    assert route["terminal_when"] == "all_required_checks_pass"
    assert "same candidate lane" in route["repair_intent"]
    assert "dead_end" in route["repair_intent"]
    assert "too-large-for-one-turn" in route["repair_intent"]


def _write_fixture(tmp_path, extension):
    skill_path = tmp_path / "skill.md"
    skill_path.write_text("---\nid: fixture\n---\n")
    (tmp_path / "planner.yaml").write_text(yaml.safe_dump(extension))
    (tmp_path / "contract.yaml").write_text(
        yaml.safe_dump({
            "schema_version": "expert-skill-contract-v1",
            "skill_id": "fixture",
            "revision": "v1",
            "plan": {"assertions": [{"id": "plan_ok"}]},
            "checks": [{"id": "source_ok"}],
        })
    )
    metadata = {
        "id": "fixture",
        "revision": "v1",
        "planner_extension_file": "planner.yaml",
        "contract_file": "contract.yaml",
    }
    return skill_path, metadata


def _minimal_extension():
    return {
        "schema_version": "expert-skill-planner-extension-v1",
        "skill_id": "fixture",
        "revision": "v1",
        "plan_version": "mega-plan-v2",
        "ir_bindings": {"work_domains": [{"id": "items"}]},
        "candidate_templates": [{
            "id": "full",
            "target_topology": {"launch_count": 1},
        }],
        "failure_routes": [{
            "id": "repair",
            "match": {"check_ids": ["source_ok"]},
            "repair_intent": "repair the declared source invariant",
        }],
    }


def test_revision_mismatch_fails_closed(tmp_path):
    extension = _minimal_extension()
    extension["revision"] = "stale"
    skill_path, metadata = _write_fixture(tmp_path, extension)
    errors = MODULE.planner_extension_errors(skill_path, metadata)
    assert any("revision must match" in error for error in errors)


def test_unknown_contract_route_and_environment_identity_are_rejected(tmp_path):
    extension = copy.deepcopy(_minimal_extension())
    extension["failure_routes"][0]["match"]["check_ids"] = ["missing_check"]
    extension["reference_path"] = "/private/reference"
    skill_path, metadata = _write_fixture(tmp_path, extension)
    errors = MODULE.planner_extension_errors(skill_path, metadata)
    assert any("unknown contract checks" in error for error in errors)
    assert any("environment identity key reference_path" in error for error in errors)
    assert any("absolute machine paths" in error for error in errors)


def test_per_bucket_targets_are_planir_bindings_checked_like_the_target(tmp_path):
    extension = _minimal_extension()
    extension["ir_bindings"].update({
        "regions": [{"id": "body"}],
        "queues": [{"id": "work"}],
        "target_by_bucket": {
            "64": {"launch_count": 1, "required_regions": ["body"],
                   "required_queues": ["work"]},
        },
    })
    skill_path, metadata = _write_fixture(tmp_path, extension)
    assert MODULE.planner_extension_errors(skill_path, metadata) == []

    extension["ir_bindings"]["target_by_bucket"]["64"]["required_regions"] = ["tail"]
    extension["ir_bindings"]["target_by_bucket"]["small"] = {"launch_count": 1}
    skill_path, metadata = _write_fixture(tmp_path, extension)
    errors = MODULE.planner_extension_errors(skill_path, metadata)
    assert any(
        "target_by_bucket[64] names unknown required_regions: tail" in error
        for error in errors
    )
    assert any("target_by_bucket['small'] must be a target" in error for error in errors)


def test_repository_large_token_skill_routes_the_small_regime_to_its_own_skill():
    extension = yaml.safe_load((SKILL_DIR / "planner_extension.yaml").read_text())
    assert "target_by_bucket" not in extension["ir_bindings"]
    assert [item["id"] for item in extension["candidate_templates"]] == ["full_persistent_pipeline"]
    skill = (SKILL_DIR / "skill.md").read_text()
    assert "megamoe_small_mtpr_fusion" in skill
    assert "Small-token profile" not in skill
    small = SKILL_DIR.parent / "megamoe_small_mtpr_fusion"
    assert (small / "skill.md").is_file() and (small / "runtime_validation.py").is_file()


def test_generated_index_exposes_planner_extension():
    index = yaml.safe_load(
        (ROOT / "perf_knowledge" / "expert_skills" / "index.yaml").read_text()
    )
    entry = next(
        item for item in index["skills"]
        if item["id"] == "megamoe_ep_mega_fusion"
    )
    assert entry["planner_extension_file"].endswith("/planner_extension.yaml")
    assert not entry.get("embedded_components")
    assert entry["validation_status"] == "experimental"
    assert entry["auto_apply"] is False


def test_bundle_identity_is_path_independent_and_content_bound(tmp_path):
    skill_path, metadata, _, _ = MODULE.load("megamoe_ep_mega_fusion")
    original = MODULE.skill_bundle_identity(skill_path, metadata)
    contract = yaml.safe_load((SKILL_DIR / "contract.yaml").read_text())
    expected_contract_sha = hashlib.sha256(
        yaml.safe_dump(contract, sort_keys=True).encode()
    ).hexdigest()
    assert original["contract_sha256"] == expected_contract_sha

    copied = tmp_path / "skill"
    shutil.copytree(SKILL_DIR, copied)
    copied_identity = MODULE.skill_bundle_identity(copied / "skill.md", metadata)
    assert copied_identity == original

    copied_extension = copied / "planner_extension.yaml"
    text = copied_extension.read_text()
    copied_extension.write_text(text.replace(
        "preferred_template: full_persistent_pipeline",
        "preferred_template: alternate_pipeline",
        1,
    ))
    _, copied_metadata, _, _ = MODULE.load("megamoe_ep_mega_fusion")
    changed = MODULE.skill_bundle_identity(copied / "skill.md", copied_metadata)
    assert changed["planner_extension_sha256"] != original["planner_extension_sha256"]
    assert changed["bundle_sha256"] != original["bundle_sha256"]
    assert changed["contract_sha256"] == original["contract_sha256"]
