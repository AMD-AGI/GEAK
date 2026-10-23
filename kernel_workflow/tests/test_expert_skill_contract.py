"""Generic declarative Expert Skill contract tests."""

import ast
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest
import yaml


TOOL = Path(__file__).resolve().parents[1] / "tools" / "expert_skill_contract.py"
SPEC = importlib.util.spec_from_file_location("expert_skill_contract", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS = {
    "complete_host_path",
    "selected_fused_host_bindings",
    "fused_bundle_flow",
    "fused_host_abi",
    "persistent_kernel_abi",
    "fused_launch_abi",
    "stage2_active_compile_flow",
    "flat_stripe_completion",
    "g1_cache_builder_flow",
    "g1_cache_epilogue_flow",
    "g1_cache_store_sink",
    "unified_g1_g2_loop",
    "g2_scheduler_cfg",
    "g2_skew_orientation",
    "shared_stage2_body",
    "scheduler_counter_domain_separation",
    "nontrivial_stage2_body",
    "stage2_descriptor_flow",
    "stage2_standalone_shared_body",
    "stage2_nw8_geometry",
    "combine_emitter_wired",
    "combine_readiness_is_strict",
    "combine_launch_bindings",
    "combine_block_fp8_decode",
    "combine_u_selection",
    "p2p_visibility",
    "p2p_cache_selection",
    "p2p_cache_store_sink",
    "p2p_close_protocol",
    "progressive_token_readiness",
    "generation_lifecycle",
    "combine_output_work_item",
    "block_claimed_combine",
    "bucket_512_payload_rows",
}

MEGAMOE_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "perf_knowledge"
    / "expert_skills"
    / "skills"
    / "megamoe_ep_mega_fusion"
    / "contract.yaml"
)
# Optional on-disk calibration inputs; the tests that need them skip when unset.
MEGAMOE_REFERENCE_ROOT = Path(os.environ.get("MEGAMOE_REFERENCE_ROOT", "/nonexistent"))
MEGAMOE_REFERENCE_PLAN = Path(os.environ.get("MEGAMOE_REFERENCE_PLAN_JSON", "/nonexistent"))
MEGAMOE_DEVIATING_CANDIDATE_ROOT = Path(
    os.environ.get("MEGAMOE_DEVIATING_CANDIDATE_ROOT", "/nonexistent")
)


def _contract():
    return {
        "schema_version": "expert-skill-contract-v1",
        "skill_id": "test_skill",
        "revision": "v1",
        "source": {
            "include": ["src/*.py"],
            "required": ["src/a.py"],
            "core_identity": ["src/a.py"],
        },
        "provenance": {"require_manifest": True},
        "copy_detection": {
            "exact_similarity": 0.985,
            "suspect_similarity": 0.90,
            "exact_function_ratio": 0.80,
            "suspect_function_ratio": 0.50,
        },
        "plan": {
            "required": True,
            "assertions": [
                {"id": "target", "path": "target.launch_count", "op": "eq", "value": 2},
                {
                    "id": "threads",
                    "left": {"path": "resource.threads"},
                    "op": "eq",
                    "right": {
                        "op": "multiply",
                        "args": [
                            {"path": "resource.wave"},
                            {"path": "resource.waves"},
                        ],
                    },
                },
            ],
        },
        "checks": [
            {
                "id": "new_logic",
                "kind": "regex",
                "file": "src/a.py",
                "patterns": [r"def new", r"return"],
            }
        ],
    }


def _root(tmp_path: Path, name: str, source: str) -> Path:
    root = tmp_path / name
    code = root / "src"
    code.mkdir(parents=True)
    (code / "a.py").write_text(source)
    return root


def _plan():
    return {
        "target": {"launch_count": 2},
        "resource": {"threads": 512, "wave": 64, "waves": 8},
    }


def _manifest(contract, candidate):
    return {
        "sealed_before_reference_comparison": True,
        "reference_exposed_during_authoring": False,
        "candidate_tree_digest": MODULE.tree_digest(
            MODULE.collect_files(candidate, contract)
        ),
    }


def test_contract_loader_rejects_invalid_schema(tmp_path):
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "bad"}))
    with pytest.raises(ValueError):
        MODULE.load_contract(path)


def test_source_collection_honors_declarative_exclusions(tmp_path):
    root = tmp_path / "tree"
    (root / "src").mkdir(parents=True)
    (root / "3rdparty").mkdir()
    (root / "src" / "gemm.py").write_text("def valid():\n    pass\n")
    (root / "3rdparty" / "gemm.py").write_text("not valid python (\n")
    contract = _contract()
    contract["source"]["include"] = ["**/gemm.py"]
    contract["source"]["exclude"] = ["3rdparty/**"]
    files = MODULE.collect_files(root, contract)
    assert set(files) == {"src/gemm.py"}
    _, errors = MODULE._parse_files(files)
    assert not errors


def test_repository_megamoe_contract_is_declarative_and_generic():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    assert contract["skill_id"] == "megamoe_ep_mega_fusion"
    assert not contract.get("revision")
    assert {item["id"] for item in contract["checks"]} == {
        "complete_host_path",
        "selected_fused_host_bindings",
        "fused_bundle_flow",
        "fused_host_abi",
        "persistent_kernel_abi",
        "fused_launch_abi",
        "stage2_active_compile_flow",
        "host_specializes_g2_chunk",
        "flat_stripe_completion",
        "g1_cache_builder_flow",
        "g1_cache_epilogue_flow",
        "g1_cache_store_sink",
        "unified_g1_g2_loop",
        "g2_scheduler_cfg",
        "g2_skew_orientation",
        "shared_stage2_body",
        "scheduler_counter_domain_separation",
        "nontrivial_stage2_body",
        "stage2_descriptor_flow",
        "stage2_standalone_shared_body",
        "stage2_nw8_geometry",
        "combine_launch_bindings",
        "combine_block_fp8_decode",
        "combine_u_selection",
        "combine_emitter_wired",
        "combine_readiness_is_strict",
        "progressive_token_readiness",
        "generation_lifecycle",
        "combine_output_work_item",
        "block_claimed_combine",
        "p2p_visibility",
        "p2p_cache_selection",
        "p2p_cache_store_sink",
        "p2p_close_protocol",
        "bucket_512_payload_rows",
    }


def test_repository_megamoe_contract_requires_selected_implementation_semantics():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    checks = {item["id"]: item for item in contract["checks"]}
    required = {
        check_id
        for check_id, check in checks.items()
        if check.get("severity", "required") == "required"
    }
    assert required == MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    assert checks["host_specializes_g2_chunk"]["severity"] == "advisory"
    assert contract["activation"] == {
        "scope": "post_selection_claimed_complete_structural_checkpoint",
        "requires_selected_or_pinned_skill": True,
        "required_checks_define_claimed_implementation_completeness": True,
    }
    assert contract["binding_policy"]["applicability_owner"] == "workflow"
    assert (
        contract["binding_policy"]["adapter_selection"]
        == "analyze_discovered_semantic_bindings"
    )


def test_missing_selected_megamoe_implementation_fails_structural_checkpoint(tmp_path):
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for root in (baseline, candidate):
        source = root / "adapter_moe.py"
        source.parent.mkdir(parents=True)
        source.write_text("def unrelated():\n    return None\n")
    plan = {
        "plan_version": "mega-plan-v2",
        "target": {"launch_count": 2},
        "resources": {
            "arch": "gfx950",
            "local_memory": {"total_bytes": 160400, "limit_bytes": 163840},
        },
        "schedule": {"policies": {
            "combine_transition":
                "after_local_g1_g2_drain_without_grid_barrier",
            "scheduler_semantics":
                "continuation_then_skew_mod6_ready_g2_else_g1_then_blocking_g2_contiguous_c1_c16",
            "combine_semantics":
                "runtime_p_direct_per_wave_wait_block_claim_one_system19_u1_u2_u4",
        }},
    }
    result = MODULE.evaluate(
        contract,
        baseline,
        candidate,
        None,
        _manifest(contract, candidate),
        plan,
    )
    assert not result["input_errors"]
    assert not result["plan_consistency_errors"]
    assert result["failed_required_checks"] == [
        item["id"]
        for item in contract["checks"]
        if item["id"] in MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    ]
    assert not result["structural_compatible"]
    assert result["verdict"] == "incomplete"


def test_false_pass_fixture_matches_current_contract_identity():
    fixture = json.loads(
        (
            Path(__file__).with_name("fixtures")
            / "megamoe_ep_mega_fusion_false_pass_v2.json"
        ).read_text()
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    contract_sha = hashlib.sha256(
        yaml.safe_dump(contract, sort_keys=True).encode()
    ).hexdigest()
    assert fixture["candidate_head"] == "08f0c3a9b974103f9fcde000925436d5bd640f82"
    assert fixture["contract_sha256"] == contract_sha
    assert fixture["required_check_count"] == len(
        MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    )
    assert (
        set(fixture["failed_required_checks"])
        | set(fixture["passing_required_checks"])
        == MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    )
    assert fixture["required_failure_count"] == len(
        fixture["failed_required_checks"]
    )
    assert fixture["required_failure_count"] == 13
    tiers = {item["id"]: item["tier"] for item in contract["checks"]}
    assert fixture["semantic_failures"] == [
        check_id for check_id in fixture["failed_required_checks"]
        if tiers[check_id] == "semantic"
    ]
    assert fixture["surface_failures"] == [
        check_id for check_id in fixture["failed_required_checks"]
        if tiers[check_id] == "surface"
    ]
    assert fixture["identity_mismatch_failures"] == []
    assert fixture["structural_compatible"] is False
    assert fixture["verdict"] == "incomplete"


def test_compact_skill_contains_no_private_evidence_identifiers():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    text = path.read_text()
    assert not re.search(r"(^|[\s'\"])/(root|home|sgl-workspace)/", text)
    assert "candidate_tree_sha256" not in text
    assert "f33b628" not in text
    assert "mega-ep-fusion-v" not in text


def test_megamoe_skill_exposes_dependency_ordered_implementation_recipe():
    skill_dir = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
    )
    skill = (skill_dir / "skill.md").read_text()
    author = (skill_dir / "author_guide.md").read_text()
    planner = (skill_dir / "planner_guide.md").read_text()
    stages = [
        "Host selection, ownership and ABI",
        "G1 production and publication",
        "shared G2 body and weighted P2P publication",
        "unified persistent G1/G2 scheduler",
        "Combine as the irreversible third queue",
        "generation, replay and acceptance",
    ]
    positions = [skill.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "Dependency-ordered build recipe" in author
    assert "Source batch order" in planner
    assert "SharedAllocator().allocate(...).peek()" in skill
    assert "A bounded diagnostic wait must be removed" in author


def test_repository_fusion_contract_accepts_operator_neutral_plan_ir_v2():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    plan = {
        "plan_version": "mega-plan-v2",
        "expert_skill_id": "megamoe_ep_mega_fusion",
        "expert_skill_revision": "",
        "expert_skill_bundle_sha256": "bundle",
        "expert_skill_planner_extension_sha256": "planner",
        "expert_skill_validation_status": "experimental",
        "expert_skill_auto_apply": False,
        "expert_skill_explicit_pin_modes": ["authoring", "candidate_validation"],
        "target": {"launch_count": 2},
        "work_domains": [{
            "id": "first_stage_m_tiles",
            "unit": "m_tile",
            "extent": "ceildiv(num_valid, stage1_block_m)",
            "index_type": "i32",
            "owned_subdomain": {
                "unit": "n_stripe",
                "extent_per_claim": "stage1_n_tiles",
                "linear_index": "m_tile * stage1_n_tiles + n_stripe",
            },
        }],
        "regions": [{
            "id": "first_compute",
            "role": "producer_consumer",
            "work_domain": "first_stage_m_tiles",
            "engine": "matrix_memory",
        }],
        "queues": [{
            "id": "first_compute_queue",
            "work_domain": "first_stage_m_tiles",
            "claim_unit": "m_tile",
            "chunk_policy": "one_m_tile_all_n_stripes",
        }],
        "buffers": [{"id": "payload"}],
        "counters": [{"id": "ready"}],
        "resources": {
            "arch": "gfx950",
            "workgroup": {
                "wave_size": 64, "wave_count": 8, "thread_count": 512,
            },
            "local_memory": {
                "total_bytes": 160400,
                "limit_bytes": 163840,
                "allocation_rule": "max",
            },
            "registers": {"scratch_bytes_max": None},
            "occupancy": {"min_workgroups_per_cu": 1},
            "parameters": {
                "stage1_pool_bytes": 131072,
                "stage2_slab_bytes": 131728,
                "role_halves": 1,
                "additive_bytes": 28672,
            },
        },
        "schedule": {
            "primary_loop": {
                "kind": "unified",
                "carried_state": ["consumer_active", "g2_pend", "g2_next"],
            },
            "policies": {
                    "combine_transition":
                        "after_local_g1_g2_drain_without_grid_barrier",
                "completion_publication_granularity": "per_m_tile",
                "first_compute_claim_domain": "m_tile",
                "first_compute_stripe_ownership": "all_n_tiles_per_claim",
                "completion_publication_operation":
                    "waitcnt_barrier_thread0_system_store_n_tiles",
                "completion_rmw_in_unified_loop": "forbidden_transitively",
                "combine_item_jit_frame":
                    "direct_decorated_item_no_extra_wrapper",
                "combine_reducer_vectorization":
                    "pressure_guarded_u1_u2_u4",
                "combine_payload_visibility":
                    "system_scope_loads_without_per_item_acquire",
                "combine_output_cache": "rank_local_slc",
                "scheduler_semantics":
                    "continuation_then_skew_mod6_ready_g2_else_g1_then_blocking_g2_contiguous_c1_c16",
                "combine_semantics":
                    "runtime_p_direct_per_wave_wait_block_claim_one_system19_u1_u2_u4",
            },
            "parameters": {
                "combine_third_queue": True,
                "g2_chunk_large": 16,
                "g2_chunk_small": 1,
                "preemption_interval": 6,
                "skew_num": 5,
                "skew_den": 4,
                "work_shards": 4,
            },
        },
        "abi": {
            "arguments": [{"id": "payload"}],
            "parameters": {
                "direct_fused_args": True,
                "stage2_pointer_count": 12,
                "combine_pointer_count": 7,
                "optional_quant_pointer_count": 0,
                "disabled_placeholders": True,
            },
        },
    }
    valid, errors, _ = MODULE.validate_plan(contract, plan)
    assert valid
    assert not errors


def test_repository_plan_contract_rejects_missing_combine_transition():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    plan = {
        "plan_version": "mega-plan-v2",
        "expert_skill_id": "megamoe_ep_mega_fusion",
        "expert_skill_revision": "",
        "expert_skill_bundle_sha256": "bundle",
        "expert_skill_planner_extension_sha256": "planner",
        "expert_skill_validation_status": "experimental",
        "expert_skill_auto_apply": False,
        "expert_skill_explicit_pin_modes": ["authoring", "candidate_validation"],
        "target": {"launch_count": 2},
        "work_domains": [{
            "id": "first_stage_m_tiles",
            "unit": "m_tile",
            "extent": "ceildiv(num_valid, stage1_block_m)",
            "owned_subdomain": {"extent_per_claim": "stage1_n_tiles"},
        }],
        "regions": [{
            "id": "first_compute",
            "work_domain": "first_stage_m_tiles",
        }],
        "queues": [{
            "id": "first_compute_queue",
            "work_domain": "first_stage_m_tiles",
            "claim_unit": "tile",
            "chunk_policy": "one_tile",
        }],
        "buffers": [{}],
        "counters": [{}],
        "resources": {
            "arch": "gfx950",
            "workgroup": {"wave_size": 64, "wave_count": 8, "thread_count": 512},
            "local_memory": {
                "total_bytes": 160400,
                "limit_bytes": 163840,
                "allocation_rule": "max",
            },
            "registers": {"scratch_bytes_max": None},
            "occupancy": {"min_workgroups_per_cu": 1},
            "parameters": {
                "stage1_pool_bytes": 131072,
                "stage2_slab_bytes": 131728,
                "role_halves": 1,
                "additive_bytes": 28672,
            },
        },
        "schedule": {
            "primary_loop": {
                "kind": "unified",
                "carried_state": ["consumer_active", "g2_pend", "g2_next"],
            },
            "policies": {
                "completion_publication_granularity": "per_m_tile",
                "first_compute_claim_domain": "m_tile",
                "first_compute_stripe_ownership": "all_n_tiles_per_claim",
                "completion_publication_operation":
                    "waitcnt_barrier_thread0_system_store_n_tiles",
                "completion_rmw_in_unified_loop": "forbidden_transitively",
            },
            "parameters": {
                "combine_third_queue": True,
                "g2_chunk_large": 16,
                "g2_chunk_small": 1,
                "preemption_interval": 6,
                "skew_num": 5,
                "skew_den": 4,
                "work_shards": 4,
            },
        },
        "abi": {
            "arguments": [{}],
            "parameters": {
                "direct_fused_args": True,
                "stage2_pointer_count": 12,
                "combine_pointer_count": 7,
                "optional_quant_pointer_count": 0,
                "disabled_placeholders": True,
            },
        },
    }
    valid, errors, _ = MODULE.validate_plan(contract, plan)
    assert not valid
    assert any("combine_in_kernel" in error for error in errors)


def test_generic_engine_contains_no_operator_specific_contract():
    source = TOOL.read_text()
    assert "megamoe_ep_mega_fusion" not in source
    assert "mega_moe_stage1.py" not in source
    assert "AITER_MEGAMOE" not in source


def test_assignment_value_relates_target_to_its_own_rhs():
    contract = _contract()
    contract["checks"] = [{
        "id": "capacity_units",
        "kind": "assignment_value",
        "file": "src/a.py",
        "scope": "build",
        "target": "capacity",
        "value": r"ceildiv\(rows, block\)",
    }]
    unrelated = ast.parse(
        "def build(rows, block):\n"
        "    blocks = ceildiv(rows, block)\n"
        "    capacity = rows\n"
        "    return capacity\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": unrelated})
    assert not result["capacity_units"]["pass"]

    related = ast.parse(
        "def build(rows, block):\n"
        "    capacity = ceildiv(rows, block)\n"
        "    return capacity\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": related})
    assert result["capacity_units"]["pass"]


def test_assignment_value_none_rejects_forbidden_rhs():
    contract = _contract()
    contract["checks"] = [{
        "id": "domain_separation",
        "kind": "assignment_value",
        "file": "src/a.py",
        "scope": "build",
        "target": "output_extent",
        "value": r"routed_extent",
        "match": "none",
    }]
    tree = ast.parse(
        "def build(output_extent, routed_extent):\n"
        "    output_extent = routed_extent\n"
        "    return output_extent\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": tree})
    assert not result["domain_separation"]["pass"]


def test_regex_sequence_rejects_correct_names_in_wrong_order():
    contract = _contract()
    contract["checks"] = [{
        "id": "publish_order",
        "kind": "regex_sequence",
        "file": "src/a.py",
        "scope": "build",
        "patterns": [r"state\s*=\s*prepare\(\)", r"publish\(state\)"],
    }]
    wrong = ast.parse(
        "def build():\n"
        "    publish(state)\n"
        "    state = prepare()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": wrong})
    assert not result["publish_order"]["pass"]

    right = ast.parse(
        "def build():\n"
        "    state = prepare()\n"
        "    publish(state)\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": right})
    assert result["publish_order"]["pass"]


def test_publication_placement_rejects_rmw_anywhere_in_hot_loop():
    contract = _contract()
    contract["checks"] = [{
        "id": "publish_frame",
        "kind": "publication_placement",
        "file": "src/a.py",
        "scope": "kernel",
        "loop_test": "consumer_active",
        "compute_call": "_compute",
        "address": "ready_addr",
        "rmw_calls": ["atomic_add_system", "atomic_add_agent"],
        "store_calls": ["store_i32_system"],
    }]
    unsafe = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.atomic_add_system(ready_addr, 1)\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "        _publish()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": unsafe})
    assert not result["publish_frame"]["pass"]
    assert any(
        "inside hot loop" in failure
        for failure in result["publish_frame"]["failures"]
    )

    flushed = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.atomic_add_system(ready_addr, 1)\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "    _publish()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": flushed})
    assert result["publish_frame"]["pass"]

    owned = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.store_i32_system(ready_addr, 0, 1)\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "        _publish()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": owned})
    assert result["publish_frame"]["pass"]

    loop_head = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.atomic_add_system(ready_addr, 1)\n"
        "    while consumer_active:\n"
        "        _publish()\n"
        "        _compute()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": loop_head})
    assert not result["publish_frame"]["pass"]

    wrapped = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _raw_publish():\n"
        "        ops.atomic_add_agent(ready_addr, 1)\n"
        "    def renamed_wrapper():\n"
        "        _raw_publish()\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "        renamed_wrapper()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": wrapped})
    assert not result["publish_frame"]["pass"]

    dead_only = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.store_i32_system(ready_addr, 0, 1)\n"
        "    if False:\n"
        "        _publish()\n"
        "    while consumer_active:\n"
        "        _compute()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": dead_only})
    assert not result["publish_frame"]["pass"]


def _owned_protocol_rule():
    return {
        "id": "owned_completion",
        "kind": "owned_completion_protocol",
        "file": "src/a.py",
        "scope": "kernel",
        "normalize_wrappers": ["Int32", "Int64", "int"],
        "region": {
            "guard": "fuse_all",
            "arm": "body",
            "loop": {"test": "consumer_active", "contains": r"\bg2_pend\b"},
        },
        "claim": {
            "extent_target": "g1_total_mtiles",
            "extent_value": "num_m_tiles",
            "source_extent_target": "num_m_tiles",
            "source_extent_value": r"ceildiv\(num_valid, sort_block_m\)",
            "helper": "_claim",
            "owner_test": "tid == 0",
            "local_target": "local_claim",
            "atomic_call": "atomic_add_agent",
            "head_address": r"a_work_head \+ work_shard \* 64",
            "increment": "1",
            "index_target": "work",
            "index_value": r"work_shard \+ local_claim \* WORK_SHARDS",
            "bound": "work < g1_total_mtiles",
        },
        "handoff": {
            "store_call": "ptr_store",
            "scratch": "scratch",
            "vector_call": "from_elements",
            "values": ["1", "work", "-1"],
            "view_target": "fdrain_view",
            "view_value": r"make_view\(scratch, make_layout\(3, 1\)\)",
            "load_target": "_vals",
            "load_value": r"Vec\(fdrain_view\.load\(\)\)",
            "barrier_call": "barrier",
            "unit_target": "unit",
            "unit_value": r"_vals\[1\]",
        },
        "stripes": {
            "mode_test": "kind == 1",
            "iterator": "stripe",
            "range_call": "range",
            "count": "N_TILES",
            "compute_call": "_compute",
            "compute_index": r"unit \* N_TILES \+ stripe",
        },
        "publication": {
            "helper": "_publish",
            "argument": "unit",
            "wait_call": "s_waitcnt",
            "wait_argument": "0",
            "barrier_call": "barrier",
            "owner_test": "tid == 0",
            "store_call": "store_i32_system",
            "address_call": "_ready_addr",
            "address_argument": "m_tile",
            "direct_address_patterns": [
                r"ready_base \+ .*\bunit\b.* \* 4",
                r"4 \* .*\bunit\b.* \+ ready_base",
            ],
            "address_base_patterns": [r"^ready_base$"],
            "address_index_patterns": [
                r"\bunit\b.* \* 4",
                r"4 \* .*\bunit\b",
            ],
            "store_offset": "0",
            "store_value": "N_TILES",
            "forbidden_rmw_calls": [
                "atomic_add_system", "atomic_add_agent",
                "atomic_add_global_at", "AtomicRMWOp",
            ],
            "forbidden_calls": ["fence_system_release"],
        },
    }


def _owned_protocol_source():
    return (
        "def kernel(fuse_all, consumer_active, g2_pend, num_valid, sort_block_m, "
        "a_work_head, work_shard, tid, peer_ready):\n"
        "    num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))\n"
        "    if fuse_all:\n"
        "        g1_total_mtiles = num_m_tiles\n"
        "        def _ready_addr(m_tile):\n"
        "            return m_tile\n"
        "        def _publish(m_tile):\n"
        "            ops.s_waitcnt(Int32(0))\n"
        "            ops.barrier()\n"
        "            if tid == Int32(0):\n"
        "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
        "Int32(N_TILES))\n"
        "        def _peer_publish(addr):\n"
        "            ops.atomic_add_system(addr, Int32(1))\n"
        "        fdrain_view = make_view(scratch, make_layout(3, 1))\n"
        "        while consumer_active:\n"
        "            g2_pend = g2_pend\n"
        "            def _claim():\n"
        "                local_claim = Int32(ops.atomic_add_agent("
        "a_work_head + Int64(work_shard) * Int64(64), Int32(1)))\n"
        "                work = work_shard + local_claim * Int32(WORK_SHARDS)\n"
        "                if work < g1_total_mtiles:\n"
        "                    ptr_store(Vec.from_elements("
        "[Int32(1), work, Int32(-1)], Int32), scratch)\n"
        "                else:\n"
        "                    ptr_store(Vec.from_elements("
        "[Int32(0), Int32(0), Int32(-1)], Int32), scratch)\n"
        "            if tid == Int32(0):\n"
        "                _claim()\n"
        "            _peer_publish(peer_ready)\n"
        "            ops.barrier()\n"
        "            _vals = Vec(fdrain_view.load())\n"
        "            kind = _vals[0]\n"
        "            unit = _vals[1]\n"
        "            if kind == Int32(1):\n"
        "                for stripe in range(int(N_TILES)):\n"
        "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n"
        "                _publish(unit)\n"
    )


def _owned_protocol_result(source):
    contract = _contract()
    contract["checks"] = [_owned_protocol_rule()]
    return MODULE.evaluate_checks(
        contract, {"src/a.py": ast.parse(source)}
    )["owned_completion"]


def test_owned_completion_protocol_accepts_owner_store_and_unrelated_atomics():
    result = _owned_protocol_result(_owned_protocol_source())
    assert result["pass"], result["failures"]
    assert result["units_passed"] == result["units_total"] == 9


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))",
            "num_m_tiles = num_valid",
        ),
        (
            "num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))",
            "num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))\n"
            "    num_m_tiles = num_valid",
        ),
        ("g1_total_mtiles = num_m_tiles", "g1_total_mtiles = num_m_tiles * N_TILES"),
        (
            "work_shard + local_claim * Int32(WORK_SHARDS)",
            "local_claim",
        ),
        (
            "                work = work_shard + local_claim * Int32(WORK_SHARDS)\n",
            "                work = work_shard + local_claim * Int32(WORK_SHARDS)\n"
            "                work = Int32(0)\n",
        ),
        ("work < g1_total_mtiles", "work < num_m_tiles * N_TILES"),
        ("range(int(N_TILES))", "range(int(N_TILES - 1))"),
        (
            "unit * Int32(N_TILES) + Int32(stripe)",
            "unit * Int32(N_TILES) + Int32(stripe) + Int32(1)",
        ),
        ("ops.s_waitcnt(Int32(0))\n            ops.barrier()",
         "ops.barrier()\n            ops.s_waitcnt(Int32(0))"),
        ("tid == Int32(0)", "tid == Int32(1)"),
        ("Int32(N_TILES))\n", "Int32(1))\n"),
        (
            "ops.store_i32_system(_ready_addr(m_tile), Int32(0), Int32(N_TILES))",
            "ops.store_i32_system(_ready_addr(m_tile + Int32(1)), Int32(0), "
            "Int32(N_TILES))",
        ),
        (
            "ops.store_i32_system(_ready_addr(m_tile), Int32(0), Int32(N_TILES))",
            "ops.atomic_add_system(_ready_addr(m_tile), Int32(1))",
        ),
    ],
)
def test_owned_completion_protocol_rejects_relation_mutations(old, new):
    source = _owned_protocol_source()
    assert old in source
    result = _owned_protocol_result(source.replace(old, new, 1))
    assert not result["pass"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            pass\n",
        ),
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            _claim()\n",
        ),
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            if tid == Int32(0):\n"
            "                if allow_claim:\n"
            "                    _claim()\n",
        ),
        (
            "[Int32(1), work, Int32(-1)], Int32), scratch)",
            "[Int32(1), work, Int32(-1)], Int32), other_scratch)",
        ),
        (
            "                    ptr_store(Vec.from_elements("
            "[Int32(1), work, Int32(-1)], Int32), scratch)\n",
            "                    if publish_scratch:\n"
            "                        ptr_store(Vec.from_elements("
            "[Int32(1), work, Int32(-1)], Int32), scratch)\n",
        ),
        (
            "            ops.barrier()\n"
            "            _vals = Vec(fdrain_view.load())\n",
            "            _vals = Vec(fdrain_view.load())\n",
        ),
        (
            "            unit = _vals[1]\n",
            "            if load_unit:\n"
            "                unit = _vals[1]\n",
        ),
        (
            "            unit = _vals[1]\n",
            "            unit = _vals[1]\n"
            "            unit = Int32(0)\n",
        ),
        (
            "                _publish(unit)\n",
            "                if unit >= Int32(0):\n"
            "                    _publish(unit)\n",
        ),
        (
            "                _publish(unit)\n",
            "                continue\n"
            "                _publish(unit)\n",
        ),
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            continue\n"
            "            if tid == Int32(0):\n"
            "                _claim()\n",
        ),
        (
            "                _publish(unit)\n",
            "                _publish = lambda value: None\n"
            "                _publish(unit)\n",
        ),
        (
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n",
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n"
            "                    break\n",
        ),
        (
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n",
            "                    ops.store_i32_system(_ready_addr(unit), Int32(0), "
            "Int32(N_TILES))\n"
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n",
        ),
        (
            "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
            "Int32(N_TILES))\n",
            "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
            "Int32(N_TILES))\n"
            "            else:\n"
            "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
            "Int32(N_TILES))\n",
        ),
        (
            "                _publish(unit)\n",
            "                ops.fence_system_release()\n"
            "                _publish(unit)\n",
        ),
        (
            "        def _publish(m_tile):\n"
            "            ops.s_waitcnt(Int32(0))\n",
            "        def _publish(m_tile):\n"
            "            m_tile = Int32(0)\n"
            "            ops.s_waitcnt(Int32(0))\n",
        ),
    ],
)
def test_owned_completion_protocol_rejects_cfg_and_handoff_bypasses(old, new):
    source = _owned_protocol_source()
    assert old in source
    result = _owned_protocol_result(source.replace(old, new, 1))
    assert not result["pass"]


@pytest.mark.parametrize(
    "extra",
    [
        (
            "                completion_alias = _ready_addr(unit)\n"
            "                ops.atomic_add_system(completion_alias, Int32(1))\n"
        ),
        (
            "                completion_alias = _ready_addr(unit)\n"
            "                _peer_publish(completion_alias)\n"
        ),
        (
            "                ops.atomic_add_system("
            "ready_base + unit * Int32(4), Int32(1))\n"
        ),
        (
            "                ops.atomic_add_system("
            "Int32(4) * unit + ready_base, Int32(1))\n"
        ),
        (
            "                ops.atomic_add_global_at("
            "_ready_addr(unit), Int32(1))\n"
        ),
        (
            "                base_alias = ready_base\n"
            "                offset_alias = unit * Int32(4)\n"
            "                completion_alias = base_alias + offset_alias\n"
            "                ops.atomic_add_system(completion_alias, Int32(1))\n"
        ),
        (
            "                address_factory = _ready_addr\n"
            "                completion_alias = address_factory(unit)\n"
            "                ops.atomic_add_system(completion_alias, Int32(1))\n"
        ),
        (
            "                bad_rmw = ops.atomic_add_system\n"
            "                bad_rmw(_ready_addr(unit), Int32(1))\n"
        ),
    ],
)
def test_owned_completion_protocol_rejects_region_rmw_aliases(extra):
    source = _owned_protocol_source().replace(
        "                _publish(unit)\n",
        "                _publish(unit)\n" + extra,
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("RMW" in failure for failure in result["failures"])


def test_owned_completion_protocol_ignores_unselected_standalone_arm():
    source = _owned_protocol_source() + (
        "    else:\n"
        "        ops.atomic_add_system(_ready_addr(Int32(0)), Int32(1))\n"
    )
    result = _owned_protocol_result(source)
    assert result["pass"], result["failures"]


def test_owned_completion_protocol_rejects_transitive_completion_rmw():
    source = _owned_protocol_source().replace(
        "        while consumer_active:\n",
        "        def _raw(addr):\n"
        "            ops.atomic_add_agent(addr, Int32(1))\n"
        "        def _wrapped(addr):\n"
        "            _raw(addr)\n"
        "        while consumer_active:\n",
    ).replace(
        "                _publish(unit)\n",
        "                _publish(unit)\n"
        "                _wrapped(_ready_addr(unit))\n",
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("RMW" in failure for failure in result["failures"])


def test_owned_completion_protocol_tracks_aliases_and_keyword_forwarding():
    source = _owned_protocol_source().replace(
        "        while consumer_active:\n",
        "        def _address_alias(m_tile):\n"
        "            return _ready_addr(m_tile)\n"
        "        def _raw(addr):\n"
        "            first = addr\n"
        "            second = first\n"
        "            ops.atomic_add_system(second, Int32(1))\n"
        "        def _wrapped(*, target):\n"
        "            _raw(addr=target)\n"
        "        while consumer_active:\n",
    ).replace(
        "                _publish(unit)\n",
        "                _publish(unit)\n"
        "                _wrapped(target=_address_alias(unit))\n",
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("RMW" in failure for failure in result["failures"])


def test_owned_completion_protocol_rejects_per_item_fence():
    source = _owned_protocol_source().replace(
        "            ops.s_waitcnt(Int32(0))\n",
        "            ops.fence_system_release()\n"
        "            ops.s_waitcnt(Int32(0))\n",
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("forbidden" in failure for failure in result["failures"])


def test_reference_self_is_calibration_only_and_never_hardware_evidence(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    result = MODULE.evaluate(
        contract, baseline, reference, reference,
        _manifest(contract, reference), _plan(),
    )
    assert result["verdict"] == "calibration_only"
    assert result["structural_exact"]
    assert result["reference_copy_detected"]
    assert not result["capability_eligible"]
    assert not result["hardware_verified"]
    assert not result["gpu_executed"]
    assert not result["device_jit_verified"]
    assert not result["path_activation_verified"]
    assert not result["launch_count_verified"]
    assert not result["accuracy_verified"]
    assert not result["graph_liveness_verified"]
    assert not result["residency_verified"]
    assert not result["performance_verified"]


def test_independent_shape_can_pass_with_valid_seal_and_plan(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "def new():\n    result = 1\n    return result\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    assert result["structural_compatible"]
    assert result["plan_consistent"]
    assert not result["reference_copy_detected"]
    assert result["independent_structure_pass"]
    assert result["capability_eligible"]
    assert result["provenance_status"] == "attested_unverified"


def test_reference_parity_gaps_are_informational_unless_declared(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(
        tmp_path, "reference",
        "def new():\n    return 1\n\ndef reference_extra():\n    return 2\n",
    )
    candidate = _root(
        tmp_path, "candidate",
        "def new():\n    value = 1\n    return value\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    assert result["missing_new_symbols"]
    assert result["required_changed_files_missing"] == []
    assert result["contract_failures"] == []
    assert result["independent_structure_pass"]
    assert result["capability_eligible"]


def test_feature_in_static_dead_code_or_comment_is_rejected(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        '"""def new(): return 1"""\n# def new(): return 1\nif False:\n'
        "    def new():\n        return 1\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    assert not result["checks"]["new_logic"]["pass"]
    assert not result["structural_compatible"]


def test_plan_expression_is_machine_checked(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    candidate = _root(tmp_path, "candidate", "def new():\n    return 1\n")
    broken = _plan()
    broken["resource"]["threads"] = 256
    result = MODULE.evaluate(
        contract, baseline, candidate, None,
        _manifest(contract, candidate), broken,
    )
    assert not result["plan_consistent"]
    assert not result["structural_compatible"]
    assert any("threads" in error for error in result["plan_consistency_errors"])


def test_plan_may_defer_compile_only_resource_measurement():
    contract = _contract()
    contract["plan"] = {
        "required": True,
        "assertions": [{
            "id": "scratch_unverified_or_zero",
            "path": "resource.scratch_bytes_max",
            "op": "in",
            "value": [None, 0],
        }],
    }
    valid, errors, _ = MODULE.validate_plan(
        contract, {"resource": {"scratch_bytes_max": None}}
    )
    assert valid
    assert not errors


def test_call_keyword_identity_rejects_payload_pointer_as_ready_pointer(tmp_path):
    contract = _contract()
    contract["checks"] = [
        {
            "id": "ready_identity",
            "category": "abi",
            "kind": "call_keywords",
            "file": "src/a.py",
            "scope": "wire",
            "call": "dict",
            "keywords": {
                "local_ready": r"_fx_tok_ready",
                "peer_ready": r"_fx_p2p_tok_ready",
            },
        }
    ]
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "def wire(op):\n"
        "    return dict(local_ready=op._fx_tok_ready, "
        "peer_ready=op._fx_p2p_comb_inp)\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, None,
        _manifest(contract, candidate), _plan(),
    )
    assert not result["checks"]["ready_identity"]["pass"]
    assert result["contract_failures"][0]["category"] == "abi"
    assert "peer_ready" in result["next_blocker"]


def test_selected_branch_calls_rejects_direct_unfused_tail():
    contract = _contract()
    contract["checks"] = [{
        "id": "host_path",
        "kind": "selected_branch_calls",
        "file": "src/a.py",
        "scope": "run",
        "condition": "fused",
        "required_calls": ["persistent"],
        "forbidden_calls": ["standalone"],
        "require_return": True,
    }]
    unsafe = ast.parse(
        "def run(fused):\n"
        "    if fused:\n"
        "        persistent()\n"
        "        output = standalone()\n"
        "        return output\n"
        "    return standalone()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": unsafe})["host_path"]
    assert not result["pass"]
    assert any("forbidden standalone" in failure for failure in result["failures"])

    safe = ast.parse(
        "def run(fused, debug):\n"
        "    if fused:\n"
        "        persistent()\n"
        "        if debug:\n"
        "            standalone()\n"
        "        return output\n"
        "    return standalone()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": safe})["host_path"]
    assert result["pass"], result["failures"]


def test_function_shape_requires_wide_consumed_abi():
    contract = _contract()
    contract["checks"] = [{
        "id": "abi",
        "kind": "function_shape",
        "file": "src/a.py",
        "scope": "kernel",
        "min_parameters": 3,
        "min_loaded_parameters": 3,
    }]
    narrow = ast.parse("def kernel(a, b):\n    return a + b\n")
    result = MODULE.evaluate_checks(contract, {"src/a.py": narrow})["abi"]
    assert not result["pass"]

    dead = ast.parse("def kernel(a, b, c):\n    return a + b\n")
    result = MODULE.evaluate_checks(contract, {"src/a.py": dead})["abi"]
    assert not result["pass"]

    complete = ast.parse("def kernel(a, b, c):\n    return a + b + c\n")
    result = MODULE.evaluate_checks(contract, {"src/a.py": complete})["abi"]
    assert result["pass"], result["failures"]


def test_call_arity_requires_forwarded_expansions():
    contract = _contract()
    contract["checks"] = [{
        "id": "forwarding",
        "kind": "call_arity",
        "file": "src/a.py",
        "scope": "launch",
        "call": "invoke",
        "min_arguments": 4,
        "min_starred": 2,
    }]
    narrow = ast.parse("def launch(a, rest):\n    invoke(a, *rest)\n")
    result = MODULE.evaluate_checks(contract, {"src/a.py": narrow})["forwarding"]
    assert not result["pass"]

    complete = ast.parse(
        "def launch(a, b, left, right):\n"
        "    invoke(a, b, *left, *right)\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": complete})["forwarding"]
    assert result["pass"], result["failures"]


def test_tuple_bindings_rejects_missing_duplicate_and_placeholder_sources():
    contract = _contract()
    contract["checks"] = [{
        "id": "host_bundle",
        "kind": "tuple_bindings",
        "file": "src/a.py",
        "scope": "wire",
        "bindings": [{
            "target": "args",
            "min_elements": 3,
            "min_distinct": 3,
            "forbidden": ["None", "0"],
        }],
    }]
    placeholder = ast.parse(
        "def wire(a):\n"
        "    args = (a, a, None)\n"
        "    return args\n"
    )
    result = MODULE.evaluate_checks(
        contract, {"src/a.py": placeholder}
    )["host_bundle"]
    assert not result["pass"]

    complete = ast.parse(
        "def wire(a, b, c):\n"
        "    args = (a, b, c)\n"
        "    return args\n"
    )
    result = MODULE.evaluate_checks(
        contract, {"src/a.py": complete}
    )["host_bundle"]
    assert result["pass"], result["failures"]


def test_bundle_flow_requires_producer_outputs_at_launch_sink():
    contract = _contract()
    contract["checks"] = [{
        "id": "flow",
        "kind": "bundle_flow",
        "file": "src/a.py",
        "scope": "launch",
        "producer_call": "build",
        "outputs": ["spec", "args"],
        "bundle_target": "kw",
        "keys": {"fused_spec": "spec", "fused_args": "args"},
        "sink_call": "invoke",
    }]
    dead = ast.parse(
        "def launch():\n"
        "    spec, args = build()\n"
        "    self.saved = args\n"
        "    invoke()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": dead})["flow"]
    assert not result["pass"]

    complete = ast.parse(
        "def launch():\n"
        "    spec, args = build()\n"
        "    kw = {'fused_spec': spec, 'fused_args': args}\n"
        "    invoke(**kw)\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": complete})["flow"]
    assert result["pass"], result["failures"]


def test_nontrivial_function_rejects_identity_adapter():
    contract = _contract()
    contract["checks"] = [{
        "id": "body",
        "kind": "nontrivial_function",
        "file": "src/a.py",
        "scope": "build.emit",
        "min_statements": 3,
        "required_calls": ["compute", "publish"],
    }]
    identity = ast.parse(
        "def build():\n"
        "    def emit(item):\n"
        "        return item\n"
        "    return emit\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": identity})["body"]
    assert not result["pass"]

    complete = ast.parse(
        "def build():\n"
        "    def emit(item):\n"
        "        value = compute(item)\n"
        "        publish(value)\n"
        "        return value\n"
        "    return emit\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": complete})["body"]
    assert result["pass"], result["failures"]


def _semantic_result(kind, source, relations, scope="kernel"):
    contract = _contract()
    contract["checks"] = [{
        "id": "semantic",
        "kind": kind,
        "file": "src/a.py",
        "scope": scope,
        "normalize_wrappers": ["I32"],
        "relations": relations,
    }]
    return MODULE.evaluate_checks(
        contract, {"src/a.py": ast.parse(source)}
    )["semantic"]


def test_structured_cfg_proves_state_transition_and_rejects_mutations():
    source = (
        "def kernel(active, pending, nxt, head, ready, g1_head):\n"
        "    kind = I32(0)\n"
        "    unit = I32(0)\n"
        "    while active:\n"
        "        kind = I32(0)\n"
        "        if pending > I32(0):\n"
        "            kind = I32(2)\n"
        "            unit = nxt\n"
        "            pending = pending - I32(1)\n"
        "            nxt = nxt + I32(1)\n"
        "        else:\n"
        "            if peek(head) < ready:\n"
        "                claim(head)\n"
        "            if kind == I32(0):\n"
        "                claim(g1_head)\n"
        "                kind = I32(1)\n"
        "        if kind == I32(1):\n"
        "            run_g1(unit)\n"
        "        if kind == I32(2):\n"
        "            run_g2(unit)\n"
        "        active = kind != I32(0)\n"
    )
    relations = [
        {"id": "loop", "op": "count", "count": 1,
         "select": {"node": "while", "test": "active"}},
        {"id": "continuation", "op": "ordered", "selectors": [
            {"node": "assign", "target": "kind", "value": "2",
             "guard": "pending > 0"},
            {"node": "assign", "target": "unit", "value": "nxt",
             "guard": "pending > 0"},
            {"node": "assign", "target": "pending", "value": "pending - 1",
             "guard": "pending > 0"},
            {"node": "assign", "target": "nxt", "value": r"nxt \+ 1",
             "guard": "pending > 0"},
        ]},
        {"id": "no_claim", "op": "forbidden",
         "region": {"node": "if", "test": "pending > 0"}, "region_arm": "body",
         "select": {"node": "call", "call": "claim|wait"}},
        {"id": "g1", "op": "guarded",
         "subject": {"node": "call", "call": "run_g1"},
         "guard": "kind == 1"},
        {"id": "g2", "op": "guarded",
         "subject": {"node": "call", "call": "run_g2"},
         "guard": "kind == 2"},
    ]
    result = _semantic_result("structured_cfg", source, relations)
    assert result["pass"], result["failures"]
    mutations = [
        ("            unit = nxt\n", "            claim(head)\n            unit = nxt\n"),
        ("pending = pending - I32(1)", "pending = pending + I32(1)"),
        ("        if kind == I32(2):\n", "        if kind == I32(1):\n"),
    ]
    for old, new in mutations:
        mutated = _semantic_result(
            "structured_cfg", source.replace(old, new, 1), relations
        )
        assert not mutated["pass"], (old, mutated["failures"])


def test_shared_callable_proves_factory_identity_and_preload_dominance():
    source = (
        "def compile():\n"
        "    consts = derive()\n"
        "    emit = make_emitter(**consts)\n"
        "    def kernel(x):\n"
        "        preload(x)\n"
        "        emit(x=x)\n"
        "    return kernel\n"
    )
    relations = [
        {"id": "identity", "op": "callable_identity",
         "factory_call": {"node": "call", "call": "make_emitter"},
         "consumers": [{"scope": "compile.kernel", "source": "emit",
                        "call": {"node": "call", "call": "emit",
                                 "keywords": {"x": "x"}}}]},
        {"id": "constants", "op": "depends",
         "sink": {"node": "call", "call": "make_emitter"},
         "star_keyword": True, "source": "consts"},
        {"id": "preload", "op": "all_precede",
         "first": {"node": "call", "call": "preload"},
         "second": {"node": "call", "call": "emit"}},
    ]
    result = _semantic_result("shared_callable", source, relations, "compile")
    assert result["pass"], result["failures"]
    for mutated_source in (
        source.replace("emit(x=x)", "other(x=x)"),
        source.replace("        preload(x)\n", ""),
        source.replace("        preload(x)\n        emit(x=x)\n",
                       "        emit(x=x)\n        preload(x)\n"),
    ):
        assert not _semantic_result(
            "shared_callable", mutated_source, relations, "compile"
        )["pass"]


def test_collective_protocol_proves_owner_close_and_parallel_arrival():
    source = (
        "def publish(tid, ctr, block, nblocks, BM, ready, token):\n"
        "    fence()\n"
        "    barrier()\n"
        "    if tid == I32(0):\n"
        "        prev = atomic(ctr + block * I32(4), I32(1))\n"
        "        store(prev, scratch)\n"
        "    barrier()\n"
        "    closes = load(scratch) == nblocks - I32(1)\n"
        "    if closes:\n"
        "        if tid == I32(0):\n"
        "            atomic(ctr + block * I32(4), I32(0) - nblocks)\n"
        "        if tid < I32(BM):\n"
        "            atomic_system(ready + token * I32(4), I32(1))\n"
    )
    relations = [
        {"id": "owner_add", "op": "guarded",
         "subject": {"node": "call", "call": "atomic",
                     "args": [r"ctr \+ block \* 4", "1"]},
         "guard": "tid == 0"},
        {"id": "same_address", "op": "same_argument",
         "left": {"node": "call", "call": "atomic",
                  "args": [r"ctr \+ block \* 4", "1"]},
         "right": {"node": "call", "call": "atomic",
                   "args": [r"ctr \+ block \* 4", "0 - nblocks"]}},
        {"id": "close_guard", "op": "guarded",
         "subject": {"node": "call", "call": "atomic_system"},
         "guard": "closes"},
        {"id": "parallel", "op": "guarded",
         "subject": {"node": "call", "call": "atomic_system"},
         "guard": "tid < BM"},
        {"id": "no_loop", "op": "forbidden",
         "region": {"node": "if", "test": "closes"}, "region_arm": "body",
         "select": {"node": "for", "target": ".*", "iter": ".*BM.*"}},
    ]
    result = _semantic_result(
        "collective_protocol", source, relations, "publish"
    )
    assert result["pass"], result["failures"]
    mutations = [
        source.replace("    if tid == I32(0):\n", "    if tid == I32(1):\n", 1),
        source.replace("ctr + block * I32(4), I32(0) - nblocks",
                       "ctr + (block + I32(1)) * I32(4), I32(0) - nblocks"),
        source.replace("        if tid < I32(BM):\n",
                       "        for row in range(BM):\n"),
    ]
    for mutated in mutations:
        assert not _semantic_result(
            "collective_protocol", mutated, relations, "publish"
        )["pass"]


def test_arithmetic_pipeline_proves_decode_scale_reduce_and_store():
    source = (
        "def reduce(inp, scales, out, lane):\n"
        "    in_cache = I32(2 | 1 | 16)\n"
        "    out_cache = I32(2)\n"
        "    raw = load(inp, dtype='i32', cache=in_cache)\n"
        "    scale_raw = load(scales, dtype='i8', cache=in_cache)\n"
        "    scale_bits = broadcast(scale_raw, lane)\n"
        "    scale = shift(scale_bits, I32(23))\n"
        "    fp32 = decode_fp8(raw) * scale\n"
        "    packed = to_bf16(fp32)\n"
        "    store(out, packed, cache=out_cache)\n"
    )
    relations = [
        {"id": "payload", "op": "count", "count": 1,
         "select": {"node": "call", "call": "load",
                    "keywords": {"dtype": "'i32'", "cache": "in_cache"}}},
        {"id": "scale", "op": "count", "count": 1,
         "select": {"node": "call", "call": "load",
                    "keywords": {"dtype": "'i8'", "cache": "in_cache"}}},
        {"id": "scale_bits", "op": "count", "count": 1,
         "select": {"node": "assign", "target": "scale",
                    "value": r"shift\(scale_bits, 23\)"}},
        {"id": "pipeline", "op": "ordered", "selectors": [
            {"node": "assign", "target": "raw"},
            {"node": "assign", "target": "scale_raw"},
            {"node": "assign", "target": "scale_bits"},
            {"node": "assign", "target": "scale"},
            {"node": "assign", "target": "fp32"},
            {"node": "assign", "target": "packed"},
            {"node": "call", "call": "store"},
        ]},
        {"id": "store", "op": "all_calls_match", "call": "store",
         "args": ["out", "packed"], "keywords": {"cache": "out_cache"}},
    ]
    result = _semantic_result(
        "arithmetic_pipeline", source, relations, "reduce"
    )
    assert result["pass"], result["failures"]
    for mutated in (
        source.replace("dtype='i8'", "dtype='i32'"),
        source.replace("I32(23)", "I32(22)"),
        source.replace("store(out, packed", "store(inp, packed"),
    ):
        assert not _semantic_result(
            "arithmetic_pipeline", mutated, relations, "reduce"
        )["pass"]


def test_value_flow_fails_closed_on_ambiguous_or_dead_sources():
    relations = [{
        "id": "cache", "op": "depends",
        "sink": {"node": "call", "call": "consume"},
        "keyword": "cache", "source": "selected",
    }]
    good = (
        "def kernel(selected):\n"
        "    forwarded = selected\n"
        "    consume(cache=forwarded)\n"
    )
    result = _semantic_result("value_flow", good, relations)
    assert result["pass"], result["failures"]
    ambiguous = good.replace(
        "    consume(cache=forwarded)\n",
        "    forwarded = other\n    consume(cache=forwarded)\n",
    )
    assert not _semantic_result("value_flow", ambiguous, relations)["pass"]
    dead = good.replace("consume(cache=forwarded)", "consume(cache=other)")
    assert not _semantic_result("value_flow", dead, relations)["pass"]


def test_interprocedural_parameter_and_starred_abi_flow_rejects_discards():
    source = (
        "def run(fused_stage2, fused_args, fused_combine, "
        "fused_combine_args, fused_quant_args):\n"
        "    launch = compile_kernel(fused_stage2=fused_stage2, "
        "fused_combine=fused_combine)\n"
        "    fa = tuple(fused_args)\n"
        "    ca = tuple(fused_combine_args)\n"
        "    qa = tuple(fused_quant_args)\n"
        "    _run_compiled(launch, *fa, *ca, *qa)\n"
    )
    relations = [
        {"id": "used", "op": "parameters_used",
         "parameters": ["fused_.*"], "min_parameters": 5},
        {"id": "spec", "op": "depends",
         "sink": {"node": "call", "call": "compile_kernel"},
         "keyword": "fused_stage2", "source": "fused_stage2"},
        {"id": "combine", "op": "depends",
         "sink": {"node": "call", "call": "compile_kernel"},
         "keyword": "fused_combine", "source": "fused_combine"},
        {"id": "runtime", "op": "depends",
         "sink": {"node": "call", "call": "_run_compiled"},
         "star_arg": 0, "source": "fused_args"},
        {"id": "abi_order", "op": "starred_arguments",
         "call": {"node": "call", "call": "_run_compiled"},
         "values": ["fa", "ca", "qa"]},
    ]
    result = _semantic_result("value_flow", source, relations, "run")
    assert result["pass"], result["failures"]
    mutations = [
        source.replace("fused_stage2=fused_stage2",
                       "fused_stage2=None"),
        source.replace("    fa = tuple(fused_args)\n", ""),
        source.replace("*fa, *ca, *qa", "*ca, *fa, *qa"),
        source.replace("fused_combine=fused_combine",
                       "fused_combine=None"),
    ]
    for mutated in mutations:
        assert not _semantic_result(
            "value_flow", mutated, relations, "run"
        )["pass"]


def test_tuple_and_mapping_provenance_reject_aliases_and_placeholders():
    source = (
        "def host(a, b, c, bm, bn):\n"
        "    roots = (a, b, c)\n"
        "    spec = dict(BM=bm, BN=bn, ready=True)\n"
        "    return roots, spec\n"
    )
    relations = [
        {"id": "roots", "op": "tuple_elements", "target": "roots",
         "length": 3, "min_distinct": 3, "values": ["a", "b", "c"]},
        {"id": "spec", "op": "mapping_entries",
         "mapping": {"node": "assign", "target": "spec"},
         "entries": {"BM": "bm", "BN": "bn", "ready": "True"},
         "forbidden": ["0", "None"]},
    ]
    result = _semantic_result("value_flow", source, relations, "host")
    assert result["pass"], result["failures"]
    for mutated in (
        source.replace("roots = (a, b, c)", "roots = (a, a, c)"),
        source.replace("BN=bn", "BN=0"),
        source.replace("ready=True", "ready=None"),
    ):
        assert not _semantic_result(
            "value_flow", mutated, relations, "host"
        )["pass"]


def test_active_callable_and_decorator_checks_reject_dead_or_unrewritten_body():
    source = (
        "def compile_body(consts):\n"
        "    emit = make_emitter(**consts)\n"
        "    @jit\n"
        "    def item(x):\n"
        "        emit(x)\n"
        "    return item\n"
    )
    relations = [
        {"id": "factory", "op": "callable_identity", "active": True,
         "factory_call": {"node": "call", "call": "make_emitter",
                          "arg_count": 0, "star_keywords": 1},
         "consumers": [{"scope": "compile_body.item", "active": True,
                        "source": "emit",
                        "call": {"node": "call", "call": "emit"}}]},
        {"id": "decorated", "op": "count", "count": 1,
         "select": {"node": "function", "name": "item",
                    "decorators": ["jit"]}},
    ]
    result = _semantic_result(
        "shared_callable", source, relations, "compile_body"
    )
    assert result["pass"], result["failures"]
    dead = source.replace(
        "    emit = make_emitter(**consts)\n",
        "    if False:\n        emit = make_emitter(**consts)\n",
    )
    assert not _semantic_result(
        "shared_callable", dead, relations, "compile_body"
    )["pass"]
    wrapped_dead = dead.replace("if False:", "if const_expr(False):")
    assert not _semantic_result(
        "shared_callable", wrapped_dead, relations, "compile_body"
    )["pass"]
    assert not _semantic_result(
        "shared_callable", source.replace("    @jit\n", ""),
        relations, "compile_body",
    )["pass"]
    assert not _semantic_result(
        "shared_callable",
        source.replace("make_emitter(**consts)", "make_emitter(0, **consts)"),
        relations, "compile_body",
    )["pass"]


def test_exclusive_effect_and_unguarded_fallback_reject_v10_shapes():
    source = (
        "def kernel(diag, ready, pref):\n"
        "    if diag:\n"
        "        waitcnt()\n"
        "        barrier()\n"
        "        publish(ready)\n"
        "    else:\n"
        "        publish(ready)\n"
        "    if idle:\n"
        "        claim(ready)\n"
    )
    relations = [
        {"id": "once", "op": "exclusive_effect",
         "select": {"node": "call", "call": "publish"}},
        {"id": "fallback", "op": "not_guarded",
         "subject": {"node": "call", "call": "claim", "guard": "idle"},
         "guard": "pref"},
    ]
    result = _semantic_result("structured_cfg", source, relations)
    assert result["pass"], result["failures"]
    sequential = source.replace(
        "    else:\n        publish(ready)\n",
        "    publish(ready)\n",
    )
    guarded = source.replace(
        "    if idle:\n        claim(ready)\n",
        "    if idle:\n        if pref:\n            claim(ready)\n",
    )
    assert not _semantic_result(
        "structured_cfg", sequential, relations
    )["pass"]
    assert not _semantic_result(
        "structured_cfg", guarded, relations
    )["pass"]


def test_per_row_address_def_use_rejects_uniform_arrival_address():
    source = (
        "def close(tx, packed_base, ready_table):\n"
        "    meta = load(packed_base + tx * 4)\n"
        "    peer = meta >> 24\n"
        "    token = meta & 255\n"
        "    ready = load(ready_table + peer * 8)\n"
        "    atomic(ready + token * 4, 1)\n"
    )
    relations = [{
        "id": "row", "op": "depends",
        "sink": {"node": "call", "call": "atomic"},
        "arg": 0, "source": "tx",
    }]
    result = _semantic_result(
        "collective_protocol", source, relations, "close"
    )
    assert result["pass"], result["failures"]
    uniform = source.replace("packed_base + tx * 4", "packed_base")
    assert not _semantic_result(
        "collective_protocol", uniform, relations, "close"
    )["pass"]


def test_group_segment_plan_relation_rejects_limit_overflow():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    plan = {
        "plan_version": "mega-plan-v2",
        "target": {"launch_count": 2},
        "resources": {
            "arch": "gfx950",
            "local_memory": {"total_bytes": 163841, "limit_bytes": 163840},
        },
        "schedule": {"policies": {
            "combine_transition":
                "after_local_g1_g2_drain_without_grid_barrier",
            "scheduler_semantics":
                "continuation_then_skew_mod6_ready_g2_else_g1_then_blocking_g2_contiguous_c1_c16",
            "combine_semantics":
                "runtime_p_direct_per_wave_wait_block_claim_one_system19_u1_u2_u4",
        }},
    }
    valid, errors, _ = MODULE.validate_plan(contract, plan)
    assert not valid
    assert any("group_segment_within_gfx950" in error for error in errors)


def test_constant_return_abstract_interpretation_checks_profile_branch():
    source = (
        "def choose(bucket, width):\n"
        "    if bucket <= 2048:\n"
        "        bm, nw = 32, 4\n"
        "    else:\n"
        "        bm, nw = 64, 8\n"
        "    bn = 512 if width >= 4096 else 256\n"
        "    return Config(BM=bm, BN=bn, NW=nw)\n"
    )
    relations = [{
        "id": "profile", "op": "constant_return",
        "inputs": {"bucket": 8192, "width": 7168},
        "call": "Config",
        "keywords": {"BM": 64, "BN": 512, "NW": 8},
    }]
    result = _semantic_result("value_flow", source, relations, "choose")
    assert result["pass"], result["failures"]
    mutated = source.replace("bm, nw = 64, 8", "bm, nw = 32, 4")
    assert not _semantic_result(
        "value_flow", mutated, relations, "choose"
    )["pass"]


def test_definite_attribute_initialization_follows_builds_and_dynamic_maps():
    source = (
        "class Owner:\n"
        "    def __init__(self):\n"
        "        self._build()\n"
        "        sources = {'peer': storage, 'ready': counters}\n"
        "        for name, value in sources.items():\n"
        "            setattr(self, name, value)\n"
        "    def _build(self):\n"
        "        self.output = allocate()\n"
    )
    relations = [{
        "id": "attrs", "op": "definite_attributes",
        "class": "Owner", "entry": "__init__",
        "attributes": ["output", "peer", "ready"],
    }]
    result = _semantic_result(
        "value_flow", source, relations, "Owner.__init__"
    )
    assert result["pass"], result["failures"]
    missing = source.replace("'ready': counters", "'other': counters")
    assert not _semantic_result(
        "value_flow", missing, relations, "Owner.__init__"
    )["pass"]
    conditional = source.replace(
        "        self._build()\n",
        "        if enabled:\n            self._build()\n",
    )
    assert not _semantic_result(
        "value_flow", conditional, relations, "Owner.__init__"
    )["pass"]


def test_cached_call_hashability_requires_canonical_immutable_specs():
    source = (
        "@cache\n"
        "def compile_kernel(spec):\n"
        "    return spec\n"
        "def run(spec):\n"
        "    canonical = None if spec is None else "
        "tuple(sorted(dict(spec).items()))\n"
        "    return compile_kernel(spec=canonical)\n"
    )
    relations = [{
        "id": "hashable", "op": "cached_call_hashable",
        "call": {"node": "call", "call": "compile_kernel"},
        "keywords": ["spec"],
    }]
    result = _semantic_result("value_flow", source, relations, "run")
    assert result["pass"], result["failures"]
    raw = source.replace("spec=canonical", "spec=spec")
    assert not _semantic_result(
        "value_flow", raw, relations, "run"
    )["pass"]
    mutable = source.replace(
        "None if spec is None else tuple(sorted(dict(spec).items()))",
        "[]",
    )
    assert not _semantic_result(
        "value_flow", mutable, relations, "run"
    )["pass"]


def test_profile_group_segment_uses_max_lifetimes_and_limit():
    source = (
        "def compile_shape():\n"
        "    pool = max(2 * a_bytes, c_bytes)\n"
        "    slab = ready_off + peer_bytes\n"
        "    scales = scale_bytes\n"
        "    return pool\n"
    )
    relations = [{
        "id": "lds", "op": "profile_group_segment",
        "inputs": {
            "a_bytes": 32768,
            "c_bytes": 131072,
            "ready_off": 131648,
            "peer_bytes": 80,
            "scale_bytes": 28672,
        },
        "base": {"node": "assign", "target": "pool"},
        "role": {"node": "assign", "target": "slab"},
        "additive": {"node": "assign", "target": "scales"},
        "role_multiplier": 1,
        "expected": 160400,
        "limit": 163840,
    }]
    result = _semantic_result(
        "value_flow", source, relations, "compile_shape"
    )
    assert result["pass"], result["failures"]
    oversized = source.replace(
        "ready_off + peer_bytes", "ready_off + peer_bytes + 32768"
    )
    assert not _semantic_result(
        "value_flow", oversized, relations, "compile_shape"
    )["pass"]
    additive_lifetimes = source.replace(
        "max(2 * a_bytes, c_bytes)", "2 * a_bytes + c_bytes"
    )
    assert not _semantic_result(
        "value_flow", additive_lifetimes, relations, "compile_shape"
    )["pass"]


def test_loop_prefix_and_helper_effect_relations_reject_noops():
    source = (
        "def kernel(active):\n"
        "    while active:\n"
        "        barrier()\n"
        "        step()\n"
        "    def helper(unit):\n"
        "        compute(unit)\n"
    )
    relations = [
        {"id": "prefix", "op": "body_prefix",
         "region": {"node": "while", "test": "active"},
         "select": {"node": "call", "call": "barrier"}},
        {"id": "effect", "op": "within_count",
         "region": {"node": "function", "name": "helper"},
         "count": 1, "select": {"node": "call", "call": "compute"}},
        {"id": "abi", "op": "function_parameters",
         "function": {"node": "function", "name": "helper"},
         "required": ["unit"], "forbidden": ["accumulator"]},
    ]
    result = _semantic_result("structured_cfg", source, relations)
    assert result["pass"], result["failures"]
    no_barrier = source.replace("        barrier()\n", "")
    assert not _semantic_result(
        "structured_cfg", no_barrier, relations
    )["pass"]
    identity = source.replace("        compute(unit)\n", "        return unit\n")
    assert not _semantic_result(
        "structured_cfg", identity, relations
    )["pass"]
    free_value = source.replace("def helper(unit):", "def helper(unit, accumulator):")
    assert not _semantic_result(
        "structured_cfg", free_value, relations
    )["pass"]


def test_topk_ancestor_and_branch_partition_reject_squared_or_duplicate_paths():
    source = (
        "def reduce(end, U, K, wide):\n"
        "    for u in range(U):\n"
        "        for k in range(K):\n"
        "            acc = acc + vals[u][k]\n"
        "    if wide:\n"
        "        loop(end, 4)\n"
        "    else:\n"
        "        loop(end, 1)\n"
    )
    relations = [
        {"id": "one_k", "op": "ancestor_count",
         "subject": {"node": "assign", "target": "acc"},
         "container": {"node": "for", "target": "k", "iter": "range\\(K\\)"},
         "count": 1},
        {"id": "paths", "op": "branch_partition",
         "nodes": {"node": "call", "call": "loop"}},
    ]
    result = _semantic_result(
        "arithmetic_pipeline", source, relations, "reduce"
    )
    assert result["pass"], result["failures"]
    squared = source.replace(
        "            acc = acc + vals[u][k]\n",
        "            for k2 in range(K):\n"
        "                acc = acc + vals[u][k]\n",
    ).replace(
        '"container": {"node": "for", "target": "k",',
        '"container": {"node": "for", "target": "k|k2",',
    )
    squared_relations = [dict(value) for value in relations]
    squared_relations[0] = {
        **relations[0],
        "container": {"node": "for", "target": "k|k2", "iter": r"range\(K\)"},
    }
    assert not _semantic_result(
        "arithmetic_pipeline", squared, squared_relations, "reduce"
    )["pass"]
    duplicate = source + "\n"
    duplicate = duplicate.replace(
        "    else:\n        loop(end, 1)\n",
        "    else:\n        loop(end, 1)\n    loop(end, 1)\n",
    )
    assert not _semantic_result(
        "arithmetic_pipeline", duplicate, relations, "reduce"
    )["pass"]


def test_contract_score_denominator_is_stable_when_scope_is_missing(tmp_path):
    contract = _contract()
    contract["checks"] = [
        {
            "id": "three_patterns",
            "kind": "regex",
            "file": "src/a.py",
            "scope": "target",
            "patterns": ["one", "two", "three"],
        }
    ]
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    present = _root(
        tmp_path, "present",
        "def target():\n    one = two = three = 1\n    return one + two + three\n",
    )
    missing = _root(tmp_path, "missing", "def other():\n    return 0\n")
    present_result = MODULE.evaluate(
        contract, baseline, present, None,
        _manifest(contract, present), _plan(),
    )
    missing_result = MODULE.evaluate(
        contract, baseline, missing, None,
        _manifest(contract, missing), _plan(),
    )
    assert present_result["semantic_features_total"] == 3
    assert missing_result["semantic_features_total"] == 3


def test_forbidden_host_counter_reset_is_rejected(tmp_path):
    contract = _contract()
    contract["checks"] = [
        {
            "id": "no_reset",
            "kind": "forbid_methods",
            "file": "src/a.py",
            "scope": "launch",
            "methods": ["zero_"],
            "receivers": ["ready"],
        }
    ]
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "def launch(token_ready):\n"
        "    token_ready.zero_()\n"
        "    return token_ready\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, None,
        _manifest(contract, candidate), _plan(),
    )
    assert not result["checks"]["no_reset"]["pass"]
    assert not result["structural_compatible"]


def test_missing_required_source_file_fails_closed(tmp_path):
    contract = _contract()
    missing = tmp_path / "missing"
    result = MODULE.evaluate(contract, missing, missing, None, None, _plan())
    assert result["input_errors"]
    assert not result["structural_compatible"]


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_reference_links_are_not_independent_evidence(tmp_path, link_kind):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(tmp_path, "candidate", "def placeholder():\n    pass\n")
    target = candidate / "src" / "a.py"
    target.unlink()
    source = reference / "src" / "a.py"
    if link_kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    evidence = result["candidate_symlinks"] + result["reference_hardlinks"]
    assert "src/a.py" in evidence
    assert not result["capability_eligible"]


def test_every_repository_megamoe_check_declares_a_tier():
    contract = MODULE.load_contract(MEGAMOE_CONTRACT)
    missing = [item["id"] for item in contract["checks"] if "tier" not in item]
    assert missing == []
    tiers = {item["id"]: item["tier"] for item in contract["checks"]}
    assert set(tiers.values()) <= {"semantic", "surface"}, tiers
    for check_id in (
        "flat_stripe_completion",
        "unified_g1_g2_loop",
        "g2_scheduler_cfg",
        "g2_skew_orientation",
        "scheduler_counter_domain_separation",
        "generation_lifecycle",
    ):
        assert tiers[check_id] == "semantic"
    assert tiers["fused_host_abi"] == "surface"


def test_contract_loader_rejects_unknown_check_tier(tmp_path):
    contract = _contract()
    path = tmp_path / "contract.yaml"
    contract["checks"][0]["tier"] = "cosmetic"
    path.write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match="tier"):
        MODULE.load_contract(path)
    contract["checks"][0]["tier"] = "surface"
    path.write_text(yaml.safe_dump(contract))
    assert MODULE.load_contract(path)["checks"][0]["tier"] == "surface"


def test_next_blocker_names_every_failing_required_check_semantic_first(tmp_path):
    contract = _contract()
    surface = [f"surface_gap_{index}" for index in range(4)]
    semantic = [f"semantic_gap_{index}" for index in range(6)]
    contract["checks"] = [
        {"id": check_id, "tier": "surface", "kind": "regex", "file": "src/a.py",
         "patterns": [f"absent_{check_id}"]}
        for check_id in surface
    ] + [
        {"id": check_id, "kind": "regex", "file": "src/a.py",
         "patterns": [f"absent_{check_id}"]}
        for check_id in semantic
    ] + [
        {"id": "satisfied", "tier": "semantic", "kind": "regex", "file": "src/a.py",
         "patterns": ["def new"]},
        {"id": "advisory_gap", "tier": "surface", "severity": "advisory",
         "kind": "regex", "file": "src/a.py", "patterns": ["absent_advisory"]},
    ]
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    candidate = _root(tmp_path, "candidate", "def new():\n    return 1\n")
    plan = _plan()
    plan["resource"]["threads"] = 256
    result = MODULE.evaluate(
        contract, baseline, candidate, None, _manifest(contract, candidate), plan,
    )
    assert result["failed_required_checks"] == surface + semantic
    assert result["semantic_failures"] == semantic
    assert result["surface_failures"] == surface
    assert result["checks"][semantic[0]]["tier"] == "semantic"
    assert result["checks"][surface[0]]["tier"] == "surface"
    assert {item["id"]: item["tier"] for item in result["contract_failures"]} == {
        **{check_id: "semantic" for check_id in semantic},
        **{check_id: "surface" for check_id in surface},
        "threads": "semantic",
    }

    blocker = result["next_blocker"]
    assert isinstance(blocker, str) and "\n" not in blocker
    assert blocker.startswith("10 required checks failing (6 semantic, 4 surface).")
    semantic_group = "semantic: " + ", ".join(semantic)
    surface_group = "surface: " + ", ".join(surface)
    assert blocker.index(semantic_group) < blocker.index(surface_group)
    assert "satisfied" not in blocker and "advisory_gap" not in blocker
    details = blocker.split("details: ", 1)[1]
    assert details.startswith(f"{semantic[0]}: absent_{semantic[0]}")
    assert details.count("absent_semantic_gap_") == MODULE.NEXT_BLOCKER_DETAIL_LIMIT
    assert "absent_surface_gap_" not in details
    assert "(+4 more unit failures)" in details
    assert blocker.endswith(
        "plan errors (1): " + "; ".join(result["plan_consistency_errors"])
    )


def test_any_of_relation_requires_one_complete_alternative():
    source = (
        "def kernel(spec, waves):\n"
        "    nw = int(waves)\n"
        "    kw = dict(spec)\n"
        "    kw['NW'] = nw\n"
        "    return derive(**kw)\n"
        "def helper():\n"
        "    only_in_helper()\n"
    )
    relations = [
        {"id": "nw_reaches", "op": "any_of", "alternatives": [
            {"id": "explicit", "relations": [
                {"op": "depends", "sink": {"node": "call", "call": "derive"},
                 "keyword": "NW", "source": r"\bwaves\b"},
            ]},
            {"id": "expanded", "relations": [
                {"op": "depends", "sink": {"node": "assign", "target": r"kw\['NW'\]"},
                 "source": r"\bwaves\b"},
                {"op": "depends", "sink": {"node": "call", "call": "derive"},
                 "star_keyword": True, "source": "^kw$"},
            ]},
        ]},
        {"id": "scoped", "op": "any_of", "scope": "helper", "alternatives": [
            {"op": "count", "count": 1,
             "select": {"node": "call", "call": "only_in_helper"}},
        ]},
    ]
    result = _semantic_result("value_flow", source, relations)
    assert result["pass"], result["failures"]
    assert result["units_total"] == 2
    explicit = source.replace(
        "    kw['NW'] = nw\n    return derive(**kw)\n", "    return derive(NW=nw)\n"
    )
    assert _semantic_result("value_flow", explicit, relations)["pass"]
    literal = _semantic_result(
        "value_flow", source.replace("kw['NW'] = nw", "kw['NW'] = 8"), relations
    )
    assert not literal["pass"]
    assert literal["units_passed"] == 1
    assert literal["failures"][0].startswith("nw_reaches: no alternative holds: [explicit]")
    assert "| [expanded]" in literal["failures"][0]
    empty = _semantic_result(
        "value_flow", source, [{"id": "none", "op": "any_of", "alternatives": []}]
    )
    assert not empty["pass"]


def test_reference_megamoe_tree_passes_every_required_check():
    if not MEGAMOE_REFERENCE_ROOT.is_dir() or not MEGAMOE_REFERENCE_PLAN.is_file():
        pytest.skip("MegaMoE reference tree or MegaPlanIR JSON is not available")
    contract = MODULE.load_contract(MEGAMOE_CONTRACT)
    plan = json.loads(MEGAMOE_REFERENCE_PLAN.read_text())
    result = MODULE.evaluate(
        contract, MEGAMOE_REFERENCE_ROOT, MEGAMOE_REFERENCE_ROOT, None, None, plan,
    )
    assert result["input_errors"] == []
    assert result["plan_consistency_errors"] == []
    assert result["failed_required_checks"] == [], result["next_blocker"]
    assert result["semantic_failures"] == []
    assert result["surface_failures"] == []
    assert result["structural_compatible"]
    assert result["next_blocker"] == ""


def test_deviating_megamoe_candidate_still_fails_its_semantic_deviations():
    if not MEGAMOE_DEVIATING_CANDIDATE_ROOT.is_dir() or not MEGAMOE_REFERENCE_PLAN.is_file():
        pytest.skip("deviating MegaMoE candidate tree or MegaPlanIR JSON is not available")
    contract = MODULE.load_contract(MEGAMOE_CONTRACT)
    plan = json.loads(MEGAMOE_REFERENCE_PLAN.read_text())
    root = MEGAMOE_DEVIATING_CANDIDATE_ROOT
    result = MODULE.evaluate(contract, root, root, None, None, plan)
    deviations = {
        "flat_stripe_completion": "one_flat_g1_execute",
        "g2_scheduler_cfg": "sharded_head_address",
        "g2_skew_orientation": "five_four_orientation",
        "unified_g1_g2_loop": "discriminator_g2",
        "generation_lifecycle": "only_combine_head_resets",
        "progressive_token_readiness": "padding_target_is_generation_topk",
        "combine_readiness_is_strict": "lane_zero_waits_for_generation_target",
    }
    for check_id, unit in deviations.items():
        assert check_id in result["semantic_failures"], check_id
        failing_units = {
            re.split(r"[:.\[]", message, maxsplit=1)[0]
            for message in result["checks"][check_id]["failures"]
        }
        assert unit in failing_units, (check_id, sorted(failing_units))
    assert "preemption_cohort_and_skew" in {
        re.split(r"[:.\[]", message, maxsplit=1)[0]
        for message in result["checks"]["g2_scheduler_cfg"]["failures"]
    }
    for check_id in result["failed_required_checks"]:
        assert check_id in result["next_blocker"]
