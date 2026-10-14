"""Contracts for backend-neutral performance-decision normalization."""

from pathlib import Path

import yaml

import _normalize_performance_decisions as NORMALIZE


HERE = Path(__file__).resolve().parent
SCHEMA = NORMALIZE.load_yaml(HERE / "performance_axes" / "gemm.yaml")


def vector(language, config):
    return NORMALIZE.normalize_implementation(
        {"language": language, "config": config},
        SCHEMA,
    )


def test_all_backends_project_tile_and_split_k_to_the_same_axis():
    configs = {
        "flydsl": {
            "tile_m": 64, "tile_n": 128, "tile_k": 64, "split_k": 2,
        },
        "triton": {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "NUM_KSPLIT": 2,
        },
        "ck": {
            "MPerBlock": 64, "NPerBlock": 128, "KPerBlock": 64,
            "splitK": 2,
        },
        "asm": {
            "tileM": 64, "tileN": 128, "tileK": 64, "selectedsplitK": 2,
        },
    }
    for language, config in configs.items():
        got = vector(language, config)
        assert got["axes"]["workgroup_tile"]["values"] == {"k": 64, "m": 64, "n": 128}
        assert got["axes"]["workgroup_tile"]["completeness"] == "complete"
        assert got["axes"]["work_partition"]["values"]["split_k"] == 2
        assert got["backend_representation"] != "unregistered_backend"


def test_backend_spelling_and_comparison_limits_are_retained():
    got = vector("triton", {
        "config": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "num_stages": 2,
        },
    })
    tile = got["axes"]["workgroup_tile"]
    assert tile["source_fields"]["m"] == "config.BLOCK_SIZE_M"
    assert "directly comparable" in tile["comparison_contract"]
    assert got["axes"]["pipeline_schedule"]["values"]["stages"] == 2
    assert "not a speed claim" in got["contract"]


def test_opaque_ck_or_asm_config_is_not_fabricated():
    got = vector("asm", "runtime_selected")
    assert got["axes"] == {}
    assert got["limitations"] == [
        "configuration is opaque; no backend knobs were decoded",
    ]
    ck = vector("ck", {
        "layout": "unshuffled",
        "config": "default_compiled_instance",
        "identity_complete": True,
    })
    assert ck["axes"]["operand_layout"]["values"]["layout"] == "unshuffled"
    assert ck["limitations"] == [
        "backend config payload is opaque: default_compiled_instance",
    ]


def test_source_categories_unify_questions_without_unifying_representations():
    source = {
        "source_evidence": [
            {
                "category": "mfma_intrinsic", "language": "flydsl",
                "match": "rocdl.mfma_f32_16x16x16f16(", "evidence_id": "src_f",
            },
            {
                "category": "mfma_intrinsic", "language": "triton",
                "match": "tl.dot(", "evidence_id": "src_t",
            },
            {
                "category": "ck_instance", "language": "ck",
                "match": "DeviceGemmMultiD_Xdl_CShuffle_V3", "evidence_id": "src_c",
            },
            {
                "category": "asm_object", "language": "asm",
                "match": "kernel.co", "evidence_id": "src_a",
            },
        ],
    }
    got = NORMALIZE.summarize_source_evidence(source, SCHEMA)
    rows = got["observations"]
    compute = [row for row in rows if row["axis"] == "compute_instruction"]
    variants = [row for row in rows if row["axis"] == "implementation_variant"]
    assert {row["language"] for row in compute} == {"flydsl", "triton", "ck", "asm"}
    assert {row["representation"] for row in compute} == {
        "instruction_or_compiler_op", "template_family", "binary_identity",
    }
    assert {row["language"] for row in variants} == {"ck", "asm"}
    assert "not asserted equivalent" in got["contract"]


def test_current_source_corpus_covers_four_backends_on_shared_axes():
    source = NORMALIZE.load_yaml(HERE / "evidence" / "gemm_source.yaml")
    coverage = NORMALIZE.summarize_source_evidence(source, SCHEMA)["coverage"]
    compute = coverage["compute_instruction"]
    assert {"flydsl", "triton", "ck", "asm"} <= set(compute)
    assert coverage["workgroup_tile"]["flydsl"] > 0
    assert coverage["workgroup_tile"]["asm"] > 0
    assert coverage["pipeline_schedule"]["ck"] > 0
    assert coverage["architecture_capability"]["flydsl"] > 0
    assert coverage["configuration_constraints"]["flydsl"] > 0
    assert coverage["tunable_surface"]["triton"] > 0
    assert coverage["runtime_contract"]["asm"] > 0


def test_every_current_source_category_maps_to_at_least_one_development_axis():
    source = NORMALIZE.load_yaml(HERE / "evidence" / "gemm_source.yaml")
    summary = NORMALIZE.summarize_source_evidence(source, SCHEMA)
    assert summary["unmapped_categories"] == {}


def test_model_covers_every_curated_card_and_required_comparison_context():
    model = NORMALIZE.model_summary(SCHEMA)
    cards = yaml.safe_load((HERE / "decisions" / "gemm.yaml").read_text())["cards"]
    referenced = {
        ref
        for dependency in model["dependencies"]
        for ref in dependency.get("decision_refs") or []
    }
    assert {card["id"] for card in cards} <= referenced
    assert {
        "math_contract", "dtype", "gfx", "shape", "baseline",
        "accounting", "aiter_commit", "toolchain",
    } <= set(model["comparison_context"]["required"])
    assert {axis["kind"] for axis in model["axes"]} >= {
        "performance_choice", "hard_constraint", "search_space",
        "structural_choice", "integration_contract",
    }


def test_shipped_triton_config_becomes_a_normalized_seed_not_a_result():
    tuned = yaml.safe_load((HERE / "evidence" / "gemm_tuned_configs.yaml").read_text())
    profile = NORMALIZE.normalize_tuned_configs(
        tuned, SCHEMA, config_id="cfg_386a49c8c892741c",
    )[0]
    decisions = profile["performance_decisions"]
    assert decisions["axes"]["workgroup_tile"]["values"] == {
        "k": 64, "m": 256, "n": 256,
    }
    assert decisions["axes"]["pipeline_schedule"]["values"]["stages"] == 2
    assert "speed" not in decisions["axes"]["workgroup_tile"]["values"]
