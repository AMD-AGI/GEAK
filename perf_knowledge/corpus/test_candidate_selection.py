"""Contracts for the corpus catalog and deterministic card matcher."""

import json
from pathlib import Path

import pytest
import yaml

import _select_candidates as SELECT


HERE = Path(__file__).resolve().parent


def test_catalog_paths_exist_and_family_names_are_unique():
    catalog = yaml.safe_load((HERE / "catalog.yaml").read_text())
    seen = set()
    for family in catalog["families"]:
        names = [family["id"], *(family.get("patterns") or [])]
        assert not (seen & set(names)), f"ambiguous catalog names: {seen & set(names)}"
        seen.update(names)
        for field in (
            "source_evidence", "tuned_evidence", "decisions", "source_document",
            "decision_document", "benchmark_manifest", "performance_axes",
            "performance_normalizer",
        ):
            assert (HERE / family[field]).is_file(), (family["id"], field, family[field])


def test_gfx_constraint_and_eligible_performance_candidates_are_separate():
    result = SELECT.select(HERE / "catalog.yaml", {
        "operator_family": "scaled_quant_gemm",
        "target_language": "flydsl",
        "gfx": "gfx942",
        "dtype": "fp8_e4m3fnuz_blockscale",
        "regime": "prefill",
        "bottleneck": "compute",
        "m": 32768,
        "n": 4096,
        "k": 2048,
        "traits": ["lds_staging"],
    })
    got = {row["id"]: row["type"] for row in result["eligible"]}
    assert got["flydsl-gfx942-path-gates"] == "constraint"
    assert got["flydsl-xor16-lds-layout-consistency"] == "performance_candidate"
    assert "flydsl-small-m-hgemm-family" not in got
    assert {axis["id"] for axis in result["performance_axes"]} >= {
        "compute_instruction", "workgroup_tile", "pipeline_schedule", "operand_layout",
    }
    assert result["performance_model"]["dependencies"]
    assert "math_contract" in result["performance_model"]["comparison_context"]["required"]


def test_small_m_family_is_rejected_on_gfx942_and_selected_on_gfx950():
    base = {
        "operator_family": "gemm",
        "target_language": "flydsl",
        "dtype": "bf16",
        "regime": "decode",
        "bottleneck": "latency",
        "m": 8,
        "n": 4096,
        "k": 7168,
        "traits": [],
    }
    rejected = SELECT.select(HERE / "catalog.yaml", {**base, "gfx": "gfx942"})
    selected = SELECT.select(HERE / "catalog.yaml", {**base, "gfx": "gfx950"})
    assert "flydsl-small-m-hgemm-family" in {row["id"] for row in rejected["rejected"]}
    assert "flydsl-small-m-hgemm-family" in {row["id"] for row in selected["eligible"]}


def test_required_trait_defers_a_card_instead_of_calling_it_slow():
    card = {
        "match": {
            "operator_families": ["gemm"],
            "target_languages": ["flydsl"],
            "requires": ["lds_staging"],
        },
    }
    status, reasons = SELECT.classify_card(card, {
        "operator_family": "gemm",
        "target_language": "flydsl",
        "traits": [],
    })
    assert status == "deferred"
    assert any("lds_staging" in reason for reason in reasons)


@pytest.mark.parametrize(("given", "stored"), [("fp16", "f16"), ("bfloat16", "bf16")])
def test_dtype_aliases_match(given, stored):
    card = {
        "match": {
            "operator_families": ["gemm"],
            "target_languages": ["flydsl"],
            "dtypes": [stored],
        },
    }
    status, _ = SELECT.classify_card(card, {
        "operator_family": "gemm",
        "target_language": "flydsl",
        "dtype": given,
        "traits": [],
    })
    assert status == "eligible"


def test_compatible_single_ref_outcomes_rank_but_bundles_do_not(tmp_path):
    context = {
        "operator_family": "scaled_quant_gemm",
        "target_language": "flydsl",
        "gfx": "gfx942",
        "flydsl_version": "0.3.0",
        "dtype": "fp8_e4m3fnuz_blockscale",
        "regime": "prefill",
        "bottleneck": "compute",
        "m": 32768,
        "n": 4096,
        "k": 1024,
        "traits": ["lds_staging", "split_k_hot_loop"],
    }
    exact = {
        "observed_language": "flydsl",
        "flydsl_version": "0.3.0",
        "gfx": "gfx942",
        "op_spec": {
            "dtype": "fp8_e4m3fnuz_blockscale", "regime": "prefill",
            "m": 32768, "n": 4096, "k": 1024,
        },
    }
    outcomes = {
        "single_ref_direction_results": [
            {
                "run": "a", "decisions": ["flydsl-xor16-lds-layout-consistency"],
                "incremental_vs_incumbent": 1.05, "speedup_vs_frozen_baseline": 1.1,
                "attribution_scope": "single_ref_direction",
                "quality_eligible_for_ranking": True, "context": exact,
            },
            {
                "run": "b", "decisions": ["flydsl-splitk-hot-loop-schedule-bundle"],
                "incremental_vs_incumbent": 1.20, "speedup_vs_frozen_baseline": 1.3,
                "attribution_scope": "single_ref_direction",
                "quality_eligible_for_ranking": True, "context": exact,
            },
            {
                "run": "wrong-gfx", "decisions": ["flydsl-xor16-lds-layout-consistency"],
                "incremental_vs_incumbent": 9.0,
                "attribution_scope": "single_ref_direction",
                "quality_eligible_for_ranking": True,
                "context": {**exact, "gfx": "gfx950"},
            },
        ],
        "bundled_direction_results": [{
            "run": "bundle",
            "decisions": [
                "flydsl-xor16-lds-layout-consistency",
                "flydsl-splitk-hot-loop-schedule-bundle",
            ],
            "incremental_vs_incumbent": 4.0,
            "quality_eligible_for_ranking": True,
            "context": exact,
        }],
    }
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps(outcomes))

    result = SELECT.select(HERE / "catalog.yaml", context, path)
    performance = [
        row for row in result["eligible"] if row["type"] == "performance_candidate"
    ]
    assert [row["id"] for row in performance[:2]] == [
        "flydsl-splitk-hot-loop-schedule-bundle",
        "flydsl-xor16-lds-layout-consistency",
    ]
    assert performance[0]["measured_prior"]["median"] == 1.20
    assert performance[1]["measured_prior"]["median"] == 1.05
    assert performance[1]["measured_prior"]["count"] == 1
    assert len(result["measured_bundles"]) == 1
    assert result["measured_bundles"][0]["incremental_vs_incumbent"] == 4.0


def test_outcome_without_matching_language_or_gfx_is_not_a_prior():
    outcomes = {
        "single_ref_direction_results": [{
            "run": "old",
            "decisions": ["card"],
            "incremental_vs_incumbent": 2.0,
            "attribution_scope": "single_ref_direction",
            "quality_eligible_for_ranking": True,
            "context": {
                "observed_language": "triton",
                "flydsl_version": "0.3.0",
                "gfx": "gfx942",
                "op_spec": {},
            },
        }],
    }
    assert SELECT.measured_priors(outcomes, {
        "target_language": "flydsl", "flydsl_version": "0.3.0", "gfx": "gfx942",
    }) == {}


def test_missing_recorded_shape_cannot_match_a_specific_request():
    row = {
        "run": "missing-shape",
        "decisions": ["card"],
        "incremental_vs_incumbent": 1.5,
        "attribution_scope": "single_ref_direction",
        "quality_eligible_for_ranking": True,
        "context": {
            "observed_language": "flydsl", "flydsl_version": "0.3.0",
            "gfx": "gfx942", "op_spec": {},
        },
    }
    assert SELECT.measured_priors(
        {"single_ref_direction_results": [row]},
        {
            "target_language": "flydsl", "flydsl_version": "0.3.0", "gfx": "gfx942",
            "m": 8, "n": 256, "k": 128,
        },
    ) == {}


def test_measured_regression_ranks_after_unmeasured_candidate(tmp_path):
    context = {
        "operator_family": "scaled_quant_gemm", "target_language": "flydsl",
        "gfx": "gfx942", "flydsl_version": "0.3.0",
        "dtype": "fp8_e4m3fnuz_blockscale", "regime": "prefill",
        "bottleneck": "compute", "m": 32768, "n": 4096, "k": 1024,
        "traits": ["lds_staging", "split_k_hot_loop"],
    }
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps({"single_ref_direction_results": [{
        "run": "loss",
        "decisions": ["flydsl-xor16-lds-layout-consistency"],
        "incremental_vs_incumbent": 0.8,
        "attribution_scope": "single_ref_direction",
        "quality_eligible_for_ranking": True,
        "context": {
            "observed_language": "flydsl", "flydsl_version": "0.3.0",
            "gfx": "gfx942",
            "op_spec": {
                "dtype": "fp8_e4m3fnuz_blockscale", "regime": "prefill",
                "m": 32768, "n": 4096, "k": 1024,
            },
        },
    }]}))
    result = SELECT.select(HERE / "catalog.yaml", context, path)
    performance = [
        row for row in result["eligible"] if row["type"] == "performance_candidate"
    ]
    ids = [row["id"] for row in performance]
    # The measured regression sorts after every unmeasured candidate, however many there are.
    assert ids[-1] == "flydsl-xor16-lds-layout-consistency"
    assert "flydsl-splitk-hot-loop-schedule-bundle" in ids[:-1]
    assert all("measured_prior" not in row for row in performance[:-1])


def test_a_both_regime_workload_keeps_decode_only_cards():
    """`both` means the workload has decode AND prefill cases; rejecting a decode card for it dropped
    the small-M family from every mixed task."""
    card = {"match": {"operator_families": ["gemm"], "target_languages": ["flydsl"],
                      "regimes": ["decode"]}}
    status, _ = SELECT.classify_card(card, {
        "operator_family": "gemm", "target_language": "flydsl", "regime": "both", "traits": [],
    })
    assert status == "eligible"


def _dtype_status(card_dtypes, given):
    card = {"match": {"operator_families": ["gemm"], "target_languages": ["flydsl"],
                      "dtypes": card_dtypes}}
    return SELECT.classify_card(card, {
        "operator_family": "gemm", "target_language": "flydsl", "dtype": given, "traits": [],
    })[0]


@pytest.mark.parametrize("given", ["fp8_e4m3fnuz", "fp8_e4m3_fnuz", "float8_e4m3fnuz", "FP8_E4M3FNUZ"])
def test_fp8_spellings_are_canonicalised_then_matched_exactly(given):
    assert _dtype_status(["fp8_e4m3fnuz"], given) == "eligible"


def test_the_scale_contract_suffix_is_a_different_dtype():
    """A per-token card read against a block-scale task is a different math contract; a `fp8*`
    family once made every A8 card eligible for block-scale tasks."""
    assert _dtype_status(["fp8_e4m3fnuz"], "fp8_e4m3fnuz_blockscale") == "rejected"
    assert _dtype_status(["fp8_e4m3fnuz_blockscale"], "fp8_e4m3_fnuz_blockscale") == "eligible"


def test_every_card_dtype_is_already_canonical():
    """A card spelling that canonicalises to something else can never match a request."""
    cards = yaml.safe_load((HERE / "decisions" / "gemm.yaml").read_text())["cards"]
    for card in cards:
        for dtype in card["match"].get("dtypes") or []:
            assert SELECT.normalize_dtype(dtype) == dtype, (card["id"], dtype)


def test_an_undefined_trait_is_named_rather_than_silently_deferring():
    result = SELECT.select(HERE / "catalog.yaml", {
        "operator_family": "gemm", "target_language": "flydsl", "gfx": "gfx950", "dtype": "bf16",
        "traits": ["lds_staging", "lds_stageing"],
    })
    assert result["undefined_traits"] == ["lds_stageing"]


def test_every_taxonomy_gemm_operator_resolves_to_the_gemm_family():
    catalog = yaml.safe_load((HERE / "catalog.yaml").read_text())
    for operator in ("dense_gemm", "scaled_quant_gemm", "batched_gemm", "splitk_streamk_gemm",
                     "skinny_gemv_decode", "gemm_epilogue_fused", "grouped_gemm_moe"):
        assert SELECT.resolve_family(catalog, operator)["id"] == "gemm", operator


def test_a_workload_is_classified_case_by_case():
    meta = {"dtype": "bf16", "cases": [
        {"sig": "decode_m8", "m": 8, "n": 4096, "k": 4096},
        {"sig": "prefill_m16384", "m": 16384, "n": 4096, "k": 4096},
    ]}
    context = {"operator_family": "dense_gemm", "target_language": "flydsl", "gfx": "gfx950",
               "dtype": "bf16", "regime": None, "traits": []}
    result = SELECT.select_workload(HERE / "catalog.yaml", context, SELECT.workload_cases(meta))
    by_sig = {case["sig"]: case for case in result["cases"]}
    assert by_sig["decode_m8"]["regime"] == "decode"
    assert "flydsl-small-m-hgemm-family" in by_sig["decode_m8"]["eligible"]
    assert "flydsl-small-m-hgemm-family" not in by_sig["prefill_m16384"]["eligible"]
    assert result["eligible_in_some_cases"]["flydsl-small-m-hgemm-family"] == ["decode_m8"]
    assert "flydsl-hgemm-config-space" in result["eligible_in_every_case"]


def test_a_seam_meta_with_model_dims_and_explicit_regimes_is_a_workload():
    """Seam tasks extracted from a served model keep N/K once in `model_dims`, name each case,
    state its regime, and mark correctness-only edges untimed; `--workload` has to read that."""
    meta = {
        "dtype": "bf16 in / bf16 weight / bf16 out, fp32 accumulate",
        "model_dims": {"N": 6144, "K": 2048, "hidden_size": 6144},
        "cases": [
            {"name": "dec_bs1", "m": 6, "regime": "decode", "timed": True},
            {"name": "pre_M1936", "m": 1936, "regime": "prefill", "timed": True},
            {"name": "edge_M1_single_token", "m": 1, "timed": False},
        ],
    }
    cases = SELECT.workload_cases(meta)
    assert [(c["sig"], c["m"], c["n"], c["k"], c["regime"], c["timed"]) for c in cases] == [
        ("dec_bs1", 6, 6144, 2048, "decode", True),
        ("pre_M1936", 1936, 6144, 2048, "prefill", True),
        ("edge_M1_single_token", 1, 6144, 2048, None, False),
    ]
    assert SELECT.workload_dtype(meta) == "bf16"
    context = {"operator_family": "dense_gemm", "target_language": "flydsl", "gfx": "gfx950",
               "dtype": SELECT.workload_dtype(meta), "regime": None, "traits": []}
    result = SELECT.select_workload(HERE / "catalog.yaml", context, cases)
    assert [c["timed"] for c in result["cases"]] == [True, True, False]
    assert "flydsl-small-m-hgemm-family" in result["eligible_in_some_cases"]
    # The untimed edge is classified but does not shrink what holds for the timed workload.
    assert result["eligible_in_some_cases"]["flydsl-small-m-hgemm-family"] == ["dec_bs1"]


def test_workload_dtype_is_canonical_single_or_absent():
    assert SELECT.workload_dtype({"dtype": "fp8_e4m3fnuz_blockscale"}) == "fp8_e4m3fnuz_blockscale"
    assert SELECT.workload_dtype({"dtype": "bfloat16"}) == "bf16"
    assert SELECT.workload_dtype({"dtype": "fp16 in, fp32 accumulate, fp16 out"}) == "f16"
    assert SELECT.workload_dtype({"dtype": "bf16 activations / int4 weight"}) is None
    assert SELECT.workload_dtype({}) is None


def test_a_seam_meta_workload_reads_model_dims_regime_and_untimed_edges():
    """Seam metas extracted from a served model: `name`/`m`/`regime` per case, N and K once in
    `model_dims`, correctness-only edges `timed: false`, and a dtype stated as a whole contract."""
    meta = {
        "dtype": "bf16 in / bf16 weight / bf16 out, fp32 accumulate",
        "model_dims": {"N": 6144, "K": 2048},
        "cases": [
            {"name": "dec_bs1", "regime": "decode", "m": 6, "timed": True},
            {"name": "pre_M16955", "regime": "prefill", "m": 16955, "timed": True},
            {"name": "edge_M1_single_token", "m": 1, "timed": False},
        ],
    }
    cases = SELECT.workload_cases(meta)
    assert [(c["sig"], c["m"], c["n"], c["k"], c["regime"], c["timed"]) for c in cases] == [
        ("dec_bs1", 6, 6144, 2048, "decode", True),
        ("pre_M16955", 16955, 6144, 2048, "prefill", True),
        ("edge_M1_single_token", 1, 6144, 2048, None, False),
    ]
    assert SELECT.workload_dtype(meta) == "bf16"
    context = {"operator_family": "dense_gemm", "target_language": "flydsl", "gfx": "gfx950",
               "dtype": SELECT.workload_dtype(meta), "regime": None, "traits": []}
    result = SELECT.select_workload(HERE / "catalog.yaml", context, cases)
    by_sig = {case["sig"]: case for case in result["cases"]}
    assert by_sig["edge_M1_single_token"]["timed"] is False
    assert "flydsl-small-m-hgemm-family" in by_sig["dec_bs1"]["eligible"]
    # The untimed M=1 edge is classified but does not narrow what holds for the timed workload.
    assert result["eligible_in_some_cases"]["flydsl-small-m-hgemm-family"] == ["dec_bs1"]


def test_workload_dtype_names_one_operand_dtype_or_none():
    assert SELECT.workload_dtype({"dtype": "fp8_e4m3fnuz_blockscale"}) == "fp8_e4m3fnuz_blockscale"
    assert SELECT.workload_dtype({"dtype": "bfloat16"}) == "bf16"
    assert SELECT.workload_dtype({"dtype": "bf16 in / fp8_e4m3fnuz weight, fp32 accumulate"}) is None
    assert SELECT.workload_dtype({}) is None


def test_a_malformed_shape_is_refused():
    with pytest.raises(ValueError):
        SELECT.workload_cases(None, ["8x4096"])


def test_missing_gfx_defers_explicit_exclusion():
    card = {
        "match": {
            "operator_families": ["gemm"], "target_languages": ["flydsl"],
            "exclude_gfx": ["gfx942"],
        },
    }
    status, reasons = SELECT.classify_card(card, {
        "operator_family": "gemm", "target_language": "flydsl", "traits": [],
    })
    assert status == "deferred"
    assert any("missing gfx" in reason for reason in reasons)
