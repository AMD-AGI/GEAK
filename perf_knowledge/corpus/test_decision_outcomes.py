#!/usr/bin/env python3
"""Tests for the standalone decision-outcome aggregator."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "aggregate_decision_outcomes", HERE / "_aggregate_decision_outcomes.py",
)
AGG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGG)
SELECT_SPEC = importlib.util.spec_from_file_location(
    "select_candidates_for_outcome_test", HERE / "_select_candidates.py",
)
SELECT = importlib.util.module_from_spec(SELECT_SPEC)
SELECT_SPEC.loader.exec_module(SELECT)


def write_yaml(run, outcomes, **context):
    run.mkdir(parents=True, exist_ok=True)
    data = {
        "kernel": "test-kernel",
        "captured_at": run.name,
        "workspace_commit": context.get("workspace_commit", "a" * 40),
        "observed_language": context.get("observed_language", "flydsl"),
        "flydsl_version": context.get("flydsl_version", "0.3.0"),
        "provider": context.get("provider", "standalone_flydsl"),
        "gfx": context.get("gfx", "gfx950"),
        "op_spec": context.get("op_spec", {"m": 128, "n": 256, "k": 64}),
        "measurement": {
            "method": "paired events",
            "warmup_iterations": 10,
            "benchmark_iterations": 100,
            "workload_aligned": False,
            "reliable": context.get("reliable", True),
        },
        "validation": {"status": context.get("validation_status", "accepted")},
        "decision_outcomes": outcomes,
    }
    path = run / "validation_environment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def outcome(decision, **overrides):
    row = {
        "decision": decision,
        "round": 1,
        "direction": "r1_d0",
        "specialty": "compute",
        "cited_then_verified": 1.25,
        "status": "verified",
        "correctness": "pass",
        "became_winner": True,
        "incremental_vs_incumbent": 1.10,
        "incremental_basis": "ratio_of_frozen_baseline_speedups",
    }
    row.update(overrides)
    return row


def test_single_card_direction_is_isolated_and_keeps_context(tmp_path):
    write_yaml(tmp_path / "run-a", [outcome("card-a")], op_spec={"dtype": "fp8"})

    got = AGG.aggregate_decision_outcomes([tmp_path])

    assert got["schema_version"] == 1
    assert got["runs_scanned"] == 1
    assert got["errors"] == []
    card = got["decisions"]["card-a"]
    assert card["attempts"] == card["unique_runs"] == card["winner_count"] == 1
    assert card["status_counts"] == {"verified": 1}
    assert card["correctness_counts"] == {"pass": 1}
    assert card["single_ref_direction_results"] == 1
    assert card["single_ref_seed_results"] == card["bundled_direction_results"] == 0
    assert card["contexts"][0]["observed_language"] == "flydsl"
    assert card["contexts"][0]["metric_kind"] == "geomean"
    assert card["contexts"][0]["measurement_reliable"] is True
    assert card["contexts"][0]["validation_status"] == "accepted"
    assert len(card["contexts"][0]["protocol_fingerprint"]) == 16
    assert card["single_ref_incremental_vs_incumbent"]["median"] == 1.1
    assert card["single_ref_speedup_vs_frozen_baseline"]["median"] == 1.25
    result = got["single_ref_direction_results"][0]
    assert result["decisions"] == ["card-a"]
    assert result["speedup_vs_frozen_baseline"] == 1.25
    assert result["incremental_vs_incumbent"] == 1.1
    assert result["attribution_scope"] == "single_ref_direction"
    assert result["quality_eligible_for_ranking"] is True
    assert "cited_then_verified" not in json.dumps(got)


def test_multi_card_direction_is_bundle_only(tmp_path):
    write_yaml(tmp_path / "run-b", [
        outcome("card-a"),
        outcome("card-b", cited_then_verified=1.25),
    ])

    got = AGG.aggregate([tmp_path])

    assert got["single_ref_direction_results"] == []
    assert len(got["bundled_direction_results"]) == 1
    bundle = got["bundled_direction_results"][0]
    assert bundle["decisions"] == ["card-a", "card-b"]
    assert bundle["decision_count"] == 2
    assert bundle["attribution_scope"] == "bundle_only"
    assert bundle["individual_decision_attribution"] is False
    assert bundle["quality_eligible_for_ranking"] is True
    context = {
        "target_language": "flydsl",
        "flydsl_version": "0.3.0",
        "gfx": "gfx950",
    }
    assert SELECT.measured_priors(got, context) == {}
    assert SELECT.compatible_bundles(got, context) == [bundle]
    for decision in ("card-a", "card-b"):
        assert got["decisions"][decision]["attempts"] == 1
        assert got["decisions"][decision]["winner_count"] == 0
        assert got["decisions"][decision]["single_ref_direction_results"] == 0
        assert got["decisions"][decision]["bundled_direction_results"] == 1


def test_bad_yaml_is_skipped_and_counted_as_an_error(tmp_path):
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    write_yaml(good, [outcome("card-a")])
    bad.mkdir()
    (bad / "validation_environment.yaml").write_text(
        "decision_outcomes: [unterminated", encoding="utf-8",
    )

    got = AGG.aggregate([tmp_path])

    assert got["runs_scanned"] == 1
    assert len(got["errors"]) == 1
    assert got["errors"][0]["path"].endswith("bad/validation_environment.yaml")
    assert got["decisions"]["card-a"]["attempts"] == 1


def test_json_manifest_is_accepted(tmp_path):
    run = tmp_path / "json-run"
    run.mkdir()
    (run / "validation_environment.json").write_text(json.dumps({
        "observed_language": "flydsl",
        "flydsl_version": "0.4.0",
        "provider": "aiter_vendored_flydsl",
        "gfx": "gfx942",
        "op_spec": {"shape": [8, 4096, 7168]},
        "decision_outcomes": [
            outcome("json-card", status="regression", correctness="pass",
                    became_winner=False, cited_then_verified=0.98),
        ],
    }), encoding="utf-8")

    got = AGG.aggregate([tmp_path])

    card = got["decisions"]["json-card"]
    assert card["status_counts"] == {"regression": 1}
    assert card["correctness_counts"] == {"pass": 1}
    assert got["single_ref_direction_results"][0]["context"]["gfx"] == "gfx942"
    assert got["single_ref_direction_results"][0]["speedup_vs_frozen_baseline"] == 0.98


def test_duplicate_and_overlapping_runs_paths_are_deduplicated(tmp_path):
    run = tmp_path / "nested" / "run"
    write_yaml(run, [outcome("card-a")])

    got = AGG.aggregate([tmp_path, tmp_path, tmp_path / "nested", run])

    assert got["runs_scanned"] == 1
    assert got["decisions"]["card-a"]["attempts"] == 1
    assert got["decisions"]["card-a"]["unique_runs"] == 1


def test_copied_manifest_is_deduplicated_by_run_identity(tmp_path):
    first = write_yaml(tmp_path / "original", [outcome("card-a")])
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "validation_environment.yaml").write_text(first.read_text())
    got = AGG.aggregate([tmp_path])
    assert got["runs_scanned"] == 1
    assert got["runs_deduplicated"] == 1


def test_cli_json_stdout_and_output_file(tmp_path, capsys):
    write_yaml(tmp_path / "run", [outcome("card-a")])

    assert AGG.main(["--runs", str(tmp_path), "--json"]) == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["runs_scanned"] == 1

    output = tmp_path / "aggregate.yaml"
    assert AGG.main(["--runs", str(tmp_path), "--output", str(output)]) == 0
    file_payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert file_payload["decisions"]["card-a"]["attempts"] == 1


def test_flagged_invalid_numbers_do_not_enter_ranking_statistics(tmp_path):
    write_yaml(
        tmp_path / "flagged",
        [outcome("card-a", incremental_vs_incumbent=float("nan"))],
        validation_status="flagged",
    )
    write_yaml(
        tmp_path / "bool",
        [outcome("card-b", incremental_vs_incumbent=True)],
    )
    got = AGG.aggregate([tmp_path])
    assert "single_ref_incremental_vs_incumbent" not in got["decisions"]["card-a"]
    assert "single_ref_incremental_vs_incumbent" not in got["decisions"]["card-b"]
    assert all(
        row["quality_eligible_for_ranking"] is False
        for row in got["single_ref_direction_results"]
    )


def test_accepted_regression_is_preserved_as_negative_evidence(tmp_path):
    write_yaml(tmp_path / "regression", [
        outcome(
            "card-a", status="regression", became_winner=False,
            cited_then_verified=0.8, incremental_vs_incumbent=0.9,
        ),
    ])
    got = AGG.aggregate([tmp_path])
    assert got["decisions"]["card-a"]["single_ref_incremental_vs_incumbent"]["median"] == 0.9
    assert got["single_ref_direction_results"][0]["quality_eligible_for_ranking"] is True


def test_unscoped_rows_do_not_accidentally_become_one_bundle(tmp_path):
    rows = [
        outcome("card-a", round=None, direction=None),
        outcome("card-b", round=None, direction=None),
    ]
    write_yaml(tmp_path / "unscoped", rows)
    got = AGG.aggregate([tmp_path])
    assert len(got["single_ref_direction_results"]) == 2
    assert got["bundled_direction_results"] == []


def test_single_ref_author_seed_stays_weak_seed_evidence(tmp_path):
    row = outcome(
        "card-a", round=None, direction="author_seed", phase="author_seed",
        attribution="single_ref_seed", incremental_vs_incumbent=None,
    )
    write_yaml(tmp_path / "seed", [row])
    got = AGG.aggregate([tmp_path])
    assert got["single_ref_direction_results"] == []
    assert got["single_ref_seed_results"][0]["attribution_scope"] == "single_ref_seed"
    assert got["decisions"]["card-a"]["single_ref_seed_results"] == 1


def _patch_manifest(path, **sections):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in sections.items():
        data.setdefault(key, {}).update(value)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _director(run, *, timing_basis="unknown", correctness="pass", claim=1.30, measured=1.31):
    (run / "director_validation.json").write_text(json.dumps({
        "validation_status": "flagged", "correctness": correctness, "timing_basis": timing_basis,
        "tech_lead_reported_speedup_geomean": claim, "director_verified_speedup_geomean": measured,
    }), encoding="utf-8")


def _task_meta(run, graph):
    workspace = run / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "meta.json").write_text(json.dumps({"cases": [
        {"name": "dec_bs1", "m": 6, "regime": "decode", "graph": graph, "timed": True},
        {"name": "edge_M1_single_token", "m": 1, "timed": False},
    ]}), encoding="utf-8")


def test_timing_receipt_flag_on_a_graph_timed_run_is_a_ranking_lower_bound(tmp_path):
    run = tmp_path / "graph"
    write_yaml(run, [outcome("card-a")], validation_status="flagged")
    _director(run)
    _task_meta(run, graph=True)
    got = AGG.aggregate([tmp_path])
    result = got["single_ref_direction_results"][0]
    assert result["context"]["flag_reasons"] == ["timing_receipt_absent"]
    assert result["context"]["flag_evidence"] == "director_validation.json"
    assert result["context"]["graph_timed"] is True
    assert result["context"]["ranking_basis"] == "graph_timed_lower_bound"
    assert result["quality_eligible_for_ranking"] is True
    assert got["decisions"]["card-a"]["single_ref_speedup_vs_frozen_baseline"]["median"] == 1.25


def test_eager_or_otherwise_flagged_runs_stay_out_of_ranking(tmp_path):
    cases = {
        "card-a": dict(graph=False, director={}),                      # eager: own host dispatch
        "card-b": dict(graph=True, director={"correctness": "fail"}),
        "card-c": dict(graph=True, director={"claim": 1.5, "measured": 1.2}),
        "card-d": dict(graph=True, director=None),                     # nothing recorded
    }
    for card, spec in cases.items():
        run = tmp_path / card
        write_yaml(run, [outcome(card)], validation_status="flagged")
        _task_meta(run, graph=spec["graph"])
        if spec["director"] is not None:
            _director(run, **spec["director"])
    got = AGG.aggregate([tmp_path])
    by_card = {row["decisions"][0]: row for row in got["single_ref_direction_results"]}
    assert by_card["card-a"]["context"]["graph_timed"] is False
    assert by_card["card-b"]["context"]["flag_reasons"] == ["correctness_fail", "timing_receipt_absent"]
    assert by_card["card-c"]["context"]["flag_reasons"] == ["below_claim", "timing_receipt_absent"]
    assert by_card["card-d"]["context"]["flag_evidence"] == "not_recorded"
    for card, row in by_card.items():
        assert row["context"]["ranking_basis"] is None, card
        assert row["quality_eligible_for_ranking"] is False, card
        assert "single_ref_speedup_vs_frozen_baseline" not in got["decisions"][card]


def test_structured_manifest_fields_take_precedence_and_name_the_metric(tmp_path):
    run = tmp_path / "structured"
    path = write_yaml(run, [outcome("card-a")], validation_status="flagged")
    _director(run, correctness="fail")          # superseded: the manifest states its own reasons
    _patch_manifest(
        path,
        validation={"timing_basis": "unprimed", "flag_reasons": ["timer_unprimed"],
                    "graph_timed": True},
        measurement={"primary_metric": "time_weighted"},
    )
    context = AGG.aggregate([tmp_path])["single_ref_direction_results"][0]["context"]
    assert context["flag_evidence"] == "manifest"
    assert context["ranking_basis"] == "graph_timed_lower_bound"
    assert context["metric_kind"] == "time_weighted"


def test_a_multi_shape_seam_run_informs_each_case_it_measured(tmp_path):
    """A seam run lists its timed M values and spells N/K upper-case; a workload query asks per
    case with lower-case keys and a bare regime. The prior must reach exactly the measured cases."""
    run = tmp_path / "seam"
    write_yaml(run, [outcome("flydsl-hgemm-config-space")], validation_status="flagged",
               op_spec={"dtype": "bf16", "regime": "decode, CUDA-graph captured",
                        "shapes": {"N": 6144, "K": 2048, "M": [3, 6, 12]}})
    _director(run)
    _task_meta(run, graph=True)
    outcomes_path = tmp_path / "outcomes.json"
    outcomes_path.write_text(json.dumps(AGG.aggregate([run])), encoding="utf-8")
    meta = {"dtype": "bf16", "model_dims": {"N": 6144, "K": 2048}, "cases": [
        {"name": "dec_bs1", "m": 6, "regime": "decode"},
        {"name": "dec_bs5", "m": 30, "regime": "decode"},
    ]}
    context = {"operator_family": "dense_gemm", "target_language": "flydsl", "gfx": "gfx950",
               "dtype": "bf16", "regime": None, "traits": []}
    result = SELECT.select_workload(HERE / "catalog.yaml", context, SELECT.workload_cases(meta),
                                    outcomes_path)
    by_sig = {case["sig"]: case for case in result["cases"]}
    assert by_sig["dec_bs1"]["measured_priors"]["flydsl-hgemm-config-space"]["median"] == 1.1
    assert by_sig["dec_bs5"]["measured_priors"] == {}


def _patch_manifest(path, **sections):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in sections.items():
        data.setdefault(key, {}).update(value)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _director(run, *, timing_basis="unknown", correctness="pass", claim=1.30, measured=1.31):
    """An older Director verdict: its reasons live in fields, not in the manifest."""
    (run / "director_validation.json").write_text(json.dumps({
        "validation_status": "flagged", "correctness": correctness, "timing_basis": timing_basis,
        "tech_lead_reported_speedup_geomean": claim, "director_verified_speedup_geomean": measured,
    }), encoding="utf-8")


def _task_meta(run, graph):
    workspace = run / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "meta.json").write_text(json.dumps({"cases": [
        {"name": "dec_bs1", "m": 6, "regime": "decode", "graph": graph, "timed": True},
        {"name": "edge_M1", "m": 1, "timed": False},
    ]}), encoding="utf-8")


def test_a_timing_receipt_flag_on_a_graph_timed_run_ranks_as_a_lower_bound(tmp_path):
    run = tmp_path / "graph"
    write_yaml(run, [outcome("card-a")], validation_status="flagged")
    _director(run)
    _task_meta(run, graph=True)

    got = AGG.aggregate([tmp_path])
    result = got["single_ref_direction_results"][0]
    assert result["context"]["flag_evidence"] == "director_validation.json"
    assert result["context"]["flag_reasons"] == ["timing_receipt_absent"]
    assert result["context"]["graph_timed"] is True
    assert result["context"]["ranking_basis"] == "graph_timed_lower_bound"
    assert result["quality_eligible_for_ranking"] is True
    assert got["decisions"]["card-a"]["single_ref_speedup_vs_frozen_baseline"]["median"] == 1.25


def test_eager_timed_or_otherwise_flagged_runs_stay_out_of_ranking(tmp_path):
    eager = tmp_path / "eager"
    write_yaml(eager, [outcome("card-a")], validation_status="flagged")
    _director(eager)
    _task_meta(eager, graph=False)
    wrong = tmp_path / "wrong"
    write_yaml(wrong, [outcome("card-b")], validation_status="flagged")
    _director(wrong, correctness="fail")
    _task_meta(wrong, graph=True)
    low = tmp_path / "low"
    write_yaml(low, [outcome("card-c")], validation_status="flagged")
    _director(low, claim=1.5, measured=1.2)
    _task_meta(low, graph=True)
    bare = tmp_path / "bare"
    write_yaml(bare, [outcome("card-d")], validation_status="flagged")

    got = AGG.aggregate([tmp_path])
    by_card = {row["decisions"][0]: row for row in got["single_ref_direction_results"]}
    assert by_card["card-a"]["context"]["graph_timed"] is False
    assert by_card["card-b"]["context"]["flag_reasons"] == ["correctness_fail", "timing_receipt_absent"]
    assert by_card["card-c"]["context"]["flag_reasons"] == ["below_claim", "timing_receipt_absent"]
    assert by_card["card-d"]["context"]["flag_evidence"] == "not_recorded"
    for card, row in by_card.items():
        assert row["context"]["ranking_basis"] is None, card
        assert row["quality_eligible_for_ranking"] is False, card
        assert "single_ref_speedup_vs_frozen_baseline" not in got["decisions"][card]


def test_structured_manifest_fields_take_precedence_and_name_the_metric(tmp_path):
    run = tmp_path / "structured"
    path = write_yaml(run, [outcome("card-a")], validation_status="flagged")
    _director(run, correctness="fail")
    _patch_manifest(
        path,
        validation={"timing_basis": "unprimed", "flag_reasons": ["timer_unprimed"], "graph_timed": True},
        measurement={"primary_metric": "time_weighted"},
    )

    context = AGG.aggregate([tmp_path])["single_ref_direction_results"][0]["context"]
    assert context["flag_evidence"] == "manifest"
    assert context["flag_reasons"] == ["timer_unprimed"]
    assert context["ranking_basis"] == "graph_timed_lower_bound"
    assert context["metric_kind"] == "time_weighted"


def test_a_multi_shape_seam_run_primes_the_cases_it_measured(tmp_path):
    """A seam run records N/K upper-case, every timed M as a list and prose after the regime."""
    run = tmp_path / "seam"
    write_yaml(run, [outcome("flydsl-hgemm-config-space")], validation_status="flagged", op_spec={
        "dtype": "bf16", "regime": "decode, CUDA-graph captured",
        "shapes": {"N": 6144, "K": 2048, "M": [3, 6, 12]},
    })
    _director(run)
    _task_meta(run, graph=True)
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps(AGG.aggregate([run])), encoding="utf-8")

    meta = {"dtype": "bf16", "model_dims": {"N": 6144, "K": 2048}, "cases": [
        {"name": "dec_bs1", "regime": "decode", "m": 6},
        {"name": "dec_bs5", "regime": "decode", "m": 30},
        {"name": "pre_M1936", "regime": "prefill", "m": 1936},
    ]}
    context = {"operator_family": "dense_gemm", "target_language": "flydsl", "gfx": "gfx950",
               "dtype": "bf16", "regime": None, "traits": []}
    result = SELECT.select_workload(HERE / "catalog.yaml", context, SELECT.workload_cases(meta),
                                    outcomes)
    priors = {case["sig"]: case["measured_priors"] for case in result["cases"]}
    assert priors["dec_bs1"]["flydsl-hgemm-config-space"]["median"] == pytest.approx(1.10)
    assert priors["dec_bs5"] == {}      # M=30 was not measured
    assert priors["pre_M1936"] == {}    # neither the shape nor the regime was measured


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
