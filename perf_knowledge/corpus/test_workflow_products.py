#!/usr/bin/env python3
"""Tests for the corpus validation against workflow products and for their registry ingest."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "benchmarks"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = load("validate_products", HERE / "_validate_products.py")
INGEST = load("ingest_workflow_products", HERE / "benchmarks" / "ingest_workflow_products.py")
SELECT_IMPL = load("select_implementations_for_products", HERE / "benchmarks" / "select_implementations.py")

KERNEL = """\
import flydsl
from flydsl.compiler import flyc
STAGES = 2
SPLIT_K = 4

def swizzle_xor16(i):
    return i

@flyc.kernel
def hgemm(w):
    rocdl.sched_barrier(0)
    load(w, cache_modifier="nt")
"""
BUCKETS = ("dec_bs1", "dec_bs6")


def seam_meta(graph):
    return {
        "short_name": "bf16_test_decode", "dtype": "bf16 in / bf16 weight / bf16 out, fp32 accumulate",
        "model_dims": {"N": 6144, "K": 2048},
        "cases": [
            {"name": "dec_bs1", "regime": "decode", "m": 6, "graph": graph, "timed": True},
            {"name": "dec_bs6", "regime": "decode", "m": 36, "graph": graph, "timed": True},
            {"name": "edge_M1", "m": 1, "timed": False},
        ],
    }


def make_run(root, *, graph=True, citations=None):
    run = root / "lane"
    (run / "workspace" / "baseline_src").mkdir(parents=True)
    (run / "workspace" / "baseline_src" / "base.py").write_text("import triton\n", encoding="utf-8")
    (run / "workspace" / "meta.json").write_text(json.dumps(seam_meta(graph)), encoding="utf-8")
    (run / "validation_workspace" / "kernel_src").mkdir(parents=True)
    (run / "validation_workspace" / "kernel_src" / "kernel.py").write_text(KERNEL, encoding="utf-8")
    patch = ["--- /dev/null", "+++ b/kernel_src/kernel.py", f"@@ -0,0 +1,{KERNEL.count(chr(10))} @@"]
    patch += ["+" + line for line in KERNEL.splitlines()]
    (run / "final_patch.diff").write_text("\n".join(patch) + "\n", encoding="utf-8")
    rows = [{"fp": "cand", "buckets": {name: {"sp": sp, "base": 0.007, "mde": 0.01, "regime": "decode"}
                                       for name, sp in zip(BUCKETS, speedups)}}
            for speedups in ((1.9, 1.3), (1.8, 1.32))]
    (run / "validation_workspace" / "_runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (run / "director_validation.json").write_text(json.dumps({
        "validation_status": "flagged", "correctness": "pass", "timing_basis": "unknown",
        "tech_lead_reported_speedup_geomean": 1.5, "director_verified_speedup_geomean": 1.52,
    }), encoding="utf-8")
    outcomes = [{"decision": card, "round": 1, "direction": f"r1_d{i}", "cited_then_verified": 1.4,
                 "status": "verified", "correctness": "pass", "became_winner": i == 0,
                 "incremental_vs_incumbent": 1.05}
                for i, card in enumerate(citations or [])]
    manifest = {
        "kernel": "test_lane", "captured_at": "t", "workspace_commit": "a" * 40,
        "observed_language": "flydsl", "flydsl_version": "0.3.2", "provider": "standalone_flydsl",
        "gfx": "gfx950", "aiter_commit": "c" * 40, "rocm_version": "7.2.0",
        "op_spec": {"dtype": "bf16", "math_contract": "C = X @ W^T", "regime": "decode",
                    "shapes": {"N": 6144, "K": 2048, "M": [6, 36]}},
        "software_detail": {"aiter_dirty": False},
        "measurement": {"method": "oracle", "reliable": True, "workload_aligned": False},
        "validation": {"status": "flagged", "correctness": "pass",
                       "per_case": [{"name": name, "speedup": 1.5} for name in BUCKETS]},
        "decision_outcomes": outcomes,
    }
    (run / "validation_environment.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return run


def test_a_run_product_is_validated_against_the_corpus(tmp_path):
    make_run(tmp_path, citations=[
        "gemm-streamed-operand-cache-policy",    # applies to any FlyDSL GEMM
        "flydsl-small-m-hgemm-family",           # M <= 16 only: one timed case of two
        "flydsl-a8-legality",                    # an 8-bit card on a bf16 task
        "cfg_0000000000000000",                  # a shipped ID nobody shipped
    ])
    result = VALIDATE.validate([str(tmp_path)], [], gfx="gfx950")
    assert result["errors"] == []
    report = result["products"][0]
    assert report["context"]["dtype"] == "bf16"
    assert report["context"]["operator_family"] == "dense_gemm"
    assert report["timed_cases"] == 2 and report["cases"] == 3
    assert report["design_traits"] == ["lds_staging", "split_k_hot_loop"]
    assert "flydsl-lds-pipeline-depth" in report["selection"]["unlocked_by_design_traits"]
    assert report["scan"]["introduced_basis"] == "patch_added_lines"
    assert {"cache_policy", "lds_stage", "lds_swizzle", "scheduling", "split_k"} <= set(
        report["scan"]["introduced_categories"])
    assert "cache_policy" not in report["coverage"]["uncovered_categories"]

    cited = report["citations"]["cited"]
    assert cited["gemm-streamed-operand-cache-policy"]["placement"] == "eligible_every_case"
    assert cited["gemm-streamed-operand-cache-policy"]["became_winner"] == 1
    assert cited["flydsl-small-m-hgemm-family"] == {
        **cited["flydsl-small-m-hgemm-family"], "placement": "eligible_some_cases", "cases": ["dec_bs1"]}
    assert cited["flydsl-a8-legality"]["placement"] == "rejected"
    assert cited["cfg_0000000000000000"]["placement"] == "unknown_shipped_id"
    assert report["citations"]["contradictions"] == ["cfg_0000000000000000", "flydsl-a8-legality"]

    ready = report["measured_readiness"]
    assert ready["ranking_basis"] == "graph_timed_lower_bound"
    assert ready["flag_reasons"] == ["timing_receipt_absent"]
    assert ready["single_ref_winners"] == ["gemm-streamed-operand-cache-policy"]
    assert result["summary"]["runs_with_ranking_basis"] == 1


def test_an_uncaptured_language_is_resolved_from_source_or_reported(tmp_path):
    run = make_run(tmp_path)
    manifest = yaml.safe_load((run / "validation_environment.yaml").read_text())
    manifest["observed_language"] = "not_captured"
    (run / "validation_environment.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    report = VALIDATE.validate([str(tmp_path)], [])["products"][0]
    assert report["context"]["target_language"] == "flydsl"
    assert report["language_basis"] == "detected"
    assert report["measured_readiness"]["observed_language"] is None
    assumed = VALIDATE.validate([str(tmp_path)], [], assumed_languages={str(run): "flydsl"})
    ready = assumed["products"][0]["measured_readiness"]
    assert ready["observed_language"] == "flydsl"
    assert ready["observed_language_basis"] == "assumed_by_operator"


def test_a_task_product_is_scanned_whole(tmp_path):
    task = tmp_path / "task"
    (task / "kernel_src").mkdir(parents=True)
    (task / "kernel_src" / "impl.py").write_text(KERNEL, encoding="utf-8")
    (task / "meta.json").write_text(json.dumps(seam_meta(True)), encoding="utf-8")
    report = VALIDATE.validate([], [str(task)], gfx="gfx950")["products"][0]
    assert report["kind"] == "task" and report["language_basis"] == "declared"
    assert report["scan"]["introduced_basis"] == "whole_source"
    assert "citations" not in report and "measured_readiness" not in report


def query(**overrides):
    values = {"suite": None, "language": "flydsl", "gfx": "gfx950", "flydsl_version": "0.3.2",
              "dtype": None, "baseline": None, "math_contract": None, "aiter_commit": "c" * 40,
              "torch_version": "2.9.1", "hip_version": "7.2", "context_fingerprint": None,
              "m": None, "n": None, "k": None, "top_k": 3, "task_dir": None}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("graph,status", [(True, "measured"), (False, "not_rankable")])
def test_ingested_products_rank_per_case_only_when_the_policy_allows(tmp_path, graph, status):
    run = make_run(tmp_path, graph=graph)
    registry = INGEST.ingest([str(tmp_path)], tmp_path / "results",
                             environment_extra={"torch": "2.9.1", "hip": "7.2"})
    assert registry["errors"] == []
    assert registry["records_count"] == 4          # two timed cases x two Director replicates
    assert {row["implementation"]["status"] for row in registry["records"]} == {status}
    assert {row["context"]["suite"] for row in registry["records"]} == {"bf16_test_decode"}

    got = SELECT_IMPL.select_task(registry, query(task_dir=run / "workspace"))
    assert [case["sig"] for case in got["cases"]] == list(BUCKETS)   # the untimed edge is not queried
    first = got["cases"][0]
    if status == "measured":
        speedup = first["candidates"][0]["implementation"]["measured_speedup"]
        assert speedup["replicates"] == 2 and speedup["median"] == pytest.approx(1.85)
    else:
        assert first["candidates"] == []


def test_a_query_without_environment_identity_returns_no_bundle(tmp_path):
    run = make_run(tmp_path)
    registry = INGEST.ingest([str(tmp_path)], tmp_path / "results",
                             environment_extra={"torch": "2.9.1", "hip": "7.2"})
    got = SELECT_IMPL.select_task(registry, query(task_dir=run / "workspace", torch_version=None))
    assert all(case["candidates"] == [] and "torch_version" in case["missing_query_context"]
               for case in got["cases"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
