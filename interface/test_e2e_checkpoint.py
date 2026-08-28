#!/usr/bin/env python3
"""CPU-only schema-v2 E2E checkpoint recovery tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e_checkpoint", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(eval_dir: Path, level: str, *, duplicate_slot: bool = False) -> dict:
    asset = eval_dir / "checkpoint_assets" / "bench_e2e.sh"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text("#!/bin/sh\n", encoding="utf-8")
    checkpoint = {
        "schema_version": 2,
        "checkpoint_type": "e2e_validation",
        "committed": True,
        "eval_dir": str(eval_dir),
        "phase": "TuningSkillset" if level == "tuning_skillset" else "Finalize",
        "validation_level": level,
        "gate": "accepted",
        "validation_status": "accepted_tuning",
        "baseline_throughput_tok_s": 1000.0,
        "final_throughput_tok_s": 1100.0,
        "throughput_speedup": 1.1,
        "baseline_config": {"flags": "", "env": ""},
        "accepted_config": {"flags": "--fp8", "env": "AITER=1"},
        "accepted_kernels": [{
            "short_name": "a8w8_gemm",
            "kernel_slot": "aiter:a8w8_gemm",
            "from_tuning_skillset": level == "tuning_skillset",
        }],
        "accepted_heads": [],
        "final_launch_script": {"snapshot": "checkpoint_assets/bench_e2e.sh"},
        "measurement": {
            "measurement_mode": "isolated_server",
            "workload": {"isl": 1024, "osl": 1024, "conc": 64},
            "legs": [{"arm": "A", "usable": True}, {"arm": "B", "usable": True}],
            "correctness": {"gate": "pass"},
            "acceptance": {"gain_exceeds_noise": True, "correctness_passed": True},
        },
        "stack": {
            "stack_after_digest": "stack-after",
            "kernel_slots": [
                {"kernel_slot": "aiter:a8w8_gemm", "selected": True},
            ],
        },
        "replay": {"requires_server_restart": True},
        "integrity": {
            "checkpoint_assets": [{
                "snapshot": "checkpoint_assets/bench_e2e.sh",
                "sha256": _sha(asset),
            }],
        },
    }
    if duplicate_slot:
        checkpoint["stack"]["kernel_slots"].append(
            {"kernel_slot": "aiter:a8w8_gemm", "selected": True}
        )
    checkpoint["checkpoint_sha256"] = rx._checkpoint_digest(checkpoint)
    return checkpoint


def _write(eval_dir: Path, relative: str, checkpoint: dict) -> None:
    path = eval_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint), encoding="utf-8")


def test_recovers_tuning_checkpoint_and_preserves_kernel_metadata(tmp_path):
    eval_dir = tmp_path / "e2e"
    checkpoint = _checkpoint(eval_dir, "tuning_skillset")
    _write(eval_dir, "tuning/e2e_validation.json", checkpoint)

    recovered = rx._recover_e2e_validation_checkpoint(eval_dir)

    assert recovered["recovered_e2e_checkpoint_level"] == "tuning_skillset"
    assert recovered["throughput_speedup"] == 1.1
    assert recovered["accepted_config"]["env"] == "AITER=1"
    assert recovered["recovery_evidence"]["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    journey = rx.build_kernel_journey(recovered, recovered)
    kernel = journey["kernels"][0]
    assert kernel["source_phase"] == "TuningSkillset"
    assert kernel["dispatch"]["task_group"] == "tuning_skillset"
    assert kernel["e2e"]["e2e_gain_scope"] == "tuning_stack_unattributed"


def test_final_checkpoint_outranks_tuning_checkpoint(tmp_path):
    eval_dir = tmp_path / "e2e"
    _write(eval_dir, "tuning/e2e_validation.json", _checkpoint(eval_dir, "tuning_skillset"))
    final = _checkpoint(eval_dir, "final_pair")
    final["validation_status"] = "provisional_final_pair"
    final["checkpoint_sha256"] = rx._checkpoint_digest(final)
    _write(eval_dir, "final/e2e_validation.json", final)

    recovered = rx._recover_e2e_validation_checkpoint(eval_dir)

    assert recovered["recovered_e2e_checkpoint_level"] == "final_pair"
    assert recovered["recovered_intermediate"] is False


def test_rejects_tampered_assets_and_ambiguous_slots(tmp_path):
    eval_dir = tmp_path / "e2e"
    checkpoint = _checkpoint(eval_dir, "tuning_skillset", duplicate_slot=True)
    _write(eval_dir, "tuning/e2e_validation.json", checkpoint)
    assert rx._recover_e2e_validation_checkpoint(eval_dir) is None


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda c: c.update(schema_version=1), "unsupported schema_version"),
        (lambda c: c.update(checkpoint_type="other"), "wrong checkpoint_type"),
        (lambda c: c.update(validation_level="config_sweep"), "unexpected validation_level"),
        (lambda c: c.update(committed=False), "checkpoint is not a committed accepted result"),
        (lambda c: c.update(eval_dir="/other"), "eval_dir does not match recovery target"),
        (lambda c: c.update(throughput_speedup=0), "non-positive throughput fields"),
        (lambda c: c.update(final_throughput_tok_s=1000.0, throughput_speedup=1.0),
         "accepted checkpoint has no positive gain"),
        (lambda c: c.update(throughput_speedup=1.2), "throughput_speedup does not match throughput pair"),
        (lambda c: c.pop("replay"), "missing replay"),
        (lambda c: c.update(accepted_heads={}), "missing accepted_heads"),
        (lambda c: c["measurement"].pop("workload"), "missing measurement.workload"),
        (lambda c: c["measurement"].update(measurement_mode="shared_server"), "unsupported measurement mode"),
        (lambda c: c["measurement"].update(legs=[]), "missing measurement legs"),
        (lambda c: c["measurement"].pop("acceptance"), "missing measurement.acceptance"),
        (lambda c: c["measurement"]["acceptance"].update(gain_exceeds_noise=False),
         "checkpoint gain did not exceed noise"),
        (lambda c: c["measurement"]["acceptance"].update(correctness_passed=False),
         "checkpoint correctness did not pass"),
        (lambda c: c["stack"].update(kernel_slots=[{"selected": True}]), "selected kernel has no kernel_slot"),
        (lambda c: c["integrity"].update(checkpoint_assets={}), "missing integrity.checkpoint_assets"),
        (lambda c: c["integrity"].update(checkpoint_assets=[None]), "invalid checkpoint asset"),
        (lambda c: c["integrity"].update(checkpoint_assets=[{"snapshot": "../outside", "sha256": "x"}]), "missing checkpoint asset"),
    ],
)
def test_checkpoint_validator_rejects_each_required_contract(tmp_path, mutate, reason):
    eval_dir = tmp_path / "e2e"
    checkpoint = _checkpoint(eval_dir, "tuning_skillset")
    mutate(checkpoint)
    checkpoint["checkpoint_sha256"] = rx._checkpoint_digest(checkpoint)
    valid, observed = rx._valid_e2e_checkpoint(checkpoint, eval_dir, {"tuning_skillset"})
    assert not valid
    assert observed == reason


def test_checkpoint_digest_path_and_path_metadata_helpers(tmp_path):
    eval_dir = tmp_path / "e2e"
    checkpoint = _checkpoint(eval_dir, "tuning_skillset")
    assert rx._checkpoint_digest(checkpoint) == checkpoint["checkpoint_sha256"]
    assert rx._checkpoint_asset_path(eval_dir, "") is None
    assert rx._checkpoint_asset_path(eval_dir, "../outside") is None
    asset = eval_dir / "checkpoint_assets" / "bench_e2e.sh"
    assert rx._checkpoint_asset_path(eval_dir, "checkpoint_assets/bench_e2e.sh") == asset.resolve()
    assert rx._checkpoint_path_value("launch.sh") == "launch.sh"
    assert rx._checkpoint_path_value({"path": "launch.sh"}) == "launch.sh"
    assert rx._checkpoint_path_value({"snapshot": "checkpoint_assets/launch.sh"}) == "checkpoint_assets/launch.sh"
    assert rx._checkpoint_path_value(None) == ""


def test_checkpoint_parent_chain_must_match_the_parent_document(tmp_path):
    eval_dir = tmp_path / "e2e"
    parent = _checkpoint(eval_dir, "config_sweep")
    _write(eval_dir, "config/e2e_validation.json", parent)
    child = _checkpoint(eval_dir, "tuning_skillset")
    child["parent_checkpoint"] = {
        "path": "config/e2e_validation.json",
        "checkpoint_sha256": parent["checkpoint_sha256"],
    }
    child["checkpoint_sha256"] = rx._checkpoint_digest(child)

    assert rx._valid_e2e_checkpoint(child, eval_dir, {"tuning_skillset"}) == (True, "")

    parent["final_throughput_tok_s"] = 1200.0
    _write(eval_dir, "config/e2e_validation.json", parent)
    valid, reason = rx._valid_e2e_checkpoint(child, eval_dir, {"tuning_skillset"})
    assert not valid
    assert reason == "parent checkpoint digest mismatch"


def test_recovers_legacy_tuning_composite_as_provisional(tmp_path):
    eval_dir = tmp_path / "e2e_interrupted"
    deploy = eval_dir / "tuning" / "deploy"
    deploy.mkdir(parents=True)
    (deploy / "deploy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (deploy / "MANIFEST.json").write_text(json.dumps({
        "apply_env": "AITER_CONFIG=/tmp/gemm.csv",
        "ops_tuned": [{
            "kernel_id": "gemm_a8w8",
            "kernel_slot": "aiter:gemm_a8w8",
            "backend": "aiter",
        }],
    }), encoding="utf-8")
    ab = eval_dir / "tuning" / "ab"
    ab.mkdir()
    (ab / "ab_summary.json").write_text(json.dumps({
        "pre_median": 1000.0, "post_median": 1100.0,
        "median_pair_delta_pct": 10.0,
        "n_pairs": 1,
        "legs": [
            {"arm": "A", "usable": True, "mode": "isolated_server", "tput": 1000.0, "hits": 0},
            {"arm": "B", "usable": True, "mode": "isolated_server", "tput": 1100.0, "hits": 5},
        ],
    }), encoding="utf-8")

    recovered = rx._recover_workflow_return(tmp_path)
    normalized = rx.normalize_result({}, recovered)

    assert recovered["validation_status"] == "recovered_tuning_skillset_legacy_provisional"
    assert recovered["accepted_kernels"][0]["kernel_id"] == "gemm_a8w8"
    assert normalized["result_source"] == "disk_tuning_skillset_legacy_provisional"

    checkpoint = _checkpoint(eval_dir, "tuning_skillset")
    asset = eval_dir / "checkpoint_assets" / "bench_e2e.sh"
    asset.write_text("changed\n", encoding="utf-8")
    _write(eval_dir, "tuning/e2e_validation.json", checkpoint)
    assert rx._recover_e2e_validation_checkpoint(eval_dir) is None


def test_recovers_legacy_tuning_from_three_raw_isolated_pairs(tmp_path):
    eval_dir = tmp_path / "e2e_interrupted"
    deploy = eval_dir / "tuning" / "deploy"
    deploy.mkdir(parents=True)
    (deploy / "deploy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (deploy / "MANIFEST.json").write_text(
        json.dumps({"apply_env": "AITER_CONFIG=/tmp/gemm.csv"}), encoding="utf-8"
    )
    for index, (pre, post) in enumerate(((1000.0, 1030.0), (1010.0, 1040.0), (990.0, 1025.0)), 1):
        for arm, throughput in (("pre", pre), ("post", post)):
            leg = eval_dir / "tuning" / "ab" / f"{arm}_{index}"
            leg.mkdir(parents=True)
            (leg / "bench_summary.json").write_text(json.dumps({
                "usable_for_acceptance": True,
                "measurement_mode": "isolated_server",
                "effective_config_digest": "same-effective-config",
                "throughput_tok_s_median": throughput,
            }), encoding="utf-8")
            attempt = leg / "replica_001" / "attempt_1"
            attempt.mkdir(parents=True)
            (attempt / "server.log").write_text(
                "is tuned on cu_num\n" if arm == "post" else "", encoding="utf-8"
            )

    recovered = rx._recover_workflow_return(tmp_path)

    assert recovered["validation_status"] == "recovered_tuning_skillset_legacy_provisional"
    assert recovered["baseline_throughput_tok_s"] == 1000.0
    assert recovered["final_throughput_tok_s"] == 1030.0
    assert recovered["recovery_evidence"]["summary_path"].endswith("bench_summary.json")


def test_verified_legacy_final_pair_outranks_tuning_and_checks_engagement(tmp_path):
    eval_dir = tmp_path / "e2e_interrupted"
    deploy = eval_dir / "tuning" / "deploy"
    deploy.mkdir(parents=True)
    (deploy / "deploy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    env = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "/tmp/tuned.csv"}
    (deploy / "MANIFEST.json").write_text(json.dumps({"extra_env": env}), encoding="utf-8")
    ab = eval_dir / "tuning" / "ab"
    ab.mkdir()
    (ab / "ab_summary.json").write_text(json.dumps({
        "pre_median": 1000.0, "post_median": 1100.0,
        "paired_mean_delta_pct": 10.0, "n_pairs": 1,
        "legs": [
            {"arm": "A", "usable": True, "mode": "isolated_server", "tput": 1000.0, "hits": 0},
            {"arm": "B", "usable": True, "mode": "isolated_server", "tput": 1100.0, "hits": 3},
        ],
    }), encoding="utf-8")
    final = eval_dir / "final"
    (final / "tuning").mkdir(parents=True)
    (final / "tuning" / "MANIFEST.json").write_text(
        json.dumps({"extra_env": env}), encoding="utf-8"
    )
    (final / "final_launch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (final / "accepted_config.json").write_text(json.dumps({
        "extra_server_args": "--final", "extra_env": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=/tmp/tuned.csv",
    }), encoding="utf-8")
    for name, throughput, hits in (("bench", 1040.0, 3), ("bench_control", 1000.0, 0)):
        attempt = final / name / "replica_001" / "attempt_1"
        attempt.mkdir(parents=True)
        (final / name / "bench_summary.json").write_text(json.dumps({
            "status": "complete", "usable_for_acceptance": True,
            "measurement_mode": "isolated_server", "effective_config_digest": "same",
            "throughput_tok_s_median": throughput, "ttft_ms_median": 10.0,
            "tpot_ms_median": 1.0,
        }), encoding="utf-8")
        (attempt / "server.log").write_text(
            "is tuned on cu_num\n" * hits, encoding="utf-8"
        )
    summary = {
        "final_bundle_tok_s": 1040.0,
        "drift_control_same_session_tok_s": 1000.0,
        "paired_in_session_speedup": 1.04,
        "tuning_engagement": {
            "final_bundle": {"tuned_hits": 3},
            "drift_control": {"tuned_hits": 0},
        },
    }
    (final / "FINAL_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")

    recovered = rx._recover_workflow_return(tmp_path)

    assert recovered["baseline_throughput_tok_s"] == 1000.0
    assert recovered["final_throughput_tok_s"] == 1040.0
    assert recovered["recovery_evidence"]["tuned_hits"] == 3
    assert recovered["accepted_kernels"][0]["kernel_id"] == "gemm_a8w8_blockscale_bpreshuffle"

    (final / "bench_control/replica_001/attempt_1/server.log").write_text(
        "is tuned on cu_num\n", encoding="utf-8"
    )
    fallback = rx._recover_workflow_return(tmp_path)
    assert fallback["baseline_throughput_tok_s"] == 1000.0
    assert fallback["final_throughput_tok_s"] == 1100.0

    (final / "bench_control/replica_001/attempt_1/server.log").write_text(
        "", encoding="utf-8"
    )
    summary["output_parity"] = "fail"
    (final / "FINAL_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
    parity_fallback = rx._recover_workflow_return(tmp_path)
    assert parity_fallback["final_throughput_tok_s"] == 1100.0

    (final / "FINAL_SUMMARY.json").unlink()
    missing_final_fallback = rx._recover_workflow_return(tmp_path)
    assert missing_final_fallback["final_throughput_tok_s"] == 1100.0
