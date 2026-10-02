# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete accepted AgentX launch evidence survives the normal handoff."""
import copy
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

import pytest

from e2e_workflow.scripts.adapters.server_args import server_semantics
from interface.effective_config import (
    ReferenceLaunchError,
    resolve_effective_config,
    resolve_reference_launch,
)

FLAGS = ["--speculative-config", '{"method":"real","nested":{"label":"a b","sizes":[1,2]}}',
         "--attention-backend", "aiter", "--compilation-config", '{"capture_sizes":[1,2,4]}']


def handoff():
    argv = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/test", "--port", "8000", *FLAGS]
    capture = {"schema": "hyperloom.serving_launch.v1", "capture_id": "a" * 32,
               "recipe_digest": "sha256:" + "b" * 64, "workspace": "/frozen/run", "started_ns": 1,
               "owner_pid": 12, "owner_start_ticks": 10, "recipe_pid": 20, "recipe_start_ticks": 20,
               "measurement": {"path": "/frozen/run/result.json", "sha256": "c" * 64,
                               "inode": 123, "mtime_ns": 2, "size": 10},
               "server": {"pid": 123, "start_ticks": 100, "boot_id": "boot", "endpoint_identity": [123, 100],
                          "launch_nonce": "a" * 32, "log_inode": 100,
                          "argv": argv, "semantic_binding": server_semantics(argv, "vllm"),
                          "serving_env_scope": "serving-knobs-v1", "serving_env": {"VLLM_USE_AITER": "1"}}}
    capture["sha256"] = hashlib.sha256(json.dumps(capture, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 3, "framework": "vllm", "model_path": "/models/test",
            "workload_spec": {"kind": "agentx_trace_replay"}, "launch_server_script": "/frozen/server.sh",
            "bench_launcher": "magpie", "baseline_env_spec": {"config": {
                "server_launch_flags": shlex.join(FLAGS), "server_env": {"VLLM_USE_AITER": "1"}}},
            "measurement_evidence": {"observed_server_launch_tokens": FLAGS.copy(),
                                     "requested_server_env": {"VLLM_USE_AITER": "1"},
                                     "observed_server_env": {"VLLM_USE_AITER": "1"},
                                     "server_launch_capture": capture, "recipe_digest": "sha256:" + "b" * 64,
                                     "server_launch_argv_complete": True}}


def test_complete_reference_and_declared_environment_are_preserved():
    value = handoff()
    effective = resolve_effective_config(value)
    assert shlex.split(resolve_reference_launch(value, effective)) == [
        "--speculative-config", '{"method":"real","nested":{"label":"a b","sizes":[1,2]}}',
        "--attention-backend", "aiter", "--compilation-config", '{"capture_sizes":[1,2,4]}']
    assert effective.final_env == {"VLLM_USE_AITER": "1"}


@pytest.mark.parametrize("index", [0, 2, 4])
def test_missing_speculation_backend_or_graph_flags_refuses_reference(index):
    value = handoff()
    flags = FLAGS.copy()
    del flags[index:index + 2]
    value["baseline_env_spec"]["config"]["server_launch_flags"] = shlex.join(flags)
    with pytest.raises(ReferenceLaunchError, match="differ"):
        resolve_reference_launch(value, resolve_effective_config(value))


@pytest.mark.parametrize("missing", ["tokens", "complete", "env", "recipe"])
def test_incomplete_reference_evidence_is_not_a_default_launch(missing):
    value = handoff()
    if missing == "tokens":
        value["measurement_evidence"].pop("observed_server_launch_tokens")
    if missing == "complete":
        value["measurement_evidence"].pop("server_launch_argv_complete")
    if missing == "env":
        value["baseline_env_spec"]["config"].pop("server_env")
    if missing == "recipe":
        value.pop("launch_server_script")
    with pytest.raises(ReferenceLaunchError, match="requires"):
        resolve_reference_launch(value, resolve_effective_config(value))


def test_legacy_handoff_does_not_acquire_strict_agentx_requirements():
    value = handoff()
    value["schema_version"] = 2
    value["measurement_evidence"] = {}
    assert resolve_reference_launch(value, resolve_effective_config(value)) is None


def test_new_candidate_does_not_mutate_the_accepted_reference():
    value = handoff()
    original = copy.deepcopy(value)
    resolve_reference_launch(value, resolve_effective_config(value))
    assert value == original


@pytest.mark.parametrize("summary", [
    {}, {"max_model_len": 1048576}, {"mem_fraction": 0.9},
    {"max_model_len": 1048576, "mem_fraction": 0.9},
])
def test_normal_bridge_carries_the_strict_reference(tmp_path, summary):
    spec = importlib.util.spec_from_file_location("recipe_run_e2e", Path(__file__).with_name("run_e2e.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    value = handoff()
    value.update(summary)
    value["exp_root"] = str(tmp_path)
    args = runner.map_args(value, timeout_s=43200)
    assert args["reference_server_args"] == args["initial_extra_server_args"]
    assert "VLLM_USE_AITER=1" in args["initial_extra_env"]
    for key, expected in summary.items():
        assert args[key] == expected


def test_reference_environment_cannot_change_after_its_measurement():
    value = handoff()
    value["accepted_env"] = "VLLM_USE_AITER=0"
    with pytest.raises(ReferenceLaunchError, match="environment differ"):
        resolve_reference_launch(value, resolve_effective_config(value))


@pytest.mark.parametrize("change", ["model", "topology", "raw_argv", "capture_digest", "observed_env", "nonce", "partial_measurement"])
def test_strict_reference_requires_process_bound_semantics_and_environment(change):
    value = handoff()
    if change == "model":
        value["model_path"] = "/models/wrong"
    elif change == "topology":
        value["tp"] = 4
    elif change == "raw_argv":
        value["measurement_evidence"]["server_launch_capture"]["server"]["argv"].append("--unexpected")
    elif change == "capture_digest":
        value["measurement_evidence"]["server_launch_capture"]["sha256"] = "wrong"
    elif change in ("nonce", "partial_measurement"):
        capture = value["measurement_evidence"]["server_launch_capture"]
        if change == "nonce":
            capture["server"]["launch_nonce"] = "f" * 32
        else:
            capture["measurement"].pop("mtime_ns")
        capture["sha256"] = hashlib.sha256(json.dumps(
            {k: v for k, v in capture.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
    else:
        value["measurement_evidence"].pop("observed_server_env")
    with pytest.raises(ReferenceLaunchError):
        resolve_reference_launch(value, resolve_effective_config(value))


def test_incomplete_reference_fails_before_launch_preparation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("recipe_main", Path(__file__).with_name("run_e2e.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    value = handoff()
    value["exp_root"] = str(tmp_path)
    value["measurement_evidence"]["server_launch_argv_complete"] = False
    source, result = tmp_path / "handoff.json", tmp_path / "result.json"
    source.write_text(json.dumps(value))

    def unexpected(*args, **kwargs):
        pytest.fail("incomplete reference reached launch preparation")

    monkeypatch.setattr(runner, "agentx_preflight", unexpected)
    monkeypatch.setattr(runner, "prepare_baseline_source", unexpected)
    assert runner.main([str(source), str(result), "--timeout-s", "43200"]) == 1
    assert json.loads(result.read_text())["error_class"] == "reference_launch_mismatch"
