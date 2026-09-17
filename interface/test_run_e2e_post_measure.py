# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The returned replay capability must describe actual, deterministically staged code."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("run_e2e_post_measure", Path(__file__).with_name("run_e2e.py"))
rx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rx)


def test_current_bench_stages_missing_helpers_and_advertises_direct_replay(tmp_path):
    shutil.copy2(rx.BENCH_SCRIPT, tmp_path / "bench_e2e.sh")
    cap = rx._stage_post_measure_lifecycle(tmp_path, "")
    assert cap["status"] == "available"
    assert cap["replay_support"] == {"bench_e2e_fallback": True, "final_launch_script": False}
    for name in cap["sources"]:
        assert (tmp_path / name).read_bytes() == (rx.BENCH_SCRIPT.parent / name).read_bytes()
    assert rx._stage_post_measure_lifecycle(tmp_path, "") == cap


def test_old_or_conflicting_bundles_are_not_overwritten(tmp_path):
    bench = tmp_path / "bench_e2e.sh"
    bench.write_text("old benchmark\n")
    assert rx._stage_post_measure_lifecycle(tmp_path, "")["reason"] == "unrecognized_bench_script"
    assert bench.read_text() == "old benchmark\n"
    shutil.copy2(rx.BENCH_SCRIPT, bench)
    helper = tmp_path / "server_teardown.sh"
    helper.write_text("different ownership policy\n")
    assert rx._stage_post_measure_lifecycle(tmp_path, "")["reason"] == "conflicting_staged_asset"
    assert helper.read_text() == "different ownership policy\n"
    assert not (tmp_path / "bench_lifecycle.py").exists()


def test_only_exact_known_recovery_launcher_earns_final_launch_capability(tmp_path):
    shutil.copy2(rx.BENCH_SCRIPT, tmp_path / "bench_e2e.sh")
    deploy = tmp_path / "tuning/deploy/deploy.sh"
    deploy.parent.mkdir(parents=True)
    deploy.write_text("#!/bin/bash\ntrue\n")
    launcher = rx._write_tuning_recovery_launcher(tmp_path)
    assert rx._stage_post_measure_lifecycle(tmp_path, launcher)["replay_support"]["final_launch_script"]
    with Path(launcher).open("a") as handle:
        handle.write("# agent modified\n")
    assert not rx._stage_post_measure_lifecycle(tmp_path, launcher)["replay_support"]["final_launch_script"]


@pytest.mark.parametrize("same_bytes", [True, False])
def test_concurrent_stager_is_accepted_only_when_its_asset_bytes_match(tmp_path, monkeypatch, same_bytes):
    shutil.copy2(rx.BENCH_SCRIPT, tmp_path / "bench_e2e.sh")
    real_link = os.link

    def raced_link(source, target):
        if Path(target).name == "bench_replica.sh":
            Path(target).write_bytes(Path(source).read_bytes() if same_bytes else b"conflicting stager")
            raise FileExistsError
        return real_link(source, target)

    monkeypatch.setattr(os, "link", raced_link)
    cap = rx._stage_post_measure_lifecycle(tmp_path, "")
    assert cap["status"] == ("available" if same_bytes else "unsupported")
    if not same_bytes:
        assert cap["reason"] == "staged_asset_changed"
        assert (tmp_path / "bench_replica.sh").read_bytes() == b"conflicting stager"


def test_unreadable_asset_returns_unsupported_without_disrupting_result_emission(tmp_path):
    shutil.copy2(rx.BENCH_SCRIPT, tmp_path / "bench_e2e.sh")
    (tmp_path / "server_teardown.sh").mkdir()
    cap = rx._stage_post_measure_lifecycle(tmp_path, "")
    assert cap["status"] == "unsupported"
    assert cap["reason"] == "staging_unavailable"
    assert not (tmp_path / "bench_lifecycle.py").exists()


def test_recovery_preserves_current_request_and_output_after_deploy_env(tmp_path):
    bench = tmp_path / "bench_e2e.sh"
    bench.write_text('printf "%s" "${GEAK_POST_MEASURE_REQUEST-unset}" > "$OUT_DIR/request_seen"\n')
    deploy = tmp_path / "tuning/deploy/deploy.sh"
    deploy.parent.mkdir(parents=True)
    deploy.write_text("printf '%s\\n' 'export GEAK_POST_MEASURE_REQUEST=stale' 'export OUT_DIR=/wrong' 'export ROOT=/wrong' 'export HERE=/wrong' > \"$GEAK_TUNING_ENV_OUT\"\n")
    launcher = rx._write_tuning_recovery_launcher(tmp_path)
    out = tmp_path / "output with spaces"
    out.mkdir()
    env = {**os.environ, "GEAK_POST_MEASURE_REQUEST": "/current/request with spaces.json"}
    subprocess.run(["bash", launcher, str(out)], env=env, check=True, timeout=5)
    assert (out / "request_seen").read_text() == env["GEAK_POST_MEASURE_REQUEST"]
    env.pop("GEAK_POST_MEASURE_REQUEST")
    subprocess.run(["bash", launcher, str(out)], env=env, check=True, timeout=5)
    assert (out / "request_seen").read_text() == "unset"
