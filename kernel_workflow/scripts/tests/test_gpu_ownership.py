# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check caller ownership against the existing allocator's real environment contract, on CPU."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from gpu_ownership import check_ownership


@pytest.mark.parametrize("spec", ["", "3", "1,3", "pool:1:1,3"])
def test_single_card_ownership_matches_selected_pool_member(spec):
    assert check_ownership(spec, {"GEAK_GPU_LOCK_HELD": "3", "HIP_VISIBLE_DEVICES": "3",
                                  "GEAK_GPU_ALLOWED": "1,3"})


@pytest.mark.parametrize("markers", [
    {"GEAK_GPU_LEASE_ACTIVE": "1", "GEAK_GPU_LEASE_IDS": "3,1"},
    {"GEAK_GPU_LOCKED_SET": "3,1", "GEAK_GPU_LOCK_HELD": "3"},
])
@pytest.mark.parametrize("spec", ["", "group:3,1", "pool:2:1,2,3"])
def test_complete_group_takes_precedence_over_last_single_marker(markers, spec):
    assert check_ownership(spec, dict(markers, HIP_VISIBLE_DEVICES="3,1", GEAK_GPU_ALLOWED="1,2,3"))


@pytest.mark.parametrize("spec, env", [
    ("2", {"GEAK_GPU_LOCK_HELD": "3", "HIP_VISIBLE_DEVICES": "3"}),
    ("1,3", {"GEAK_GPU_LOCK_HELD": "3", "HIP_VISIBLE_DEVICES": "3", "GEAK_GPU_ALLOWED": "3"}),
    ("", {"GEAK_GPU_LOCK_HELD": "3", "HIP_VISIBLE_DEVICES": "3", "GEAK_GPU_ALLOWED": "1"}),
    ("", {"GEAK_GPU_LOCK_HELD": "3", "HIP_VISIBLE_DEVICES": "2"}),
    ("", {"GEAK_GPU_LOCK_HELD": "3"}),
    ("", {"GEAK_GPU_LEASE_ACTIVE": "1"}),
    ("", {"GEAK_GPU_LEASE_IDS": "1,3"}),
    ("", {"GEAK_GPU_LEASE_ACTIVE": "yes"}),
    ("", {"GEAK_GPU_LOCKED_SET": "1,3", "GEAK_GPU_LOCK_HELD": "2", "HIP_VISIBLE_DEVICES": "1,3"}),
    ("group:1,3", {"GEAK_GPU_LOCKED_SET": "3,1", "HIP_VISIBLE_DEVICES": "3,1"}),
    ("3", {"GEAK_GPU_LOCKED_SET": "3,1", "HIP_VISIBLE_DEVICES": "3,1"}),
    ("pool:2:1,2", {"GEAK_GPU_LOCKED_SET": "3,1", "HIP_VISIBLE_DEVICES": "3,1"}),
])
def test_inconsistent_or_mismatched_ownership_is_refused(spec, env):
    with pytest.raises(ValueError):
        check_ownership(spec, env)


def test_no_markers_requires_acquisition_even_if_visibility_is_set():
    assert not check_ownership("3", {"HIP_VISIBLE_DEVICES": "3", "GEAK_GPU_ALLOWED": "3"})


@pytest.mark.parametrize("markers, expected", [
    ({}, 1),
    ({"GEAK_GPU_LOCK_HELD": "3", "HIP_VISIBLE_DEVICES": "3"}, 0),
    ({"GEAK_GPU_LOCK_HELD": "2", "HIP_VISIBLE_DEVICES": "2"}, 2),
])
def test_cli_distinguishes_acquire_reuse_and_refuse(markers, expected):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GEAK_GPU_")}
    env.update(markers)
    result = subprocess.run([sys.executable, str(SCRIPTS / "gpu_ownership.py"), "--gpu-spec", "3"],
                            env=env, text=True, check=False, capture_output=True, timeout=5)
    assert result.returncode == expected
    assert bool(result.stderr) == (expected == 2)


@pytest.mark.parametrize("entry, spec", [
    ("single", "3"), ("single", "1,3"), ("lease", "group:3,1"),
    ("lease", "pool:2:1,2,3"), ("set", "group:3,1"),
])
def test_existing_lock_and_lease_entry_points_publish_compatible_ownership(tmp_path, entry, spec):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GEAK_GPU_")}
    env.update(GEAK_GPU_LOCK_DIR=str(tmp_path / "locks"), GEAK_GPU_REQUIRE_IDLE="0",
               KERNEL_ENV_KEEP_ARCH="1", KERNEL_ENV_SKIP_ENUM_REAP="1", GEAK_GPU_ALLOWED="1,2,3")
    command = [sys.executable, str(SCRIPTS / "gpu_ownership.py"), "--gpu-spec", spec]
    if entry == "set":
        runner = SCRIPTS.parents[1] / "e2e_workflow/scripts/gpu_lock_set.py"
        argv = [sys.executable, str(runner), "3,1", *command]
    else:
        argv = ["bash", str(SCRIPTS / "gpu_lock.sh"), spec, *command]
    result = subprocess.run(argv, cwd=tmp_path, env=env, text=True, check=False, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
