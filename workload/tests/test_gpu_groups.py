import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


WORKLOAD = Path(__file__).resolve().parents[1]
ENTRY = WORKLOAD / "stages/optimize/geak-kernel-gpu-groups/run_groups.py"
spec = importlib.util.spec_from_file_location("gpu_groups", ENTRY)
groups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groups)


def test_mixed_groups_reserve_two_serial_waves():
    plan = groups.plan_groups({"a": 1, "b": 1, "collective": 8}, list(range(8)),
                              requested=6000, available=14400)
    assert plan == [
        {"required_gpus": 1, "tasks": ["a", "b"], "slots": 8,
         "waves": 1, "target_sec": 4800, "wall_sec": 7200},
        {"required_gpus": 8, "tasks": ["collective"], "slots": 1,
         "waves": 1, "target_sec": 4800, "wall_sec": 7200},
    ]


@pytest.mark.parametrize("requirement", [None, True, 0, 9])
def test_missing_or_impossible_gpu_requirement_is_refused(requirement):
    with pytest.raises(ValueError):
        groups.plan_groups({"task": requirement}, list(range(8)), 6000, 14400)


def test_worker_cleanup_requires_the_exact_work_root():
    own = {"Name": "/wl_opt_0_case_a", "Config": {"Env": ["WORK_ROOT=/run/group_1/work"]}}
    foreign = {"Name": "/wl_opt_0_case_b", "Config": {"Env": ["WORK_ROOT=/run/group_10/work"]}}
    ordinary = {"Name": "/service", "Config": {"Env": ["WORK_ROOT=/run/group_1/work"]}}
    assert groups.owned_worker(own, "/run/group_1/work")
    assert not groups.owned_worker(foreign, "/run/group_1/work")
    assert not groups.owned_worker(ordinary, "/run/group_1/work")


def test_timeouts_leave_the_protected_validation_window():
    assert groups.bounded_timeout(3600, deadline=3600, protected_seconds=1200, now=1200) == 1200
    assert groups.bounded_timeout(600, deadline=3600, protected_seconds=1200, now=1200) == 600
    with pytest.raises(ValueError, match="protected"):
        groups.bounded_timeout(3600, deadline=3600, protected_seconds=1200, now=2400)


def test_plan_only_executes_without_contacting_docker(tmp_path):
    root = tmp_path / "campaign"
    out = root / "optimize/out"
    out.mkdir(parents=True)
    for name, n in (("case_a", 1), ("case_b", 1), ("case_collective", 8)):
        task = root / "profile/out/tasks" / name
        (task / "kernel_src").mkdir(parents=True)
        (task / "meta.json").write_text(json.dumps({"required_gpus": n}))
    params = out / "params.json"
    params.write_text(json.dumps({"gpus": "0,1,2,3,4,5,6,7", "task_name_prefix": "case_",
                                 "time_budget_sec": 6000, "max_wall_sec": 14400}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "docker_was_called"
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\ntouch "$DOCKER_MARKER"\nexit 99\n')
    docker.chmod(0o755)
    env = {**os.environ, "GEAK_ROOT": str(WORKLOAD.parent), "WORKLOAD_ROOT": str(WORKLOAD),
           "RUN_ROOT": str(root), "OUT": str(out), "PARAMS": str(params),
           "STAGE": "optimize", "STAGE_SKILL": "geak-kernel-gpu-groups",
           "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
           "DOCKER_MARKER": str(marker)}
    result = subprocess.run([sys.executable, str(ENTRY), "--plan-only"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads((out / "group_plan.json").read_text())
    assert [g["required_gpus"] for g in plan["groups"]] == [1, 8]
    assert [g["wall_sec"] for g in plan["groups"]] == [7200, 7200]
    assert not marker.exists()


def test_group_specific_verification_preserves_the_total_reservation():
    windows = groups.verification_budgets({'attention': 1, 'moe': 1, 'collective': 8},
                                          7200, {'1': 9000, '8': 5400})
    assert windows == {1: 9000, 8: 5400}
    assert sum(windows.values()) == 14400
    assert groups.verification_budgets({'a': 1, 'b': 8}, 7200) == {1: 7200, 8: 7200}
    # At the first verifier, the later group and E2E must still retain their own time.
    assert groups.bounded_timeout(windows[1], 32400, 18000 + windows[8], now=0) == 9000
    assert groups.bounded_timeout(windows[8], 32400, 18000, now=9000) == 5400


@pytest.mark.parametrize('value', [0, -1, True, '9000'])
def test_invalid_group_verification_allowance_is_refused(value):
    with pytest.raises(ValueError):
        groups.verification_budgets({'a': 1}, 7200, {'1': value})


@pytest.mark.parametrize('overrides', [[], False, '', [1]])
def test_non_mapping_group_allowances_cannot_fall_back(overrides):
    with pytest.raises(ValueError, match='mapping'):
        groups.verification_budgets({'a': 1, 'b': 8}, 3600, overrides)
