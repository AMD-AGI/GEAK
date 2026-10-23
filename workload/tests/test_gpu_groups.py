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
        {"required_gpus": 1, "tasks": ["a", "b"], "slots": 2, "batch": 0, "gpus": [0, 1],
         "waves": 1, "target_sec": 4800, "wall_sec": 7200},
        {"required_gpus": 8, "tasks": ["collective"], "slots": 1, "batch": 1, "gpus": list(range(8)),
         "waves": 1, "target_sec": 4800, "wall_sec": 7200},
    ]


def test_disjoint_mixed_width_groups_share_a_batch():
    plan = groups.plan_groups({"a": 1, "b": 1, "collective": 4}, list(range(8)),
                             requested=6000, available=9000)
    assert [g["batch"] for g in plan] == [0, 0]
    assert [g["target_sec"] for g in plan] == [6000, 6000]
    assert plan[0]["gpus"] == [0, 1]
    assert plan[1]["gpus"] == [2, 3, 4, 5]


def test_partition_leaves_whole_slots_and_queues_excess_tasks():
    requirements = {"a%d" % i: 1 for i in range(10)}
    requirements["collective"] = 4
    devices = [2, 4, 6, 8, 10, 12, 14, 16]
    plan = groups.plan_groups(requirements, devices, requested=6000, available=9000)
    assert [g["batch"] for g in plan] == [0, 0]
    assert [g["waves"] for g in plan] == [3, 1]
    assert [g["target_sec"] for g in plan] == [2000, 2000]
    assigned = [gpu for g in plan for gpu in g["gpus"]]
    assert assigned == devices
    assert len(set(assigned)) == len(assigned)
    assert all(len(g["gpus"]) == g["slots"] * g["required_gpus"] for g in plan)


def test_batches_preserve_ranking_and_every_task_exactly_once():
    requirements = {"high_collective": 8, "single": 1, "small_collective": 4}
    plan = groups.plan_groups(requirements, list(range(8)), 6000, 18000)
    assert [g["tasks"] for g in plan] == [["high_collective"], ["single"], ["small_collective"]]
    assert [g["batch"] for g in plan] == [0, 1, 1]
    assert sum(max(g["wall_sec"] for g in plan if g["batch"] == b)
               for b in {g["batch"] for g in plan}) <= 18000



@pytest.mark.parametrize("limit", [1, 2, 3, 8])
def test_inflight_limit_bounds_slots_across_disjoint_groups(limit):
    requirements = {"single_%d" % i: 1 for i in range(11)}
    requirements.update({"pair_%d" % i: 2 for i in range(5)})
    requirements["full_node"] = 8
    plan = groups.plan_groups(requirements, list(range(8)), 172800, 1000000,
                              require_full_budget=True, task_limit=limit)
    assert sorted(t for group in plan for t in group["tasks"]) == sorted(requirements)
    assert all(group["target_sec"] == 172800 for group in plan)
    for batch in {group["batch"] for group in plan}:
        peers = [group for group in plan if group["batch"] == batch]
        assert sum(group["slots"] for group in peers) <= limit
        assigned = [gpu for group in peers for gpu in group["gpus"]]
        assert len(assigned) == len(set(assigned)) <= 8
        assert all(len(group["gpus"]) == group["slots"] * group["required_gpus"]
                   for group in peers)
    collective = next(group for group in plan if group["required_gpus"] == 8)
    assert collective["gpus"] == list(range(8))
    assert sum(group["batch"] == collective["batch"] for group in plan) == 1


@pytest.mark.parametrize("limit", [True, 0, -1, "8"])
def test_invalid_inflight_limit_cannot_expand_capacity(limit):
    with pytest.raises(ValueError, match="task_limit"):
        groups.plan_groups({"a": 1}, list(range(8)), 6000, 9000, task_limit=limit)

def _job(tmp_path, name, command, timeout=5):
    return {"command": command, "log": str(tmp_path / (name + ".log")),
            "work_root": str(tmp_path / name), "timeout_sec": timeout}


def test_peer_schedulers_really_run_concurrently(tmp_path, monkeypatch):
    # Each process requires its peer to start before it can exit. Sequential launch
    # cannot pass this test, even if the plan claims the pools run in parallel.
    script = tmp_path / "barrier.py"
    script.write_text('''import pathlib, sys, time
own, peer = map(pathlib.Path, sys.argv[1:])
own.touch()
deadline = time.monotonic() + 3
while not peer.exists():
    if time.monotonic() > deadline:
        sys.exit(9)
    time.sleep(0.01)
''')
    a, b = str(tmp_path / "a.ready"), str(tmp_path / "b.ready")
    jobs = [_job(tmp_path, "a", [sys.executable, str(script), a, b]),
            _job(tmp_path, "b", [sys.executable, str(script), b, a])]
    stopped, checkpoints = [], []
    monkeypatch.setattr(groups, "stop_workers", stopped.append)
    groups.run_schedulers(jobs, lambda: checkpoints.append(json.loads(json.dumps(jobs))))
    assert [j["exit_code"] for j in jobs] == [0, 0]
    assert stopped == [j["work_root"] for j in jobs]
    assert all(j["started_at"] and j["finished_at"] and j["elapsed_sec"] >= 0 for j in jobs)
    assert len(checkpoints) == 4


def test_parallel_timeout_keeps_peer_result_and_cleans_owned_roots(tmp_path, monkeypatch):
    jobs = [_job(tmp_path, "stalled", [sys.executable, "-c", "import time; time.sleep(30)"], 0.2),
            _job(tmp_path, "done", [sys.executable, "-c", "pass"])]
    stopped = []
    monkeypatch.setattr(groups, "stop_workers", stopped.append)
    groups.run_schedulers(jobs, lambda: None)
    assert [j["exit_code"] for j in jobs] == [124, 0]
    assert set(stopped) == {j["work_root"] for j in jobs}


def test_failed_peer_launch_reaps_existing_scheduler(tmp_path, monkeypatch):
    jobs = [_job(tmp_path, "started", [sys.executable, "-c", "import time; time.sleep(30)"]),
            _job(tmp_path, "missing", [str(tmp_path / "missing-binary")])]
    stopped = []
    monkeypatch.setattr(groups, "stop_workers", stopped.append)
    with pytest.raises(FileNotFoundError):
        groups.run_schedulers(jobs, lambda: None)
    assert jobs[0]["exit_code"] != 0
    assert "finished_at" in jobs[0]
    assert "started_at" not in jobs[1]
    assert stopped == [jobs[0]["work_root"]]


def _campaign(tmp_path, monkeypatch, params):
    root = tmp_path / "campaign"
    out = root / "optimize/out"
    out.mkdir(parents=True)
    for name, n in (("case_a", 1), ("case_b", 1), ("case_collective", 8)):
        task = root / "profile/out/tasks" / name
        (task / "kernel_src").mkdir(parents=True)
        (task / "meta.json").write_text(json.dumps({"required_gpus": n}))
    config = {"gpus": "0,1,2,3,4,5,6,7", "task_name_prefix": "case_",
              "time_budget_sec": 6000, "max_wall_sec": 14400,
              "image": "fake:cpu-only", **params}
    path = out / "params.json"
    path.write_text(json.dumps(config))
    for k, v in {"GEAK_ROOT": WORKLOAD.parent, "WORKLOAD_ROOT": WORKLOAD,
                 "RUN_ROOT": root, "OUT": out, "PARAMS": path,
                 "STAGE": "optimize", "STAGE_SKILL": "geak-kernel-gpu-groups",
                 "RUNNER": WORKLOAD / "runners/stub.sh"}.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.delenv("TASKS_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", [str(ENTRY)])
    return out


def test_dispatch_preserves_reservations_runner_and_resume_receipts(tmp_path, monkeypatch):
    import datetime
    now = [1000000.0]
    deadline = datetime.datetime.fromtimestamp(now[0] + 12000, datetime.timezone.utc).isoformat()
    out = _campaign(tmp_path, monkeypatch, {"deadline_utc": deadline,
                    "reserve_after_optimization_sec": 6000, "reserve_e2e_sec": 2000,
                    "verify_wall_sec": 1000, "wall_overhead_factor": 1})
    monkeypatch.setattr(groups.time, "time", lambda: now[0])
    monkeypatch.setattr(groups.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(groups, "preflight_names", lambda *args: None)
    launched = []

    def dispatch(jobs, checkpoint):
        launched.append(json.loads(json.dumps(jobs)))
        for job in jobs:
            job.update(exit_code=0, started_at=str(now[0]), finished_at=str(now[0] + 1))
        checkpoint()
        if len(launched) % 2 == 1:
            now[0] += 5000

    monkeypatch.setattr(groups, "run_schedulers", dispatch)
    assert groups.main() == 0
    assert len(launched) == 2
    assert launched[1][0]["timeout_sec"] == 1000
    for batch in launched:
        for job in batch:
            cmd = job["command"]
            assert cmd[cmd.index("--runner") + 1] == str(WORKLOAD / "runners/stub.sh")
            assert cmd[cmd.index("--gpus") + 1] == ",".join(map(str, job["gpus"]))
    receipt = out / json.loads((out / "dispatch.json").read_text())["execution_receipt"]
    original = receipt.read_bytes()
    credit = out / "group_1gpu/work/state/case_a.credit"
    credit.parent.mkdir(parents=True)
    credit.write_text("321")
    now[0] = 1000000.0
    assert groups.main() == 0
    assert len(list((out / "dispatch_attempts").glob('*.json'))) == 2
    assert receipt.read_bytes() == original
    assert credit.read_text() == "321"


def test_duplicate_dispatcher_cannot_launch_or_replace_results(tmp_path, monkeypatch):
    out = _campaign(tmp_path, monkeypatch, {})
    result = out / "result.json"
    result.write_text('{"existing": true}')
    with (out / "group_dispatch.lock").open("a") as lock:
        groups.fcntl.flock(lock, groups.fcntl.LOCK_EX | groups.fcntl.LOCK_NB)
        assert groups.main() == 78
    assert result.read_text() == '{"existing": true}'
    assert not (out / "group_plan.json").exists()


@pytest.mark.parametrize("params", [{"max_tasks": 2}, {"exclude_tasks": ["case_b"]}])
def test_group_dispatch_cannot_silently_skip_extracted_tasks(tmp_path, monkeypatch, params):
    out = _campaign(tmp_path, monkeypatch, params)
    monkeypatch.setattr(groups, "preflight_names", lambda *args: pytest.fail("contacted Docker"))
    assert groups.main() == 1
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "blocked"
    assert "every extracted task" in result["notes"]


def test_the_ninth_task_enters_the_complete_scheduler_queue(tmp_path, monkeypatch):
    """Later waves belong to the existing scheduler, never a truncated invocation."""
    out = _campaign(tmp_path, monkeypatch, {})
    tasks = out.parents[1] / "profile/out/tasks"
    for n in range(6):
        task = tasks / ("case_extra_%d" % n)
        (task / "kernel_src").mkdir(parents=True)
        (task / "meta.json").write_text('{"required_gpus": 1}')
    monkeypatch.setattr(sys, "argv", [str(ENTRY), "--plan-only"])
    assert groups.main() == 0
    plan = json.loads((out / "group_plan.json").read_text())
    scheduled = [k for g in plan["groups"] for k in g["tasks"]]
    assert len(scheduled) == len(set(scheduled)) == 9
    assert not plan["capacity_deferred"]
    assert plan["task_limit"] == 8 and plan["tasks_total"] == 9
    assert plan["groups"][0]["slots"] == 8
    assert plan["groups"][1]["gpus"] == list(range(8))
    assert len(list(tasks.iterdir())) == 9



@pytest.mark.parametrize("completed_head", [0, 8])
def test_real_scheduler_drains_full_queue_and_resumes_without_repeating_completed_head(
        tmp_path, monkeypatch, completed_head):
    # Only the Docker worker boundary is fake. Use scheduler.main/tick to prove that
    # more than a wave of tasks really reaches dispatch, including after a restart.
    sys.path.insert(0, str(WORKLOAD / "scripts"))
    import scheduler
    out = _campaign(tmp_path, monkeypatch, {
        "require_full_budget": True, "task_limit": 3, "time_budget_sec": 172800})
    tasks_dir = out.parents[1] / "profile/out/tasks"
    for name, width in [("case_extra_%d" % i, 1) for i in range(8)] + [("case_other_collective", 8)]:
        (tasks_dir / name / "kernel_src").mkdir(parents=True)
        (tasks_dir / name / "meta.json").write_text(json.dumps({"required_gpus": width}))
    tasks = scheduler.discover_tasks(str(tasks_dir))
    assert len(tasks) == 12
    widths = {name: json.loads((tasks_dir / name / "meta.json").read_text())["required_gpus"]
              for name in tasks}
    preserved = {}
    for name in tasks[:completed_head]:
        state = out / ("group_%dgpu/work/state" % widths[name])
        state.mkdir(parents=True, exist_ok=True)
        path = state / (name + ".status.json")
        path.write_text('{"status":"done","credited_seconds":172800,"attempts":7}')
        preserved[path] = path.read_bytes()
    started, running, queues = [], {}, []

    class FakeDocker(scheduler.Backend):
        def alive(self, kernel, gpus):
            return running.get(kernel) == gpus

        def start(self, kernel, gpus, env):
            assert kernel not in running
            assert not set(gpus).intersection(g for pool in running.values() for g in pool)
            assert len(running) < 3
            assert len(gpus) == widths[kernel]
            assert env["TARGET"] == "172800" and env["REQUIRE_FULL_BUDGET"] == "1"
            running[kernel] = gpus
            started.append(kernel)
            path = Path(self.work_root) / "state" / (kernel + ".status.json")
            path.write_text('{"status":"done","credited_seconds":172800,"attempts":1}')
            return True

        def stop(self, kernel, gpus):
            running.pop(kernel, None)

    def dispatch(jobs, checkpoint):
        for job in jobs:
            command = job["command"]
            excluded = command[command.index("--exclude") + 1].split(",")
            queues.append(scheduler.discover_tasks(str(tasks_dir), exclude=excluded))
            with monkeypatch.context() as context:
                context.setattr(sys, "argv", command[1:])
                job["exit_code"] = scheduler.main()
            running.clear()
        checkpoint()

    monkeypatch.setattr(scheduler, "DockerBackend", FakeDocker)
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)
    monkeypatch.setattr(groups, "preflight_names", lambda *args: None)
    monkeypatch.setattr(groups, "run_schedulers", dispatch)
    assert groups.main() == 0
    assert sorted(started) == sorted(tasks[completed_head:])
    assert len(started) == len(set(started))
    assert any(len(queue) > 8 for queue in queues)
    assert all(path.read_bytes() == original for path, original in preserved.items())
    dispatch_result = json.loads((out / "dispatch.json").read_text())
    assert dispatch_result["full_budget_complete"]
    assert set(dispatch_result["remaining_task_seconds"]) == set(tasks)
    assert set(dispatch_result["remaining_task_seconds"].values()) == {0}
    for name in tasks:
        assert (out / "work/state" / (name + ".status.json")).resolve() == (
            out / ("group_%dgpu/work/state" % widths[name]) / (name + ".status.json"))
    receipt = out / dispatch_result["execution_receipt"]
    first_receipt = receipt.read_bytes()
    started.clear()
    assert groups.main() == 0
    assert not started
    assert receipt.read_bytes() == first_receipt
    assert len(list((out / "dispatch_attempts").glob("*.json"))) == 2

def test_broken_extracted_directory_is_not_silently_ignored(tmp_path, monkeypatch):
    out = _campaign(tmp_path, monkeypatch, {})
    (out.parents[1] / "profile/out/tasks/case_incomplete").mkdir()
    monkeypatch.setattr(groups, "preflight_names", lambda *args: pytest.fail("contacted Docker"))
    assert groups.main() == 1
    assert "case_incomplete" in json.loads((out / "result.json").read_text())["notes"]


def test_profile_resource_map_avoids_mutating_qualified_metadata(tmp_path, monkeypatch):
    out = _campaign(tmp_path, monkeypatch, {})
    profile = out.parents[1]/'profile/out'
    for p in (profile/'tasks').glob('*/meta.json'):p.write_text('{}')
    (profile/'task_gpu_requirements.json').write_text(json.dumps({'case_a':1,'case_b':1,'case_collective':8}))
    monkeypatch.setattr(sys,'argv',[str(ENTRY),'--plan-only'])
    assert groups.main() == 0
    plan = json.loads((out/'group_plan.json').read_text())
    assert [g['required_gpus'] for g in plan['groups']] == [1,8]
    assert all(p.read_text()=='{}' for p in (profile/'tasks').glob('*/meta.json'))
    params=json.loads((out/'params.json').read_text());params['task_gpu_requirements']={'case_collective':1}
    (out/'params.json').write_text(json.dumps(params))
    assert groups.main() == 1
    assert 'contradicts' in json.loads((out/'result.json').read_text())['notes']


@pytest.mark.parametrize("requirement", [None, True, 0, 9])
def test_missing_or_impossible_gpu_requirement_is_refused(requirement):
    with pytest.raises(ValueError):
        groups.plan_groups({"task": requirement}, list(range(8)), 6000, 14400)


def test_full_budget_is_not_silently_shortened():
    short = groups.plan_groups({"a": 1, "collective": 8}, list(range(8)),
                               43200, 50000, require_full_budget=True)
    assert [g['target_sec'] for g in short] == [43200, 43200]
    plan = groups.plan_groups({"a": 1, "b": 1}, list(range(8)),
                             43200, 64800, require_full_budget=True)
    assert plan[0]["target_sec"] == 43200


def test_full_target_checkpoints_a_short_window_and_resumes_credit(tmp_path, monkeypatch):
    import datetime
    now = [1000000.0]
    out = _campaign(tmp_path, monkeypatch, {
        'require_full_budget': True, 'time_budget_sec': 6000, 'max_wall_sec': 8000,
        'deadline_utc': datetime.datetime.fromtimestamp(now[0]+12000, datetime.timezone.utc).isoformat(),
        'reserve_after_optimization_sec': 6000, 'reserve_e2e_sec': 2000,
        'verify_wall_sec': 1000, 'checkpoint_reserve_sec': 600})
    monkeypatch.setattr(groups.time, 'time', lambda: now[0])
    monkeypatch.setattr(groups.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(groups, 'preflight_names', lambda *args: None)
    calls = []

    def checkpoint_dispatch(jobs, save):
        calls.extend(jobs)
        assert len(jobs) == 1 and jobs[0]['required_gpus'] == 1
        cmd = jobs[0]['command']
        assert cmd[cmd.index('--target')+1] == '6000'
        work = Path(jobs[0]['work_root']) / 'state'
        work.mkdir(parents=True, exist_ok=True)
        for name in ('case_a', 'case_b'):
            (work / (name+'.status.json')).write_text('{"status":"running","credited_seconds":4000}')
        jobs[0]['exit_code'] = 124
        now[0] += jobs[0]['timeout_sec'] + 1
        save()

    monkeypatch.setattr(groups, 'run_schedulers', checkpoint_dispatch)
    assert groups.main() == 1
    dispatch = json.loads((out/'dispatch.json').read_text())
    assert dispatch['full_budget_complete'] is False
    assert dispatch['remaining_task_seconds'] == {'case_a':2000,'case_b':2000,'case_collective':6000}
    assert dispatch['scheduler_exit_codes'] == [124, None]
    assert len(calls) == 1
    result = json.loads((out/'result.json').read_text())
    assert result['status'] == 'blocked' and 'target remains 6000' in result['notes']
    assert (out/'work/state/case_a.status.json').is_symlink()
    old_receipt = (out/dispatch['execution_receipt']).read_bytes()

    params = json.loads((out/'params.json').read_text())
    params['deadline_utc'] = datetime.datetime.fromtimestamp(now[0]+30000, datetime.timezone.utc).isoformat()
    params['max_wall_sec'] = 30000
    (out/'params.json').write_text(json.dumps(params))

    def finish_dispatch(jobs, save):
        for job in jobs:
            work = Path(job['work_root'])/'state'
            work.mkdir(parents=True, exist_ok=True)
            names = ('case_a','case_b') if job['required_gpus']==1 else ('case_collective',)
            if job['required_gpus']==1:
                assert json.loads((work/'case_a.status.json').read_text())['credited_seconds']==4000
            for name in names:
                (work/(name+'.status.json')).write_text('{"status":"done","credited_seconds":6000}')
            job['exit_code'] = 0
        now[0] += 1
        save()

    monkeypatch.setattr(groups, 'run_schedulers', finish_dispatch)
    assert groups.main() == 0
    resumed = json.loads((out/'dispatch.json').read_text())
    assert resumed['full_budget_complete'] is True
    assert set(resumed['remaining_task_seconds'].values()) == {0}
    assert (out/dispatch['execution_receipt']).read_bytes() == old_receipt


def test_live_scheduler_prevents_dispatch_without_stopping_workers(tmp_path):
    with (tmp_path / "scheduler.lock").open("a") as lock:
        groups.fcntl.flock(lock, groups.fcntl.LOCK_EX | groups.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="preserve its running workers"):
            groups.preflight_scheduler(tmp_path)


def test_scheduler_ownership_refusal_does_not_clean_existing_workers(tmp_path, monkeypatch):
    jobs = [_job(tmp_path, "busy", [sys.executable, "-c", "raise SystemExit(78)"])]
    stopped = []
    monkeypatch.setattr(groups, "stop_workers", stopped.append)
    with pytest.raises(RuntimeError, match="existing workers were preserved"):
        groups.run_schedulers(jobs, lambda: None)
    assert jobs[0]["exit_code"] == 78
    assert not stopped


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


@pytest.mark.parametrize("message", ["Error: No such object: missing", "error: no such object: missing"])
def test_docker_absence_handles_cli_capitalization(monkeypatch, message):
    from types import SimpleNamespace
    monkeypatch.setattr(groups.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr=message, stdout=""))
    assert groups.docker_inspect(["missing"]) == []


def test_docker_permission_error_is_not_absence(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(groups.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr="permission denied", stdout=""))
    with pytest.raises(RuntimeError, match="permission denied"):
        groups.docker_inspect(["owned-looking-name"])
