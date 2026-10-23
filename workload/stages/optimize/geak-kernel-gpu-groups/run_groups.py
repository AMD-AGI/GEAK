#!/usr/bin/env python3
"""Group frozen tasks by GPU count and compose the existing workload tools."""

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def plan_groups(requirements, gpus, requested, available, overhead=1.5, require_full_budget=False,
                task_limit=None):
    if not isinstance(require_full_budget, bool):
        raise ValueError("require_full_budget must be a boolean")
    if (not gpus or len(set(gpus)) != len(gpus)
            or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in gpus)):
        raise ValueError("GPU ids must be nonnegative, nonempty and unique")
    if task_limit is None:
        task_limit = len(gpus)
    if isinstance(task_limit, bool) or not isinstance(task_limit, int) or task_limit < 1:
        raise ValueError("task_limit must be a positive integer")
    if requested <= 0 or available <= 0 or overhead < 1:
        raise ValueError("A positive budget and overhead factor >= 1 are required")
    grouped = {}
    for task, count in requirements.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Invalid GPU requirement for %s: %r" % (task, count))
        if count > len(gpus):
            raise ValueError("%s requires %d GPUs; only %d are available" % (task, count, len(gpus)))
        grouped.setdefault(count, []).append(task)
    if not grouped:
        raise ValueError("No tasks to schedule")
    # Reserve at least one whole slot per group in a batch. Different widths can
    # share a node; a collective that needs the entire node still runs separately.
    batches = []
    for n, tasks in grouped.items():
        group = {"required_gpus": n, "tasks": tasks}
        batch = next((b for b in batches
                      if len(b) < task_limit
                      and sum(g["required_gpus"] for g in b) + n <= len(gpus)), None)
        if batch is None:
            batch = []
            batches.append(batch)
        batch.append(group)

    for index, batch in enumerate(batches):
        # Find the shortest fixed partition: ceil(tasks / waves) slots per group.
        # This only composes existing schedulers, which still own task placement.
        low, high = 1, max(len(g["tasks"]) for g in batch)
        while low < high:
            mid = (low + high) // 2
            needed = sum(g["required_gpus"] * math.ceil(len(g["tasks"]) / mid)
                         for g in batch)
            needed_slots = sum(math.ceil(len(g["tasks"]) / mid) for g in batch)
            if needed <= len(gpus) and needed_slots <= task_limit:
                high = mid
            else:
                low = mid + 1
        for group in batch:
            group["slots"] = math.ceil(len(group["tasks"]) / low)
        spare = len(gpus) - sum(g["slots"] * g["required_gpus"] for g in batch)
        spare_slots = task_limit - sum(g["slots"] for g in batch)
        for group in batch:
            extra = min(len(group["tasks"]) - group["slots"],
                        spare // group["required_gpus"], spare_slots)
            group["slots"] += extra
            spare -= extra * group["required_gpus"]
            spare_slots -= extra
        offset = 0
        for group in batch:
            end = offset + group["slots"] * group["required_gpus"]
            group.update(batch=index, gpus=gpus[offset:end],
                         waves=math.ceil(len(group["tasks"]) / group["slots"]))
            offset = end

    waves = sum(max(g["waves"] for g in batch) for batch in batches)
    target = requested if require_full_budget else min(requested, math.floor(available / (overhead * waves)))
    if target < 1:
        raise ValueError("No optimization time remains after reservations")
    return [dict(g, target_sec=target, wall_sec=math.floor(g["waves"] * target * overhead))
            for batch in batches for g in batch]


def verification_budgets(requirements, default, overrides=None):
    """Reserve each group's actual validation workload before allocating agent time."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("verify_wall_sec_by_gpu_count must be a mapping")
    budgets = {}
    for count in set(requirements.values()):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Invalid GPU requirement: %r" % count)
        value = overrides.get(str(count), default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Invalid verification allowance for GPU count %d" % count)
        budgets[count] = value
    return budgets


def bounded_timeout(cap, deadline, protected_seconds, now=None):
    if deadline is None:
        return cap
    remaining = math.floor(deadline - (time.time() if now is None else now) - protected_seconds)
    if remaining < 1:
        raise ValueError("The allocation deadline and protected validation time leave no execution budget")
    return min(cap, remaining)


def container_work_root(info):
    entries = (info.get("Config") or {}).get("Env") or []
    values = dict(item.split("=", 1) for item in entries if "=" in item)
    return values.get("WORK_ROOT")


def owned_worker(info, work_root):
    return (info.get("Name", "").lstrip("/").startswith("wl_opt_")
            and container_work_root(info) == str(work_root))


def docker_inspect(names):
    if not names:
        return []
    result = subprocess.run(["docker", "inspect", *names], capture_output=True, text=True)
    if result.returncode:
        # Docker <= 28 says "Error: No such object:"; 29 says "error: no such object:".
        if len(names) == 1 and "no such object" in result.stderr.lower():
            return []
        raise RuntimeError("Docker inspect failed: " + result.stderr[:400])
    return json.loads(result.stdout)


def stop_workers(work_root):
    ids = subprocess.check_output(["docker", "ps", "-aq"], text=True).split()
    for cid in ids:
        records = docker_inspect([cid])
        if records and owned_worker(records[0], work_root):
            info = records[0]
            cid = info["Id"]
            print("Stopping owned worker", info.get("Name"), flush=True)
            subprocess.run(["docker", "stop", "--timeout", "15", cid], check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run(["docker", "rm", cid], check=True, stdout=subprocess.DEVNULL)


def preflight_names(backend, tasks, slots, work_root):
    for task in tasks:
        for slot in slots:
            name = backend.name_of(task, slot)
            existing = docker_inspect([name])
            if existing and not owned_worker(existing[0], work_root):
                raise RuntimeError("Refusing Docker-name collision with another work root: " + name)


def preflight_scheduler(work_root):
    Path(work_root).mkdir(parents=True, exist_ok=True)
    with (Path(work_root) / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A scheduler still owns %s; preserve its running workers "
                               "and resume after it finishes" % work_root) from exc


def link_artifact(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise RuntimeError("Refusing to overwrite another work artifact: " + str(target))
    else:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def run_schedulers(jobs, checkpoint):
    """Launch disjoint GPU groups together; retain receipts and stop only their workers."""
    running = []

    def finish(job, started, rc):
        job.update(exit_code=rc, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   elapsed_sec=time.monotonic() - started)
        checkpoint()
        stop_workers(job["work_root"])

    try:
        for job in jobs:
            with Path(job["log"]).open("a") as output:
                proc = subprocess.Popen(job["command"], stdout=output, stderr=subprocess.STDOUT)
            started = time.monotonic()
            running.append((job, proc, started))
            job.update(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       scheduler_pid=proc.pid)
            checkpoint()
        while running:
            for entry in list(running):
                job, proc, started = entry
                rc = proc.poll()
                if rc is None and time.monotonic() - started >= job["timeout_sec"]:
                    proc.kill()
                    proc.wait()
                    rc = 124
                if rc is not None:
                    if rc == 78:
                        # The work-root lock belongs to a pre-existing scheduler.
                        # It must retain its workers even if it won a preflight race.
                        job.update(exit_code=rc, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   elapsed_sec=time.monotonic() - started)
                        running.remove(entry)
                        checkpoint()
                        raise RuntimeError("Scheduler refused ownership; its existing workers were preserved")
                    finish(job, started, rc)
                    running.remove(entry)
            if running:
                time.sleep(0.1)
    finally:
        # A failed launch or an interrupted coordinator must not leave a peer
        # scheduler dispatching new workers while its GPU pool is being cleaned up.
        for job, proc, started in running:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        errors = []
        for job, proc, started in running:
            try:
                finish(job, started, proc.returncode)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    workload = Path(os.environ["WORKLOAD_ROOT"])
    geak = Path(os.environ["GEAK_ROOT"])
    root, out = Path(os.environ["RUN_ROOT"]), Path(os.environ["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    params = read_json(os.environ["PARAMS"])
    sys.path.insert(0, str(workload / "scripts"))
    sys.path.insert(0, str(workload / "stages/optimize"))
    import scheduler
    from check_scope import check_accounting, valid_limit
    if params.get("container_runtime_file"):
        os.environ["WORKLOAD_DOCKER_RUNTIME"] = params["container_runtime_file"]
    from emit_result import emit

    def fail(reason):
        return emit("blocked", notes=reason, artifacts={"plan": "group_plan.json"}
                    if (out / "group_plan.json").exists() else {})

    lock = None
    try:
        if not args.plan_only:
            lock = (out / "group_dispatch.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another group dispatcher owns this work root; leave it running", file=sys.stderr)
                return 78
        tasks_dir = Path(os.environ.get("TASKS_DIR", str(root / "profile/out/tasks")))
        order_file = root / "roofline/out/roofline.json"
        all_tasks = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
        tasks = scheduler.discover_tasks(str(tasks_dir), str(order_file) if order_file.exists() else None,
                                         params.get("max_tasks"), params.get("exclude_tasks") or [])
        # The scheduler drains each complete queue as slots become free. Truncating
        # it here strands the tail on every resume, even after the head is complete.
        limit = valid_limit(params.get("task_limit", 8))
        check_accounting(all_tasks, tasks)
        prefix = params.get("task_name_prefix")
        if not prefix or any(not name.startswith(prefix) for name in tasks):
            raise ValueError("Every task must use the configured unique task_name_prefix")
        declarations = dict(params.get("task_gpu_requirements") or {})
        if params.get("task_gpu_requirements_file"):
            path = Path(params["task_gpu_requirements_file"])
            if not path.is_absolute(): path = geak / path
            declarations.update(read_json(path))
        requirements = {}
        resource_map = root / "profile/out/task_gpu_requirements.json"
        upstream_requirements = read_json(resource_map) if resource_map.exists() else {}
        if not isinstance(upstream_requirements, dict):
            raise ValueError("profile task_gpu_requirements.json must be a mapping")
        for name in tasks:
            meta = read_json(tasks_dir / name / "meta.json")
            declared = declarations.get(name)
            recorded = meta.get("required_gpus")
            upstream = upstream_requirements.get(name)
            if recorded is not None and upstream is not None and recorded != upstream:
                raise ValueError("GPU requirement contradicts the profile resource map: " + name)
            if recorded is None:
                recorded = upstream
            if declared is not None and recorded is not None and declared != recorded:
                raise ValueError("GPU requirement contradicts frozen metadata: " + name)
            requirements[name] = recorded if recorded is not None else declared
        gpus = [int(g) for g in str(params.get("gpus", "")).split(",") if g.strip()]
        reserve = int(params.get("reserve_after_optimization_sec", 21600))
        e2e_reserve = int(params.get("reserve_e2e_sec", min(14400, reserve)))
        verify_wall = int(params.get("verify_wall_sec", 3600))
        if reserve < 0 or e2e_reserve < 0 or verify_wall < 1:
            raise ValueError("Reservations must be nonnegative and verification needs a positive budget")
        verify_budgets = verification_budgets(requirements, verify_wall,
                                              params.get("verify_wall_sec_by_gpu_count"))
        reserve = max(reserve, e2e_reserve + sum(verify_budgets.values()))
        single_method_name = params.get("single_gpu_method", "geak-kernel-workflow")
        if single_method_name not in ("geak-kernel-workflow", "native-frozen"):
            raise ValueError("Unsupported single_gpu_method: " + str(single_method_name))
        requested = int(params.get("time_budget_sec", 172800))
        overhead = float(params.get("wall_overhead_factor", 1.5))
        full_budget = params.get("require_full_budget", False)
        checkpoint_reserve = params.get("checkpoint_reserve_sec", 600)
        if isinstance(checkpoint_reserve, bool) or not isinstance(checkpoint_reserve, int) or checkpoint_reserve < 1:
            raise ValueError("checkpoint_reserve_sec must be a positive integer")
        deadline = None
        if params.get("deadline_utc"):
            deadline = datetime.datetime.fromisoformat(params["deadline_utc"].replace("Z", "+00:00")).timestamp()
        window = int(deadline - time.time()) if deadline else requested * max(1, len(tasks)) * 2 + reserve
        available = window - reserve
        protected = reserve
        checkpoint_window = False
        if full_budget:
            full_plan = plan_groups(requirements, gpus, requested, max(1, window), overhead, True,
                                    task_limit=limit)
            need = sum(max(g["wall_sec"] for g in full_plan if g["batch"] == b)
                       for b in {g["batch"] for g in full_plan})
            if need > available:
                # A short allocation can still contribute real work to a fixed target.
                # This pass checkpoints; cold/E2E require a later completion window.
                checkpoint_window = True
                protected = checkpoint_reserve
                available = window - protected
        if params.get("max_wall_sec"):
            available = min(available, int(params["max_wall_sec"]))
        groups = plan_groups(requirements, gpus, requested, available, overhead,
                             require_full_budget=full_budget, task_limit=limit)
        if min(g["target_sec"] for g in groups) < int(params.get("attempt_min_sec", 600)):
            raise ValueError("Remaining lease cannot cover a useful attempt and reserved verification/A/B")
        plan = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "available_optimization_wall_sec": available, "reserve_after_optimization_sec": reserve,
                "deadline_utc": params.get("deadline_utc"), "reserve_e2e_sec": e2e_reserve,
                "verification_allowance_per_group_sec": verify_wall,
                "verification_allowances_by_gpu_count": verify_budgets, "groups": groups,
                "require_full_budget": full_budget, "checkpoint_window": checkpoint_window,
                "protected_current_window_sec": protected,
                "allocation_operations": "none", "single_gpu_method": single_method_name,
                "collective_method": "native-collective",
                "task_limit": limit, "tasks_total": len(tasks),
                "capacity_deferred": []}
        write_json(out / "group_plan.json", plan)
        if args.plan_only:
            print(json.dumps(plan, indent=2))
            return 0

        single_method = (Path(__file__).resolve().parent / "native-task"
                         if single_method_name == "native-frozen"
                         else workload / "stages/optimize/geak-kernel-workflow")
        collective_method = Path(__file__).resolve().parent / "collective-method"
        image = params.get("image")
        if not image:
            raise ValueError("A pinned image is required for physical GPU isolation")
        threshold = float(params.get("keep_threshold", 1.045))
        overrides = ",".join("%s=%s" % pair for pair in sorted((params.get("keep_threshold_overrides") or {}).items()))
        jobs = []
        receipt = out / "dispatch_attempts" / (str(time.time_ns()) + ".json")

        def checkpoint():
            write_json(receipt, {"plan": plan, "schedulers": jobs})

        for group in groups:
            n = group["required_gpus"]
            method = collective_method if n > 1 else single_method
            group["optimization_method"] = "native-collective" if n > 1 else single_method_name
            group_out = out / ("group_%dgpu" % n)
            work = group_out / "work"
            group_out.mkdir(parents=True, exist_ok=True)
            preflight_scheduler(work)
            backend = scheduler.DockerBackend(str(work), image)
            slots, _ = scheduler.make_slots(group["gpus"], n)
            preflight_names(backend, group["tasks"], slots, work)
            group["verify_wall_sec"] = verify_budgets[n]
            wall = bounded_timeout(group["wall_sec"], deadline, protected)
            command = [sys.executable, str(workload / "scripts/scheduler.py"), "run",
                       "--work-root", str(work), "--tasks-dir", str(tasks_dir),
                       "--skill-root", str(method), "--runner", os.environ["RUNNER"],
                       "--gpus", ",".join(map(str, group["gpus"])), "--gpus-per-task", str(n),
                       "--target", str(group["target_sec"]), "--max-wall-sec", str(wall),
                       "--backend", "docker", "--image", image,
                       "--exclude", ",".join(t for t in all_tasks if t not in group["tasks"]),
                       "--keep-threshold", str(threshold)]
            for flag, key, default in [("attempt-cap", "attempt_cap_sec", 1500),
                                       ("attempt-min", "attempt_min_sec", 600),
                                       ("measure-timeout", "measure_timeout_sec", 1800),
                                       ("budget", "budget", 4), ("min-improve", "min_improve", 0.05),
                                       ("stale-sec", "heartbeat_stale_sec", 900),
                                       ("bad-node-limit", "bad_node_limit", 2)]:
                command += ["--" + flag, str(params.get(key, default))]
            if order_file.exists(): command += ["--order-file", str(order_file)]
            if overrides: command += ["--keep-threshold-override", overrides]
            if full_budget:
                command += ["--require-full-budget"]
            jobs.append({"required_gpus": n, "batch": group["batch"], "gpus": group["gpus"],
                         "command": command, "log": str(group_out / "scheduler.log"),
                         "work_root": str(work), "timeout_sec": wall + 300,
                         "already_complete": full_budget and all(scheduler.is_done(str(work), t, requested, True)
                                                                 for t in group["tasks"])})

        checkpoint()
        optimization_end = time.monotonic() + available
        for batch in sorted({g["batch"] for g in groups}):
            peers = [j for j in jobs if j["batch"] == batch and not j["already_complete"]]
            if not peers:
                continue
            window_left = math.floor(optimization_end - time.monotonic())
            if deadline is not None:
                window_left = min(window_left, math.floor(deadline - time.time() - protected))
            if window_left < 1:
                break
            for job in peers:
                job["timeout_sec"] = min(job["timeout_sec"], window_left)
            run_schedulers(peers, checkpoint)

        for group in groups:
            work = out / ("group_%dgpu" % group["required_gpus"]) / "work"
            group["verification"] = ("step 2 of stages/optimize/README.md: "
                                     "rerun this group's unit tests from cold")
            for name in group["tasks"]:
                for area, suffix in (("state", ".status.json"), ("runs", "")):
                    source = work / area / (name + suffix)
                    if source.exists(): link_artifact(source, out / "work" / area / source.name)

        remaining = {name: max(0, requested - scheduler.read_credit(
            str(out / ("group_%dgpu" % count) / "work"), name)) for name, count in requirements.items()}
        complete = all(scheduler.is_done(str(out / ("group_%dgpu" % count) / "work"), name, requested, True)
                       for name, count in requirements.items()) if full_budget else None
        validation_fits = deadline is None or deadline - time.time() >= reserve
        ready_for_verification = not full_budget or (complete and validation_fits)
        write_json(out / "group_plan.json", plan)
        write_json(out / "dispatch.json", {
            "scheduler_exit_codes": [job.get("exit_code") for job in jobs],
            "execution_receipt": str(receipt.relative_to(out)),
            "groups": groups,
            "task_gpu_requirements": requirements,
            "full_budget_complete": complete, "remaining_task_seconds": remaining,
            "validation_window_available": validation_fits,
            "next": "steps 2 and 3 of stages/optimize/README.md" if ready_for_verification
                    else "resume this work root in a sufficient allocated window; do not reset credit"})
        if not ready_for_verification:
            artifacts = {"plan": "group_plan.json", "dispatch": "dispatch.json"}
            if (out / "work").exists(): artifacts["work"] = "work"
            reason = ("Fixed optimization target remains %ds per task; checkpointed remaining seconds: %s. "
                      % (requested, remaining)) if not complete else "All search targets are complete; cold verification and E2E still need their reserved window. "
            return emit("blocked", notes=reason + "Resume with these artifacts and credit preserved. No allocation was changed.",
                        artifacts=artifacts)
        print("dispatch complete; verification and result.json are steps 2-3 "
              "of stages/optimize/README.md", flush=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError, InterruptedError) as exc:
        return fail(str(exc))
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise InterruptedError("Optimization interrupted; owned workers will be stopped")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    sys.exit(main())
