#!/usr/bin/env python3
"""Group frozen tasks by GPU count and compose the existing workload tools."""

import argparse
import datetime
import hashlib
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


def plan_groups(requirements, gpus, requested, available, overhead=1.5):
    if (not gpus or len(set(gpus)) != len(gpus)
            or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in gpus)):
        raise ValueError("GPU ids must be nonnegative, nonempty and unique")
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
    waves = sum(math.ceil(len(tasks) / (len(gpus) // n)) for n, tasks in grouped.items())
    target = min(requested, math.floor(available / (overhead * waves)))
    if target < 1:
        raise ValueError("No optimization time remains after reservations")
    return [
        {"required_gpus": n, "tasks": tasks, "slots": len(gpus) // n,
         "waves": math.ceil(len(tasks) / (len(gpus) // n)),
         "target_sec": target,
         "wall_sec": math.floor(math.ceil(len(tasks) / (len(gpus) // n)) * target * overhead)}
        for n, tasks in sorted(grouped.items())
    ]


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
        if len(names) == 1 and "No such" in result.stderr:
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


def link_artifact(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise RuntimeError("Refusing to overwrite another work artifact: " + str(target))
    else:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def log_run(argv, log, timeout=None):
    print("Running", argv[0], Path(argv[1]).name if len(argv) > 1 else "", flush=True)
    with Path(log).open("a") as output:
        return subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT,
                              timeout=timeout).returncode


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
    import scheduler
    from container_runtime import runtime_options
    if params.get("container_runtime_file"):
        os.environ["WORKLOAD_DOCKER_RUNTIME"] = params["container_runtime_file"]
    from emit_result import emit

    def fail(reason):
        return emit("blocked", notes=reason, artifacts={"plan": "group_plan.json"}
                    if (out / "group_plan.json").exists() else {})

    active_work = None
    try:
        tasks_dir = Path(os.environ.get("TASKS_DIR", str(root / "profile/out/tasks")))
        order_file = root / "roofline/out/roofline.json"
        all_tasks = scheduler.discover_tasks(str(tasks_dir), None, None, [])
        tasks = scheduler.discover_tasks(str(tasks_dir), str(order_file) if order_file.exists() else None,
                                         params.get("max_tasks"), params.get("exclude_tasks") or [])
        prefix = params.get("task_name_prefix")
        if not prefix or any(not name.startswith(prefix) for name in tasks):
            raise ValueError("Every task must use the configured unique task_name_prefix")
        declarations = dict(params.get("task_gpu_requirements") or {})
        if params.get("task_gpu_requirements_file"):
            path = Path(params["task_gpu_requirements_file"])
            if not path.is_absolute(): path = geak / path
            declarations.update(read_json(path))
        requirements = {}
        for name in tasks:
            meta = read_json(tasks_dir / name / "meta.json")
            declared = declarations.get(name)
            recorded = meta.get("required_gpus")
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
        deadline = None
        if params.get("deadline_utc"):
            deadline = datetime.datetime.fromisoformat(params["deadline_utc"].replace("Z", "+00:00")).timestamp()
        available = int(deadline - time.time() - reserve) if deadline else requested * max(1, len(tasks)) * 2
        if params.get("max_wall_sec"):
            available = min(available, int(params["max_wall_sec"]))
        groups = plan_groups(requirements, gpus, requested, available, overhead)
        if min(g["target_sec"] for g in groups) < int(params.get("attempt_min_sec", 600)):
            raise ValueError("Remaining lease cannot cover a useful attempt and reserved verification/A/B")
        plan = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "available_optimization_wall_sec": available, "reserve_after_optimization_sec": reserve,
                "deadline_utc": params.get("deadline_utc"), "reserve_e2e_sec": e2e_reserve,
                "verification_allowance_per_group_sec": verify_wall,
                "verification_allowances_by_gpu_count": verify_budgets, "groups": groups,
                "allocation_operations": "none", "single_gpu_method": single_method_name,
                "collective_method": "native-collective"}
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
        scheduler_codes = []
        for index, group in enumerate(groups):
            n = group["required_gpus"]
            method = collective_method if n > 1 else single_method
            group["optimization_method"] = "native-collective" if n > 1 else single_method_name
            group_out = out / ("group_%dgpu" % n)
            work = group_out / "work"
            group_out.mkdir(parents=True, exist_ok=True)
            active_work = work
            backend = scheduler.DockerBackend(str(work), image)
            slots, _ = scheduler.make_slots(gpus, n)
            preflight_names(backend, group["tasks"], slots, work)
            group["verify_wall_sec"] = verify_budgets[n]
            future_verification = sum(verify_budgets[g["required_gpus"]] for g in groups[index:])
            wall = bounded_timeout(group["wall_sec"], deadline, e2e_reserve + future_verification)
            command = [sys.executable, str(workload / "scripts/scheduler.py"), "run",
                       "--work-root", str(work), "--tasks-dir", str(tasks_dir),
                       "--skill-root", str(method), "--runner", os.environ["RUNNER"],
                       "--gpus", ",".join(map(str, gpus)), "--gpus-per-task", str(n),
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
            try:
                try:
                    rc = log_run(command, group_out / "scheduler.log",
                                 timeout=bounded_timeout(wall + 300, deadline,
                                                         e2e_reserve + future_verification))
                except subprocess.TimeoutExpired:
                    rc = 124
                    print("Scheduler deadline reached; verifying retained work after cleanup", flush=True)
            finally:
                stop_workers(work)
            scheduler_codes.append(rc)

            group["verification"] = ("step 2 of stages/optimize/README.md: "
                                     "rerun this group's unit tests from cold")
            for name in group["tasks"]:
                for area, suffix in (("state", ".status.json"), ("runs", "")):
                    source = work / area / (name + suffix)
                    if source.exists(): link_artifact(source, out / "work" / area / source.name)
            active_work = None

        write_json(out / "group_plan.json", plan)
        write_json(out / "dispatch.json", {
            "scheduler_exit_codes": scheduler_codes,
            "groups": groups,
            "task_gpu_requirements": requirements,
            "next": "steps 2 and 3 of stages/optimize/README.md"})
        print("dispatch complete; verification and result.json are steps 2-3 "
              "of stages/optimize/README.md", flush=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError, InterruptedError) as exc:
        return fail(str(exc))
    finally:
        if active_work is not None:
            stop_workers(active_work)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise InterruptedError("Optimization interrupted; owned workers will be stopped")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    sys.exit(main())
