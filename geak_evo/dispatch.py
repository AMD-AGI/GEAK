"""Adapt the common optimizer task/result files to an admitted native owner.

The installed owner binds ``main`` to its existing services. No provider,
admission, process manager or device allocation is constructed here.
"""

import argparse
import copy
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from ._util import EvoError, canonical_json, require, sha256_hex
from .archive.refs import file_ref, read_ref
from .archive.store import _path, _read, _write
from .command import CommandInterrupted, run_admitted_task
from .planner.model import decode


def _need(value, reason):
    require(value, reason, ValueError)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _document(reference):
    return decode(read_ref(reference))


def _git(root, *args, data=None, deadline=None):
    """Bounded Git plumbing, with hooks, signing and global filters disabled."""
    seconds = 30 if deadline is None else min(30, deadline - time.monotonic())
    _need(seconds > 0, "common command deadline expired")
    value = subprocess.run(["/usr/bin/git", "-C", str(root), "-c", "core.hooksPath=/dev/null",
        "-c", "core.fsmonitor=false", "-c", "commit.gpgsign=false", *args],
        input=data, capture_output=True, timeout=seconds, check=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
             "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
             "GIT_AUTHOR_NAME": "GEAK-EVO", "GIT_AUTHOR_EMAIL": "geak-evo@localhost",
             "GIT_COMMITTER_NAME": "GEAK-EVO", "GIT_COMMITTER_EMAIL": "geak-evo@localhost"})
    return value.stdout


def _clean(root, base, deadline=None):
    _need(_git(root, "rev-parse", "HEAD", deadline=deadline).decode().strip() == base
          and _git(root, "status", "--porcelain=v1", "--untracked-files=all", deadline=deadline) == b"",
          "owned clone moved or has uncommitted files")


def _check_inputs(task, binding, config, evaluator, budget):
    _need(type(task) is dict and type(task.get("schema_version")) is int and task["schema_version"] == 1,
          "common optimizer task schema_version 1 required")
    _need(set(binding) == {"schema", "owner_config_ref", "task_ref", "case_map", "dispatch_root", "max_steps", "device", "export_seconds"}
          and binding["schema"] == "geak-evo-dispatch-owner-v1", "exact dispatch owner binding required")
    native = _document(config["search"]["templates"]["config"])
    _need(task.get("task_ref") == binding["task_ref"] == native["contract_ref"]
          and _document(binding["task_ref"]) == evaluator.contract,
          "dispatcher task is not the admitted native task")
    mapping = binding["case_map"]
    cases = evaluator.contract["benchmark_case_ids"]
    lane_cases = task["baseline"]["per_case"]
    _need(type(mapping) is dict and mapping and set(mapping) == set(lane_cases)
          and len(mapping) == len(cases) and set(mapping.values()) == set(cases)
          and all(type(name) is str and name for name in mapping)
          and all(_positive(value) for value in lane_cases.values()),
          "complete one-to-one lane/native benchmark-case mapping required")
    _need(task["device"] == binding["device"], "dispatcher device differs from the owner's admitted mapping")
    objective = copy.deepcopy(task["objective"])
    if "weights" in objective:
        _need(type(objective["weights"]) is dict and set(objective["weights"]) == set(mapping),
              "lane workload weights do not cover the mapped cases")
        objective["weights"] = {mapping[name]: value for name, value in objective["weights"].items()}
    _need(objective == config["objective"], "lane objective differs from admitted command objective")
    _need(_positive(task["incumbent_speedup"]) and type(task["min_improvement"]) in (int, float)
          and math.isfinite(task["min_improvement"]) and task["min_improvement"] >= 0,
          "finite incumbent and nonnegative improvement threshold required")
    limit = task["budget"]
    _need(type(binding["max_steps"]) is int and 0 < binding["max_steps"] <= 10000
          and type(limit["max_rounds"]) is int and binding["max_steps"] <= limit["max_rounds"]
          and limit["min_improvement"] == task["min_improvement"], "admitted search bound exceeds common task bound")
    deadline = limit["deadline_epoch"]
    _need(_positive(binding["export_seconds"]), "explicit export/conversion margin required")
    cutoff = None
    if deadline is not None:
        _need(_positive(deadline), "finite common command deadline required")
        now_unix, now_ns = time.time(), time.monotonic_ns()
        cutoff_ns = now_ns + int((deadline - now_unix) * 1e9)
        _need(budget.config["deadline_ns"] + int(binding["export_seconds"] * 1e9) <= cutoff_ns,
              "owner clock and export margin exceed the common command deadline")
        _need(now_ns < cutoff_ns, "common command deadline expired")
        cutoff = cutoff_ns / 1e9
    _need(budget.config == config["search"]["shared_budget"], "dispatch and search must use the same native budget")
    return mapping, objective, cutoff


def _clone(task, binding, evaluator, deadline):
    directory, workspace = Path(task["eval_dir"]), Path(task["workspace"])
    dispatch_root = Path(binding["dispatch_root"])
    _need(dispatch_root.is_absolute() and dispatch_root.resolve(strict=True) == dispatch_root
          and directory.parent == dispatch_root and directory.resolve(strict=True) == directory,
          "task evaluation directory is outside the admitted dispatch root")
    root = directory / "repo"
    _need(root.resolve(strict=True) == root and workspace.resolve(strict=True) == workspace
          and (workspace == root or root in workspace.parents)
          and (root / ".git").is_dir() and not (root / ".git").is_symlink()
          and all(path.stat().st_uid == os.geteuid() for path in (directory, root, workspace, root / ".git")),
          "dispatcher must supply its canonical owned clone")
    _need(Path(_git(workspace, "rev-parse", "--show-toplevel", deadline=deadline).decode().strip()) == root,
          "workspace is not inside the owned dispatcher clone")
    base = task["base_commit"]
    _need(type(base) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base), "exact base commit required")
    paths = evaluator.contract["source_paths"]
    _need(type(task["editable_paths"]) is list and len(task["editable_paths"]) == len(paths)
          and set(task["editable_paths"]) == set(paths), "lane editable files differ from the native source closure")
    for name in evaluator.contract["artifacts"]:
        _path(name)
        _need(not any(ord(c) < 32 for c in name), "control characters in task paths are unsupported")
    _clean(root, base, deadline)
    _baseline(workspace, evaluator)
    return root, workspace, base


def _baseline(workspace, evaluator):
    for name, digest in evaluator.contract["artifacts"].items():
        raw, mode = _read(workspace, name)
        _need(sha256_hex(raw) == digest and (raw, mode) == _read(evaluator.artifact_root, name),
              "clone source or protected task artifact differs from native admission: " + name)


def _commit(store, source_id, root, workspace, base, deadline):
    """Write the store's exact bytes/modes, then commit only those source paths."""
    source = store.source(source_id)
    prefix = workspace.relative_to(root).as_posix()
    prefix = "" if prefix == "." else prefix + "/"
    entries = []
    for row in source["files"]:
        name, identity, mode = row["path"], row["blob_id"], row["executable"]
        _path(name)
        _need(mode in (0, 0o111) and not any(ord(c) < 32 for c in name), "unsupported selected source mode/path")
        target = workspace / name
        if identity is None:
            target.unlink(missing_ok=True)
            entries.append("0 " + "0" * len(base) + "\t" + prefix + name + "\0")
        else:
            raw = _read(store.root / "objects", identity)[0]
            _write(target, raw)
            target.chmod(0o644 | mode)
            blob = _git(root, "hash-object", "-w", "--stdin", data=raw, deadline=deadline).decode().strip()
            entries.append(("100755" if mode else "100644") + " " + blob + "\t" + prefix + name + "\0")
    _need(store.capture_source(workspace, [row["path"] for row in source["files"]]) == source_id,
          "written clone differs from selected source")
    # Plumbing writes raw blobs without invoking attribute filters or hooks.
    _git(root, "update-index", "-z", "--index-info", data="".join(entries).encode(), deadline=deadline)
    tree = _git(root, "write-tree", deadline=deadline).decode().strip()
    commit = _git(root, "commit-tree", tree, "-p", base,
                  data=("GEAK-EVO selected source " + source_id + "\n").encode(), deadline=deadline).decode().strip()
    changed = set(_git(root, "diff", "--no-renames", "--name-only", "-z", base, commit,
                       deadline=deadline).decode().rstrip("\0").split("\0"))
    _need(changed and changed <= {prefix + row["path"] for row in source["files"]},
          "selected commit changed files outside the native source closure")
    _git(root, "update-ref", "HEAD", commit, base, deadline=deadline)
    _clean(root, commit, deadline)
    return commit


def run_dispatch_task(task_path, out_path, *, owner_binding_ref, store, services, evaluator, budget, episode):
    """Run one common-lane task through the owner's admitted command and clone."""
    task_path, out = Path(task_path), Path(out_path)
    task_ref = file_ref(task_path)
    task = _document(task_ref)
    _need(out.is_absolute() and out.parent.resolve(strict=True) == out.parent
          and not out.exists() and not out.is_symlink()
          and task_path == out.parent / "task.json" and out.name == "result.json"
          and out.parent == Path(task["eval_dir"]), "new common task/result paths required")
    result = {"schema_version": 1, "status": "error", "stop_reason": "budget", "rounds": 0,
              "best": None, "measurement_valid": False, "custody_proven": False,
              "reward_eligible": False, "framework_final_pass": False}
    try:
        binding = _document(owner_binding_ref)
        config = _document(binding["owner_config_ref"])
        mapping, objective, deadline = _check_inputs(task, binding, config, evaluator, budget)
        root, workspace, base = _clone(task, binding, evaluator, deadline)
        campaign = config["search"]["campaign_id"]
        try:
            prior = store.read_current(campaign)["note"]["evolution"]
        except KeyError:
            prior = {}
        before = sum(row.get("engineer_id") == "planner" for row in prior.get("opportunities", []))
        bound = Path(config["search"]["campaign_root"]) / "evo-dispatch.json"
        identity = {"task_ref": task_ref, "owner_binding_ref": owner_binding_ref}
        if bound.exists():
            _need(decode(_read(bound.parent, bound.name)[0]) == identity, "campaign is bound to another dispatcher task")
        else:
            _write(bound, canonical_json(identity), immutable=True)
        native_out = out.parent / "native-result"
        native = run_admitted_task(binding["owner_config_ref"], store=store, services=services,
            evaluator=evaluator, budget=budget, episode=episode, output_dir=native_out,
            max_steps=binding["max_steps"])
        result["native_result_ref"] = file_ref(native_out / "result.json")
        result["task_ref"], result["owner_binding_ref"] = task_ref, owner_binding_ref
        revision = native["command"]["campaign_revision"]
        if revision is not None:
            state = store.get_revision(campaign, revision)["note"]["evolution"]
            result["rounds"] = sum(row.get("engineer_id") == "planner" for row in state.get("opportunities", [])) - before
        _need(native["outcome"] in ("improved", "no_improvement"), native.get("error", "admitted command did not complete"))
        result.update(status="no_improvement", stop_reason="no_improvement")
        if native["outcome"] == "improved":
            _need(native["validation_status"] == "provisional_native_observation" and native["objective"] == objective
                  and all(native[key] is False for key in ("measurement_valid", "custody_proven", "reward_eligible", "framework_final_pass")),
                  "native result is not a checked provisional comparison")
            rows = native["per_case"]
            _need(len(rows) == len(mapping) and {row["case_id"] for row in rows} == set(mapping.values()),
                  "native result does not cover every mapped case")
            timings = {row["case_id"]: row for row in rows}
            pairs = {name: {"baseline_ms": timings[case]["baseline_ms"], "optimized_ms": timings[case]["candidate_ms"]}
                     for name, case in mapping.items()}
            _need(all(_positive(value) for pair in pairs.values() for value in pair.values()), "invalid measured pair")
            ratios = {name: row["baseline_ms"] / row["optimized_ms"] for name, row in pairs.items()}
            weights = task["objective"].get("weights")
            primary = (math.fsum(weights.values()) / math.fsum(weights[name] / ratios[name] for name in mapping)
                       if objective["kind"] == "time_weighted" else
                       math.exp(math.fsum(math.log(ratio) for ratio in ratios.values()) / len(ratios)))
            if objective["kind"] == "external_primary":
                # The checked UT reader owns this method's phase reduction and
                # repeatability gate; per-case medians are only descriptive.
                from .evaluate.external import validate_objective
                validate_objective(evaluator.contract, objective)
                _need(native.get("native_reproducible_gain") is True, "external native gain gate did not pass")
                primary = native["timing_context"]["candidate"]["diagnostics"]["primary_speedup"]
            _need(_positive(primary) and math.isclose(primary, native["final_speedup"], rel_tol=1e-12),
                  "native and common-lane metrics disagree")
            if primary > task["incumbent_speedup"] * (1 + task["min_improvement"]):
                _need(_document(task_ref) == task, "dispatcher task changed during search")
                _clean(root, base, deadline)
                _baseline(workspace, evaluator)
                source_id = native["selected_source_id"]
                _need(store.get_candidate(native["selected_candidate_id"])["source_id"] == source_id,
                      "selected candidate source differs")
                commit = _commit(store, source_id, root, workspace, base, deadline)
                result.update(status="improved", stop_reason="budget",
                              best={"commit": commit, "speedup": primary, "per_case": pairs})
            else:
                result["notes"] = "Selected source did not beat the incumbent by the common improvement threshold."
    except (CommandInterrupted, EvoError, ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result.update(status="error", best=None, notes=type(exc).__name__ + ": " + str(exc))
    _write(out, canonical_json(result) + b"\n", immutable=True)
    return result


def main(argv=None, *, owner_binding_ref, store, services, evaluator, budget, episode):
    """The existing installed owner passes services, and the lane supplies argv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise CommandInterrupted("common dispatcher termination cutoff; retain pending work and costs")
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        result = run_dispatch_task(args.task, args.out, owner_binding_ref=owner_binding_ref,
            store=store, services=services, evaluator=evaluator, budget=budget, episode=episode)
        return 1 if result["status"] == "error" else 0
    finally:
        signal.signal(signal.SIGTERM, previous)
