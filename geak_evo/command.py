"""Run an already-admitted native task with the owner's existing services.

The owner calls ``main`` from its installed entrypoint. This module does not
load providers, create admissions or budgets, or own processes and devices.
"""

import argparse
import fcntl
import math
import os
from pathlib import Path
import time

from ._util import EvoError, canonical_json, require, sha256_hex
from .archive.refs import read_ref
from .archive.store import _read, _write
from .evaluate.evaluator import ArenaEvaluator, ArenaEvaluationUnsettled
from .host import HostServices, prepare_native_search
from .model_profile import ModelProfile
from .planner.model import decode
from .result import export_final_result, read_evaluation
from .selection.choices import authenticate_choice
from .selection.final import validate_final_recommendation


class CommandInterrupted(BaseException):
    """Stop the command after owned workers retain their settlement records."""


def _need(value, reason):
    require(value, reason, ValueError)


def _document(reference):
    return decode(read_ref(reference))


def _measure(evaluator, budget, candidate_id, operation_id, seconds, kind):
    """Reserve once; recover only a durable evaluator dispatch with the same ID."""
    fresh = budget.reserve(operation_id, seconds, kind)
    row = budget.read()["operations"][operation_id]
    if row["status"] == "settled":
        record = read_evaluation(evaluator, candidate_id, evaluation_id=operation_id)["receipt"]
        _need(not row.get("envelope_exceeded") and record["settled"] is True
              and record["native_suite_pass_observed"] is True
              and not record["collection_errors"] and not record["host_errors"],
              "settled final measurement failed or exceeded its envelope")
        return
    root = evaluator.root / operation_id
    if not fresh:
        _need((root / "request.json").is_file() and (root / "dispatch_intent.json").is_file(),
              "reservation has no durable evaluator dispatch; owner reconciliation required")
    executed, settled, cost = row["executed"], False, row["elapsed_seconds"] or 0
    try:
        try:
            response = evaluator.evaluate(candidate_id, evaluation_id=operation_id)
        except ArenaEvaluationUnsettled as exc:
            response = exc.result
        # Reread the complete owner collection, including artifacts omitted from
        # the response projection. Arena's existing validators own these facts.
        receipt = response["receipt"]
        collection = Path(receipt["collection"])
        _need(collection.parent == root, "measurement collection belongs to another operation")
        request = evaluator._request(candidate_id, operation_id)
        _need(decode(_read(root, "request.json")[0]) == request, "measurement request changed")
        host = decode(_read(collection, "host_receipt.json")[0])
        artifacts = {name: _read(collection / "raw", name)[0] for name in host["artifacts_sha256"]}
        settled = evaluator._envelope(request, host, artifacts)
        executed = executed or host["execution_started"]
        assessed = evaluator._observations(request, host, artifacts)
        native_cost = assessed["costs"]["wall_seconds"]
        if native_cost is not None:
            cost = max(cost, native_cost)
        _need(settled and not receipt["collection_errors"]
              and receipt["native_suite_pass_observed"] is True,
              "final measurement is unsettled or did not pass the complete native suite")
    finally:
        # Include recovery/downtime in the original reservation, never erase
        # known execution, and keep unknown cleanup charged and blocking.
        cost = max(cost, (time.monotonic_ns() - row["reserved_ns"]) / 1e9)
        budget.settle(operation_id, cost, executed, settled)
    _need(not budget.read()["operations"][operation_id].get("envelope_exceeded"),
          "final native operation exceeded its admitted envelope")


def _configuration(reference, store, evaluator, budget):
    config = _document(reference)
    _need(set(config) == {"schema", "search", "baseline_candidate_id", "final_measurements", "objective"}
          and config["schema"] == "geak-evo-admitted-command-v1", "exact admitted command config required")
    search = config["search"]
    _need(isinstance(evaluator, ArenaEvaluator) and evaluator.store is store,
          "existing Arena evaluator for this store required")
    _need(all(callable(getattr(budget, name, None)) for name in ("reserve", "settle", "read"))
          and budget.config == search["shared_budget"], "existing shared native budget differs")
    clock = _document(search["clock_ref"])
    _need(budget.config["boot_id"] == clock["boot_id"]
          == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
          and budget.config["session_id"] == clock["session_id"]
          and budget.config["deadline_ns"] == clock["started_monotonic_ns"] + int(clock["wall_seconds"] * 1e9),
          "shared budget must retain the original clock and boot")
    native = _document(search["templates"]["config"])
    probe = native["backend_config"]["probe_config"]
    _need(evaluator.contract == _document(native["contract_ref"])
          and evaluator.contract_id == probe["job_id"] and evaluator.epoch == probe["measurement_epoch_id"]
          and evaluator.root == Path(native["evaluation_root"])
          and evaluator.artifact_root == Path(native["artifact_root"]), "evaluator differs from native admission")
    protocol = evaluator.contract.get("report_contract", {}).get("published_protocol", {})
    _need(evaluator.contract["schema"] == "geak-arena-native-operator-v2"
          and protocol.get("schema") == "geak-published-task-protocol-v1"
          and protocol.get("report_format") == "task-report-v1" and protocol.get("private_cache_scope") == "phase",
          "command currently supports admitted native task-report-v1 with phase cache receipts")
    final = config["final_measurements"]
    _need(set(final) == {"baseline_seconds", "selected_seconds"}
          and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                  and value == probe["native_envelope_seconds"] for value in final.values()),
          "final measurements require the existing native operation envelope")
    baseline = store.get_candidate(config["baseline_candidate_id"])
    _need(baseline["job_id"] == evaluator.contract_id and baseline["measurement_epoch_id"] == evaluator.epoch,
          "baseline belongs to another task or epoch")
    source = store.source(baseline["source_id"])
    _need({row["path"] for row in source["files"]} == set(evaluator.contract["source_paths"]),
          "baseline source closure differs")
    for row in source["files"]:
        _need(row["blob_id"] is not None and
              (_read(store.root / "objects", row["blob_id"])[0], row["executable"])
              == _read(evaluator.artifact_root, row["path"]), "baseline differs from admitted task source")
    objective = config["objective"]
    _need(type(objective) is dict and objective.get("kind") in ("geomean", "time_weighted"),
          "invalid common measurement objective")
    weights = objective.get("weights")
    _need((weights is None or (type(weights) is dict and set(weights) == set(evaluator.contract["benchmark_case_ids"])
          and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in weights.values())))
          and (objective["kind"] != "time_weighted" or weights is not None), "invalid common workload weights")
    return config


def run_admitted_task(owner_config_ref, *, store, services, evaluator, budget, episode, output_dir, max_steps):
    """Search, measure ORIGINAL and the checked recommendation, then export.

    ``episode`` is the owner's existing NativeEvidence.episode reader. It also
    permits same-ID final settlement after the search clock has expired, without
    constructing a new search or launching another model call. An accrued
    operation-envelope overrun still returns incomplete. Config and max steps
    are fixed on first use. Each invocation writes to a new output folder.
    """
    _need(isinstance(services, HostServices) and callable(episode), "existing host services and native episode reader required")
    _need(type(max_steps) is int and 0 < max_steps <= 10000, "bounded positive search steps required")
    out = Path(output_dir)
    _need(out.is_absolute() and out.parent.resolve(strict=True) == out.parent
          and not out.exists() and not out.is_symlink(), "new canonical output directory required")
    config = _configuration(owner_config_ref, store, evaluator, budget)
    search_args = dict(config["search"])
    campaign = search_args.pop("campaign_id")
    seed = search_args.pop("seed_candidate_id")
    search_args["profile"] = ModelProfile(**search_args["profile"])
    root = Path(search_args["campaign_root"])
    _need(root.is_absolute() and root.resolve(strict=True) == root and root in store.root.resolve().parents,
          "canonical existing campaign root required")
    identity = {"owner_config_ref": owner_config_ref, "max_steps": max_steps}
    fd = os.open(root / "evo-command.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding_path, snapshot_path = root / "evo-command.json", root / "evo-command-search.json"
        if binding_path.exists():
            _need(decode(_read(root, binding_path.name)[0]) == identity, "saved command config or search bound changed")
        else:
            _write(binding_path, canonical_json(identity), immutable=True)
        snapshot, baseline_id, selected_id, stage = None, None, None, "search"
        try:
            if snapshot_path.exists():
                saved = decode(_read(root, snapshot_path.name)[0])
                _need(saved["command"] == identity, "saved search belongs to another command")
                revision = saved["revision"]
                snapshot = {"campaign_id": campaign, "revision": revision,
                            "state": store.get_revision(campaign, revision)["note"]["evolution"]}
            else:
                intent = root / "evo-command-search-intent.json"
                _need(not intent.exists(), "search started without a frozen return; owner reconciliation required")
                _need(all(row["status"] == "settled" and not row.get("envelope_exceeded")
                          for row in budget.read()["operations"].values()),
                      "pending or overrun native operation blocks search")
                search = prepare_native_search(store, campaign, seed, services=services, **search_args)
                _write(intent, canonical_json(identity), immutable=True)
                snapshot = search.run(max_steps=max_steps)
                _need(snapshot["state"]["phase"] in ("idle", "stopped"),
                      "pending search needs owner reconciliation before final measurement")
                _write(snapshot_path, canonical_json({"command": identity, "revision": snapshot["revision"]}), immutable=True)
            def current():
                head = store.read_current(campaign)
                _need(head["revision"] == snapshot["revision"] and head["note"]["evolution"] == snapshot["state"],
                      "campaign advanced beyond the frozen search; owner reconciliation required")
            current()
            stage = "selection"
            options = {"expected_binding": {"campaign_id": campaign, "job_id": evaluator.contract_id,
                                           "measurement_epoch_id": evaluator.epoch},
                       "clock_ref": search_args["clock_ref"], "read_ref": read_ref,
                       "authenticate_choice": lambda context, choice: authenticate_choice(store, context, choice, episode),
                       "runtime": services.model_runtime}
            selection = validate_final_recommendation(store, snapshot, **options)
            if selection["status"] == "selected":
                choice = selection["choice"]
                def operation(role, candidate):
                    return "evo-" + role + "-" + sha256_hex(canonical_json({
                        "command": identity, "candidate_id": candidate, "revision": snapshot["revision"]}))[:48]
                baseline_id = operation("baseline", config["baseline_candidate_id"])
                selected_id = operation("final", choice["candidate_id"])
                stage = "baseline"
                _measure(evaluator, budget, config["baseline_candidate_id"], baseline_id,
                         config["final_measurements"]["baseline_seconds"], "baseline")
                current()
                stage = "final"
                _measure(evaluator, budget, choice["candidate_id"], selected_id,
                         config["final_measurements"]["selected_seconds"], "final")
            current()
            stage = "export"
            result = export_final_result(store, snapshot, evaluator=evaluator, selection_options=options,
                baseline_candidate_id=config["baseline_candidate_id"], baseline_evaluation_id=baseline_id,
                selected_evaluation_id=selected_id, output_dir=out,
                objective=config["objective"])
        except (CommandInterrupted, EvoError, ValueError, OSError, KeyError, TypeError) as exc:
            result = {"eval_dir": str(out), "outcome": "error", "validation_status": "command_incomplete",
                      "final_patch": "", "final_geomean": None, "final_weighted": None,
                      "final_arithmetic": None, "final_speedup": None,
                      "workload_aligned": config["objective"]["kind"] == "time_weighted", "timing_basis": "unknown",
                      "measurement_valid": False, "custody_proven": False, "reward_eligible": False,
                      "framework_final_pass": False, "error_kind": type(exc).__name__, "error": str(exc)}
            out.mkdir(mode=0o700, exist_ok=True)
        result["command"] = {**identity, "stage": stage, "campaign_id": campaign,
                             "campaign_revision": None if snapshot is None else snapshot["revision"],
                             "baseline_evaluation_id": baseline_id, "selected_evaluation_id": selected_id}
        _write(out / "result.json", canonical_json(result) + b"\n")
        return result
    finally:
        os.close(fd)


def main(argv=None, *, store, services, evaluator, budget, episode):
    """Callable CLI for an installed owner entrypoint with already-bound services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-config", nargs=2, metavar=("PATH", "SHA256"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args(argv)
    result = run_admitted_task({"path": args.owner_config[0], "sha256": args.owner_config[1]},
        store=store, services=services, evaluator=evaluator, budget=budget, episode=episode,
        output_dir=args.output_dir, max_steps=args.max_steps)
    return 0 if result["outcome"] in ("improved", "no_improvement") else 1
