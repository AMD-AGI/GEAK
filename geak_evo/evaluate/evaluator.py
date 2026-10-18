"""Collect provisional Arena observations through explicit host callbacks.

No task Python is imported and no process or device is launched by this module.
The host must bind its existing owned lifecycle to ``launch`` and ``settle``.
Neither native reports nor the supplemental exported-value veto prove that the
timed graph, inputs and Python judge have independent custody.
"""

import copy
import fcntl
import json

from .._util import EvoError, canonical_json, require, sha256_hex
import math
import os
from pathlib import Path
import re

from .jobs import declared_call_counts, job_digest, validate_arena_job
from .artifacts import pack
from ..archive.store import _path, _read, _write


class ArenaEvaluationError(EvoError, ValueError):
    pass


class ArenaEvaluationUnsettled(ArenaEvaluationError):
    """Raw collection is retained; reconcile this exact evaluation operation."""

    def __init__(self, result):
        super().__init__("Arena evaluator cleanup is unproven; reconcile the same evaluation_id")
        self.result = result


def canonical(value):
    return canonical_json(value)


def sha256(raw):
    return sha256_hex(raw)




def _number(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _save(path, value):
    _write(path, canonical(value) + b"\n", immutable=True)


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key", ArenaEvaluationError)
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=pairs)
    canonical(result)
    return result


class ArenaEvaluator:
    """One durable operation per candidate/evaluation ID, including recovery.

    ``launch(request, operation_dir) -> opaque_handle`` receives a detached JSON
    request after request.json and dispatch_intent.json are durable.
    ``settle(handle_or_None, request, operation_dir) -> {receipt, artifacts}``
    must always reconcile the same operation. None on recovery never permits a
    relaunch. Artifacts map safe relative names to bytes. See ``_envelope`` and
    ``_observations`` for the exact settlement-v1 shape; cleanup evidence names
    must identify retained host-owned lifecycle artifacts. Callbacks, including
    their interpretation of raw Arena reports, are a trusted host boundary.

    ``evaluate`` returns {receipt, artifacts, costs}. Missing or unauthenticated
    positive evaluator cleanup raises ArenaEvaluationUnsettled *after* retaining
    raw artifacts. A positively settled failed run returns a failure observation.
    Non-Exception interruptions propagate after the collection receipt is saved.
    All observations remain ineligible for reward and anchor promotion.
    """

    def __init__(self, store, *, contract, contract_id, artifact_root, output_root,
                 measurement_epoch_id, launch=None, settle=None):
        self.store = store
        self.contract = validate_arena_job(contract, artifact_root)
        self.contract_id = job_digest(self.contract)
        require(contract_id == self.contract_id, "Arena contract digest mismatch", ArenaEvaluationError)
        self.artifact_root = Path(artifact_root).resolve(strict=True)
        self.epoch = measurement_epoch_id
        require(isinstance(self.epoch, str) and self.epoch.strip(), "explicit Arena epoch required", ArenaEvaluationError)
        self.root = Path(output_root).absolute()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(self.root.resolve(strict=True) == self.root, "canonical evaluator root required", ArenaEvaluationError)
        self.launch, self.settle = launch, settle
        self.evaluator_sha256 = sha256(_read(Path(__file__).parent, Path(__file__).name)[0])

    def declared_call_counts(self):
        """Timer invocations per case within one process/leg, warmups separate."""
        return declared_call_counts(self.contract)

    def _request(self, candidate_id, evaluation_id):
        require(isinstance(evaluation_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", evaluation_id),
              "invalid Arena evaluation ID", ArenaEvaluationError)
        validate_arena_job(self.contract, self.artifact_root)
        candidate = self.store.get_candidate(candidate_id)
        require(candidate["job_id"] == self.contract_id and candidate["measurement_epoch_id"] == self.epoch,
              "candidate contract/epoch mismatch", ArenaEvaluationError)
        source = self.store.source(candidate["source_id"])
        require([item["path"] for item in source["files"]] == sorted(self.contract["source_paths"]),
              "candidate source closure mismatch", ArenaEvaluationError)
        # Missing editable files remain an explicit candidate failure for the
        # host to report; they never fall back to the founding source bytes.
        objects = {item["blob_id"]: str(self.store.root / "objects" / item["blob_id"])
                   for item in source["files"] if item["blob_id"] is not None}
        return {"schema": "geak-arena-evaluation-request-v1", "operation_id": evaluation_id,
                "binding": {"job_id": self.contract_id, "measurement_epoch_id": self.epoch,
                            "candidate_id": candidate_id, "candidate_source_id": candidate["source_id"]},
                "evaluator_sha256": self.evaluator_sha256, "contract": self.contract,
                "artifact_root": str(self.artifact_root), "source": source, "source_objects": objects,
                "correctness_case_ids": self.contract["correctness_case_ids"],
                "benchmark_case_ids": self.contract["benchmark_case_ids"],
                "benchmark_report_ids": self.contract["benchmark_report_ids"],
                "phases": self.contract["phases"], "measurement_valid": False,
                "custody_proven": False, "reward_eligible": False}

    def _prepare(self, candidate_id, evaluation_id):
        request = self._request(candidate_id, evaluation_id)
        root = self.root / evaluation_id
        existing = root.exists()
        if existing:
            require(root.resolve(strict=True) == root, "unsafe operation directory", ArenaEvaluationError)
            # An interrupted preparation with no request is ambiguous. Preserve
            # it, reject reuse and never interpret absence as no dispatch.
            retained = _parse(_read(root, "request.json")[0])
            require(retained == request, "evaluation ID belongs to a different contract/source/epoch or evaluator", ArenaEvaluationError)
        else:
            root.mkdir(mode=0o700)
            _save(root / "request.json", request)
        return request, root, existing

    def prepare(self, candidate_id, *, evaluation_id):
        """Persist identity without dispatch; later evaluate only reconciles it.

        A prepared ID is never a launch ticket. This conservative rule also
        covers interruption between durable request and dispatch intent.
        """
        with self._lock():
            request, _, _ = self._prepare(candidate_id, evaluation_id)
            return request

    def _lock(self):
        from contextlib import contextmanager
        @contextmanager
        def locked():
            fd = os.open(self.root / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
            finally:
                os.close(fd)
        return locked()

    @staticmethod
    def _envelope(request, receipt, artifacts):
        fields = {"schema", "operation_id", "request_sha256", "binding", "execution_started",
                  "cleanup", "runtime", "phase_exit_codes", "observations", "artifacts_sha256", "costs", "errors"}
        require(type(receipt) is dict and set(receipt) == fields, "invalid Arena settlement fields", ArenaEvaluationError)
        require(receipt["schema"] == "geak-arena-settlement-v1"
              and receipt["operation_id"] == request["operation_id"]
              and receipt["request_sha256"] == sha256(canonical(request))
              and receipt["binding"] == request["binding"], "Arena settlement identity mismatch", ArenaEvaluationError)
        require(receipt["artifacts_sha256"] == {name: sha256(raw) for name, raw in artifacts.items()},
              "Arena settlement artifact hashes differ", ArenaEvaluationError)
        require(type(receipt["execution_started"]) is bool, "explicit execution state required", ArenaEvaluationError)
        cleanup = receipt["cleanup"]
        require(type(cleanup) is dict and set(cleanup) == {"settled", "evidence"}
              and type(cleanup["settled"]) is bool, "explicit evaluator cleanup required", ArenaEvaluationError)
        require(type(cleanup["evidence"]) is list
              and all(type(name) is str and name in artifacts for name in cleanup["evidence"]),
              "cleanup evidence missing from retained artifacts", ArenaEvaluationError)
        require(not cleanup["settled"] or bool(cleanup["evidence"]), "positive cleanup needs host evidence", ArenaEvaluationError)
        return cleanup["settled"]

    @staticmethod
    def _observations(request, receipt, artifacts):
        runtime = receipt["runtime"]
        require(type(runtime) is dict and set(runtime) == {"image", "config_digest", "physical_devices", "evidence"},
              "explicit observed runtime/device state required, with null for unknown", ArenaEvaluationError)
        for key in ("image", "config_digest"):
            require(runtime[key] is None or runtime[key] == request["contract"]["runtime"][key],
                  "observed runtime differs from the declared public runtime", ArenaEvaluationError)
        devices = runtime["physical_devices"]
        require(devices is None or (type(devices) is list and devices
              and all(type(device) is str and device.strip() for device in devices)
              and len(devices) == len(set(devices))), "invalid observed physical devices", ArenaEvaluationError)
        require(type(runtime["evidence"]) is list
              and all(type(name) is str and name in artifacts for name in runtime["evidence"]),
              "runtime/device evidence missing", ArenaEvaluationError)
        require(all(runtime[key] is None for key in ("image", "config_digest", "physical_devices"))
              or bool(runtime["evidence"]), "observed runtime/device identity requires host evidence", ArenaEvaluationError)
        phases = receipt["phase_exit_codes"]
        require(type(phases) is dict and set(phases) == set(request["phases"])
              and all(value is None or type(value) is int for value in phases.values()),
              "invalid phase exit-code map", ArenaEvaluationError)
        errors = receipt["errors"]
        require(type(errors) is list and all(type(item) is str for item in errors), "invalid host error list", ArenaEvaluationError)
        costs = receipt["costs"]
        require(type(costs) is dict and set(costs) == {"wall_seconds", "evaluations"}
              and (costs["wall_seconds"] is None or _number(costs["wall_seconds"]))
              and type(costs["evaluations"]) is int and costs["evaluations"] in (0, 1), "invalid evaluator cost vector", ArenaEvaluationError)
        require(costs["evaluations"] == int(receipt["execution_started"]), "evaluation count differs from execution state", ArenaEvaluationError)
        obs = receipt["observations"]
        require(type(obs) is dict and set(obs) == {"correctness", "benchmark"}, "invalid observation groups", ArenaEvaluationError)
        correct, bench = obs["correctness"], obs["benchmark"]
        for row, fields in ((correct, {"status", "case_ids", "evidence"}),
                            (bench, {"status", "times_ms", "evidence"})):
            require(type(row) is dict and set(row) == fields
                  and row["status"] in ("passed", "failed", "not_run", "unknown"), "invalid native observation", ArenaEvaluationError)
            require(type(row["evidence"]) is list and all(type(name) is str and name in artifacts for name in row["evidence"]),
                  "native observation evidence missing", ArenaEvaluationError)
            require(row["status"] in ("not_run", "unknown") or bool(row["evidence"]), "native outcome needs raw evidence", ArenaEvaluationError)
        ids = correct["case_ids"]
        require(type(ids) is list and all(type(case) is str for case in ids)
              and len(ids) == len(set(ids)) and set(ids) <= set(request["correctness_case_ids"]), "wrong correctness cases", ArenaEvaluationError)
        require(correct["status"] != "passed" or ids == request["correctness_case_ids"], "native pass must cover every correctness case", ArenaEvaluationError)
        times = bench["times_ms"]
        require(type(times) is dict and set(times) <= set(request["benchmark_report_ids"].values())
              and all(_number(value) and value > 0 for value in times.values()), "wrong native benchmark IDs or times", ArenaEvaluationError)
        require(bench["status"] != "passed" or set(times) == set(request["benchmark_report_ids"].values()),
              "native benchmark pass must cover exact scored domain", ArenaEvaluationError)
        passed = (receipt["execution_started"] and not errors and all(value == 0 for value in phases.values())
                  and correct["status"] == "passed" and bench["status"] == "passed")
        return {"native_suite_pass_observed": bool(passed), "native_observations": obs,
                "observed_runtime": runtime,
                "per_case_times_ms": {case: times[report] for case, report in request["benchmark_report_ids"].items()
                                      if report in times}, "costs": costs, "host_errors": errors}

    def evaluate(self, candidate_id, *, evaluation_id):
        require(callable(self.launch) and callable(self.settle), "explicit owned evaluator callbacks required", ArenaEvaluationError)
        with self._lock():
            request, root, existing = self._prepare(candidate_id, evaluation_id)
            intent = root / "dispatch_intent.json"
            recovered, handle, errors = existing, None, []
            interruption = None
            if recovered:
                require(not intent.exists() or _parse(_read(root, intent.name)[0]) == {"request_sha256": sha256(canonical(request))},
                      "dispatch intent differs from retained request", ArenaEvaluationError)
            else:
                _save(intent, {"request_sha256": sha256(canonical(request))})
                try:
                    handle = self.launch(copy.deepcopy(request), root)
                except BaseException as exc:
                    errors.append({"stage": "launch", "error_kind": type(exc).__name__})
                    if not isinstance(exc, Exception):
                        interruption = exc
            artifacts, receipt = {}, None
            try:
                collected = self.settle(handle, copy.deepcopy(request), root)
                require(type(collected) is dict and set(collected) == {"receipt", "artifacts"}, "invalid callback response", ArenaEvaluationError)
                receipt = collected["receipt"]
                require(type(collected["artifacts"]) is dict, "artifact bytes map required", ArenaEvaluationError)
                for name, raw in collected["artifacts"].items():
                    _path(name)
                    require(type(raw) is bytes, "artifact must contain bytes", ArenaEvaluationError)
                    artifacts[name] = raw
            except BaseException as exc:
                errors.append({"stage": "settle", "error_kind": type(exc).__name__})
                if not isinstance(exc, Exception):
                    interruption = exc
            indices = [int(path.name[11:]) for path in root.glob("collection-*")
                       if re.fullmatch(r"collection-[0-9]+", path.name)]
            collection = root / ("collection-%04d" % (max(indices, default=0) + 1))
            collection.mkdir(mode=0o700)
            retained = {"raw/" + name: raw for name, raw in artifacts.items()}
            for name, raw in retained.items():
                target = collection / name
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                _write(target, raw, immutable=True)
            if receipt is not None:
                try:
                    _save(collection / "unpacked_host_receipt.json", receipt)
                    retained["host_receipt.json"] = canonical(receipt) + b"\n"
                except (TypeError, ValueError):
                    errors.append({"stage": "receipt", "error_kind": "non_json_receipt"})
            settled, assessed = False, {}
            try:
                settled = self._envelope(request, receipt, artifacts)
                assessed = self._observations(request, receipt, artifacts)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append({"stage": "validation", "error_kind": type(exc).__name__, "reason": str(exc)})
            result = {"schema": "geak-arena-evaluation-v1", "evaluation_id": evaluation_id,
                      "binding": request["binding"], "request_sha256": sha256(canonical(request)),
                      "collection": str(collection), "reconciled": recovered, "settled": settled,
                      "native_suite_pass_observed": False, "per_case_times_ms": {},
                      "observed_runtime": {"image": None, "config_digest": None, "physical_devices": None, "evidence": []},
                      "measurement_valid": False, "custody_proven": False, "reward_eligible": False,
                      "framework_final_pass": False, "observed_frequency_aggregate": None,
                      "limitations": ["Native task Python is not an independent hostile-code judge.",
                                      "Exported-value numerical veto does not bind timed graph or input state.",
                                      "Current public runtime is not exact historical serving capture."],
                      **assessed, "collection_errors": errors}
            if errors or not settled:
                result["native_suite_pass_observed"] = False
            costs = assessed.get("costs", {"wall_seconds": None, "evaluations": None})
            if receipt is not None and not errors:
                # The original full callback map and both validators above are
                # authoritative. Packing only projects transport; raw files and
                # unpacked_host_receipt.json remain in this private collection.
                response = pack(artifacts, receipt, result, request)["response"]
                result, retained = response["receipt"], response["artifacts"]
            else:
                retained["arena_evaluation.json"] = canonical(result) + b"\n"
                retained["request.json"] = canonical(request) + b"\n"
                response = {"receipt": result, "artifacts": retained, "costs": costs}
            if "host_receipt.json" in retained:
                _write(collection / "host_receipt.json", retained["host_receipt.json"], immutable=True)
            _save(collection / "arena_evaluation.json", result)
            if interruption is not None:
                raise interruption
            if not settled:
                raise ArenaEvaluationUnsettled(response)
            return response
