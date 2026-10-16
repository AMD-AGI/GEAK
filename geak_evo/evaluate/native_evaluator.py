"""Frozen operator evaluation through explicitly supplied host authority.

This adapter never grants devices or starts a subprocess. ``launch`` and
``settle`` are trusted host callbacks, not values supplied by an engineer. The
old GEAK operator supplies descriptor validation and numeric comparison; its
separate Python processes alone are not an adversarial execution sandbox.
"""

import fcntl
from contextlib import ExitStack
import json

from .._util import EvoError, canonical_json, require, sha256_hex
import math
from pathlib import Path
import random
import re
import statistics
import types

from .jobs_v1 import job_digest, validate_job
from ..archive.store import _read, _write
from ..clock import SystemClock


class NativeEvaluationError(EvoError, ValueError):
    pass


class NativeEvaluationUnsettled(NativeEvaluationError):
    """Collection is retained; the caller must keep its opportunity pending."""

    def __init__(self, result):
        super().__init__("native evaluation remains unsettled; reconcile the same evaluation_id")
        self.result = result


def canonical(value):
    return canonical_json(value)


def sha256(data):
    return sha256_hex(data)




def _parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key", NativeEvaluationError)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           NativeEvaluationError("nonfinite JSON")))
    canonical(value)
    return value


def _number(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _save(path, value):
    _write(path, canonical(value) + b"\n", immutable=True)


def _new_collection(root):
    # An interrupted write can leave a directory without its final receipt.
    # Preserve it and advance; never reuse an incomplete collection slot.
    indices = [int(path.name[11:]) for path in root.glob("collection-*")
               if re.fullmatch(r"collection-[0-9]+", path.name)]
    path = root / ("collection-%04d" % (max(indices, default=0) + 1))
    path.mkdir(mode=0o700)
    return path


class NativeEvaluator:
    """Collect correctness then repeated timing for one exact archived source.

    ``operator_tool`` is a host-selected {path, sha256} reference to the existing
    GEAK operator_task.py. Only that pinned trusted module is imported, for its
    offline descriptor validator; candidate and adapter modules are never
    imported by this host process.

    launch(request, attempt_dir) returns an opaque owner handle. Always called
    settlement, including after launch failure, is
    settle(handle_or_None, request, attempt_dir) -> {receipt, artifacts}.
    Artifacts are bytes; the receipt authenticates their exact hashes and the
    request hash. On recovery None means reconcile the existing operation, never
    relaunch it. The assessment methods below validate its exact fields.

    evaluate returns {receipt, artifacts, costs}. The driver combines these costs
    with engineer/planner costs; an unreported cost remains None. This adapter
    does not author a scientific archive card, research note or claim.
    """

    def __init__(self, store, *, job, artifact_root, operator_tool, output_root,
                 measurement_epoch_id, launch=None, settle=None, repeats=2,
                 max_relative_spread=0.10, performance=True, phase_timeout_seconds=300,
                 clock=None):
        self.store = store
        self.job = validate_job(job, artifact_root)
        self.job_id = job_digest(self.job)
        self.artifact_root = Path(artifact_root).resolve(strict=True)
        self.epoch = measurement_epoch_id
        require(isinstance(self.epoch, str) and self.epoch.strip(), "epoch required", NativeEvaluationError)
        require(type(repeats) is int and repeats >= 2, "timing requires at least two separate operations", NativeEvaluationError)
        require(_number(max_relative_spread), "invalid timing spread limit", NativeEvaluationError)
        require(type(performance) is bool, "explicit performance selection required", NativeEvaluationError)
        require(_number(phase_timeout_seconds) and phase_timeout_seconds > 0, "positive phase timeout required", NativeEvaluationError)
        self.repeats, self.max_spread, self.performance = repeats, max_relative_spread, performance
        self.phase_timeout = phase_timeout_seconds
        self.launch, self.settle = launch, settle
        self.clock = clock if clock is not None else SystemClock()
        self.root = Path(output_root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(not self.root.is_symlink(), "evaluator storage cannot be a symlink", NativeEvaluationError)
        path = Path(operator_tool["path"])
        self.operator_bytes = _read(path.parent, path.name)[0]
        require(sha256(self.operator_bytes) == operator_tool["sha256"], "pinned operator changed", NativeEvaluationError)
        self.operator_sha = operator_tool["sha256"]
        module = types.ModuleType("_geak_evo_pinned_operator")
        module.__file__ = str(path)
        exec(compile(self.operator_bytes, str(path), "exec"), module.__dict__)
        self.operator = module
        self.descriptor_raw = _read(self.artifact_root, "operator_task.json")[0]
        self.descriptor = module.validate_descriptor(_parse(self.descriptor_raw))
        desc = self.descriptor
        require(self.job["artifacts"].get("operator_task.json") == sha256(self.descriptor_raw),
              "operator descriptor must be in the immutable job", NativeEvaluationError)
        require(all(self.job["artifacts"].get(name) == digest for name, digest in desc["frozen_files"].items()),
              "operator frozen files must be exact job artifacts", NativeEvaluationError)
        require(set(desc["editable_files"]) == set(self.job["baseline"]["source_paths"]),
              "operator editable closure differs from job source closure", NativeEvaluationError)
        self.case_ids = sorted(case["case_id"] for case in self.job["cases"])
        for group in ("correctness_cases", "performance_cases"):
            require(sorted(case["id"] for case in desc[group]) == self.case_ids,
                  "operator must cover every frozen job case", NativeEvaluationError)
        require(self.job["graph"]["mode"] == "disabled", "operator does not implement graph timing", NativeEvaluationError)
        require(self.job["timing"]["aggregation"] == "mean"
              and self.job["timing"]["warmup_calls"] == desc["timing"]["warmup"]
              and self.job["timing"]["measured_calls"] == desc["timing"]["iterations"],
              "job timing differs from per-call synchronized operator mean", NativeEvaluationError)

    def prepare(self, candidate_id, *, evaluation_id):
        """Freeze exact bytes and intent without invoking launch or settlement."""
        require(isinstance(evaluation_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", evaluation_id),
              "invalid evaluation_id", NativeEvaluationError)
        candidate = self.store.get_candidate(candidate_id)
        require(candidate["job_id"] == self.job_id and candidate["measurement_epoch_id"] == self.epoch,
              "candidate job/epoch differs from evaluator", NativeEvaluationError)
        source = self.store.source(candidate["source_id"])
        require([item["path"] for item in source["files"]] == sorted(self.descriptor["editable_files"]),
              "candidate source closure differs from descriptor", NativeEvaluationError)
        require(all(item["blob_id"] is not None for item in source["files"]), "candidate deleted a required source", NativeEvaluationError)
        root = self.root / evaluation_id
        if root.exists():
            prepared = _parse(_read(root, "prepared.json")[0])
            require(prepared["binding"]["candidate_id"] == candidate_id
                  and prepared["binding"]["job_id"] == self.job_id
                  and prepared["binding"]["measurement_epoch_id"] == self.epoch
                  and prepared["binding"]["operator_sha256"] == self.operator_sha
                  and prepared["policy"] == self._policy(), "evaluation_id belongs to another intent", NativeEvaluationError)
            self._verify_snapshot(prepared)
            return prepared
        root.mkdir(mode=0o700)
        snapshot = root / "snapshot"
        self.store.materialize(candidate["source_id"], snapshot)
        for name, digest in self.job["artifacts"].items():
            if name in self.descriptor["editable_files"]:
                continue
            raw = _read(self.artifact_root, name)[0]
            require(sha256(raw) == digest, "job artifact changed during snapshot", NativeEvaluationError)
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _write(target, raw, immutable=True)
        manifest = {name: sha256(_read(snapshot, name)[0]) for name in self.job["artifacts"]}
        for item in source["files"]:
            manifest[item["path"]] = item["blob_id"]
        identity = {"descriptor_sha256": sha256(self.descriptor_raw),
                    "frozen_files": self.descriptor["frozen_files"],
                    "source_files": {item["path"]: item["blob_id"] for item in source["files"]}}
        require(self.operator.identity(snapshot) == identity, "snapshot differs from declared operator identity", NativeEvaluationError)
        _write(root / "operator_task.py", self.operator_bytes, immutable=True)
        _save(root / "source.json", source)
        _save(root / "job.json", self.job)
        binding = {"job_id": self.job_id, "measurement_epoch_id": self.epoch,
                   "candidate_id": candidate_id, "candidate_source_id": candidate["source_id"],
                   "case_ids": self.case_ids, "descriptor_sha256": identity["descriptor_sha256"],
                   "operator_sha256": self.operator_sha, "snapshot_sha256": sha256(canonical(manifest))}
        prepared = {"schema": "geak-evo-native-prepared-v1", "evaluation_id": evaluation_id,
                    "binding": binding, "workspace": str(snapshot), "snapshot_manifest": manifest,
                    "operator_tool": {"path": str(root / "operator_task.py"), "sha256": self.operator_sha},
                    "operator_identity": identity, "policy": self._policy()}
        _save(root / "prepared.json", prepared)
        return prepared

    def _policy(self):
        return {"performance": self.performance, "repeats": self.repeats,
                "max_relative_spread": self.max_spread, "phase_timeout_seconds": self.phase_timeout}

    def _timeout(self, mode):
        return min(self.phase_timeout, self.descriptor["limits"]["worker_timeout_s"] * len(self.case_ids)
                   * (2 if mode == "correctness" else 1) + 30)

    @property
    def max_payload_seconds(self):
        """Declared payload ceiling, excluding trusted setup and positive cleanup."""
        return self._timeout("correctness") + (self.repeats * self._timeout("performance") if self.performance else 0)

    @property
    def max_pair_payload_seconds(self):
        """Both sources' payload ceilings; reserve setup/cleanup separately."""
        return 2 * self.max_payload_seconds

    def _verify_snapshot(self, prepared):
        for name, digest in prepared["snapshot_manifest"].items():
            require(sha256(_read(prepared["workspace"], name)[0]) == digest, "executed snapshot changed", NativeEvaluationError)
        ref = prepared["operator_tool"]
        path = Path(ref["path"])
        require(sha256(_read(path.parent, path.name)[0]) == self.operator_sha == ref["sha256"],
              "retained evaluator changed", NativeEvaluationError)
        require(self.store.capture_source(prepared["workspace"], self.descriptor["editable_files"])
              == prepared["binding"]["candidate_source_id"], "source bytes or executable modes changed", NativeEvaluationError)

    def _assess(self, request, receipt, artifacts, errors):
        """Interpret authenticated collection, never infer causes from exit code."""
        result = {"success": False, "settled": False, "correctness_observed": False,
                  "times_ms": None, "errors": list(errors)}
        try:
            require(receipt.get("schema") == "geak-evo-native-settlement-v1"
                  and receipt.get("operation_id") == request["operation_id"]
                  and receipt.get("request_sha256") == sha256(canonical(request)), "settlement request mismatch", NativeEvaluationError)
            for flag in ("execution_started", "settled", "evaluation_integrity_valid", "measurement_valid", "peer_exception_active"):
                require(type(receipt.get(flag)) is bool, "missing explicit settlement flag: " + flag, NativeEvaluationError)
            require(receipt.get("artifacts_sha256") == {name: sha256(raw) for name, raw in artifacts.items()},
                  "settlement artifact hashes differ", NativeEvaluationError)
            require(type(receipt.get("errors")) is list, "missing boundary errors", NativeEvaluationError)
            result["settled"] = receipt["settled"]
            require(receipt["settled"], "owned execution remains unsettled", NativeEvaluationError)
            require(receipt["execution_started"] and type(receipt.get("exit_code")) is int,
                  "native execution was not confirmed", NativeEvaluationError)
            require(not receipt["errors"] and not errors, "launch or boundary failure; inspect raw evidence", NativeEvaluationError)
            require(receipt.get("qualification_ref") is not None, "no launch qualification reference", NativeEvaluationError)
            devices = receipt.get("physical_devices")
            require(type(devices) is list and devices and all(isinstance(device, str) and device for device in devices)
                  and len(devices) == len(set(devices)), "physical device identity missing", NativeEvaluationError)
            require("stdout.bin" in artifacts and "stderr.bin" in artifacts, "native streams missing", NativeEvaluationError)
            require(receipt["exit_code"] == 0, "native command failed; exit code does not identify numerical failure", NativeEvaluationError)
            if request["mode"] == "correctness":
                prefix = b"GEAK_CORRECTNESS_PROGRESS: "
                records = [line[len(prefix):] for line in artifacts["stdout.bin"].splitlines() if line.startswith(prefix)]
                require(len(records) == 1, "exactly one correctness progress record required", NativeEvaluationError)
                progress = _parse(records[0])
                require(progress.get("schema") == "geak-operator-correctness-progress-v1"
                      and progress.get("identity_before") == request["operator_identity"]
                      and progress.get("identity_after") == request["operator_identity"]
                      and progress.get("completed") is True and progress.get("failure") is None,
                      "correctness report identity/completion mismatch", NativeEvaluationError)
                rows = progress.get("cases")
                require(type(rows) is list and all(type(row) is dict for row in rows)
                      and sorted(row.get("case_id", "") for row in rows) == self.case_ids
                      and all(row.get("status") == "pass" for row in rows), "correctness case set did not pass", NativeEvaluationError)
                result["correctness_observed"] = True
            else:
                report = _parse(artifacts["performance_report.json"])
                rows, times = report.get("test_cases"), {}
                require(type(rows) is list and rows, "performance cases missing", NativeEvaluationError)
                for row in rows:
                    case, value = row.get("test_case_id"), row.get("execution_time_ms")
                    require(isinstance(case, str) and case not in times and _number(value) and value > 0,
                          "invalid performance case", NativeEvaluationError)
                    require(row.get("params", {}).get("timing_scope") == "complete_callable_synchronized_wall",
                          "performance timing scope mismatch", NativeEvaluationError)
                    times[case] = value
                require(sorted(times) == self.case_ids, "performance case set changed", NativeEvaluationError)
                result["times_ms"] = times
            result["success"] = True
        except Exception as exc:
            result["errors"].append(type(exc).__name__ + ": " + str(exc))
        return result

    def operation_requests(self, prepared):
        """Expose exact requests for protected launch review before execution."""
        requests = []
        modes = ["correctness"] + (["performance"] * self.repeats if self.performance else [])
        for index, mode in enumerate(modes):
            request = {key: prepared[key] for key in ("binding", "workspace", "operator_tool", "snapshot_manifest", "operator_identity")}
            request.update(schema="geak-evo-native-operation-v1", operation_id=prepared["evaluation_id"] + "-" + str(index),
                           mode=mode, timeout_seconds=self._timeout(mode))
            requests.append(_parse(canonical(request)))
        return requests

    def _operation(self, prepared, index, mode):
        root = self.root / prepared["evaluation_id"] / ("operation-%03d" % index)
        request = self.operation_requests(prepared)[index]
        require(request["mode"] == mode, "operation mode differs from prepared policy", NativeEvaluationError)
        fresh = not root.exists()
        if fresh:
            root.mkdir(mode=0o700)
            _save(root / "request.json", request)  # Commit before crossing the launch boundary.
        else:
            require(_parse(_read(root, "request.json")[0]) == request, "retained native request changed", NativeEvaluationError)
        prior = sorted(root.glob("collection-*/result.json"))
        if prior:
            row = _parse(_read(prior[-1].parent, "result.json")[0])
            if row["assessment"]["settled"]:
                return row
        handle, errors, interruption = None, [], None
        started = self.clock.monotonic()
        if fresh:
            try:
                handle = self.launch(request, root)
            except BaseException as exc:
                errors.append("launch:" + type(exc).__name__)
                interruption = exc if not isinstance(exc, Exception) else None
        collection = _new_collection(root)
        receipt, artifacts = {}, {}
        try:
            result = self.settle(handle, request, root)
            require(type(result) is dict and set(result) == {"receipt", "artifacts"}, "invalid trusted settlement result", NativeEvaluationError)
            receipt, artifacts = result["receipt"], result["artifacts"]
            require(type(receipt) is dict and type(artifacts) is dict
                  and all(isinstance(value, bytes) for value in artifacts.values()), "settlement requires artifact bytes", NativeEvaluationError)
            canonical(receipt)
            for name, raw in artifacts.items():
                # Reuse the operator's strict relative-path validator before writes.
                self.operator.relative(name)
                target = collection / "artifacts" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                _write(target, raw, immutable=True)
        except BaseException as exc:
            errors.append("settlement:" + type(exc).__name__)
            interruption = exc if not isinstance(exc, Exception) else interruption
            try:
                canonical(receipt)
            except (TypeError, ValueError, OverflowError):
                receipt = {"invalid_receipt": "trusted callback did not return finite JSON"}
            if type(receipt) is not dict:
                receipt = {"invalid_receipt": "trusted callback did not return an object"}
            if type(artifacts) is not dict or not all(isinstance(raw, bytes) for raw in artifacts.values()):
                artifacts = {}
        try:
            self._verify_snapshot(prepared)
        except Exception as exc:
            errors.append("snapshot:" + type(exc).__name__)
        assessment = self._assess(request, receipt, artifacts, errors)
        row = {"request": request, "receipt": receipt, "assessment": assessment,
               "adapter_errors": errors, "collection": str(collection), "wall_seconds": self.clock.monotonic() - started}
        _save(collection / "result.json", row)
        if interruption is not None:
            raise interruption
        return row

    def _artifacts(self, prepared):
        root = self.root / prepared["evaluation_id"]
        artifacts = {}
        for path in root.glob("operation-*/collection-*/result.json"):
            row = _parse(_read(path.parent, path.name)[0])
            request = _parse(_read(path.parent.parent, "request.json")[0])
            require(row["request"] == request, "retained collection request changed", NativeEvaluationError)
            raw = {name: _read(path.parent / "artifacts", name)[0]
                   for name in row["receipt"].get("artifacts_sha256", {})}
            require(row["assessment"] == self._assess(request, row["receipt"], raw, row["adapter_errors"]),
                  "retained collection artifacts or assessment changed", NativeEvaluationError)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "lock":
                name = path.relative_to(root).as_posix()
                artifacts["native/" + name] = _read(root, name)[0]
        return artifacts

    def evaluate(self, candidate_id, *, evaluation_id):
        """Run once, or reconcile the same operation without duplicate execution."""
        require(callable(self.launch) and callable(self.settle), "explicit trusted launch and settlement callbacks required", NativeEvaluationError)
        prepared = self.prepare(candidate_id, evaluation_id=evaluation_id)
        root = self.root / evaluation_id
        with (root / "lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            prior = sorted(root.glob("evaluation-*.json"))
            if prior:
                receipt = _parse(_read(root, prior[-1].name)[0])
                if receipt["settled"]:
                    artifacts = self._artifacts(prepared)
                    artifacts["native_evaluation.json"] = canonical(receipt) + b"\n"
                    return {"receipt": receipt, "artifacts": artifacts, "costs": receipt["costs"]}
            operations = []
            for index, mode in enumerate(["correctness"] + (["performance"] * self.repeats if self.performance else [])):
                self._verify_snapshot(prepared)
                row = self._operation(prepared, index, mode)
                operations.append(row)
                if not row["assessment"]["success"]:
                    break
            return self._finish_evaluation(prepared, operations)

    def _finish_evaluation(self, prepared, operations):
        """Finalize collected operations while the caller holds the source lock."""
        root = self.root / prepared["evaluation_id"]
        prior = sorted(root.glob("evaluation-*.json"))
        if prior:
            receipt = _parse(_read(root, prior[-1].name)[0])
            if receipt["settled"]:
                require(receipt["operations"] == operations, "terminal evaluation operations changed", NativeEvaluationError)
                artifacts = self._artifacts(prepared)
                artifacts["native_evaluation.json"] = canonical(receipt) + b"\n"
                return {"receipt": receipt, "artifacts": artifacts, "costs": receipt["costs"]}
        require(bool(operations), "cannot finalize an evaluation without an operation", NativeEvaluationError)
        success = len(operations) == 1 + (self.repeats if self.performance else 0) and all(
            row["assessment"]["success"] for row in operations)
        settled = all(row["assessment"]["settled"] for row in operations)
        independent = all(row["receipt"].get("evaluation_integrity_valid") is True for row in operations)
        provisional = any(row["receipt"].get("peer_exception_active") is not False
                          or row["receipt"].get("measurement_valid") is not True for row in operations)
        devices = [row["receipt"].get("physical_devices") for row in operations]
        same_devices = bool(devices[0]) and all(value == devices[0] for value in devices)
        times, spreads, stable = {}, {}, False
        if success and self.performance:
            for case in self.case_ids:
                values = [row["assessment"]["times_ms"][case] for row in operations[1:]]
                times[case] = statistics.median(values)
                spreads[case] = (max(values) - min(values)) / times[case]
            stable = all(value <= self.max_spread for value in spreads.values())
        costs = self._evaluation_costs(prepared, operations)
        receipt = {"schema": "geak-evo-native-evaluation-v1", "evaluation_id": prepared["evaluation_id"],
                   "binding": prepared["binding"], "settled": settled, "independent": independent,
                   "correctness_observed": operations[0]["assessment"]["correctness_observed"],
                   "correct": operations[0]["assessment"]["correctness_observed"] and independent,
                   "eligible": bool(success and independent and stable and same_devices and not provisional),
                   "provisional": provisional, "physical_devices": devices[0], "timing_stable": stable,
                   "times_ms": times, "relative_spreads": spreads, "costs": costs,
                   "operations": operations,
                   "scope": "Frozen complete-call operator comparison; each timed call is synchronized. "
                   "Repeated timing is not independent search replication. Peer-exception timing is provisional. "
                   "A failing command alone does not establish numerical mismatch or refute its algorithm."}
        _save(root / ("evaluation-%04d.json" % (len(prior) + 1)), receipt)
        artifacts = self._artifacts(prepared)
        artifacts["native_evaluation.json"] = canonical(receipt) + b"\n"
        result = {"receipt": receipt, "artifacts": artifacts, "costs": costs}
        if not settled:
            raise NativeEvaluationUnsettled(result)
        return result

    def _evaluation_costs(self, prepared, operations):
        # Reconciliation replaces cumulative GPU totals; collection wall costs
        # remain additive. Paired and ordinary evaluation use this same reducer.
        root = self.root / prepared["evaluation_id"]
        rows = [_parse(_read(path.parent, path.name)[0]) for path in root.glob("operation-*/collection-*/result.json")]
        gpu = [row["receipt"].get("costs", {}).get("gpu_seconds") for row in operations]
        return {"wall_seconds": sum(row["wall_seconds"] for row in rows),
                "gpu_seconds": sum(gpu) if all(_number(value) for value in gpu) else None}

    def evaluate_pair(self, baseline_candidate_id, candidate_id, *, comparison_id, work_ids, order_seed=0):
        """Acquire actual paired operations, preserving each native interval mean.

        ``repeats`` is the number of whole-case-set AB/BA blocks. Correctness
        runs first for both sources. The balanced randomized timing order is
        committed before any native dispatch. A retry reconciles those same
        operations; it never pairs previously independent evaluations.

        ``work_ids`` maps baseline/candidate to {case_id: derivation SHA256}.
        Work-model validity remains external. The returned measurement record
        is descriptive unless the separate paired receipt is eligible.
        """
        require(callable(self.launch) and callable(self.settle), "explicit trusted launch and settlement callbacks required", NativeEvaluationError)
        require(self.performance, "paired acquisition requires performance evaluation", NativeEvaluationError)
        require(isinstance(comparison_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,89}", comparison_id),
              "invalid comparison_id", NativeEvaluationError)
        require(type(order_seed) is int, "pair order seed must be an integer", NativeEvaluationError)
        roles = ("baseline", "candidate")
        require(type(work_ids) is dict and set(work_ids) == set(roles), "both work-model maps required", NativeEvaluationError)
        for role in roles:
            require(type(work_ids[role]) is dict and set(work_ids[role]) == set(self.case_ids)
                  and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                          for value in work_ids[role].values()), "work models must cover every registered case", NativeEvaluationError)
        work_ids = _parse(canonical(work_ids))
        rng = random.Random(order_seed)
        orders = ["AB", "BA"] * (self.repeats // 2)
        if self.repeats % 2:
            orders.append(rng.choice(("AB", "BA")))
        rng.shuffle(orders)
        root = self.root / ("pair-" + comparison_id)
        root.mkdir(exist_ok=True, mode=0o700)
        require(not root.is_symlink(), "pair storage cannot be a symlink", NativeEvaluationError)
        with (root / "lock").open("a+b") as lock, ExitStack() as locks:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            prepared = {role: self.prepare(identifier, evaluation_id="pair-" + comparison_id + "-" + role)
                        for role, identifier in zip(roles, (baseline_candidate_id, candidate_id))}
            for role in roles:
                handle = locks.enter_context((self.root / prepared[role]["evaluation_id"] / "lock").open("a+b"))
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            intent = {"schema": "geak-evo-native-pair-intent-v1", "comparison_id": comparison_id,
                      "sources": {role: prepared[role]["binding"] for role in roles},
                      "evaluation_ids": {role: prepared[role]["evaluation_id"] for role in roles},
                      "work_ids": work_ids, "order_seed": order_seed, "orders": orders,
                      "timed_calls_per_case_side": self.descriptor["timing"]["iterations"],
                      "policy": self._policy(), "block_scope": "adjacent operations covering every registered case"}
            if (root / "intent.json").exists():
                require(_parse(_read(root, "intent.json")[0]) == intent, "comparison_id belongs to another pair intent", NativeEvaluationError)
            else:
                require(not any(any((self.root / value["evaluation_id"]).glob("operation-*"))
                              for value in prepared.values()), "cannot adopt independently acquired operations as paired", NativeEvaluationError)
                _save(root / "intent.json", intent)
            previous = sorted(root.glob("collection-*/receipt.json"))
            terminal = previous and _parse(_read(previous[-1].parent, "receipt.json")[0])["settled"]
            completion = {"intent_sha256": sha256(canonical(intent) + b"\n"),
                          "native_operation_count": 2 * (1 + self.repeats)}
            acquisition_complete = (root / "acquisition_complete.json").exists()
            if acquisition_complete:
                require(_parse(_read(root, "acquisition_complete.json")[0]) == completion, "pair acquisition marker changed", NativeEvaluationError)
            had_operations = any(any((self.root / value["evaluation_id"]).glob("operation-*"))
                                 for value in prepared.values())
            if not terminal and not acquisition_complete and had_operations and not (root / "interrupted.json").exists():
                _save(root / "interrupted.json", {"reason": "resumed partial acquisition; timing adjacency is unqualified"})
            operations = {role: [] for role in roles}
            schedule = [(role, 0, "correctness") for role in roles]
            for index, order in enumerate(orders, 1):
                schedule.extend((roles[letter == "B"], index, "performance") for letter in order)
            for role, index, mode in schedule:
                self._verify_snapshot(prepared[role])
                row = self._operation(prepared[role], index, mode)
                operations[role].append(row)
                if not row["assessment"]["success"]:
                    break
            if (not acquisition_complete and not had_operations and not (root / "interrupted.json").exists()
                    and sum(len(rows) for rows in operations.values()) == len(schedule)
                    and all(row["assessment"]["success"] for rows in operations.values() for row in rows)):
                # Only this uninterrupted pair-owned schedule can attest order.
                # Independent per-source summaries cannot substitute for it.
                _save(root / "acquisition_complete.json", completion)
            pending = any(not row["assessment"]["settled"] for rows in operations.values() for row in rows)
            results, artifacts = {}, {}
            for role in roles:
                if not operations[role]:
                    continue
                if pending:
                    # Do not seal a successful source's partial schedule while
                    # the other source still needs reconciliation/continuation.
                    results[role] = {"artifacts": self._artifacts(prepared[role]),
                                     "costs": self._evaluation_costs(prepared[role], operations[role])}
                else:
                    results[role] = self._finish_evaluation(prepared[role], operations[role])
                artifacts.update({role + "/" + name: raw for name, raw in results[role]["artifacts"].items()})
            completed = min(sum(row["request"]["mode"] == "performance" and row["assessment"]["success"]
                                for row in operations[role]) for role in roles)
            binding = {"job_id": self.job_id, "measurement_epoch_id": self.epoch,
                       **{role + "_source_id": prepared[role]["binding"]["candidate_source_id"] for role in roles}}
            record = {"schema_version": 1, **binding, "block_schedule": "shared", "cases": []}
            for case in self.job["cases"]:
                case_id = case["case_id"]
                item = {"case_id": case_id, "observed_frequency": case["observed_frequency"], "blocks": [],
                        **{role + "_work_id": work_ids[role][case_id] for role in roles}}
                for index in range(1, completed + 1):
                    block = {"block_id": "block-%04d" % index, "job_id": self.job_id, "case_id": case_id,
                             "measurement_epoch_id": self.epoch, "order": orders[index - 1]}
                    for role in roles:
                        row = operations[role][index]
                        block[role] = {"source_id": binding[role + "_source_id"], "work_id": work_ids[role][case_id],
                            "kind": "unprofiled_interval_mean", "unit": "ms",
                            "value": row["assessment"]["times_ms"][case_id],
                            "timed_calls": self.descriptor["timing"]["iterations"],
                            "raw_report_ref": {"path": str(Path(row["collection"]) / "artifacts/performance_report.json"),
                                               "sha256": row["receipt"]["artifacts_sha256"]["performance_report.json"]},
                            "operation_id": row["request"]["operation_id"],
                            "timing_scope": "complete_callable_synchronized_wall"}
                    item["blocks"].append(block)
                if not completed:
                    item["unknown_reason"] = "no complete successful pair; inspect retained native operations"
                record["cases"].append(item)
            receipts = [value["receipt"] for value in results.values() if "receipt" in value]
            settled = not pending
            continuous = (root / "acquisition_complete.json").exists() and not (root / "interrupted.json").exists()
            costs = {"wall_seconds": sum(value["costs"]["wall_seconds"] for value in results.values())}
            gpu_costs = [value["costs"]["gpu_seconds"] for value in results.values()]
            costs["gpu_seconds"] = sum(gpu_costs) if all(_number(value) for value in gpu_costs) else None
            receipt = {"schema": "geak-evo-native-pair-v1", "comparison_id": comparison_id,
                       "binding": binding, "intent_sha256": sha256(canonical(intent) + b"\n"),
                       "settled": settled, "completed_blocks": completed, "requested_blocks": self.repeats,
                       "continuous_acquisition": continuous,
                       "native_operation_count": sum(len(value) for value in operations.values()),
                       "requested_native_operations": 2 * (1 + self.repeats),
                       "eligible": continuous and len(receipts) == 2 and completed == self.repeats
                           and all(value["eligible"] for value in receipts)
                           and receipts[0]["physical_devices"] == receipts[1]["physical_devices"],
                       "source_evaluations": {role: value["receipt"]["evaluation_id"] for role, value in results.items()
                                              if "receipt" in value},
                       "costs": costs, "measurement_sha256": sha256(canonical(record) + b"\n"),
                       "scope": "Paired unprofiled native interval means with shared case schedule; "
                                "eligibility is separate from descriptive timing and no ceiling is inferred."}
            prior = sorted(root.glob("collection-*/receipt.json"))
            if prior and _parse(_read(prior[-1].parent, "receipt.json")[0]) == receipt:
                collection = prior[-1].parent
                require(_parse(_read(collection, "measurements.json")[0]) == record, "retained paired measurements changed", NativeEvaluationError)
            else:
                collection = _new_collection(root)
                _save(collection / "measurements.json", record)
                _save(collection / "receipt.json", receipt)
            artifacts["native_pair/intent.json"] = canonical(intent) + b"\n"
            if (root / "acquisition_complete.json").exists():
                artifacts["native_pair/acquisition_complete.json"] = _read(root, "acquisition_complete.json")[0]
            if (root / "interrupted.json").exists():
                artifacts["native_pair/interrupted.json"] = _read(root, "interrupted.json")[0]
            artifacts["native_pair/receipt.json"] = canonical(receipt) + b"\n"
            artifacts["native_pair/measurements.json"] = canonical(record) + b"\n"
            result = {"receipt": receipt, "record": record, "artifacts": artifacts, "costs": costs}
            if not settled:
                raise NativeEvaluationUnsettled(result)
            return result

    def verify_pair_evidence(self, artifacts):
        """Authenticate a retained pair, including both acquired source bundles.

        This returns observed eligibility/settlement, not an acceptance decision.
        A settled reference failure may legitimately have no candidate operation.
        """
        try:
            pair = _parse(artifacts["native_pair/receipt.json"])
            comparison_id = pair.get("comparison_id") if type(pair) is dict else None
            require(isinstance(comparison_id, str)
                  and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,89}", comparison_id), "invalid paired evidence identity", NativeEvaluationError)
            root = self.root / ("pair-" + comparison_id)
            receipts = sorted(root.glob("collection-*/receipt.json"))
            require(bool(receipts), "pair has no retained host acquisition", NativeEvaluationError)
            paths = {"receipt.json": receipts[-1], "measurements.json": receipts[-1].parent / "measurements.json",
                     "intent.json": root / "intent.json"}
            for name in ("acquisition_complete.json", "interrupted.json"):
                if (root / name).exists():
                    paths[name] = root / name
                else:
                    require("native_pair/" + name not in artifacts, "pair contains an unretained acquisition marker", NativeEvaluationError)
            require(all(artifacts.get("native_pair/" + name) == _read(path.parent, path.name)[0]
                      for name, path in paths.items()), "paired evidence differs from retained host acquisition", NativeEvaluationError)
            require(pair["binding"]["job_id"] == self.job_id and pair["binding"]["measurement_epoch_id"] == self.epoch,
                  "pair belongs to another job or epoch", NativeEvaluationError)
            require(not pair["eligible"] or ("acquisition_complete.json" in paths and "interrupted.json" not in paths),
                  "eligible pair lacks uninterrupted acquisition", NativeEvaluationError)
            intent = _parse(artifacts["native_pair/intent.json"])
            for role in ("baseline", "candidate"):
                evaluation_id = intent["evaluation_ids"][role]
                if role not in pair["source_evaluations"]:
                    require(not pair["settled"] or not any((self.root / evaluation_id).glob("operation-*")),
                          "settled unacquired source has retained operations", NativeEvaluationError)
                    continue
                source_artifacts = {name[len(role) + 1:]: raw for name, raw in artifacts.items()
                                    if name.startswith(role + "/")}
                source_receipt = _parse(source_artifacts["native_evaluation.json"])
                require(source_receipt["evaluation_id"] == evaluation_id == pair["source_evaluations"][role],
                      "paired source evaluation identity differs", NativeEvaluationError)
                source_binding = {"job_id": self.job_id, "measurement_epoch_id": self.epoch,
                                  "candidate_source_id": pair["binding"][role + "_source_id"]}
                self.verify_evidence(source_artifacts, source_binding)
            return pair
        except (KeyError, TypeError, OSError) as exc:
            raise NativeEvaluationError("paired evidence is incomplete or malformed") from exc

    def verify_evidence(self, artifacts, expected_binding):
        """Accept only bytes matching this host's own retained evaluation."""
        receipt = _parse(artifacts["native_evaluation.json"])
        require(all(receipt["binding"].get(key) == value for key, value in expected_binding.items()),
              "archive evidence binding mismatch", NativeEvaluationError)
        prepared = self.prepare(receipt["binding"]["candidate_id"], evaluation_id=receipt["evaluation_id"])
        root = self.root / receipt["evaluation_id"]
        finals = sorted(root.glob("evaluation-*.json"))
        require(finals and _parse(_read(root, finals[-1].name)[0]) == receipt, "evaluation is not the current retained host result", NativeEvaluationError)
        retained = self._artifacts(prepared)
        require(all(artifacts.get(name) == raw for name, raw in retained.items()), "native evidence bytes differ from host collection", NativeEvaluationError)
        return {key: receipt[key] for key in ("independent", "settled", "correct", "eligible")}
