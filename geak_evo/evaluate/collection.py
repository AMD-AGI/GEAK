"""Host collection and anchor checks for the workload-native evaluator.

The host supplies its existing store, NativeEvaluator and research loop. This
adapter preserves native_job's collection boundary without constructing a
model runtime, execution provider, campaign clock or search policy.
"""

import json
from pathlib import Path

from .._util import sha256_hex
from .jobs_v1 import job_digest, validate_job


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


class NativeJobCollector:
    """Join harvested engineer sources to native evaluation and archive checks.

    Pass ``collect`` as the loop's collection callback. Construction installs
    ``verify_anchor_evidence`` as the store's verifier, as NativeJobCampaign did.
    ``loop`` supplies the existing snapshot only when an engineer omitted a
    card for a new direction; it may be attached before collection. No pending
    native exception is caught or converted to a successful collection.
    """

    def __init__(self, store, *, evaluator, job, job_id, measurement_epoch_id,
                 paired_timing=None, loop=None):
        self.job = validate_job(job)
        if job_digest(self.job) != job_id:
            raise ValueError("collector job identity differs from its validated contract")
        if (evaluator.store is not store or evaluator.job_id != job_id
                or evaluator.epoch != measurement_epoch_id or evaluator.job != self.job):
            raise ValueError("collector and evaluator store/job/epoch differ")
        if paired_timing is not None and (type(paired_timing) is not dict
                or set(paired_timing) != {"order_seed"}
                or type(paired_timing["order_seed"]) is not int or not evaluator.performance):
            raise ValueError("paired_timing requires performance=true and one explicit integer order_seed")
        self.store, self.evaluator = store, evaluator
        self.job_id, self.epoch = job_id, measurement_epoch_id
        self.source_paths = self.job["baseline"]["source_paths"]
        self.paired_timing = None if paired_timing is None else dict(paired_timing)
        self.evaluations_per_turn = 2 if paired_timing is not None else 1
        self.loop = loop
        self.store.verify_evidence = self.verify_anchor_evidence

    def verify_anchor_evidence(self, artifacts, expected_binding):
        try:
            checks = self.evaluator.verify_evidence(artifacts, expected_binding)
            if self.paired_timing is None:
                return checks
            pair = self.evaluator.verify_pair_evidence(artifacts)
            native = json.loads(artifacts["native_evaluation.json"])
            roles = [role for role, identity in pair["source_evaluations"].items() if identity == native["evaluation_id"]]
            if (len(roles) != 1 or pair.get("eligible") is not True or pair.get("continuous_acquisition") is not True
                    or pair.get("settled") is not True
                    or pair["binding"][roles[0] + "_source_id"] != expected_binding["candidate_source_id"]
                    or any(pair["binding"][key] != expected_binding[key] for key in ("job_id", "measurement_epoch_id"))):
                raise ValueError("anchor requires its own eligible uninterrupted paired acquisition")
            return checks
        except (KeyError, TypeError, OSError) as exc:
            raise ValueError("paired/native anchor evidence is incomplete or malformed") from exc


    def collect_pair(self, candidate_id, attempt_id):
        """Keep adaptive paired feedback and both exact source evaluations."""
        baseline_id = self.store.get("campaign")["tip"]
        work_ids, models = {}, {}
        for role, identity in (("baseline", baseline_id), ("candidate", candidate_id)):
            source = self.store.get_candidate(identity)["source_id"]
            work_ids[role] = {}
            for case in self.job["cases"]:
                model = {"schema": "geak-evo-unmodeled-work-v1", "job_id": self.job_id,
                         "measurement_epoch_id": self.epoch, "source_id": source, "case_id": case["case_id"],
                         "status": "unknown", "work": None, "traffic": None,
                         "scope": "Identity only; no work/traffic derivation or roofline measurement exists."}
                raw = encoded(model)
                digest = sha256_hex(raw)
                work_ids[role][case["case_id"]] = digest
                models["work_models/" + digest + ".json"] = raw
        pair = self.evaluator.evaluate_pair(baseline_id, candidate_id, comparison_id=attempt_id,
            work_ids=work_ids, order_seed=self.paired_timing["order_seed"])
        artifacts = dict(pair["artifacts"], **models)
        baseline = {name[len("baseline/"):]: raw for name, raw in artifacts.items() if name.startswith("baseline/")}
        baseline_ref = None
        if "native_evaluation.json" in baseline:
            evidence_id = self.store.archive_evidence(baseline_id, dict(artifacts, **baseline))
            baseline_ref = {"evidence_id": evidence_id, "artifact": "native_evaluation.json"}
        candidate = {name[len("candidate/"):]: raw for name, raw in artifacts.items() if name.startswith("candidate/")}
        artifacts.update(candidate)
        if "native_evaluation.json" not in candidate:
            artifacts["native_not_launched.json"] = encoded({"reason": "Paired acquisition stopped before candidate native launch."})
        from .timing import analyze_measurements
        analysis = analyze_measurements(pair["record"], timing_aggregation=self.job["timing"]["aggregation"],
            expected_binding=pair["receipt"]["binding"], observed_cases=self.job["cases"])
        artifacts["paired_feedback.json"] = encoded({"measurement_role": "adaptive_search_feedback",
            "eligible": pair["receipt"]["eligible"], "baseline_evidence_ref": baseline_ref,
            "comparison": analysis, "limitation": "Timing repetitions are not independent searches or held-out final validation; work/traffic remain unknown."})
        return {"artifacts": artifacts, "costs": dict(pair["costs"],
            evaluations=len(pair["receipt"]["source_evaluations"]))}


    def collect(self, result, candidate_id):
        receipt = result["receipt"]
        if receipt["settlement"].get("settled") is not True or receipt["captured_tip_source_id"] is None:
            return {"costs": {"wall_seconds": receipt["elapsed_seconds"], "evaluations": 0},
                    "artifacts": {"native_not_launched.json": encoded({"reason": "engineer/source unsettled"})},
                    "card": None, "exchange_receipt": None}
        workspace = Path(receipt["request"]["workspace"])
        if self.store.capture_source(workspace, self.source_paths) != receipt["captured_tip_source_id"]:
            raise ValueError("settled workspace changed after harvest; retain original native collection intent")
        evaluation = (self.collect_pair(candidate_id, receipt["attempt_id"]) if self.paired_timing is not None else
                      self.evaluator.evaluate(candidate_id, evaluation_id=receipt["attempt_id"]))
        metadata = {}
        artifacts = dict(evaluation["artifacts"])
        for name, key in (("geak_card.json", "card"), ("geak_exchange_receipt.json", "exchange_receipt")):
            path = workspace / name
            if path.exists():
                if path.is_symlink() or path.stat().st_size > 262144:
                    raise ValueError("invalid or oversized agent metadata")
                raw = path.read_bytes()
                metadata[key] = read_json(path)
                if path.read_bytes() != raw:
                    raise ValueError("agent metadata changed while collecting")
                artifacts["agent_" + name] = raw
        if self.store.capture_source(workspace, self.source_paths) != receipt["captured_tip_source_id"]:
            raise ValueError("workspace changed during collection; source/evidence association requires reconciliation")
        if "receipt" in evaluation:
            artifacts["native_evaluation.json"] = encoded(evaluation["receipt"])
        artifacts["usage_accounting.json"] = encoded({"raw_terminal_usage": receipt.get("usage"),
            "usage_status": receipt.get("usage_status"), "continuation": receipt["request"]["continuation"]["mode"],
            "per_episode_tokens": None, "dollars": None,
            "limitation": "Resumed CLI terminal usage may be cumulative; no complete upstream billing attested."})
        if metadata.get("card") is None and receipt["request"]["direction_id"] not in self.store.get():
            pending = self.loop.snapshot()["state"]["pending"]
            metadata["card"] = {"idea": {"idea_id": pending["proposal"]["direction_id"],
                "statement": "No engineer-authored implementation summary was supplied. The linked planner assignment records intent, not an implemented result.",
                "status": "unassessed"},
                "author": {"engineer_id": "planner-proposal-intent", "attempt_ref": pending["packet_ref"]}}
        return {"costs": {"wall_seconds": receipt["elapsed_seconds"] + evaluation["costs"]["wall_seconds"],
                          "evaluations": evaluation["costs"].get("evaluations", 1)},
                "artifacts": artifacts, "card": metadata.get("card"), "exchange_receipt": metadata.get("exchange_receipt")}

