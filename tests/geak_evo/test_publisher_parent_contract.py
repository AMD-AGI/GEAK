"""Real cleaned parent evidence through the retained publisher's snapshot check.

CPU-only, temporary ResearchStore fixtures. The producer is parent_prompt_context
over EvolutionLoop.snapshot; the consumer is the patched retained Publisher
method with its real retained accounting/archive/claim readers. Constructor,
root custody, operation dispatch and GPU/model execution are outside this test.
"""

import ast
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest

from geak_evo.archive import succession
from geak_evo.archive.refs import file_ref
from geak_evo.archive.store import ResearchStore
from geak_evo.engineer.assignment import archive_files
from geak_evo.engineer.investigation import retain_trial_sources
from geak_evo.loop import EvolutionLoop


def retained_publisher(test):
    """Compose pinned patches and load actual retained readers without aliases."""
    repository = Path(__file__).resolve().parents[2]
    phase = json.loads((repository / "docs/evo/native-search-phase-base.json").read_bytes())
    row = next(item for item in phase["files"] if item["name"] == "bridge/finite_publisher.py")
    original = Path(row["source_path"])
    if not original.is_file():
        raise unittest.SkipTest("retained publisher source is not installed in this CPU checkout")
    raw = original.read_bytes()
    test.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
    temporary = tempfile.TemporaryDirectory()
    test.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    source = root / row["name"]
    source.parent.mkdir()
    source.write_bytes(raw)
    subprocess.run(["git", "apply", str(repository / "docs/evo/native-search-phase.patch")],
                   cwd=root, check=True, capture_output=True)
    test.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), row["patched_sha256"])
    contract = json.loads((repository / "docs/evo/publisher-contract-base.json").read_bytes())
    contract_row = next(item for item in contract["files"] if item["name"] == row["name"])
    test.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), contract_row["sha256"])
    subprocess.run(["git", "apply", "--include=bridge/finite_publisher.py",
                    str(repository / "docs/evo/publisher-contract.patch")],
                   cwd=root, check=True, capture_output=True)
    test.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), contract_row["patched_sha256"])

    package_name = "_retained_parent_contract_fixture"
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    modules = {}
    retained = original.parents[1] / "evolution/src/geak_evo"
    dependencies = {
        "claims": "ef99952575681da3324b3838e9c27f3508d44d19a4de2130476bb42026441c89",
        "selection": "488cca61172d410886c6b1d7b870ea97fa15600f232a56e938d94743c0a22f41",
        "program_lease": "004dd3128b5d15a147e27ec3328816fd86cfcb7bb464b99d4980769506484196",
        "evo_archive": "2584e8a890386f4cbb32ad07cf2ddbdc62021db8afb50a716827cd654ccc6060",
        "maturation": "710c71cc38e9eb804233e2e9d2a614341f5df6f2644a1e5b8a2af7db4059c869",
        "research_state": "54a77f97a7d6db1c1d1d2adc045924f2a10968a3a082371e8355930aa2148f51",
    }
    for name, digest in dependencies.items():
        qualified = package_name + "." + name
        path = retained / (name + ".py")
        raw = path.read_bytes()
        test.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        module = ModuleType(qualified)
        module.__file__ = str(path)
        module.__package__ = package_name
        sys.modules[qualified] = module
        test.addCleanup(sys.modules.pop, qualified, None)
        exec(compile(raw, str(path), "exec"), module.__dict__)
        modules[name] = module
    test.addCleanup(sys.modules.pop, package_name, None)
    namespace = dict(contextlib=contextlib, copy=copy, hashlib=hashlib, json=json,
        math=math, os=os, Path=Path, re=re, stat=stat, time=time, need=succession.need,
        ClaimLog=modules["claims"].ClaimLog,
        read_archive=modules["evo_archive"].read_archive,
        account_opportunities=modules["selection"].account_opportunities,
        program_lease=modules["program_lease"],
        read_maturation=modules["maturation"].read_maturation,
        pending_maturation_reviews=modules["maturation"].pending_maturation_reviews,
        ResearchStore=modules["research_state"].ResearchStore)
    tree = ast.parse(source.read_bytes(), filename=str(source))
    selected = {"Publisher", "_HistoricalView", "canonical", "trusted_path", "read_ref"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
    test.assertEqual({node.name for node in nodes}, selected)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class PublisherParentContractTests(unittest.TestCase):
    def setUp(self):
        self.retained = retained_publisher(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.job = "a" * 64
        self.store = ResearchStore(self.root / "store", allowed_epochs=[(self.job, "epoch")],
                                   source_paths={self.job: ["kernel.py"]})
        self.seed = self.candidate("seed")
        self.loop = EvolutionLoop.create(self.store, "campaign", self.seed,
            limits={"wall_seconds": 100, "evaluations": 10}, deadline=1000, clock=lambda: 100)
        self.reader = self.retained["ResearchStore"](self.store.root,
            allowed_epochs=[(self.job, "epoch")], source_paths={self.job: ["kernel.py"]})
        self.publisher = object.__new__(self.retained["Publisher"])
        self.publisher.store = self.reader
        self.publisher.store_root = self.store.root
        self.publisher.campaign_id = "campaign"
        self.publisher.parent_contract = SimpleNamespace(
            parent_snapshot=succession.parent_snapshot, archive_files=succession.archive_files,
            selected_parent_material=succession.selected_parent_material,
            bound_evidence=ResearchStore._bound_evidence)
        self.publisher.verify_quantity = None
        self.publisher.clock = {"started_unix": 101, "started_monotonic_ns": time.monotonic_ns()}

    def candidate(self, label, *, parents=()):
        (self.work / "kernel.py").write_text("value = " + repr(label) + "\n")
        source = self.store.capture_source(self.work, ["kernel.py"])
        return self.store.candidate(self.job, "epoch", source, parents=list(parents))

    def artifact(self, candidate, name, value):
        evidence = self.store.archive_evidence(candidate, {name: succession.encoded(value)})
        return {"evidence_id": evidence, "artifact": name}

    def retained_trials(self):
        """One reviewed episode, two repeated failed trials and one other source."""
        owner = self.candidate("terminal", parents=[self.seed])
        owner_source = self.store.get_candidate(owner)["source_id"]
        trial_candidate = self.candidate("failed trial", parents=[self.seed])
        trial_source = self.store.get_candidate(trial_candidate)["source_id"]
        other = self.candidate("other trial", parents=[self.seed])
        other_source = self.store.get_candidate(other)["source_id"]
        trials, feedbacks = [], {}
        for index, source in enumerate((trial_source, trial_source, other_source)):
            saved = self.root / ("trial-source-" + str(index))
            self.store.materialize(source, saved)
            feedback = dict(schema="geak-native-probe-feedback-v1", profile="native",
                source_id=source, operation_id="probe-" + str(index), turn_id="op1",
                native_pass=False, settled=True, error="correctness failed",
                receipt=dict(schema="geak-arena-evaluation-v1", evaluation_id="probe-" + str(index),
                    binding=dict(candidate_id=trial_candidate if index < 2 else other,
                        candidate_source_id=source, job_id=self.job, measurement_epoch_id="epoch"),
                    native_suite_pass_observed=False, host_errors=[], collection_errors=[],
                    native_observations={"correctness": {"status": "failed"},
                                         "benchmark": {"status": "not_run"}}))
            path = self.root / ("feedback-" + str(index) + ".json")
            path.write_bytes(succession.encoded(feedback))
            feedbacks[path.name] = path.read_bytes()
            trials.append(dict(source_archive=dict(source_id=source, source_dir=str(saved),
                    manifest_ref=file_ref(self.store.root / "objects" / source)),
                archive_ref=file_ref(path), verified=None, error="correctness failed"))
        roles = {"selected": owner, "terminal": owner}
        harvest = {role: {"source_id": owner_source} for role in roles}
        harvest["trials"] = trials
        inventory = retain_trial_sources(self.store, harvest,
            {"attempt_id": "op1", "request": {"tip": {"candidate_id": self.seed}, "donors": []}},
            {"job_id": self.job, "measurement_epoch_id": "epoch"})
        common = {"native_source_roles.json": succession.encoded(roles),
                  "native_trial_sources.json": succession.encoded(inventory)}
        raw = self.store.archive_evidence(owner, dict(common, **feedbacks,
            **{"archive/manifest.json": succession.encoded(harvest)}))
        links = [{"original_ref": trial["archive_ref"], "evidence_ref": {
            "evidence_id": raw, "artifact": "feedback-" + str(index) + ".json"}}
            for index, trial in enumerate(trials)]
        collected = self.store.archive_evidence(owner, dict(common, **{
            "collection.json": succession.encoded({"costs": {"wall_seconds": 3, "evaluations": 3}}),
            "native_idea_harvest.json": succession.encoded(harvest),
            "native_evidence_index.json": succession.encoded(links)}))
        collection = {"evidence_id": collected, "artifact": "collection.json"}
        transport = self.artifact(owner, "transport.json", {"attempt_id": "op1", "settled": True})
        packet = self.artifact(owner, "packet.json", {"proposal": {"engineer_id": "engineer-1"}})
        note = {"claim": "Synthetic failed trial", "scope": "CPU contract fixture", "next_test": "Continue"}
        self.store.checkpoint("d1", self.seed, note=note, expected_revision=0)
        review = dict(note, research_status="reported", research_review_id="review-1",
            reviewed_outcome_refs=[transport, collection], research_update_ref=collection,
            latest_transport_ref=transport)
        self.store.checkpoint("d1", owner, note=review, expected_revision=1,
                              evidence=[transport["evidence_id"], collected])
        record, state = self.loop._state()
        state["opportunities"].append(dict(state["binding"], opportunity_id="op1", direction_id="d1",
            island_id="unassigned", engineer_id="engineer-1", question="Synthetic trial",
            opened_at=10, review_at=20, settled=True, completed_sequence=0, progress_complete=False,
            requested={"wall_seconds": 5, "evaluations": 3}, reserved={"wall_seconds": 5, "evaluations": 3},
            receipts=[{"receipt_id": "cost-1", "costs": {"wall_seconds": 3, "evaluations": 3}}]))
        state["history"] += [dict(attempt_id="op1", settled=True, candidate_id=owner,
            transport_ref=transport, collection_ref=collection, packet_ref=packet),
            dict(review_kind="research_review", direction_revision=2,
                proposal=dict(direction_id="d1", transport_ref=transport, collection_ref=collection,
                    expected_revision=1, note={"research_review_id": "review-1"}))]
        self.loop._save(record, state)
        return trial_candidate

    def produce(self, parent):
        proposal = {"parent_candidate_id": parent, "direction_id": "next", "expected_revision": 0}
        planner = self.artifact(self.seed, "planner.json", proposal)
        record, state = self.loop._state()
        state.update(phase="planned", pending={"proposal": proposal, "planner_ref": planner})
        self.loop._save(record, state)
        snapshot = self.loop.snapshot()
        planned_record = self.store.get("campaign")
        corpus = archive_files(self.store, snapshot)
        prompt, snapshot_ref = succession.parent_prompt_context(self.store, snapshot, parent,
            direction_id="next", search_policy=None, campaign_root=self.root)
        admitted = copy.deepcopy(snapshot["state"])
        admitted["phase"] = "dispatching"
        admitted["opportunities"].append({"opportunity_id": "op2", "opened_at": 100})
        # Save a later campaign revision so the real historical view must restore
        # the pre-grant revision when its retained archive reader reconstructs it.
        self.loop._save(planned_record, admitted)
        view = self.retained["_HistoricalView"](self.reader, {"campaign": planned_record},
            bound_evidence=self.publisher.parent_contract.bound_evidence)
        consumer_corpus = self.publisher.parent_contract.archive_files(view, snapshot)
        self.assertEqual(consumer_corpus, corpus)
        return dict(prompt=prompt, snapshot_ref=snapshot_ref,
            request={"read_only": corpus + [snapshot_ref]}, state=admitted,
            packet={"proposal": proposal, "planner_ref": planner},
            allowed_objects={ref["sha256"] for ref in consumer_corpus}, planned_record=planned_record, view=view)

    def consume(self, produced):
        return self.publisher._parent_snapshot(**{key: produced[key] for key in
            ("request", "state", "packet", "allowed_objects", "planned_record", "view")})

    def test_initial_seed_snapshot_with_empty_trial_history_is_accepted(self):
        produced = self.produce(self.seed)
        supplied = json.loads(Path(produced["snapshot_ref"]["path"]).read_bytes())
        self.assertEqual(self.consume(produced), supplied)
        self.assertEqual(supplied["native_parent_outcomes"], [])
        self.assertEqual(supplied["native_trial_history"]["trials"], [])
        self.assertIn(produced["snapshot_ref"]["sha256"], produced["allowed_objects"])

    def test_reviewed_trial_history_survives_real_producer_and_consumer(self):
        parent = self.retained_trials()
        produced = self.produce(parent)
        supplied = json.loads(Path(produced["snapshot_ref"]["path"]).read_bytes())
        self.assertEqual(self.consume(produced), supplied)
        outcome, = supplied["native_parent_outcomes"]
        self.assertEqual(outcome["roles"], [])
        self.assertEqual(outcome["trial_indexes"], [0, 1])
        history = supplied["native_trial_history"]
        self.assertEqual(history["coverage"]["trial_count"], 2)
        self.assertEqual({row["candidate_id"] for row in history["trials"]}, {parent})
        self.assertEqual({row["engineer_id"] for row in history["trials"]}, {"engineer-1"})
        self.assertTrue(all(row["feedback_state"] == "available" for row in history["trials"]))
        self.assertTrue(all(row["stages"]["correctness"] == "failed" for row in history["trials"]))
        self.assertTrue(all(row["times_ms"] is None for row in history["trials"]))
        candidate = self.store.get_candidate(parent)
        source = self.store.source(candidate["source_id"])
        self.assertTrue({parent, candidate["source_id"], source["files"][0]["blob_id"]}
                        <= produced["allowed_objects"])
        proposal = produced["packet"]["proposal"]
        expected_material = succession.selected_parent_material(self.store, supplied, proposal)
        self.assertEqual(self.publisher.parent_contract.selected_parent_material(
            produced["view"], supplied, proposal), expected_material)

    def test_changed_trial_history_is_rejected_even_with_valid_evidence_envelope(self):
        parent = self.retained_trials()
        produced = self.produce(parent)
        supplied = json.loads(Path(produced["snapshot_ref"]["path"]).read_bytes())
        supplied["native_trial_history"]["trials"][0]["outcome"]["error"] = "invented success"
        evidence = self.store.archive_evidence(parent, {"native_parent_snapshot.json": succession.encoded(supplied)})
        envelope, _ = self.store.evidence(evidence)
        changed = file_ref(self.store.root / "objects" / envelope["artifacts"]["native_parent_snapshot.json"])
        produced["request"]["read_only"][-1] = changed
        with self.assertRaisesRegex(ValueError, "parent snapshot"):
            self.consume(produced)


if __name__ == "__main__":
    unittest.main()
