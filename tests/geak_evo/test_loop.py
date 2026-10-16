"""Tests for geak_evo.loop (ported from tests/test_evolution.py).

Real CPU subprocess and immutable-artifact integration, not model research
evidence. No GPU, network or model calls.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_loop.py
Parity run on the original modules: set EVO_PARITY_ORIGINAL=1 and put the
original ``evolution/src`` on PYTHONPATH instead.

Tests that dispatch an engineer turn use the engineer request validator and
runner (``geak_evo.engineer.request``, ``geak_evo.engineer.runner``) and the job
digest (``geak_evo.evaluate.jobs_v1``); these modules are required.
"""

import copy
from model_profile_fixture import profile
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ORIGINAL = bool(os.environ.get("EVO_PARITY_ORIGINAL"))

if ORIGINAL:
    from geak_evo.claims import ClaimError, ClaimLog
    from geak_evo.engineer import file_ref, source_reference
    from geak_evo.evolution import EvolutionError, EvolutionLoop
    from geak_evo.evo_archive import archive_checkpoint, archive_record, read_record
    from geak_evo.maturation import checkpoint_maturation, read_maturation
    from geak_evo.research_state import ResearchStore
else:
    from geak_evo.archive.claims import ClaimError, ClaimLog
    from geak_evo.archive.records import archive_checkpoint, archive_record, read_record
    from geak_evo.archive.maturation import checkpoint_maturation, read_maturation
    from geak_evo.archive.refs import file_ref, source_reference
    from geak_evo.archive.store import ResearchStore
    from geak_evo.loop import EvolutionError, EvolutionLoop


RUNNER_MODULE = "geak_evo.engineer" if ORIGINAL else "geak_evo.engineer.runner"
if ORIGINAL:
    from geak_evo.engineer import recover_turn, run_turn, validate_request
    from geak_evo.jobs import job_digest as JOB_DIGEST
else:
    from geak_evo.engineer.request import validate_request
    from geak_evo.engineer.runner import recover_turn, run_turn
    from geak_evo.evaluate.jobs_v1 import job_digest as JOB_DIGEST



THREAD = "01990000-0000-7000-8000-000000000001"

# Synthetic instruction files; the engineer and crossover tests look for these phrases.
PROMPTS = {
    "planner.md": b"Synthetic planner instructions for tests. Use primary evidence. "
                  b"Choose research_review or maturation_review or an engineer.\n",
    "engineer.md": b"Synthetic engineer instructions for tests. Use primary evidence.\n",
    "crossover.md": b"Synthetic crossover instructions. Report every offer.\n",
    "program_lease_engineer.md": b"Synthetic program lease engineer instructions.\n",
}


def fixture():
    """A complete CPU job contract and its artifact files (copied from test_jobs)."""
    files = {
        "source/baseline.py": b"def complete_call(x):\n    return x\n",
        "binding.json": b'{"entry":"source/baseline.py:complete_call"}',
        "framework.json": b'{"kind":"cpu_test_fixture"}',
        "workload.json": b'{"model":"cpu-fixture","tp":1}',
        "capture.json": b'{"fixture_events":["case-1","case-1"]}',
        "inputs.json": b'{"x":[1,2],"scale":1}',
        "numerical.json": b'{"comparison":"exact against captured baseline"}',
        "timing.json": b'{"timer":"fixture only","synchronize":"before and after"}',
        "graph.json": b'{"mode":"disabled"}',
        "state.json": b'{"counter":0}',
        "state_contract.json": b'{"reset":"before each call","compare":"output and counter"}',
        "inference.json": b'{"basis":"fixture extrapolation only"}',
    }
    job = {
        "schema_version": 1, "job_version": 1,
        "workload": {
            "model": "cpu-fixture/identity", "model_revision": "1" * 40,
            "framework_source": {"repository": "fixture://framework", "version": "fixture-v1",
                                 "revision": "2" * 40, "manifest_ref": "framework.json"},
            "device": "CPU fixture", "tp": 1, "isl_tokens": 2, "osl_tokens": 1,
            "concurrency": 1, "precision": "int64"},
        "accepted_stack_digest": "3" * 64,
        "callable": {"source_path": "source/baseline.py", "qualname": "complete_call",
                     "binding_ref": "binding.json", "stateful": False},
        "baseline": {"source_paths": ["source/baseline.py"], "support_paths": ["binding.json"]},
        "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "provenance": {"kind": "test_fixture", "capture_ref": "capture.json",
                       "workload_ref": "workload.json"},
        "cases": [{"case_id": "case-1", "phase": "fixture", "rank": 0,
                   "tensors": {"x": {"shape": [2], "dtype": "int64", "strides": [1],
                                     "stride_unit": "elements", "layout": "contiguous"}},
                   "scalars": {"scale": 1}, "inputs_ref": "inputs.json", "state_refs": [],
                   "routing_refs": [],
                   "observed_frequency": {"count": 2, "unit": "calls", "provenance_ref": "capture.json"},
                   "inferred_frequency": None}],
        "numerical": {"reference": "baseline", "case_ids": ["case-1"],
                      "protocol_ref": "numerical.json", "state_contract_ref": None},
        "timing": {"case_ids": ["case-1"], "unit": "us", "protocol_ref": "timing.json",
                   "warmup_calls": 1, "measured_calls": 2, "aggregation": "median",
                   "scope": "complete_callable"},
        "graph": {"mode": "disabled", "protocol_ref": "graph.json"},
    }
    return job, files


def job_id(job):
    """The real job digest when available; otherwise a fixture-only identity for planner-only tests."""
    if JOB_DIGEST is not None:
        return JOB_DIGEST(job)
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.attempts = self.root / "attempts"
        self.attempts.mkdir()
        job, files = fixture()
        for name, data in files.items():
            path = self.artifacts / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.job_path = self.root / "job.json"
        self.job_path.write_text(json.dumps(job))
        self.job_id = job_id(job)
        self.store = ResearchStore(self.root / "store", allowed_epochs=[(self.job_id, "cpu-epoch")],
                                   source_paths={self.job_id: ["implementation.py"]})
        (self.workspace / "implementation.py").write_text("value = 1\n")
        self.parent = self.store.candidate(self.job_id, "cpu-epoch", self.capture(self.workspace))
        self.card = {"idea": {"idea_id": "smaller-state", "statement": "Try a smaller live state.", "status": "hypothesis"},
                     "membership_refs": [self.membership(self.parent, "state")]}
        archive_checkpoint(self.store, "direction", self.parent, card=self.card,
                           note={"claim": "CPU hypothesis", "scope": "Fixture only", "next_test": "Inspect"}, expected_revision=0)
        self.authorized = set()
        self.prompt_dir = self.root / "prompts"
        self.prompt_dir.mkdir()
        for name, data in PROMPTS.items():
            (self.prompt_dir / name).write_bytes(data)
        self.loop = EvolutionLoop.create(self.store, "campaign", self.parent, **self.options(),
                                         limits={"wall_seconds": 100, "tokens": 10000},
                                         deadline=200, clock=lambda: 100,
                                         frozen_cases={self.job_id: ["case"]},
                                         verify_quantity=lambda ref, payload, bound: ref["evidence_id"] in self.authorized)
        self.advice = {"weights": {"wall_seconds": 0.5, "tokens": 0.5},
                       "c_ref": 0.01, "half_life": 1, "epsilon": 0.1, "beta": 1}
        self.note = self.root / "note.txt"
        self.note.write_text("CPU transport fixture only; no GPU or model calls.\n")
        self.launches = 0

    def options(self):
        """Prompt files are injected in the port; the original reads its package prompts."""
        return {} if ORIGINAL else {"prompt_dir": self.prompt_dir}

    def reopen(self, **options):
        return EvolutionLoop(self.store, "campaign", clock=lambda: 100, **options, **self.options())

    @unittest.skipIf(ORIGINAL, "the original set program_lease and read prompts on the host side")
    def test_host_options_default_to_package_prompts_and_policy_bound(self):
        from geak_evo import loop as loop_module
        from geak_evo.archive.policy import program_lease_bound
        loop = EvolutionLoop(self.store, "campaign", clock=lambda: 100)
        self.assertEqual(loop.prompt_dir, Path(loop_module.__file__).resolve().parent / "prompts")
        self.assertEqual(loop.program_lease, 0)
        opted_in = {"native_capabilities": ["program_lease"]}
        self.assertEqual(self.reopen(program_lease=program_lease_bound(opted_in)).program_lease, 2)
        self.assertEqual(self.reopen(program_lease=program_lease_bound({})).program_lease, 0)
        with self.assertRaisesRegex(EvolutionError, "program lease bound must be 0 or 2"):
            self.reopen(program_lease=True)

    def capture(self, workspace):
        return self.store.capture_source(workspace, ["implementation.py"])

    def proposal(self, **changes):
        direction = self.store.get("direction")
        card = read_record(self.store, direction["note"]["archive_card_ref"], frozen_cases=self.loop.frozen_cases)
        membership_ref = next(iter(card.get("membership_refs", [])), None)
        island_id = read_record(self.store, membership_ref)["island_id"] if membership_ref else "unassigned"
        result = {"direction_id": "direction", "expected_revision": direction["revision"],
                  "parent_candidate_id": direction["tip"], "island_id": island_id, "engineer_id": "engineer",
                  "membership_ref": membership_ref,
                  "question": "Inspect the exact fixture before trying a smaller live state.",
                  "reason": "Distinguish implementation failure from the proposed mechanism.",
                  "evidence_uptake": "No native evidence yet; inspect the CPU result only.",
                  "reservation": {"wall_seconds": 5, "tokens": 200}, "review_at": 150,
                  "premises": [], "repeat_delta": None, "exchange_ref": None,
                  "protection_ref": None, "override_reason": None}
        result.update(changes)
        if result["protection_ref"] is not None and "grant_intent" not in changes:
            result["grant_intent"] = {"reason": "Repair despite unknown competitiveness", "next_test": "Inspect source",
                "evidence_refs": [], "release_condition": "Review after finite work",
                "displaced_work": "Ordinary refinement has less of this budget"}
        return result

    def plan(self, proposal=None, callback=None):
        proposal = self.proposal() if proposal is None else proposal
        def planner(context, operation_id):
            if callback:
                callback(context)
            return {"operation_id": operation_id, "proposal": proposal,
                    "costs": {"wall_seconds": 0.1, "tokens": 3}, "settled": True, "artifacts": {}}
        return self.loop.plan(planner, reservation={"wall_seconds": 1, "tokens": 10},
                              review_at=120, advice_config=self.advice)

    def bindings(self, candidate_id):
        candidate = self.store.get_candidate(candidate_id)
        common = {key: candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")}
        claim = dict(common, case_id="case", work_id="work", hardware_id="CPU-fixture", scope="complete-call")
        attempt = dict(common, case_id="case", inputs_id="inputs", configuration_id="configuration", diagnostic_id="fixture")
        return claim, attempt

    def arguments(self, body="", collected=None):
        snapshot = self.loop.snapshot()
        proposal = snapshot["state"]["pending"]["proposal"]
        parent = proposal["parent_candidate_id"]
        prior = snapshot["state"]["latest_receipts"].get("direction")
        current = self.store.get("direction")
        request = {"schema_version": 1, "direction_id": "direction",
                   "job": {"path": str(self.job_path), "job_id": self.job_id, "artifact_root": str(self.artifacts)},
                   "measurement_epoch_id": "cpu-epoch", "workspace": str(self.workspace),
                   "anchor": source_reference(self.store, current["anchor"]["candidate_id"]) if current["anchor"] else None,
                   "tip": source_reference(self.store, parent), "donors": [], "read_only": [file_ref(self.note)],
                   "next_question": proposal["question"], "budget": {"wall_seconds": 3, "max_tokens": None},
                   "continuation": {"mode": "resume" if prior else "fresh", "prior_receipt": prior}}
        script = self.root / ("worker-%s.py" % self.launches)
        events = [{"type": "thread.started", "thread_id": THREAD}, {"type": "turn.started"},
                  {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 4}}]
        script.write_text("import json, pathlib, sys\n"
                          "prompt = sys.stdin.read()\n"
                          "assignment = json.loads(prompt.split('Host assignment (JSON):\\n', 1)[1])\n"
                          "assert assignment['next_question']\n" + body + "\n" +
                          "\n".join("print(" + repr(json.dumps(event)) + ", flush=True)" for event in events))
        def launch(command, **kwargs):
            self.launches += 1
            return subprocess.Popen([sys.executable, "-B", str(script)], **kwargs)
        def settle(process, attempt):
            process.wait(timeout=3)
            return {"settled": True, "scope": "CPU fixture; this child creates no descendants"}
        def collect(result, candidate):
            self.assertIsNotNone(candidate)
            self.assertTrue(result["receipt"]["process_reaped"])
            return collected or {"costs": {"wall_seconds": result["receipt"]["elapsed_seconds"], "tokens": 14},
                                 "artifacts": {"inspection.txt": b"Host CPU fixture source inspection, no performance claim."},
                                 "card": None, "exchange_receipt": None,
                                 "research_update": {"note": self.research_note(),
                                    "claim_head": self.loop.snapshot()["state"]["claim_head"]}}
        binding, attempt_binding = self.bindings(parent)
        return {"request": request, "binding": binding, "attempt_binding": attempt_binding,
                "attempts_root": self.attempts, "launch_worker": launch, "settle_worker": settle,
                "backend_options": {"model_profile": profile(self.root)},
                "capture_tip": self.capture, "collect": collect}

    def execute(self, body="", **kwargs):
        arguments = self.arguments(body)
        arguments.update(kwargs)
        return self.loop.execute(**arguments)

    def research_note(self):
        return {"claim": "The CPU fixture source and receipt were inspected; no native finding.",
                "scope": "Synthetic CPU behavior only", "next_test": "Inspect the next fixture input",
                "evidence_uptake": "Retain the source observation without inferring a speedup."}

    def review_latest(self, note=None, head=None):
        direction = self.store.get("direction")
        state = self.loop.snapshot()["state"]
        history = next(item for item in reversed(state["history"]) if item.get("attempt_id") == direction["note"]["evolution_operation_id"])
        return self.loop.review_research("direction", expected_revision=direction["revision"],
            transport_ref=history["transport_ref"], collection_ref=history["collection_ref"],
            research_update={"note": note or self.research_note(), "claim_head": head or state["claim_head"]})

    def archive(self, candidate, value, name="record.json"):
        identity = self.store.archive_evidence(candidate, {name: json.dumps(value).encode()})
        return {"evidence_id": identity, "artifact": name}

    def quantity_claim(self, value):
        binding, _ = self.bindings(self.parent)
        quantity = {"value": value, "unit": "us", "denominator": "call", "population": "CPU fixture",
                    "aggregation": "mean", "measurement_kind": "synthetic fixture observation"}
        ref = self.archive(self.parent, {"binding": binding, "quantity": quantity})
        self.authorized.add(ref["evidence_id"])
        record = {"kind": "reported_quantity", "statement": "Fixture reports a quantity; not a native observation.",
                  "binding": binding, "evidence": [ref], "depends_on": [], "assumptions": [],
                  "applicability": "CPU fixture", "next_test": "Inspect underlying fixture",
                  "quantity": quantity, "extraction": {"ref": ref, "pointer": ""}}
        return self.loop.update_claims(lambda log: log.append(self.parent, record, binding))

    def test_multi_turn_real_transport_preserves_tip_and_feedback(self):
        self.plan()
        first = self.execute("pathlib.Path('implementation.py').write_text('value = 2\\n')")
        first_tip = self.store.get("direction")["tip"]
        self.assertNotEqual(first_tip, self.parent)
        self.assertEqual(first["state"]["phase"], "idle")
        self.assertEqual(first["accounting"]["known_delivered"]["tokens"], 17)
        seen = []
        self.plan(callback=lambda context: seen.append(context["archive"]["directions"]["direction"]["tip"]))
        self.execute("pathlib.Path('implementation.py').write_text('value = 3\\n')")
        self.assertEqual(seen, [first_tip])
        self.assertEqual(self.launches, 2)
        self.assertEqual(self.loop.snapshot()["accounting"]["known_delivered"]["tokens"], 34)
        child = self.store.get_candidate(self.store.get("direction")["tip"])
        self.assertEqual(child["parents"], [first_tip])
        self.assertIsNone(self.store.get("direction")["anchor"])
        self.assertEqual(len(self.loop.snapshot()["state"]["history"]), 2)

    @unittest.skipIf(ORIGINAL, "prospective authoritative engineer attribution")
    def test_card_author_uses_admitted_engineer_and_current_transport(self):
        self.plan(self.proposal(engineer_id="assigned-engineer"))
        collected = {"costs": {"wall_seconds": 0.2, "tokens": 14}, "artifacts": {},
                     "card": {"idea": self.card["idea"],
                              "author": {"engineer_id": "self-reported", "attempt_ref": None}},
                     "exchange_receipt": None,
                     "research_update": {"note": self.research_note(), "claim_head": None}}
        self.loop.execute(**self.arguments("pathlib.Path('implementation.py').write_text('value = 2\\n')", collected))
        direction = self.store.get("direction")
        card = read_record(self.store, direction["note"]["archive_card_ref"])
        history = self.loop.snapshot()["state"]["history"][-1]
        self.assertEqual(card["author"], {"engineer_id": "assigned-engineer", "attempt_ref": history["transport_ref"]})
        envelope, _ = self.store.evidence(card["author"]["attempt_ref"]["evidence_id"])
        self.assertEqual(envelope["candidate_id"], direction["tip"])
        self.assertEqual(collected["card"]["author"]["engineer_id"], "self-reported")
        # An omitted next card inherits the idea, never the previous author.
        self.plan(self.proposal(engineer_id="next-engineer"))
        self.execute("pathlib.Path('implementation.py').write_text('value = 3\\n')")
        direction = self.store.get("direction")
        card = read_record(self.store, direction["note"]["archive_card_ref"])
        history = self.loop.snapshot()["state"]["history"][-1]
        self.assertEqual(card["author"], {"engineer_id": "next-engineer", "attempt_ref": history["transport_ref"]})

    def test_collector_card_omission_preserves_last_faithful_after_fallback(self):
        witness = {"candidate_id": self.parent, "claim": "Scoped CPU fixture mechanism",
                   "evidence_refs": [self.archive(self.parent, {"inspection": "Original source retains the fixture mechanism"})]}
        direction = self.store.get("direction")
        archive_checkpoint(self.store, "direction", self.parent,
                           card=dict(self.card, last_faithful=witness), note=direction["note"],
                           expected_revision=direction["revision"])
        self.plan()
        collected = {"costs": {"wall_seconds": 0.2, "tokens": 14}, "artifacts": {},
                     "card": {"idea": self.card["idea"]}, "exchange_receipt": None,
                     "research_update": {"note": self.research_note(), "claim_head": None}}
        self.loop.execute(**self.arguments("pathlib.Path('implementation.py').write_text('fallback = True\\n')", collected))
        reopened = self.reopen()
        row = reopened.snapshot()["archive"]["directions"]["direction"]
        self.assertNotEqual(row["tip"], self.parent)
        self.assertEqual(row["card"]["last_faithful"], witness)
        self.assertEqual(self.store.get_candidate(row["tip"])["parents"], [self.parent])
        self.assertEqual(self.launches, 1)

    def test_exact_repeat_requires_review_and_correction_changes_admission(self):
        old = self.quantity_claim("12")
        premise = {"claim": old, "use": "fact", "reason": "Inspect exact fixture quantity", "next_test": "Read fixture"}
        self.plan(self.proposal(premises=[premise]))
        self.execute()  # No edit: the exact source/protocol remains unchanged.
        self.plan(self.proposal(premises=[premise]))
        with self.assertRaisesRegex(ClaimError, "explicit delta"):
            self.execute()
        self.assertEqual(self.launches, 1)
        self.loop.reject_plan("Unchanged experiment lacks a replication question.")
        matching = self.loop.snapshot()["state"]["history"][0]["attempt_id"]
        delta = {"kind": "replication", "reason": "Distinguish transient transport from reproducible behavior.",
                 "expected_observation": "The same source yields the same fixture value.", "reviewed_attempts": [matching], "evidence": []}
        self.plan(self.proposal(premises=[premise], repeat_delta=delta))
        replacement = self.quantity_claim("11")
        binding, _ = self.bindings(self.parent)
        self.loop.update_claims(lambda log: log.correct(self.parent, [old], scope=binding,
                                reason="Fixture source corrected", consequence="Use replacement in next plan",
                                replacement=replacement, status="superseded"))
        with self.assertRaisesRegex(EvolutionError, "corrected factual premise"):
            self.execute()
        self.assertEqual(self.launches, 1)
        self.loop.reject_plan("Old quantity superseded after planning; recompute premise.")
        self.plan(self.proposal(premises=[dict(premise, claim=replacement)], repeat_delta=delta,
                               evidence_uptake="The original quantity was superseded; inspect replacement evidence."))
        self.execute("pathlib.Path('implementation.py').write_text('value = 2\\n')")
        self.assertEqual(self.launches, 2)
        self.assertIn("superseded", self.store.get("direction")["note"]["planner_intent"]["evidence_uptake"])
        self.assertEqual(self.store.get("direction")["note"]["research_status"], "reported")

    def membership(self, candidate_id, island):
        charter = archive_record(self.store, candidate_id, {"kind": "evo_island_charter", "island_id": island,
                                                           "question": "Inspect a distinct representation", "descriptor_spec": {}})
        return archive_record(self.store, candidate_id, {"kind": "evo_membership", "island_id": island,
                                                        "charter_ref": charter, "assignment": {"status": "tentative",
                                                        "descriptors": {}, "evidence_refs": [], "reason": "CPU hypothesis"}})

    def test_source_exchange_enters_actual_engineer_and_preserves_unfinished_bridge(self):
        donor_work = self.root / "donor"
        donor_work.mkdir()
        (donor_work / "implementation.py").write_text("value = 7\n")
        donor = self.store.candidate(self.job_id, "cpu-epoch", self.capture(donor_work))
        donor_source = self.store.get_candidate(donor)["source_id"]
        question = "Adapt the donor's smaller live-state representation and inspect its precondition."
        exchange = {"schema_version": 1, "exchange_id": "exchange", "job_id": self.job_id,
                    "measurement_epoch_id": "cpu-epoch", "question": question,
                    "recipient": {"direction_id": "direction", "revision": 1, "candidate_id": self.parent,
                                  "source_id": self.store.get_candidate(self.parent)["source_id"], "island_id": "state",
                                  "membership_ref": self.membership(self.parent, "state")},
                    "offers": [{"offer_id": "donor", "kind": "source", "candidate_id": donor, "source_id": donor_source,
                                "origin_island_id": "layout", "selected_island_id": "layout",
                                "origin_ref": self.membership(donor, "layout"), "membership_ref": self.membership(donor, "layout"),
                                "claim": "Smaller live state", "scope": "CPU fixture only", "carrier_refs": [],
                                "obligations": [{"question": "Preserve the original value", "status": "unknown", "evidence_refs": []}]}]}
        ref = self.archive(self.parent, exchange)
        self.plan(self.proposal(question=question, exchange_ref=ref, island_id="wrong-island"))
        with self.assertRaisesRegex(EvolutionError, "island"):
            self.execute()
        self.assertEqual(self.launches, 0)
        self.loop.reject_plan("Use the actual selection-time recipient island.")
        self.plan(self.proposal(question=question, exchange_ref=ref))
        report = {"schema_version": 1, "exchange_id": "exchange", "dispositions": [{"offer_id": "donor", "decision": "adapt", "reason": "Needs a bridge"}],
                  "realizations": [], "bridge_repairs": [{"offer_ids": ["donor"], "status": "unfinished", "claim": "Bridge needs repair", "scope": "CPU fixture", "evidence_refs": []}],
                  "next_test": "Repair the bridge before evaluating the idea"}
        collected = {"costs": {"wall_seconds": 0.2, "tokens": 14}, "artifacts": {}, "card": None, "exchange_receipt": report}
        args = self.arguments("assert 'Report every offer' in prompt\n"
                              "assert 'primary evidence' in prompt\n"
                              "assert assignment['donors'][0]['source_id'] == " + repr(donor_source) + "\n"
                              "pathlib.Path('implementation.py').write_text('unfinished_bridge = True\\n')", collected)
        result = self.loop.execute(**args)
        candidate = self.store.get_candidate(self.store.get("direction")["tip"])
        self.assertEqual(candidate["donors"], [donor])
        self.assertIsNone(self.store.get("direction")["anchor"])
        history = result["state"]["history"][-1]
        _, artifacts = self.store.evidence(history["collection_ref"]["evidence_id"])
        self.assertEqual(json.loads(artifacts["collection.json"])["exchange_receipt"]["bridge_repairs"][0]["status"], "unfinished")

    def test_planner_interruption_is_not_redispatched_and_recovery_is_once(self):
        pending = {}
        def planner(context, operation_id):
            pending.update(operation_id=operation_id, proposal=self.proposal(), costs={"wall_seconds": 0.1, "tokens": 3}, settled=True, artifacts={})
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.loop.plan(planner, reservation={"wall_seconds": 1, "tokens": 10}, review_at=120, advice_config=self.advice)
        resumed = self.reopen()
        with self.assertRaisesRegex(EvolutionError, "never redispatched"):
            resumed.plan(lambda *a: self.fail("duplicate planner"), reservation={"wall_seconds": 1, "tokens": 10}, review_at=120, advice_config=self.advice)
        resumed.recover_planner(pending, verify_recovery=lambda intent, result: intent["operation_id"] == result["operation_id"])
        with self.assertRaisesRegex(EvolutionError, "pending intent"):
            resumed.recover_planner(pending, verify_recovery=lambda *a: True)
        self.assertEqual(resumed.snapshot()["accounting"]["known_delivered"]["tokens"], 3)

    def test_completed_transport_lost_return_is_recovered_without_second_launch(self):
        self.plan()
        arguments = self.arguments("pathlib.Path('implementation.py').write_text('unfinished = True\\n')")
        # The port imports the transport lazily from its own module; the original bound it at import.
        module = importlib.import_module("geak_evo.evolution" if ORIGINAL else RUNNER_MODULE)
        original = module.run_turn
        saved = {}
        def interrupted(*args, **kwargs):
            saved.update(original(*args, **kwargs))
            raise KeyboardInterrupt()
        with mock.patch.object(module, "run_turn", interrupted), self.assertRaises(KeyboardInterrupt):
            self.loop.execute(**arguments)
        resumed = self.reopen()
        with self.assertRaisesRegex(EvolutionError, "ambiguous"):
            resumed.execute(**arguments)
        collected = arguments["collect"](saved, self.parent)
        resumed.recover_engineer(result=saved, collected=collected,
                                 verify_recovery=lambda intent, result, collected: result["receipt"]["request"] == intent["request"])
        self.assertEqual(self.launches, 1)
        self.assertEqual(resumed.snapshot()["accounting"]["known_delivered"]["tokens"], 17)
        self.assertEqual(self.store.get("direction")["status"], "active")
        self.assertEqual(len(resumed.snapshot()["state"]["history"]), 1)

    def test_settlement_crash_after_direction_checkpoint_does_not_double_charge(self):
        self.plan()
        original = self.loop._save
        fired = []
        def crash(record, state):
            if state["phase"] == "idle" and state["history"] and not fired:
                fired.append(True)
                raise KeyboardInterrupt()
            return original(record, state)
        with mock.patch.object(self.loop, "_save", crash), self.assertRaises(KeyboardInterrupt):
            self.execute("pathlib.Path('implementation.py').write_text('value = 4\\n')")
        revision = self.store.get("direction")["revision"]
        resources = self.store.get("direction")["resources"]
        resumed = self.reopen()
        self.assertEqual(resumed.snapshot()["state"]["phase"], "settling")
        resumed.resume_settlement()
        self.assertEqual(self.store.get("direction")["revision"], revision)
        self.assertEqual(self.store.get("direction")["resources"], resources)
        self.assertEqual(resumed.snapshot()["accounting"]["known_delivered"]["tokens"], 17)
        with self.assertRaisesRegex(EvolutionError, "no replayable"):
            resumed.resume_settlement()

    def test_collected_correction_survives_settlement_crash_and_reaches_next_planner(self):
        old, replacement = self.quantity_claim("12"), self.quantity_claim("11")
        base = self.loop.snapshot()["state"]["claim_head"]
        log = ClaimLog(self.store, base, verify_quantity=self.loop.verify_quantity)
        binding, _ = self.bindings(self.parent)
        head = log.correct(self.parent, [old], scope=binding, reason="Fixture field was corrected",
                           consequence="Use the new quantity in the next experiment", replacement=replacement,
                           status="superseded")
        note = {"claim": "The fixture quantity is 11 us; the prior 12 us field was superseded.",
                "scope": "Synthetic complete-call fixture only", "next_test": "Check the corrected field against the next input",
                "evidence_uptake": "The correction changes the comparison and next input test."}
        self.plan(self.proposal(reason="Pre-turn prediction based on 12 us"))
        collected = {"costs": {"wall_seconds": 0.2, "tokens": 14}, "artifacts": {}, "card": None,
                     "exchange_receipt": None, "research_update": {"note": note, "claim_head": head}}
        original = self.loop._save
        def crash(record, state):
            if state["phase"] == "idle" and state["history"]:
                raise KeyboardInterrupt()
            return original(record, state)
        with mock.patch.object(self.loop, "_save", crash), self.assertRaises(KeyboardInterrupt):
            self.loop.execute(**self.arguments("pathlib.Path('implementation.py').write_text('value = 2\\n')", collected))
        direction = self.store.get("direction")
        self.assertEqual(self.loop.snapshot()["state"]["claim_head"], base)
        resumed = self.reopen(verify_quantity=self.loop.verify_quantity)
        resumed.resume_settlement()
        self.loop = resumed
        self.assertEqual(self.store.get("direction")["revision"], direction["revision"])
        self.assertEqual(self.store.get("direction")["resources"], direction["resources"])
        self.assertEqual(self.loop.snapshot()["state"]["claim_head"], head)
        self.assertEqual(len(self.loop.snapshot()["claims"]["corrections"]), 1)
        self.assertEqual(self.store.get("direction")["note"]["claim"], note["claim"])
        self.assertEqual(self.store.get("direction")["note"]["research_status"], "reported")
        delivered = []
        self.plan(callback=lambda context: delivered.append(context))
        self.assertEqual(delivered[0]["direction_notes"]["direction"]["note"]["next_test"], note["next_test"])
        old_entry = [entry for entry in delivered[0]["claims"]["claims"].values() if entry["ref"] == old][0]
        self.assertEqual(old_entry["state"], "superseded")
        self.assertEqual(self.launches, 1)

    def test_collected_claim_head_cannot_rollback_or_import_another_epoch(self):
        old, current = self.quantity_claim("12"), self.quantity_claim("11")
        note = {"claim": "Correction requires review", "scope": "CPU fixture",
                "next_test": "Inspect the raw field", "evidence_uptake": "A recorded discrepancy matters"}
        self.plan()
        collected = {"costs": {"wall_seconds": 0.1, "tokens": 14}, "artifacts": {}, "card": None,
                     "exchange_receipt": None, "research_update": {"note": note, "claim_head": old}}
        with self.assertRaisesRegex(EvolutionError, "descend"):
            self.loop.execute(**self.arguments(collected=collected))
        self.assertEqual(self.loop.snapshot()["state"]["claim_head"], current)
        self.assertEqual(self.loop.snapshot()["state"]["phase"], "collecting")
        self.store.allowed_epochs |= {(self.job_id, "other-epoch")}
        source = self.store.get_candidate(self.parent)["source_id"]
        other = self.store.candidate(self.job_id, "other-epoch", source)
        binding, _ = self.bindings(other)
        log = ClaimLog(self.store, current, verify_quantity=self.loop.verify_quantity)
        record = {"kind": "scientific_claim", "statement": "Another epoch's hypothesis",
                  "binding": binding, "evidence": [], "depends_on": [], "assumptions": [],
                  "applicability": "Another epoch", "next_test": "Inspect evidence"}
        foreign = log.append(other, record, binding)
        with self.assertRaisesRegex(EvolutionError, "another job/epoch"):
            self.loop._research_update({"note": note, "claim_head": foreign}, current, self.loop.snapshot()["state"]["binding"])
        collected["research_update"]["claim_head"] = current
        self.loop.recover_engineer(collected=collected, verify_recovery=lambda *args: True)
        self.assertEqual(self.launches, 1)

    def test_pending_report_review_requires_exact_refs_and_replays_once(self):
        self.plan()
        collected = {"costs": {"wall_seconds": 0.1, "tokens": 14}, "artifacts": {}, "card": None, "exchange_receipt": None}
        self.loop.execute(**self.arguments(collected=collected))
        state = self.loop.snapshot()["state"]
        history = state["history"][-1]
        direction = self.store.get("direction")
        update = {"note": self.research_note(), "claim_head": state["claim_head"]}
        with self.assertRaisesRegex(EvolutionError, "exact settled outcome"):
            self.loop.review_research("direction", expected_revision=direction["revision"],
                transport_ref=history["collection_ref"], collection_ref=history["collection_ref"], research_update=update)
        original = self.loop._save
        def crash(record, value):
            if value["phase"] == "idle":
                raise KeyboardInterrupt()
            return original(record, value)
        with mock.patch.object(self.loop, "_save", crash), self.assertRaises(KeyboardInterrupt):
            self.loop.review_research("direction", expected_revision=direction["revision"],
                transport_ref=history["transport_ref"], collection_ref=history["collection_ref"], research_update=update)
        reviewed = self.store.get("direction")
        resumed = self.reopen()
        self.assertEqual(resumed.snapshot()["state"]["phase"], "reviewing")
        resumed.resume_research_review()
        self.assertEqual(self.store.get("direction")["revision"], reviewed["revision"])
        self.assertEqual(self.store.get("direction")["resources"], reviewed["resources"])
        self.assertEqual(resumed.snapshot()["state"]["phase"], "idle")
        self.assertEqual(resumed.snapshot()["accounting"]["known_delivered"]["tokens"], 17)
        self.assertEqual(self.launches, 1)

    def test_unknown_cost_is_not_zero_or_released_and_overrun_is_retained(self):
        self.plan()
        arguments = self.arguments(collected={"costs": {"wall_seconds": None, "tokens": None},
                                             "artifacts": {}, "card": None, "exchange_receipt": None})
        result = self.loop.execute(**arguments)
        self.assertIsNone(result["accounting"]["delivered"]["tokens"])
        self.assertEqual(result["accounting"]["remaining_reservation"]["tokens"], 200)
        self.review_latest()
        self.plan(self.proposal(repeat_delta={"kind": "replication", "reason": "Check fixture again", "expected_observation": "Same value",
                                             "reviewed_attempts": [result["state"]["history"][0]["attempt_id"]], "evidence": []}))
        arguments = self.arguments(collected={"costs": {"wall_seconds": 0.2, "tokens": 201},
                                             "artifacts": {"overrun.txt": b"Actual cost exceeded reserved cost."}, "card": None, "exchange_receipt": None})
        result = self.loop.execute(**arguments)
        self.assertEqual(result["state"]["phase"], "accounting_blocked")
        self.assertIn("delivery exceeds reservation", result["accounting_error"])
        self.assertEqual(result["state"]["opportunities"][-1]["receipts"][0]["costs"]["tokens"], 201)

    def test_run_drives_two_planner_decisions_using_settled_evidence(self):
        contexts = []
        def planner(context, operation_id):
            contexts.append(context)
            return {"operation_id": operation_id, "proposal": self.proposal(),
                    "costs": {"wall_seconds": 0.1, "tokens": 3}, "settled": True, "artifacts": {}}
        def assignment(proposal, snapshot):
            return self.arguments("pathlib.Path('implementation.py').write_text('value = %s\\n')" % (self.launches + 2))
        result = self.loop.run(planner, assignment, planning_reservation={"wall_seconds": 1, "tokens": 10},
                               planning_seconds=20, advice_config=self.advice, max_steps=2)
        self.assertEqual(self.launches, 2)
        self.assertEqual(len(contexts[1]["state"]["history"]), 1)
        self.assertEqual(result["accounting"]["known_delivered"]["tokens"], 34)

    def test_run_selects_research_correction_then_engineer_without_manual_review(self):
        old, replacement = self.quantity_claim("12"), self.quantity_claim("11")
        binding, _ = self.bindings(self.parent)
        self.plan()
        unreviewed = {"costs": {"wall_seconds": 0.1, "tokens": 14}, "artifacts": {}, "card": None, "exchange_receipt": None}
        self.loop.execute(**self.arguments("pathlib.Path('implementation.py').write_text('value = 2\\n')", unreviewed))
        views, choices = [], []
        corrected_note = {"claim": "The fixture field is 11 us; 12 us was superseded.", "scope": "CPU fixture only",
                          "next_test": "Inspect the corrected input", "evidence_uptake": "The correction changes the next comparison."}
        def planner(context, operation_id):
            views.append(context)
            entry = context["direction_notes"]["direction"]
            if entry["note"].get("research_status") == "pending_review":
                proposal = {"kind": "research_review", "direction_id": "direction", "expected_revision": entry["revision"],
                    "transport_ref": entry["note"]["latest_transport_ref"], "collection_ref": entry["note"]["research_update_ref"],
                    "note": corrected_note, "claim_requests": [{"targets": [old], "replacement": replacement,
                        "scope": binding, "status": "superseded", "reason": "The fixture field was corrected",
                        "consequence": "Change the next input comparison"}]}
            else:
                proposal = self.proposal()
            choices.append(proposal.get("kind", "engineer"))
            return {"operation_id": operation_id, "proposal": proposal,
                    "costs": {"wall_seconds": 0.1, "tokens": 3}, "settled": True, "artifacts": {}}
        def assignment(proposal, snapshot):
            if proposal.get("kind") == "research_review":
                log = ClaimLog(self.store, snapshot["state"]["claim_head"], verify_quantity=self.loop.verify_quantity)
                for change in proposal["claim_requests"]:
                    log.correct(self.parent, **change)
                return {"research_update": {"note": proposal["note"], "claim_head": log.head}}
            return self.arguments("pathlib.Path('implementation.py').write_text('value = 3\\n')")
        result = self.loop.run(planner, assignment, planning_reservation={"wall_seconds": 1, "tokens": 10},
                               planning_seconds=20, advice_config=self.advice, max_steps=2)
        self.assertEqual(choices, ["research_review", "engineer"])
        self.assertIn("research_review", views[0]["instructions"]["text"])
        self.assertEqual(views[1]["direction_notes"]["direction"]["note"]["next_test"], corrected_note["next_test"])
        self.assertEqual(len(views[1]["claims"]["corrections"]), 1)
        self.assertEqual(self.launches, 2)
        self.assertEqual(sum(item.get("review_kind") == "research_review" for item in result["state"]["history"]), 1)
        self.assertEqual(result["accounting"]["known_delivered"]["tokens"], 37)

    def test_run_maturation_observation_replays_then_selects_review_and_engineer(self):
        witness = self.archive(self.parent, {"witness": "Synthetic source-level faithful seed"})
        nomination = checkpoint_maturation(self.store, "direction", expected_revision=self.store.get("direction")["revision"],
            decision={"action": "nominate", "reason": "Develop the fixture", "next_test": "Inspect source", "evidence_refs": [],
                      "proposition": "Smaller live state", "scope": "CPU fixture", "last_faithful": {
                          "candidate_id": self.parent, "claim": "Synthetic witness only", "evidence_refs": [witness]}})
        self.plan(self.proposal(protection_ref=nomination["ref"]))
        def grant(proposal, opportunity, accounting):
            return checkpoint_maturation(self.store, "direction", expected_revision=self.store.get("direction")["revision"],
                decision=dict(proposal["grant_intent"], action="grant", opportunity_id=opportunity["opportunity_id"]),
                accounting=accounting)["ref"]
        self.execute("pathlib.Path('implementation.py').write_text('value = 2\\n')", admit_protection=grant)
        choices, verified = [], []
        def planner(context, operation_id):
            owed = context["owed_maturation_reviews"]
            if owed:
                entry = context["direction_notes"]["direction"]
                decision = {"action": "observe" if owed[0]["needs_observation"] else "review", "reason": "Inspect finite fixture work",
                            "next_test": "Test the next source obligation", "evidence_refs": []}
                if decision["action"] == "review":
                    decision.update(outcome="continue", changed_since_prior="CPU delivery is now recorded", revival_condition="Changed compiler evidence")
                proposal = {"kind": "maturation_review", "direction_id": "direction", "expected_revision": entry["revision"],
                            "maturation_ref": entry["note"]["maturation_ref"], "decision": decision}
                choices.append(decision["action"])
            else:
                proposal = self.proposal()
                choices.append("engineer")
            return {"operation_id": operation_id, "proposal": proposal,
                    "costs": {"wall_seconds": 0.1, "tokens": 3}, "settled": True, "artifacts": {}}
        def assignment(proposal, snapshot):
            if proposal.get("kind") != "maturation_review":
                return self.arguments("pathlib.Path('implementation.py').write_text('value = 3\\n')")
            if proposal["decision"]["action"] == "review":
                return {}
            current = self.store.get("direction")["tip"]
            grant = snapshot["maturation"]["direction"]["grant"]
            observation = self.archive(current, {"kind": "maturation_observation", "binding": snapshot["state"]["binding"],
                "direction_id": "direction", "opportunity_id": grant["opportunity_id"], "assigned_candidate_id": grant["start_candidate_id"],
                "candidate_id": current, "delivered": True, "work_kind": "implementation_or_diagnosis", "settled": True, "interrupted": False})
            def verify(ref, data, expected):
                verified.append(ref)
                return ref == observation and all(json.loads(data).get(k) == v for k, v in expected.items())
            return {"observation_ref": observation, "verify_host": verify}
        original = self.loop._save
        def crash(record, state):
            if state["phase"] == "idle" and state["history"][-1].get("review_kind") == "maturation_review":
                raise KeyboardInterrupt()
            return original(record, state)
        with mock.patch.object(self.loop, "_save", crash), self.assertRaises(KeyboardInterrupt):
            self.loop.run(planner, assignment, planning_reservation={"wall_seconds": 1, "tokens": 10},
                          planning_seconds=20, advice_config=self.advice, max_steps=1)
        revision = self.store.get("direction")["revision"]
        self.loop = self.reopen()
        self.loop.resume_maturation_review()
        self.assertEqual(self.store.get("direction")["revision"], revision)
        result = self.loop.run(planner, assignment, planning_reservation={"wall_seconds": 1, "tokens": 10},
                               planning_seconds=20, advice_config=self.advice, max_steps=2)
        self.assertEqual(choices, ["observe", "review", "engineer"])
        self.assertEqual(len(verified), 1)
        self.assertEqual(self.launches, 2)
        self.assertEqual(result["owed_maturation_reviews"], [])
        self.assertEqual(sum(item.get("review_kind") == "maturation_review" for item in result["state"]["history"]), 2)
        self.assertEqual(result["accounting"]["known_delivered"]["tokens"], 40)

    def test_protected_continuation_requires_host_contract_not_agent_badge(self):
        witness = self.archive(self.parent, {"scope": "CPU source fixture, not native faithfulness"})
        nomination = checkpoint_maturation(self.store, "direction", expected_revision=self.store.get("direction")["revision"],
                           decision={"action": "nominate", "reason": "Develop a slower faithful intermediate", "next_test": "Inspect source",
                                     "evidence_refs": [], "proposition": "Smaller live state", "scope": "CPU fixture",
                                     "last_faithful": {"candidate_id": self.parent, "claim": "Fixture mechanism present", "evidence_refs": [witness]}})
        self.plan(self.proposal(protection_ref=nomination["ref"], grant_intent=None))
        with self.assertRaisesRegex(EvolutionError, "agent grant intent"):
            self.execute()
        self.assertEqual(self.launches, 0)
        self.loop.reject_plan("The planner must choose the scientific grant conditions.")
        self.plan(self.proposal(protection_ref=nomination["ref"]))
        admitted = []
        def grant(proposal, opportunity, accounting):
            admitted.append((proposal["parent_candidate_id"], opportunity["opportunity_id"], accounting["remaining_reservation"]["tokens"]))
            return checkpoint_maturation(self.store, "direction", expected_revision=self.store.get("direction")["revision"],
                       decision=dict(proposal["grant_intent"], action="grant", opportunity_id=opportunity["opportunity_id"]),
                       accounting=accounting)["ref"]
        result = self.execute(admit_protection=grant)
        self.assertEqual(admitted[0][0], self.parent)
        self.assertGreaterEqual(admitted[0][2], 200)
        self.assertEqual(result["state"]["opportunities"][-1]["opportunity_id"], admitted[0][1])
        self.assertTrue(result["state"]["opportunities"][-1]["settled"])
        self.assertEqual(read_maturation(self.store, "direction")["lifecycle"], "active")
        self.assertTrue(result["owed_maturation_reviews"][0]["needs_observation"])
        history = result["state"]["history"][-1]
        _, transport = self.store.evidence(history["transport_ref"]["evidence_id"])
        request = json.loads(transport["transport.json"])["request"]
        protected = []
        for ref in request["read_only"]:
            try:
                value = json.loads(Path(ref["path"]).read_bytes())
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict) and "faithful_seed" in value:
                protected.append(value)
        self.assertEqual(protected[0]["faithful_seed"]["candidate_id"], self.parent)
        # Allocation plus transport does not forge the separate faithful-work observation.
        candidate = self.store.get("direction")["tip"]
        observation = self.archive(candidate, {"kind": "maturation_observation", "binding": result["state"]["binding"],
                            "direction_id": "direction", "opportunity_id": admitted[0][1], "assigned_candidate_id": self.parent,
                            "candidate_id": candidate, "delivered": True, "work_kind": "implementation_or_diagnosis",
                            "settled": True, "interrupted": False})
        observed = self.loop.review_maturation("direction", expected_revision=self.store.get("direction")["revision"],
                         decision={"action": "observe", "reason": "Inspect actual fixture delivery and work", "next_test": "Review the obligation", "evidence_refs": []},
                         observation_ref=observation,
                         verify_host=lambda ref, data, expected: ref == observation and all(json.loads(data).get(k) == v for k, v in expected.items()))
        self.assertEqual(observed["record"]["lifecycle"], "review_due")
        reviewed = self.loop.review_maturation("direction", expected_revision=self.store.get("direction")["revision"],
                         decision={"action": "review", "reason": "The next obligation remains useful", "next_test": "Inspect the next source-bound case",
                                   "evidence_refs": [], "outcome": "continue", "changed_since_prior": "Source delivery and CPU work now established",
                                   "revival_condition": "A changed compiler observation would reopen shelved work"})
        self.assertEqual(reviewed["record"]["lifecycle"], "reviewed")
        self.assertEqual(self.loop.snapshot()["owed_maturation_reviews"], [])

    def test_returned_unsettled_planner_reconciles_same_op_and_keeps_original(self):
        saved = {}
        def planner(context, operation_id):
            saved.update(operation_id=operation_id, proposal=self.proposal(), costs={"wall_seconds": 0.1, "tokens": None}, settled=False, artifacts={})
            return copy.deepcopy(saved)
        self.loop.plan(planner, reservation={"wall_seconds": 1, "tokens": 10}, review_at=120, advice_config=self.advice)
        original_ref = self.loop.snapshot()["state"]["pending"]["planner_ref"]
        saved.update(costs={"wall_seconds": 0.2, "tokens": 5}, settled=True)
        self.loop.recover_planner(saved, verify_recovery=lambda pending, result: pending["operation_id"] == result["operation_id"])
        snapshot = self.loop.snapshot()
        self.assertEqual(snapshot["state"]["phase"], "planned")
        self.assertEqual(len(snapshot["state"]["opportunities"]), 1)
        self.assertEqual(snapshot["accounting"]["known_delivered"]["tokens"], 5)
        self.assertEqual(snapshot["state"]["pending"]["prior_planner_refs"], [original_ref])
        self.assertEqual(len(snapshot["state"]["opportunities"][0]["superseded_receipts"]), 1)

    def test_frozen_diagnostic_registry_and_planner_instructions_reach_callback(self):
        registry = {self.job_id: ["case"]}
        raw = self.archive(self.parent, {"scope": "CPU source fixture"})
        diagnostic = archive_record(self.store, self.parent, {"kind": "evo_diagnostic",
                         "case_ids": ["case"], "claim": "Inspect source only", "artifacts": {"source": raw},
                         "compilation_ref": None, "unknown_reason": "CPU fixture; no compiler output"}, frozen_cases=registry)
        state = self.store.get("direction")
        archive_checkpoint(self.store, "direction", self.parent, card=dict(self.card, observation_refs=[diagnostic]),
                           note=state["note"], expected_revision=state["revision"], frozen_cases=registry)
        self.loop.frozen_cases = registry
        seen = []
        self.plan(callback=lambda context: seen.append(context))
        self.assertEqual(seen[0]["frozen_cases"], registry)
        self.assertIn("primary evidence", seen[0]["instructions"]["text"])
        self.assertEqual(file_ref(seen[0]["instructions"]["ref"]["path"]), seen[0]["instructions"]["ref"])
        ref = self.loop.snapshot()["state"]["pending"]["planner_ref"]
        _, artifacts = self.store.evidence(ref["evidence_id"])
        archived = json.loads(artifacts["planner_context.json"])
        self.assertEqual(archived["instructions"], seen[0]["instructions"])

    def test_driver_supplies_nonempty_advice_and_verified_complete_review(self):
        self.plan()
        result = self.execute("pathlib.Path('implementation.py').write_text('value = 2\\n')")
        operation = result["state"]["history"][-1]["attempt_id"]
        ref = self.archive(self.store.get("direction")["tip"], {"kind": "evolution_progress_review",
                     "binding": result["state"]["binding"], "opportunity_id": operation, "complete": True})
        with self.assertRaisesRegex(EvolutionError, "coverage verification"):
            self.loop.record_progress_review(operation, ref, verify_review=lambda *a: False)
        self.loop.record_progress_review(operation, ref, verify_review=lambda *a: True)
        seen = []
        def planner(context, operation_id):
            seen.append(context["advice"]["rows"])
            return {"operation_id": operation_id, "proposal": self.proposal(), "costs": {"wall_seconds": 0.1, "tokens": 3}, "settled": True, "artifacts": {}}
        def advice(snapshot):
            return {"offers": [dict(snapshot["state"]["binding"], direction_id="direction", island_id="state", engineer_id="engineer",
                                     priority_weight=1, estimated_costs={"wall_seconds": 3, "tokens": 20})]}
        self.loop.run(planner, lambda p, s: self.arguments("pathlib.Path('implementation.py').write_text('value = 3\\n')"),
                      planning_reservation={"wall_seconds": 1, "tokens": 10}, planning_seconds=20,
                      advice_config=self.advice, max_steps=1, advice_inputs=advice)
        self.assertEqual(seen[0][0]["direction_id"], "direction")
        self.assertIsNotNone(seen[0][0]["score"])

    def test_unsettled_engineer_later_settles_exact_owner_with_one_cumulative_charge(self):
        self.plan()
        arguments = self.arguments("pathlib.Path('implementation.py').write_text('value = 8\\n')")
        arguments["settle_worker"] = lambda process, attempt: {"settled": False, "scope": "CPU fixture deliberately unconfirmed"}
        arguments["collect"] = lambda result, candidate: {"costs": {"wall_seconds": 0.1, "tokens": None},
                                                         "artifacts": {}, "card": None, "exchange_receipt": None}
        result = self.loop.execute(**arguments)
        self.assertEqual(result["state"]["phase"], "unsettled")
        prior = result["state"]["pending"]
        original_bytes = Path(prior["receipt_ref"]["path"]).read_bytes()
        recovered = importlib.import_module(RUNNER_MODULE).recover_turn(prior["receipt_ref"], capture_tip=self.capture,
                                 verify_recovery=lambda receipt, attempt: {"settled": True, "process_reaped": True,
                                                                            "scope": "Known fixture child has exited; no descendants"})
        collection = {"costs": {"wall_seconds": 0.2, "tokens": 14}, "artifacts": {}, "card": None, "exchange_receipt": None}
        recovered_state = self.loop.recover_engineer(result=recovered, collected=collection,
                       verify_recovery=lambda pending, result, collected: result["receipt"]["recovery_of"] == pending["receipt_ref"])
        self.assertEqual(recovered_state["state"]["phase"], "idle")
        self.assertEqual(recovered_state["accounting"]["known_delivered"]["tokens"], 17)
        self.assertEqual(len(recovered_state["state"]["history"]), 1)
        self.assertEqual(len(recovered_state["state"]["superseded_attempts"]), 1)
        self.assertEqual(len(recovered_state["state"]["opportunities"][-1]["superseded_receipts"]), 1)
        self.assertEqual(Path(prior["receipt_ref"]["path"]).read_bytes(), original_bytes)
        self.assertEqual(self.launches, 1)
        # The transport owner registry, not just the campaign state, now permits continuation.
        self.review_latest()
        self.plan()
        self.execute("pathlib.Path('implementation.py').write_text('value = 9\\n')")
        self.assertEqual(self.launches, 2)

    def test_admitted_grant_crash_resumes_same_opportunity_without_duplicate_allocation(self):
        self.plan()
        arguments = self.arguments("pathlib.Path('implementation.py').write_text('value = 5\\n')")
        with mock.patch.object(self.loop, "_dispatch", side_effect=KeyboardInterrupt()), self.assertRaises(KeyboardInterrupt):
            self.loop.execute(**arguments)
        state = self.loop.snapshot()["state"]
        operation = state["pending"]["operation_id"]
        self.assertEqual(state["phase"], "admitted")
        self.assertEqual(self.launches, 0)
        runtime = {key: value for key, value in arguments.items() if key not in ("request", "binding", "attempt_binding")}
        self.loop.dispatch_admitted(**runtime)
        self.assertEqual(self.launches, 1)
        self.assertEqual(self.loop.snapshot()["state"]["opportunities"][-1]["opportunity_id"], operation)
        self.assertEqual(len(self.loop.snapshot()["state"]["opportunities"]), 2)

    def test_invalid_unfinished_tip_does_not_replace_validated_anchor(self):
        # Synthetic eligibility is scoped to this CPU lifecycle fixture only.
        self.store.verify_evidence = lambda artifacts, bound: {key: True for key in ("independent", "settled", "correct", "eligible")}
        proof = self.archive(self.parent, {"fixture": "CPU synthetic anchor"})
        self.store.promote("direction", self.parent, proof["evidence_id"], expected_revision=self.store.get("direction")["revision"])
        self.plan()
        self.execute("pathlib.Path('implementation.py').write_text('incomplete = !!!\\n')\nsys.exit(7)")
        state = self.store.get("direction")
        self.assertEqual(state["anchor"]["candidate_id"], self.parent)
        self.assertNotEqual(state["tip"], self.parent)
        self.assertEqual(state["status"], "unfinished")


if __name__ == "__main__":
    unittest.main()
