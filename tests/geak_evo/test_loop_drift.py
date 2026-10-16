"""Drift tests for geak_evo.loop, on synthetic CPU fixtures only.

1  resume in every phase: an operation is never started twice and its cost is
   counted exactly once.
5  golden ledger: reservations, program lease, reviews and the review_at
   clipping preserve the Oct-15 original apart from explicit author metadata.
6  continuation rejects a stale revision, a parent that is not the tip, and a
   request that does not continue the latest receipt.
10 deadlines and review horizons use the injected clock, never time.time.

The engineer is a test double: an in-process native investigation callback
that captures a new source and writes a receipt. validate_request is the
identity here; the real request checks live in the engineer modules.

The scenario functions take a namespace of module handles so that the same
code produced the golden on the original (geak_evo.evolution etc.).
"""

import contextlib
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock


JOB = hashlib.sha256(b"synthetic loop drift job").hexdigest()
EPOCH = "cpu-epoch"
LIMITS = {"wall_seconds": 100, "tokens": 10000}
ADVICE = {"weights": {"wall_seconds": 0.5, "tokens": 0.5}, "c_ref": 0.01, "half_life": 1, "epsilon": 0.1, "beta": 1}
PROMPTS = {
    "planner.md": b"Synthetic planner instructions for tests. Use primary evidence. "
                  b"Choose research_review or maturation_review or an engineer.\n",
    "engineer.md": b"Synthetic engineer instructions for tests. Use primary evidence.\n",
    "crossover.md": b"Synthetic crossover instructions. Report every offer.\n",
    "program_lease_engineer.md": b"Synthetic program lease engineer instructions.\n",
}
LEASE_TERMS = {"episodes": 2, "release_condition": "Release after two reviewed episodes.",
               "displaced_work": "Other directions wait for this commitment."}
PLANNER_RESERVATION = {"wall_seconds": 2, "tokens": 10}
PLANNER_COSTS = {"wall_seconds": 1, "tokens": 3}
ENGINEER_COSTS = {"wall_seconds": 2, "tokens": 14}


def note(text):
    return {"claim": "Synthetic CPU observation " + text + "; no performance claim.",
            "scope": "Synthetic CPU fixture only.", "next_test": "Inspect the next fixture input.",
            "evidence_uptake": "Use the retained receipt only."}


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Counter:
    """Deterministic replacement for uuid.uuid4 inside the loop module."""

    def __init__(self):
        self.count = 0

    def uuid4(self):
        self.count += 1
        return "00000000-0000-4000-8000-%012d" % self.count


def port_namespace():
    from geak_evo import loop
    from geak_evo.archive.records import archive_checkpoint
    from geak_evo.archive.refs import file_ref
    from geak_evo.archive.store import ResearchStore

    def create(store, seed, clock, prompt_dir, *, deadline, program_lease):
        return loop.EvolutionLoop.create(store, "campaign", seed, limits=LIMITS, deadline=deadline, clock=clock,
                                         prompt_dir=prompt_dir, program_lease=program_lease)

    def reopen(store, clock, prompt_dir, *, program_lease):
        return loop.EvolutionLoop(store, "campaign", clock=clock, prompt_dir=prompt_dir, program_lease=program_lease)

    @contextlib.contextmanager
    def engineer_double(counter):
        package = types.ModuleType("geak_evo.engineer")
        package.__path__ = []
        request = types.ModuleType("geak_evo.engineer.request")
        request.validate_request = lambda value: value
        runner = types.ModuleType("geak_evo.engineer.runner")

        def run_turn(*args, **kwargs):
            raise AssertionError("drift tests use the native investigation callback")
        runner.run_turn = run_turn
        modules = {"geak_evo.engineer": package, "geak_evo.engineer.request": request,
                   "geak_evo.engineer.runner": runner}
        with mock.patch.dict(sys.modules, modules), mock.patch.object(loop, "uuid", counter):
            yield

    return types.SimpleNamespace(ResearchStore=ResearchStore, archive_checkpoint=archive_checkpoint,
                                 file_ref=file_ref, create=create, reopen=reopen, engineer_double=engineer_double,
                                 Error=loop.EvolutionError)


class Env:
    """One store, one direction and one campaign, with a counting engineer double."""

    def __init__(self, ns, root, *, deadline=1000, program_lease=0, start=100):
        self.ns, self.root = ns, Path(root)
        self.clock = Clock(start)
        self.prompts = self.root / "prompts"
        self.prompts.mkdir()
        for name, data in PROMPTS.items():
            (self.prompts / name).write_bytes(data)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.attempts = self.root / "attempts"
        self.attempts.mkdir()
        self.store = ns.ResearchStore(self.root / "store", allowed_epochs=[(JOB, EPOCH)],
                                      source_paths={JOB: ["implementation.py"]})
        (self.workspace / "implementation.py").write_text("value = 0\n")
        self.seed = self.store.candidate(JOB, EPOCH, self.capture())
        ns.archive_checkpoint(self.store, "direction", self.seed,
                              card={"idea": {"idea_id": "smaller-state", "statement": "Try a smaller live state.",
                                             "status": "hypothesis"}},
                              note={"claim": "CPU hypothesis", "scope": "Fixture only", "next_test": "Inspect"},
                              expected_revision=0)
        self.program_lease = program_lease
        self.loop = ns.create(self.store, self.seed, self.clock, self.prompts,
                              deadline=deadline, program_lease=program_lease)
        self.launches = {}
        self.turns = 0

    def reopen(self):
        self.loop = self.ns.reopen(self.store, self.clock, self.prompts, program_lease=self.program_lease)
        return self.loop

    def capture(self):
        return self.store.capture_source(self.workspace, ["implementation.py"])

    def state(self):
        return self.loop.snapshot()["state"]

    def engineer(self, **changes):
        direction = self.store.get("direction")
        proposal = {"direction_id": "direction", "expected_revision": direction["revision"],
                    "parent_candidate_id": direction["tip"], "island_id": "unassigned", "engineer_id": "engineer",
                    "membership_ref": None, "question": "Inspect the fixture before a smaller live state.",
                    "reason": "Separate implementation failure from the mechanism.",
                    "evidence_uptake": "Only CPU receipts so far.",
                    "reservation": {"wall_seconds": 5, "tokens": 200}, "review_at": self.clock() + 50,
                    "premises": [], "repeat_delta": None, "exchange_ref": None, "protection_ref": None,
                    "override_reason": None}
        proposal.update(changes)
        return proposal

    def review(self, text):
        direction = self.store.get("direction")
        history = next(item for item in reversed(self.state()["history"])
                       if item.get("attempt_id") == direction["note"]["evolution_operation_id"])
        return {"kind": "research_review", "direction_id": "direction", "expected_revision": direction["revision"],
                "transport_ref": history["transport_ref"], "collection_ref": history["collection_ref"],
                "note": note(text), "claim_requests": []}

    def planner(self, proposal, *, failure=None):
        def call(context, operation_id):
            self.clock.advance(2)
            value = proposal(context) if callable(proposal) else proposal
            result = {"operation_id": operation_id, "proposal": value, "costs": dict(PLANNER_COSTS),
                      "settled": True, "artifacts": {}}
            if failure is not None:
                result["host_disposition"] = {"kind": "planner_response_rejected", "failure": failure}
            return result
        return call

    def plan(self, proposal, *, capacity=None, failure=None):
        return self.loop.plan(self.planner(proposal, failure=failure), reservation=dict(PLANNER_RESERVATION),
                              review_at=self.clock() + 10, advice_config=ADVICE, planner_capacity=capacity,
                              planning_window_seconds=10 if capacity is not None else None)

    def bindings(self, candidate_id):
        candidate = self.store.get_candidate(candidate_id)
        common = {key: candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")}
        claim = dict(common, case_id="case", work_id="work", hardware_id="CPU-fixture", scope="complete-call")
        attempt = dict(common, case_id="case", inputs_id="inputs", configuration_id="configuration",
                       diagnostic_id="fixture")
        return claim, attempt

    def request(self, proposal, *, prior=None, use_latest=True):
        state = self.state()
        latest = state["latest_receipts"].get(proposal["direction_id"]) if use_latest else prior
        return {"schema_version": 1, "direction_id": proposal["direction_id"],
                "job": {"job_id": JOB}, "measurement_epoch_id": EPOCH, "workspace": str(self.workspace),
                "anchor": None, "tip": {"candidate_id": proposal["parent_candidate_id"]}, "donors": [],
                "read_only": [], "next_question": proposal["question"],
                "budget": {"wall_seconds": 3, "max_tokens": None},
                "continuation": {"mode": "resume" if latest else "fresh", "prior_receipt": latest}}

    def investigation(self, *, interrupt=False):
        def run(request, attempts_root, *, prompt_path, operation_id):
            self.launches[operation_id] = self.launches.get(operation_id, 0) + 1
            self.turns += 1
            self.clock.advance(3)
            (self.workspace / "implementation.py").write_text("value = %d\n" % self.turns)
            source = self.capture()
            attempt = Path(attempts_root) / ("attempt-%d" % self.turns)
            attempt.mkdir()
            prompt = Path(prompt_path).read_bytes()
            (attempt / "prompt.txt").write_bytes(prompt)
            receipt = {"attempt_id": operation_id, "request": request, "captured_tip_source_id": source,
                       "settlement": {"settled": True, "scope": "in-process CPU double"},
                       "transport_complete": True, "prompt_sha256": hashlib.sha256(prompt).hexdigest()}
            path = attempt / "receipt.json"
            path.write_text(json.dumps(receipt, sort_keys=True))
            self.last_result = {"receipt": receipt, "receipt_ref": self.ns.file_ref(path), "attempt_dir": str(attempt)}
            if interrupt:
                raise KeyboardInterrupt()
            return self.last_result
        return run

    def collect(self, update=None, *, interrupt=False):
        def collect(result, candidate_id):
            if interrupt:
                raise KeyboardInterrupt()
            return self.collected(update)
        return collect

    def collected(self, update=None):
        value = {"costs": dict(ENGINEER_COSTS), "artifacts": {"inspection.txt": b"CPU fixture inspection."},
                 "card": None, "exchange_receipt": None}
        if update is not None:
            value["research_update"] = {"note": note(update), "claim_head": self.state()["claim_head"]}
        return value

    def execute_arguments(self, proposal, *, update=None, interrupt_launch=False, interrupt_collect=False,
                          request=None):
        binding, attempt_binding = self.bindings(proposal["parent_candidate_id"])
        return {"request": request or self.request(proposal), "binding": binding, "attempt_binding": attempt_binding,
                "collect": self.collect(update, interrupt=interrupt_collect), "attempts_root": str(self.attempts),
                "run_investigation": self.investigation(interrupt=interrupt_launch)}


# --- golden scenarios ----------------------------------------------------------

def lease_scenario(ns, root):
    """run() over two committed episodes, a rejected action, a failed planner and a stop."""
    env = Env(ns, root, program_lease=2)
    script = [
        (lambda context: env.engineer(lease=copy.deepcopy(LEASE_TERMS)), None, None),
        (lambda context: env.review("first"), None, None),
        ({"kind": "research_action", "action": "unknown"}, None, None),
        (lambda context: env.engineer(), None, "second"),
        (None, "synthetic rejected response", None),
        (lambda context: env.review("second"), None, None),
        (None, None, None),
    ]
    updates = []

    def planner(context, operation_id):
        proposal, failure, update = script.pop(0)
        updates.append(update)
        return env.planner(proposal, failure=failure)(context, operation_id)

    def host_assignment(proposal, snapshot):
        kind = proposal.get("kind")
        if kind == "research_action":
            return {"binding": env.bindings(env.store.get("direction")["tip"])[0]}
        if kind == "research_review":
            return {"research_update": {"note": proposal["note"], "claim_head": snapshot["state"]["claim_head"]}}
        return env.execute_arguments(proposal, update=updates[-1])

    env.loop.run(planner, host_assignment, planning_reservation=dict(PLANNER_RESERVATION), planning_seconds=10,
                 advice_config=ADVICE, max_steps=8, planner_decision_limit=10)
    return env


def clipping_scenario(ns, root):
    """run() clips the planner review_at to the campaign deadline."""
    env = Env(ns, root, deadline=105, program_lease=2)
    env.loop.run(env.planner(None), lambda proposal, snapshot: {}, planning_reservation=dict(PLANNER_RESERVATION),
                 planning_seconds=10, advice_config=ADVICE, max_steps=1, planner_decision_limit=3)
    return env


HEX = re.compile(r"\b[0-9a-f]{64}\b")
PROMPT_PATH = re.compile(r'"[^"]*/prompts/(' + "|".join(re.escape(name) for name in PROMPTS) + r')"')
STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00")


def normalize(value, root):
    """Canonical text with paths, timestamps and hashes replaced by stable labels.

    The prompt directory is a constructor option of the port and a fixed
    package path in the original, so prompt paths are labelled too. Hashes
    depend on paths and timestamps; replacing each distinct hash by its order
    of first appearance keeps every equality relation.
    """
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    text = PROMPT_PATH.sub(r'"<PROMPTS>/\1"', text).replace(str(root), "<ROOT>")
    text = STAMP.sub("<TS>", text)
    labels = {}
    return HEX.sub(lambda match: labels.setdefault(match.group(0), "#%d" % len(labels)), text)


def golden_trace(ns, scenario):
    """Every saved campaign state of one scenario, normalized."""
    counter = Counter()
    saved = []
    with tempfile.TemporaryDirectory() as temp, ns.engineer_double(counter):
        root = Path(temp).resolve()
        holder = {}
        original = ns.ResearchStore.checkpoint

        def checkpoint(store, direction_id, *args, **kwargs):
            result = original(store, direction_id, *args, **kwargs)
            if direction_id == "campaign":
                saved.append(copy.deepcopy(result["note"]["evolution"]))
            return result
        with mock.patch.object(ns.ResearchStore, "checkpoint", checkpoint):
            env = scenario(ns, root)
            holder["direction"] = env.store.get("direction")
        final = {"states": saved, "direction": {key: holder["direction"][key]
                                                for key in ("revision", "tip", "resources", "status", "note")}}
        return normalize(final, root)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


# Recorded from the Oct-15 original (geak_evo.evolution with program_lease set
# by attribute, prompts from the same synthetic bytes). Regenerate only when a
# deliberate saved-record change is agreed.
GOLDEN = {
    "lease_scenario": {"sha256": "b7ab69b4f541cb9edfb124b544caaded4bbd129ffb3c9ce03740fc6562a7d56a",
                       "length": 347493},
    "clipping_scenario": {"sha256": "6d419dba309ac84d6262156266702a4fb5f551fa8e4cf7619fe5a637baaca3db",
                          "length": 7857},
}

# Prospective author metadata changes archive-card identities. Preserve the
# original golden above; the comparison below isolates this one added field.
AUTHOR_GOLDEN = {"sha256": "82a47fb413c743560455b05a82638e005c7abdb64ccdb83758c5123a3efa6fb8",
                 "length": 348787}


# --- tests -------------------------------------------------------------------------

class DriftTestCase(unittest.TestCase):
    def setUp(self):
        self.ns = port_namespace()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.counter = Counter()
        double = self.ns.engineer_double(self.counter)
        double.__enter__()
        self.addCleanup(double.__exit__, None, None, None)

    def env(self, **options):
        return Env(self.ns, self.root, **options)

    def delivered(self, env):
        return env.loop.snapshot()["accounting"]["known_delivered"]

    def assert_once(self, env):
        self.assertTrue(env.launches and all(count == 1 for count in env.launches.values()), env.launches)
        for row in env.state()["opportunities"]:
            self.assertLessEqual(len(row["receipts"]), 1)


class GoldenLedgerTests(DriftTestCase):
    """Drift 5: preserve original accounting with an explicit attribution delta."""

    def check(self, scenario, expected=None):
        expected = expected or GOLDEN[scenario.__name__]
        self.assertIsNotNone(expected, "golden not recorded")
        actual = golden_trace(self.ns, scenario)
        self.assertEqual(digest(actual), expected["sha256"])
        self.assertEqual(len(actual), expected["length"])

    def test_lease_scenario_with_author_metadata(self):
        self.check(lease_scenario, AUTHOR_GOLDEN)

    def test_author_is_the_only_change_to_original_lease_scenario(self):
        from geak_evo import loop
        checkpoint = loop.archive_checkpoint

        def without_author(store, direction_id, candidate_id, *, card, **kwargs):
            return checkpoint(store, direction_id, candidate_id,
                card={key: value for key, value in card.items() if key != "author"}, **kwargs)

        with mock.patch.object(loop, "archive_checkpoint", without_author):
            self.check(lease_scenario)

    def test_review_clipping_matches_original(self):
        self.check(clipping_scenario)

    def test_lease_scenario_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            env = lease_scenario(self.ns, Path(temp).resolve())
            state = env.state()
            self.assertEqual(state["phase"], "stopped")
            self.assertEqual([lease["status"] for lease in state["program_leases"]], ["completed"])
            self.assertEqual(sum(env.launches.values()), 2)
            self.assert_once(env)

    def test_review_at_is_clipped_to_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            env = clipping_scenario(self.ns, Path(temp).resolve())
            (row,) = env.state()["opportunities"]
            self.assertEqual((row["opened_at"], row["review_at"]), (100, 105))
            self.assertEqual(row["program_lease_planning_seconds"], 10)


class ResumeTests(DriftTestCase):
    """Drift 1: a crash in any phase never relaunches the operation or loses/doubles cost."""

    def expected_delivered(self, planners, engineers):
        return {key: planners * PLANNER_COSTS[key] + engineers * ENGINEER_COSTS[key] for key in LIMITS}

    def test_planning_interrupt_recovers_without_redispatch(self):
        env = self.env()
        calls = []

        def planner(context, operation_id):
            calls.append(operation_id)
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            env.loop.plan(planner, reservation=dict(PLANNER_RESERVATION), review_at=110, advice_config=ADVICE)
        loop = env.reopen()
        with self.assertRaisesRegex(self.ns.Error, "never redispatched"):
            loop.plan(planner, reservation=dict(PLANNER_RESERVATION), review_at=110, advice_config=ADVICE)
        result = {"operation_id": calls[0], "proposal": env.engineer(), "costs": dict(PLANNER_COSTS),
                  "settled": True, "artifacts": {}}
        loop.recover_planner(result, verify_recovery=lambda pending, value: value["operation_id"] == calls[0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(env.state()["phase"], "planned")
        with self.assertRaisesRegex(self.ns.Error, "planner receipt does not match"):
            loop.recover_planner(result, verify_recovery=lambda pending, value: True)
        self.assertEqual(self.delivered(env), self.expected_delivered(1, 0))

    def test_admitted_interrupt_dispatches_once(self):
        env = self.env()
        env.plan(env.engineer())
        arguments = env.execute_arguments(env.engineer(), update="a")
        with mock.patch.object(env.loop, "_dispatch", side_effect=KeyboardInterrupt()), \
                self.assertRaises(KeyboardInterrupt):
            env.loop.execute(**arguments)
        self.assertEqual(env.state()["phase"], "admitted")
        self.assertEqual(env.launches, {})
        loop = env.reopen()
        with self.assertRaisesRegex(self.ns.Error, "no admitted plan"):
            loop.execute(**arguments)
        runtime = {key: arguments[key] for key in ("collect", "attempts_root", "run_investigation")}
        loop.dispatch_admitted(**runtime)
        with self.assertRaisesRegex(self.ns.Error, "only an unstarted admitted call"):
            loop.dispatch_admitted(**runtime)
        self.assert_once(env)
        self.assertEqual(env.state()["phase"], "idle")
        self.assertEqual(self.delivered(env), self.expected_delivered(1, 1))

    def test_dispatching_interrupt_requires_recovery_not_relaunch(self):
        env = self.env()
        env.plan(env.engineer())
        arguments = env.execute_arguments(env.engineer(), update="a", interrupt_launch=True)
        with self.assertRaises(KeyboardInterrupt):
            env.loop.execute(**arguments)
        self.assertEqual(env.state()["phase"], "dispatching")
        loop = env.reopen()
        runtime = {key: arguments[key] for key in ("collect", "attempts_root", "run_investigation")}
        with self.assertRaisesRegex(self.ns.Error, "only an unstarted admitted call"):
            loop.dispatch_admitted(**runtime)
        loop.recover_engineer(result=env.last_result, collected=env.collected("a"),
                              verify_recovery=lambda pending, result, collected: True)
        with self.assertRaisesRegex(self.ns.Error, "transport was already recorded|collection was already"):
            loop.recover_engineer(result=env.last_result, collected=env.collected("a"),
                                  verify_recovery=lambda pending, result, collected: True)
        self.assert_once(env)
        self.assertEqual(env.state()["phase"], "idle")
        self.assertEqual(self.delivered(env), self.expected_delivered(1, 1))

    def test_collecting_interrupt_recovers_collection_once(self):
        env = self.env()
        env.plan(env.engineer())
        with self.assertRaises(KeyboardInterrupt):
            env.loop.execute(**env.execute_arguments(env.engineer(), update="a", interrupt_collect=True))
        self.assertEqual(env.state()["phase"], "collecting")
        loop = env.reopen()
        loop.recover_engineer(collected=env.collected("a"), verify_recovery=lambda pending, result, collected: True)
        self.assert_once(env)
        self.assertEqual(env.state()["phase"], "idle")
        self.assertEqual(self.delivered(env), self.expected_delivered(1, 1))

    def test_settling_interrupt_replays_without_double_cost(self):
        env = self.env()
        env.plan(env.engineer())
        save = env.loop._save

        def crash(record, state):
            if state["phase"] == "idle":
                raise KeyboardInterrupt()
            return save(record, state)
        with mock.patch.object(env.loop, "_save", crash), self.assertRaises(KeyboardInterrupt):
            env.loop.execute(**env.execute_arguments(env.engineer(), update="a"))
        self.assertEqual(env.state()["phase"], "settling")
        tip = env.store.get("direction")["tip"]
        loop = env.reopen()
        loop.resume_settlement()
        with self.assertRaisesRegex(self.ns.Error, "no replayable settlement"):
            loop.resume_settlement()
        self.assertEqual(env.store.get("direction")["tip"], tip)
        self.assert_once(env)
        (row,) = [item for item in env.state()["opportunities"] if item["engineer_id"] == "engineer"]
        self.assertEqual(row["receipts"][0]["costs"], ENGINEER_COSTS)
        self.assertEqual(self.delivered(env), self.expected_delivered(1, 1))
        self.assertEqual(env.store.get("direction")["resources"], ENGINEER_COSTS)

    def test_review_interrupt_replays_once(self):
        env = self.env()
        env.plan(env.engineer())
        env.loop.execute(**env.execute_arguments(env.engineer()))
        env.plan(lambda context: env.review("reviewed"))
        save = env.loop._save
        state = env.state()

        def crash(record, value):
            if value["phase"] == "idle":
                raise KeyboardInterrupt()
            return save(record, value)
        update = {"note": note("reviewed"), "claim_head": state["claim_head"]}
        with mock.patch.object(env.loop, "_save", crash), self.assertRaises(KeyboardInterrupt):
            env.loop.execute_review(planner_ref=state["pending"]["planner_ref"], research_update=update)
        self.assertEqual(env.state()["phase"], "reviewing")
        loop = env.reopen()
        loop.resume_research_review()
        self.assertEqual(env.state()["phase"], "idle")
        self.assertEqual(env.store.get("direction")["note"]["research_status"], "reported")
        self.assertEqual(sum(item.get("review_kind") == "research_review" for item in env.state()["history"]), 1)
        self.assertEqual(self.delivered(env), self.expected_delivered(2, 1))


class ContinuationTests(DriftTestCase):
    """Drift 6: a continuation needs the current revision, the tip and the latest receipt."""

    def first_turn(self):
        env = self.env()
        env.plan(env.engineer())
        env.loop.execute(**env.execute_arguments(env.engineer(), update="a"))
        return env

    def rejected(self, env, proposal, message, **arguments):
        env.plan(proposal)
        with self.assertRaisesRegex(self.ns.Error, message):
            env.loop.execute(**env.execute_arguments(proposal, **arguments))
        self.assertEqual(sum(env.launches.values()), 1)
        self.assertEqual(env.state()["phase"], "planned")
        env.loop.reject_plan("Synthetic continuation rejected.")

    def test_stale_revision_is_rejected(self):
        env = self.first_turn()
        current = env.store.get("direction")["revision"]
        self.rejected(env, env.engineer(expected_revision=current - 1), "stale selected direction")

    def test_parent_that_is_not_the_tip_is_rejected(self):
        env = self.first_turn()
        self.rejected(env, env.engineer(parent_candidate_id=env.seed), "continuation must preserve current tip")

    def test_missing_prior_receipt_is_rejected(self):
        env = self.first_turn()
        proposal = env.engineer()
        self.rejected(env, proposal, "continue latest registered engineer receipt",
                      request=env.request(proposal, prior=None, use_latest=False))

    def test_current_continuation_is_admitted(self):
        env = self.first_turn()
        env.plan(env.engineer())
        env.loop.execute(**env.execute_arguments(env.engineer(), update="b"))
        self.assert_once(env)
        self.assertEqual(len(env.launches), 2)


class ClockTests(DriftTestCase):
    """Drift 10: every horizon uses the injected clock."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch("time.time", side_effect=AssertionError("loop read time.time"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_deadline_stops_new_planning(self):
        env = self.env(deadline=200)
        env.clock.now = 200
        with self.assertRaises(ValueError):
            env.plan(env.engineer())
        self.assertEqual(env.state()["opportunities"], [])

    def test_expired_review_at_blocks_reservation(self):
        env = self.env()
        env.plan(env.engineer(review_at=env.clock() + 10))
        env.clock.advance(20)
        with self.assertRaisesRegex(ValueError, "campaign-bounded review"):
            env.loop.execute(**env.execute_arguments(env.engineer()))
        self.assertEqual(env.launches, {})
        self.assertEqual(env.state()["phase"], "planned")

    def test_expired_review_at_blocks_resumed_dispatch(self):
        env = self.env()
        env.plan(env.engineer(review_at=env.clock() + 10))
        arguments = env.execute_arguments(env.engineer())
        with mock.patch.object(env.loop, "_dispatch", side_effect=KeyboardInterrupt()), \
                self.assertRaises(KeyboardInterrupt):
            env.loop.execute(**arguments)
        env.clock.advance(20)
        runtime = {key: arguments[key] for key in ("collect", "attempts_root", "run_investigation")}
        with self.assertRaisesRegex(self.ns.Error, "admitted opportunity expired"):
            env.reopen().dispatch_admitted(**runtime)
        self.assertEqual(env.launches, {})
        self.assertEqual(env.state()["phase"], "admitted")

    def test_run_uses_injected_clock_for_review_at(self):
        env = self.env(deadline=500, start=300)
        env.loop.run(env.planner(None), lambda proposal, snapshot: {}, planning_reservation=dict(PLANNER_RESERVATION),
                     planning_seconds=10, advice_config=ADVICE, max_steps=1)
        (row,) = env.state()["opportunities"]
        self.assertEqual((row["opened_at"], row["review_at"]), (300, 310))

    def test_advice_preparation_uses_injected_monotonic(self):
        env = self.env()
        ticks = iter([5.0, 7.5])
        env.loop.monotonic = lambda: next(ticks)
        env.loop.plan(env.planner(None), reservation=dict(PLANNER_RESERVATION), review_at=110, advice_config=ADVICE,
                      advice_inputs=lambda context: {})
        self.assertEqual(env.state()["pending"]["advice"]["preparation_elapsed_seconds"], 2.5)


if __name__ == "__main__":
    unittest.main()
