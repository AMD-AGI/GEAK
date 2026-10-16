"""Tests for geak_evo.budget (program lease and planner follow-up reserve).

Pure CPU arithmetic on synthetic ledgers; no model, GPU or clock.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_budget.py
Parity run on the original modules: set EVO_PARITY_ORIGINAL=1 and put the
original ``evolution/src`` on PYTHONPATH instead. The golden digests were made
from the original program_lease.py and planner_reserve.py; the port must give
the same bytes. Error class names are not part of the trace (the port raises
LeaseError where planner_reserve raised a plain ValueError); messages are.
"""

import copy
import hashlib
import json
import os
import types
import unittest

ORIGINAL = bool(os.environ.get("EVO_PARITY_ORIGINAL"))

if ORIGINAL:
    from geak_evo import planner_reserve, program_lease
    budget = types.SimpleNamespace(**{name: getattr(module, name) for module in (program_lease, planner_reserve)
                                      for name in dir(module) if not name.startswith("__")})
    budget.NS = planner_reserve.NS
    EvoError = None
else:
    from geak_evo import budget
    from geak_evo._util import EvoError

JOB = hashlib.sha256(b"synthetic budget job").hexdigest()
BINDING = {"job_id": JOB, "measurement_epoch_id": "cpu-epoch"}
LIMITS = {"wall_seconds": 1000, "tokens": 10000}
TERMS = {"episodes": 2, "release_condition": "Release when the second review is done",
         "displaced_work": "Ordinary refinement waits"}
PLANNER = {"wall_seconds": 10, "tokens": 100}
ENGINEER = {"wall_seconds": 50, "tokens": 500}

GOLDEN = {
    "lifecycle": "e2895d60db1e3b02ddc0f3b60ae85ac3fdcd15795e188d56a78d72101b6c2b8d",
    "release_and_reject": "eda01b83688943664ff373174786441fa2dac4958e793473150a5654c641bc7a",
    "commitment_draw": "da2c3b456dd8fd1396893d717055d89805346a7c5650c7eb64edd002c2b6bd78",
    "followup": "14e552770521ba4dc78ff725efcb7380b2598e4f94070018effbacb38ca599e7",
}


def row(operation_id, engineer_id, opened, review, reserved, sequence=None, costs=None, direction="direction"):
    """One complete ordinary ledger row; settled when a sequence is given."""
    settled = sequence is not None
    receipts = [] if costs is None else [{"receipt_id": operation_id + ":receipt", "costs": dict(costs)}]
    return dict(BINDING, opportunity_id=operation_id, direction_id=direction, island_id="island",
                engineer_id=engineer_id, question="Synthetic question", opened_at=opened, review_at=review,
                settled=settled, progress_complete=False, completed_sequence=sequence,
                requested=dict(reserved), reserved=dict(reserved), receipts=receipts)


def new_state(deadline=1000, total_limit=10):
    return {"binding": dict(BINDING), "limits": dict(LIMITS), "deadline": deadline, "opportunities": [],
            "pending": {"operation_id": None, "context": {"planner_capacity": {"total_limit": total_limit,
                                                                                "steps_after_current": total_limit}}},
            "program_leases": []}


def capacity(state, total_limit=10):
    used = sum(item["engineer_id"] == "planner" for item in state["opportunities"])
    return {"total_limit": total_limit, "steps_after_current": total_limit - used - 1}


def proposal(direction="direction", review_at=None, **changes):
    result = {"direction_id": direction, "reservation": dict(ENGINEER), "review_at": review_at,
              "protection_ref": None}
    result.update(changes)
    return result


class Trace:
    """Record each call's result or error message, and the state after it."""

    def __init__(self):
        self.steps = []

    def call(self, label, function, *args, **kwargs):
        try:
            value = function(*args, **kwargs)
            self.steps.append({"step": label, "result": copy.deepcopy(value)})
            return value
        except ValueError as exc:
            self.steps.append({"step": label, "error": str(exc)})
            return exc

    def state(self, label, state):
        self.steps.append({"step": label, "state": copy.deepcopy(state)})

    def digest(self):
        data = json.dumps(self.steps, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(data).hexdigest()


def plan_row(trace, state, operation_id, opened, sequence, total_limit=10, reserved=PLANNER, now=None):
    """Prepare and append one settled planner row, as the loop does."""
    planner = row(operation_id, "planner", opened, opened + 10, reserved, sequence, {"wall_seconds": 1, "tokens": 10})
    draw = trace.call("prepare " + operation_id, budget.prepare_planner, state, planner,
                      capacity(state, total_limit), opened if now is None else now)
    state["opportunities"].append(planner)
    state["pending"]["operation_id"] = operation_id
    state["pending"]["context"]["planner_capacity"] = capacity(state, total_limit)
    state["pending"]["context"]["planner_capacity"]["steps_after_current"] += 1
    return draw


def engineer_row(trace, state, operation_id, opened, sequence, offer, bound=2):
    engineer = row(operation_id, "engineer", opened, offer["review_at"], offer["reservation"],
                   direction=offer["direction_id"])
    lease = trace.call("admit " + operation_id, budget.admit_engineer, state, offer, engineer, bound=bound, now=opened)
    if isinstance(lease, Exception):
        return lease
    engineer.update(settled=True, completed_sequence=sequence,
                    receipts=[{"receipt_id": operation_id + ":receipt", "costs": {"wall_seconds": 5, "tokens": 50}}])
    state["opportunities"].append(engineer)
    return lease


def lifecycle():
    """Open a lease, review, continue, review again: the lease completes."""
    trace, state = Trace(), new_state()
    trace.call("no lease draw", budget.prepare_planner, state, row("P0", "planner", 0, 10, PLANNER), capacity(state), 0)
    plan_row(trace, state, "P1", 0, 0)
    trace.call("disabled bound", budget.admit_engineer, copy.deepcopy(state),
               proposal(review_at=70, lease=TERMS), row("E0", "engineer", 10, 70, ENGINEER), bound=0, now=10)
    trace.call("three episodes", budget.admit_engineer, copy.deepcopy(state),
               proposal(review_at=70, lease=dict(TERMS, episodes=3)), row("E0", "engineer", 10, 70, ENGINEER), bound=2, now=10)
    engineer_row(trace, state, "E1", 10, 1, proposal(review_at=70, lease=TERMS))
    trace.state("opened", state)
    trace.call("commitments", budget.commitments, state)
    trace.call("held calendar", budget.held_calendar, budget.active_lease(state))
    plan_row(trace, state, "P2", 70, 2)
    trace.call("metadata on mandatory draw", budget.planner_action, copy.deepcopy(state),
               {"kind": "note", "direction_id": "direction", "lease_release": "stop"}, "P2", attempt_id=None, now=80)
    trace.call("review E1", budget.planner_action, state, {"kind": "research_review", "direction_id": "direction"},
               "P2", attempt_id="E1", now=80)
    plan_row(trace, state, "P3", 80, 3)
    trace.call("reset lease", budget.admit_engineer, copy.deepcopy(state),
               proposal(review_at=150, lease=TERMS), row("E9", "engineer", 90, 150, ENGINEER), bound=2, now=90)
    engineer_row(trace, state, "E2", 90, 4, proposal(review_at=150))
    plan_row(trace, state, "P4", 150, 5)
    trace.call("review E2", budget.planner_action, state, {"kind": "research_review", "direction_id": "direction"},
               "P4", attempt_id="E2", now=160)
    trace.state("completed", state)
    trace.call("active after completion", budget.active_lease, state)
    trace.call("release without lease", budget.planner_action, state, {"kind": "lease_release"}, "P4", now=160)
    return trace


def release_and_reject():
    """Explicit release, change of investigation, failed planner and calendar block."""
    trace, state = Trace(), new_state()
    plan_row(trace, state, "P1", 0, 0)
    engineer_row(trace, state, "E1", 10, 1, proposal(review_at=70, lease=TERMS))
    base = copy.deepcopy(state)
    plan_row(trace, state, "P2", 70, 2)
    trace.call("release before review", budget.planner_action, copy.deepcopy(state),
               {"kind": "lease_release", "direction_id": "direction", "reason": "Stop now"}, "P2", now=80)
    trace.call("review with release", budget.planner_action, state,
               {"kind": "research_review", "direction_id": "direction", "lease_release": "Evidence is enough"},
               "P2", attempt_id="E1", now=80)
    trace.state("released", state)
    state = copy.deepcopy(base)
    plan_row(trace, state, "P2", 70, 2)
    trace.call("review E1", budget.planner_action, state, {"kind": "research_review", "direction_id": "direction"},
               "P2", attempt_id="E1", now=80)
    plan_row(trace, state, "P3", 80, 3)
    engineer_row(trace, state, "E2", 90, 4, proposal(direction="other", review_at=150))
    trace.state("changed investigation", state)
    state = copy.deepcopy(base)
    plan_row(trace, state, "P2", 70, 2)
    trace.call("reject empty reason", budget.reject_planner, copy.deepcopy(state), "P2", " ", 80)
    trace.call("reject feasible", budget.reject_planner, state, "P2", "Planner reply failed", 80)
    trace.state("rejected", state)
    trace.call("remaining fits", budget.check_remaining, state, 85)
    trace.call("remaining late", budget.check_remaining, state, 990)
    trace.state("blocked", state)
    trace.call("prepare while blocked", budget.prepare_planner, state, row("P3", "planner", 990, 995, PLANNER),
               capacity(state), 990)
    trace.call("reject while blocked", budget.reject_planner, state, "P2", "again", 990)
    return trace


def commitment_draw():
    """A tight calendar forces the planner to draw from the held tranche."""
    trace, state = Trace(), new_state(deadline=155)
    plan_row(trace, state, "P1", 0, 0)
    engineer_row(trace, state, "E1", 10, 1, proposal(review_at=70, lease=TERMS))
    trace.state("opened", state)
    trace.call("other envelope", budget.prepare_planner, copy.deepcopy(state),
               row("P2", "planner", 70, 75, PLANNER), capacity(state), 70)
    plan_row(trace, state, "P2", 70, 2)
    trace.state("drawn", state)
    trace.call("reject commitment", budget.reject_planner, copy.deepcopy(state), "P2", "Planner reply failed", 80)
    trace.call("review E1", budget.planner_action, state, {"kind": "research_review", "direction_id": "direction"},
               "P2", attempt_id="E1", now=80)
    trace.state("reviewed", state)
    protected = new_state()
    plan_row(trace, protected, "Q1", 0, 0)
    engineer_row(trace, protected, "F1", 10, 1, proposal(review_at=70, lease=TERMS, protection_ref={"ref": "x"}))
    trace.state("protected", protected)
    plan_row(trace, protected, "Q2", 70, 2)
    trace.call("review before observe", budget.planner_action, copy.deepcopy(protected),
               {"kind": "maturation_review", "direction_id": "direction", "decision": {"action": "review"}},
               "Q2", attempt_id="F1", now=80)
    trace.call("observe", budget.planner_action, protected,
               {"kind": "maturation_review", "direction_id": "direction", "decision": {"action": "observe"}},
               "Q2", attempt_id="F1", now=80)
    trace.state("observed", protected)
    return trace


def followup():
    """planner_followup_reserve and bounded_engineer_reservation, including errors."""
    trace = Trace()
    reservation = {"wall_seconds": 12.5, "evaluations": 0}
    context = {"planner_capacity": {"total_limit": 4, "steps_after_current": 5},
               "state": {"opportunities": [{"engineer_id": "planner", "opportunity_id": "a"},
                                           {"engineer_id": "engineer", "opportunity_id": "b"}]}}
    intent = {"operation_id": "current", "reservation": reservation}
    trace.call("reserve", budget.planner_followup_reserve, context, intent)
    trace.call("reserve no capacity", budget.planner_followup_reserve, dict(context, planner_capacity=None), intent)
    trace.call("reserve current present", budget.planner_followup_reserve, context, dict(intent, operation_id="a"))
    trace.call("reserve over ceiling", budget.planner_followup_reserve,
               dict(context, planner_capacity={"total_limit": 1, "steps_after_current": 0}), intent)
    trace.call("reserve evaluations", budget.planner_followup_reserve, context,
               dict(intent, reservation={"wall_seconds": 10, "evaluations": 1}))
    trace.call("reserve infinite", budget.planner_followup_reserve, context,
               dict(intent, reservation={"wall_seconds": float("inf"), "evaluations": 0}))
    limit = {"wall_seconds": 600, "evaluations": 8}
    follow = {"reservation": reservation, "decisions_after_current": 2}
    for label, available, campaign_end, planner_end, envelope, extra in (
            ("fits", {"wall_seconds": 900, "evaluations": 20}, 2000 * budget.NS, 100 * budget.NS, 120, follow),
            ("calendar bound", {"wall_seconds": 900, "evaluations": 20}, 400 * budget.NS, 100 * budget.NS, 120, follow),
            ("strict boundary", {"wall_seconds": 900, "evaluations": 3.7}, 300 * budget.NS + 12_500_000_001,
             100 * budget.NS, 150, follow),
            ("envelope too small", {"wall_seconds": 900, "evaluations": 20}, 2000 * budget.NS, 100 * budget.NS, 571, follow),
            ("no evaluations", {"wall_seconds": 900, "evaluations": 0.5}, 2000 * budget.NS, 100 * budget.NS, 120, follow),
            ("no later decision", {"wall_seconds": 900, "evaluations": 20}, 2000 * budget.NS, 100 * budget.NS, 120,
             dict(follow, decisions_after_current=0)),
            ("negative available", {"wall_seconds": -1, "evaluations": 20}, 2000 * budget.NS, 100 * budget.NS, 120, follow),
            ("float campaign end", {"wall_seconds": 900, "evaluations": 20}, 2000.0, 100 * budget.NS, 120, follow),
            ("bad followup", {"wall_seconds": 900, "evaluations": 20}, 2000 * budget.NS, 100 * budget.NS, 120,
             dict(follow, reservation={"wall_seconds": 10, "evaluations": 1}))):
        trace.call(label, budget.bounded_engineer_reservation, available, campaign_end, planner_end, envelope,
                   engineer_limit=limit, followup=extra)
    return trace


SCENARIOS = {"lifecycle": lifecycle, "release_and_reject": release_and_reject,
             "commitment_draw": commitment_draw, "followup": followup}


class GoldenTraceTests(unittest.TestCase):
    def test_traces_match_the_original_bytes(self):
        for name, scenario in SCENARIOS.items():
            with self.subTest(name):
                self.assertEqual(scenario().digest(), GOLDEN[name])

    def test_traces_are_deterministic(self):
        for name, scenario in SCENARIOS.items():
            with self.subTest(name):
                self.assertEqual(scenario().digest(), scenario().digest())


class LeaseLifecycleTests(unittest.TestCase):
    def test_two_episodes_and_reviews_complete_the_lease(self):
        trace = lifecycle()
        steps = {step["step"]: step for step in trace.steps}
        self.assertIsNone(steps["no lease draw"]["result"])
        self.assertEqual(steps["disabled bound"]["error"], "program commitment is disabled")
        self.assertIn("exactly two episodes", steps["three episodes"]["error"])
        opened = steps["admit E1"]["result"]
        self.assertEqual((opened["status"], opened["planner_tranches"], opened["engineers_remaining"]), ("active", 3, 1))
        self.assertEqual(opened["remaining_reservation"], {"wall_seconds": 80.0, "tokens": 800.0})
        self.assertEqual(steps["commitments"]["result"],
                         [{"commitment_id": "lease:P1", "reserved": {"wall_seconds": 80.0, "tokens": 800.0}}])
        self.assertEqual(steps["held calendar"]["result"], 80.0)
        self.assertEqual(steps["metadata on mandatory draw"]["error"], "release must name a committed research review")
        self.assertIn("cannot reset its lease", steps["reset lease"]["error"])
        final = steps["review E2"]["result"]
        self.assertEqual((final["status"], final["planner_tranches"]), ("completed", 0))
        self.assertEqual(final["attempt_ids"], final["reviewed_attempt_ids"])
        self.assertIsNone(steps["active after completion"]["result"])
        self.assertEqual(steps["release without lease"]["error"], "no program lease to release")

    def test_release_reject_and_block_keep_holds(self):
        steps = {step["step"]: step for step in release_and_reject().steps}
        self.assertEqual(steps["release before review"]["error"], "research review is owed before release")
        self.assertEqual(steps["review with release"]["result"]["status"], "released")
        self.assertIsNone(steps["admit E2"]["result"])
        self.assertEqual(steps["changed investigation"]["state"]["program_leases"][0]["status"], "released")
        self.assertEqual(steps["reject empty reason"]["error"], "rejection reason must be nonempty text")
        self.assertIs(steps["reject feasible"]["result"], False)
        self.assertIs(steps["remaining fits"]["result"], False)
        self.assertIs(steps["remaining late"]["result"], True)
        lease = steps["blocked"]["state"]["program_leases"][0]
        self.assertEqual(lease["status"], "blocked")
        self.assertTrue(any(lease["remaining_reservation"].values()))
        self.assertEqual(steps["prepare while blocked"]["error"], "program lease is blocked")
        self.assertIs(steps["reject while blocked"]["result"], True)

    def test_commitment_draw_and_protected_order(self):
        steps = {step["step"]: step for step in commitment_draw().steps}
        self.assertEqual(steps["prepare P2"]["result"]["funding"], "commitment")
        self.assertEqual(steps["other envelope"]["error"], "commitment draw requires the original planner envelope")
        self.assertEqual(steps["opened"]["state"]["program_leases"][0]["planner_tranches"], 3)
        self.assertEqual(steps["drawn"]["state"]["program_leases"][0]["planner_tranches"], 2)
        self.assertIs(steps["reject commitment"]["result"], True)
        self.assertEqual(steps["reviewed"]["state"]["program_leases"][0]["planner_tranches"], 2)
        protected = steps["protected"]["state"]["program_leases"][0]
        self.assertEqual(protected["planner_tranches"], 5)
        self.assertEqual(len(protected["protection_obligations"]), 2)
        self.assertEqual(steps["review before observe"]["error"], "protected review must follow its observation")
        self.assertEqual(steps["observed"]["state"]["program_leases"][0]["protection_obligations"],
                         [{"attempt_id": "F1", "action": "review"}])


class FollowupReserveTests(unittest.TestCase):
    def test_followup_reserve_and_engineer_fit(self):
        steps = {step["step"]: step for step in followup().steps}
        self.assertEqual(steps["reserve"]["result"],
                         {"reservation": {"wall_seconds": 12.5, "evaluations": 0}, "decisions_after_current": 2})
        self.assertEqual(steps["reserve no capacity"]["error"], "frozen ordinary planner capacity required")
        self.assertEqual(steps["reserve over ceiling"]["error"], "current planner exceeds its original decision ceiling")
        self.assertEqual(steps["fits"]["result"], ({"wall_seconds": 600, "evaluations": 8}, True))
        self.assertEqual(steps["calendar bound"]["result"], ({"wall_seconds": 287, "evaluations": 8}, True))
        self.assertEqual(steps["strict boundary"]["result"], ({"wall_seconds": 200, "evaluations": 3}, True))
        self.assertIs(steps["envelope too small"]["result"][1], False)
        self.assertEqual(steps["no evaluations"]["result"], ({"wall_seconds": 600, "evaluations": 0}, False))
        self.assertEqual(steps["no later decision"]["result"], ({"wall_seconds": 0, "evaluations": 0}, False))
        self.assertEqual(steps["negative available"]["error"], "finite remaining allowance and native envelope required")
        self.assertEqual(steps["float campaign end"]["error"], "finite remaining allowance and native envelope required")
        self.assertEqual(steps["bad followup"]["error"], "configured followup planner reservation required")


@unittest.skipIf(ORIGINAL, "port-only error classes")
class ErrorClassTests(unittest.TestCase):
    def test_lease_error_is_evo_error_and_value_error(self):
        self.assertTrue(issubclass(budget.LeaseError, EvoError))
        self.assertTrue(issubclass(budget.LeaseError, ValueError))
        # The planner-reserve checks raised a plain ValueError before the port.
        with self.assertRaises(ValueError) as caught:
            budget.planner_followup_reserve({"planner_capacity": None}, {"operation_id": "x"})
        self.assertIs(type(caught.exception), ValueError)
        with self.assertRaises(budget.LeaseError):
            budget.admit_engineer(new_state(), proposal(lease=TERMS), {}, bound=0, now=0)


if __name__ == "__main__":
    unittest.main()
