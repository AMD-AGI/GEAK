"""Golden test of selection.advice (drift test 8: allocation advice inputs).

GOLDEN holds the SHA-256 of the canonical JSON output (or error text) of the
ORIGINAL pre-port functions (oct15 bridge/native_advice.py and
native_advice_inputs.py) on the synthetic inputs below. The port must give
byte-identical outputs. Errors are compared by message only: the original
raised plain ValueError, the port raises ValueError subclasses.

The campaign clock file lives in a scoped temporary directory; its reference
is replaced by "<clock_ref>" before hashing. The original reads the running
kernel's boot ID, so ``api["boot_id"]`` is that ID for the original and a fixed
test ID (injected) for the port.

Run from the repo root:  python3 -B tests/geak_evo/test_advice_golden.py
"""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

JOB = {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1"}
LIMITS = {"wall_seconds": 3600, "evaluations": 40}
ENGINEER = {"wall_seconds": 900, "evaluations": 8}
START_NS = 5_000_000_000_000
TEST_BOOT = "test-boot-id"


def _clock_ref(directory, boot_id):
    data = json.dumps({"boot_id": boot_id, "started_monotonic_ns": START_NS, "started_unix": 900.0,
                       "wall_seconds": 3600}, sort_keys=True).encode()
    path = os.path.join(directory, "clock.json")
    with open(path, "wb") as handle:
        handle.write(data)
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}


def _snapshot(steps_after=5, committed=(300, 5), upcoming=None, binding=None):
    upcoming = upcoming or {}
    planned = dict(JOB, engineer_id="planner", direction_id="campaign", opportunity_id="planner-3",
                   opened_at=1000.0, review_at=1100.0, reserved={"wall_seconds": 120, "evaluations": 0})
    planned.update(upcoming)
    state = {"binding": dict(binding or JOB), "deadline": 4500.0,
             "opportunities": [{"opportunity_id": "planner-1", "engineer_id": "planner"},
                               {"opportunity_id": "eng-1", "engineer_id": "engineer-1"}],
             "latest_receipts": {}, "history": []}
    rows = [{"committed": {"wall_seconds": committed[0], "evaluations": committed[1]}},
            {"committed": {"wall_seconds": 100, "evaluations": 0}}]
    return {"campaign_id": "campaign", "revision": 7, "planned_advice_offer": planned, "state": state,
            "accounting": {"limits": dict(LIMITS), "opportunities": rows},
            "planner_capacity": {"total_limit": 10, "steps_after_current": steps_after},
            "archive": {"directions": {}}}


def _offer(api, ctx, snapshot, envelope=600, boot=None):
    ref = _clock_ref(ctx, api["boot_id"] if boot is None else boot)
    return api["offer"](snapshot, ref, envelope, copy.deepcopy(ENGINEER))


class _Store:
    def __init__(self, revisions=None, candidates=None):
        self.revisions, self.candidates = revisions or {}, candidates or {}

    def get_revision(self, name, revision):
        return copy.deepcopy(self.revisions[(name, revision)])

    def get_candidate(self, candidate_id):
        return copy.deepcopy(self.candidates[candidate_id])


class _Development:
    def __init__(self, value):
        self.value = value

    def inputs(self, snapshot):
        return copy.deepcopy(self.value)


def _inputs(api, ctx, snapshot, development=None, committed_state=None):
    ref = _clock_ref(ctx, api["boot_id"])
    store = _Store({("campaign", 7): {"note": {"evolution": committed_state or snapshot["state"]}}})
    loop = type("Loop", (), {"store": store})()
    adapter = api["inputs"](loop, _Development(development or {"progress": []}), clock_ref=ref,
                            native_envelope_seconds=600, engineer_limit=copy.deepcopy(ENGINEER))
    return adapter(snapshot)


def _advice_row(direction="dir-a", revision=3, tip="cand-a", source="src-a", reservation=None, clock="CLOCK"):
    return {"direction_id": direction,
            "offer_binding": {"snapshot_revision": 7, "direction_revision": revision, "parent_candidate_id": tip,
                              "source_id": source, "membership_ref": None, "clock_ref": clock,
                              "reservation": reservation or {"wall_seconds": 600, "evaluations": 4}}}


def _verify(api, rows, *, held=None, gain_kind=None, available=True, advisory=True):
    advice = {"gain_kind": gain_kind or "native_local_frontier_observed_log_v1", "advisory_only": advisory,
              "rows": rows}
    actual = {"clock_ref": "CLOCK", "reservation": {"wall_seconds": 600, "evaluations": 4},
              "available": available}
    if held is not None:
        actual["program_lease_offer"] = held
    store = _Store({("dir-a", 3): {"tip": "cand-a"}, ("dir-b", 2): {"tip": "cand-b"}},
                   {"cand-a": {"source_id": "src-a"}, "cand-b": {"source_id": "src-b"}})
    return api["verify"](advice, {"revision": 7}, actual, store)


HELD = {"direction_id": "dir-b", "reservation": {"wall_seconds": 800, "evaluations": 6}, "available": True}

BASE = {"selected_source_id": "s", "selected_quality": {"value_ms": 1.0}, "baseline": None,
        "signed_frontier_log_ratio": None, "nonnegative_frontier_log_ratio": None, "credit": None,
        "unknown_reasons": []}


def _proof(kind, **changes):
    proof = dict(BASE, outcome_kind=kind)
    proof.update(changes)
    return proof


CASES = {
    "config_ok": lambda api, ctx: api["config"](LIMITS, ENGINEER),
    "config_float": lambda api, ctx: api["config"]({"wall_seconds": 1800.5, "evaluations": 33}, ENGINEER),
    "config_missing_key": lambda api, ctx: api["config"]({"wall_seconds": 10}, ENGINEER),
    "config_nonpositive": lambda api, ctx: api["config"]({"wall_seconds": 0, "evaluations": 3}, ENGINEER),
    "outcome_legacy": lambda api, ctx: api["outcome"]({}, {}),
    "outcome_legacy_conflict": lambda api, ctx: api["outcome"]({}, {"outcome_kind": "unknown"}),
    "outcome_baseline": lambda api, ctx: api["outcome"](
        {"outcome_kind": "baseline_initialization", "complete": True}, _proof("baseline_initialization")),
    "outcome_baseline_with_ratio": lambda api, ctx: api["outcome"](
        {"outcome_kind": "baseline_initialization", "complete": True},
        _proof("baseline_initialization", signed_frontier_log_ratio=0.1)),
    "outcome_observed": lambda api, ctx: api["outcome"](
        {"outcome_kind": "observed_comparison", "complete": True},
        _proof("observed_comparison", baseline={"value_ms": 2.0}, signed_frontier_log_ratio=0.5,
               nonnegative_frontier_log_ratio=0.5, credit=0.5)),
    "outcome_observed_incomplete": lambda api, ctx: api["outcome"](
        {"outcome_kind": "observed_comparison", "complete": True}, _proof("observed_comparison")),
    "outcome_unknown": lambda api, ctx: api["outcome"](
        {"outcome_kind": "unknown", "complete": False}, _proof("unknown", unknown_reasons=["timing missing"])),
    "outcome_unknown_with_credit": lambda api, ctx: api["outcome"](
        {"outcome_kind": "unknown", "complete": False},
        _proof("unknown", unknown_reasons=["timing missing"], credit=0.2)),
    "outcome_kind_mismatch": lambda api, ctx: api["outcome"](
        {"outcome_kind": "unknown", "complete": True}, _proof("unknown", unknown_reasons=["x"])),
    "offer_ok": lambda api, ctx: _offer(api, ctx, _snapshot()),
    "offer_no_later_decision": lambda api, ctx: _offer(api, ctx, _snapshot(steps_after=0)),
    "offer_short_envelope": lambda api, ctx: _offer(api, ctx, _snapshot(), envelope=1000),
    "offer_tight_allowance": lambda api, ctx: _offer(api, ctx, _snapshot(committed=(3200, 39))),
    "offer_over_allowance": lambda api, ctx: _offer(api, ctx, _snapshot(committed=(3500, 5))),
    "offer_wrong_boot": lambda api, ctx: _offer(api, ctx, _snapshot(), boot="another-boot"),
    "offer_not_planner": lambda api, ctx: _offer(api, ctx, _snapshot(upcoming={"engineer_id": "e"})),
    "offer_reused_id": lambda api, ctx: _offer(api, ctx, _snapshot(upcoming={"opportunity_id": "eng-1"})),
    "offer_long_window": lambda api, ctx: _offer(api, ctx, _snapshot(upcoming={"review_at": 1500.0})),
    "offer_other_binding": lambda api, ctx: _offer(
        api, ctx, _snapshot(binding=dict(JOB, measurement_epoch_id="epoch-2"))),
    "offer_bad_planner_reservation": lambda api, ctx: _offer(
        api, ctx, _snapshot(upcoming={"reserved": {"wall_seconds": 120, "evaluations": 1}})),
    "inputs_no_directions": lambda api, ctx: _inputs(api, ctx, _snapshot(), {"progress": [1]}),
    "inputs_unavailable": lambda api, ctx: _inputs(api, ctx, _snapshot(steps_after=0)),
    "inputs_stale_snapshot": lambda api, ctx: _inputs(
        api, ctx, _snapshot(), committed_state={"other": True}),
    "inputs_bad_development": lambda api, ctx: _inputs(api, ctx, _snapshot(), {"progress": [], "extra": 1}),
    "verify_ok": lambda api, ctx: _verify(api, [_advice_row()]),
    "verify_held": lambda api, ctx: _verify(api, [_advice_row(), _advice_row(
        "dir-b", 2, "cand-b", "src-b", reservation={"wall_seconds": 800, "evaluations": 6})], held=HELD),
    "verify_held_mismatch": lambda api, ctx: _verify(api, [_advice_row("dir-b", 2, "cand-b", "src-b")], held=HELD),
    "verify_wrong_kind": lambda api, ctx: _verify(api, [], gain_kind="other"),
    "verify_not_advisory": lambda api, ctx: _verify(api, [], advisory=None),
    "verify_unavailable": lambda api, ctx: _verify(api, [_advice_row()], available=False),
    "verify_source_changed": lambda api, ctx: _verify(api, [_advice_row(source="src-z")]),
}

GOLDEN = {
    "config_float": "55254014add400cade39cd31d301f4b0cba24a136f81ebbedce559fb73d57bda",
    "config_missing_key": "e7d890c9a8d1f2f89cd8efa26c5a7a032a75c6a07d493bfb5877428e0139a192",
    "config_nonpositive": "e7d890c9a8d1f2f89cd8efa26c5a7a032a75c6a07d493bfb5877428e0139a192",
    "config_ok": "ae1250500125404419253d0c769b56f8a2d531f4a5a6a3ea9f0555440f2af000",
    "inputs_bad_development": "1e8a42b57485c19254ebd6f8333cc8e465eca51f94c8fc2559cf4b68e00741d0",
    "inputs_no_directions": "e31beab2dae61cb297b962fd9792df291bf9b0056d9acde4f511568d23cc4a3e",
    "inputs_stale_snapshot": "8d2c7ee4cc0a4fe1607990f6fa60f4c1b990a3c6f3e8b5a29bda70ac1c44cb5f",
    "inputs_unavailable": "65adf52b25a61d41794dd931f9b9166b0bea39965d6ca1e7c05cbe3449905a3b",
    "offer_bad_planner_reservation": "12a2b7fe1dc43da9178255b49192d1baa8cf3aaaf19c1c6ae141059a5cc6fc7e",
    "offer_long_window": "fc4280be0d742ac0646eb5431c0fe61524cbf4c99c61d603fe02529d08a0d882",
    "offer_no_later_decision": "01d39ebfae7572dae4f86c47cb7bb411ca4c4ae7418f31520d57ad38243ae87a",
    "offer_not_planner": "4bd56991cfcd3b2fd90433a0905eb7393705641e9190d6fc634ad0f93739f6a6",
    "offer_ok": "ac494ceb50f7c127ece2306fd0dfbdcb1af54dd7af5edf7743a611b16b56d235",
    "offer_other_binding": "5c30c8a68bdc319c1ad8802d0ee3c8553886f644bf344b9150f956d4b051e060",
    "offer_over_allowance": "2f2e8a3db6973ab15e2713cb94ca7eb8541068194a28c5584681fdc18b69358e",
    "offer_reused_id": "4bd56991cfcd3b2fd90433a0905eb7393705641e9190d6fc634ad0f93739f6a6",
    "offer_short_envelope": "a4d189a9cccc4e16a6a99ab90533cd8d9d9ae99a4d303dee7560ebf68ca8722d",
    "offer_tight_allowance": "ca3dce66117a315101143e477bb1fe7bf96454e5114e46d9d8d4e509772598a8",
    "offer_wrong_boot": "236ce71168fecec20a4b09d544b7c0d32ba4f4dd95786b2573ae1e58d65869d1",
    "outcome_baseline": "7d4249d86fe17da74bb863bde8318fafd9187f90637d09dacf3123be63cf1df1",
    "outcome_baseline_with_ratio": "0e4959ff06474f9cb13a04efc535df3a00085de128b639dc879dabdbc696cff2",
    "outcome_kind_mismatch": "768b922ed7c8a9e21cf147953be864a1c7f8274e0039a4bfeee730efc31993a5",
    "outcome_legacy": "78fa2886f5e9367182124d9465d8ad8827856f6bce3db3b362e0e90257cf6733",
    "outcome_legacy_conflict": "0aefa340924ec400a9c98ea762a1ff6e0a1eaa059e8eebc622dee667b9644125",
    "outcome_observed": "5d42c360c5470f42e412c287ddb5b5dc6aceecdd3588ca01a0f7a49499695aa6",
    "outcome_observed_incomplete": "ee64347b383df5d471a0f54a5793315911933091a9ca1fcd49b7e1879670377e",
    "outcome_unknown": "aacdb2772008a774f8d8ba3d594a3d733046fb4c59161895f450b66023ee0c5f",
    "outcome_unknown_with_credit": "ca315e645cb02db6e614cc8351f69debdb88f6d1f5da65f816b9c1922ef3dabc",
    "verify_held": "02bc14e338fefa3bfb96c6c3a6d38be2a421d97b2219039fcb2df730ad53a331",
    "verify_held_mismatch": "48b5ad77afb141a9d0bfe89926626171d1af5429487a2547c05d37309817e9c4",
    "verify_not_advisory": "71cabccfdd469b1b79d0520b02f4ecfcb328edfc6bb5de91912bc3a141d99b90",
    "verify_ok": "b809000e18333b279fecfc593cb2b20fda0c3b205374ed3d68a0b392b57aead8",
    "verify_source_changed": "48b5ad77afb141a9d0bfe89926626171d1af5429487a2547c05d37309817e9c4",
    "verify_unavailable": "48b5ad77afb141a9d0bfe89926626171d1af5429487a2547c05d37309817e9c4",
    "verify_wrong_kind": "71cabccfdd469b1b79d0520b02f4ecfcb328edfc6bb5de91912bc3a141d99b90",
}


def outputs(api):
    """Map case name -> sha256 of canonical JSON of its output or error message."""
    result = {}
    for name, case in CASES.items():
        with tempfile.TemporaryDirectory() as directory:
            ctx = os.path.realpath(directory)
            try:
                value = {"ok": case(api, ctx)}
            except ValueError as error:
                value = {"error": True, "message": str(error)}
            text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            clock = os.path.join(ctx, "clock.json")
            if os.path.exists(clock):
                with open(clock, "rb") as handle:
                    ref = {"path": clock, "sha256": hashlib.sha256(handle.read()).hexdigest()}
                text = text.replace(json.dumps(ref, sort_keys=True, separators=(",", ":")), '"<clock_ref>"')
        result[name] = hashlib.sha256(text.encode()).hexdigest()
    return result


def port_api(boot_id=TEST_BOOT):
    from geak_evo.selection import advice

    def inputs(loop, development, **kwargs):
        return advice.NativeAdviceInputs(loop, development, boot_id=boot_id, **kwargs)

    return {"config": advice.native_advice_config, "outcome": advice._outcome,
            "offer": lambda *args: advice.engineer_offer_binding(*args, boot_id=boot_id),
            "inputs": inputs, "verify": advice.verify_delivered_advice, "boot_id": boot_id}


class AdviceGoldenTest(unittest.TestCase):
    def test_outputs_match_original(self):
        actual = outputs(port_api())
        self.assertEqual(set(actual), set(GOLDEN))
        for name in sorted(GOLDEN):
            with self.subTest(case=name):
                self.assertEqual(actual[name], GOLDEN[name])

    def test_default_boot_id_is_running_kernel(self):
        from geak_evo.selection import advice
        boot = advice._current_boot_id()
        injected = outputs(port_api(boot))
        api = port_api(boot)
        api["offer"] = advice.engineer_offer_binding
        api["inputs"] = advice.NativeAdviceInputs
        self.assertEqual(outputs(api), injected)

    def test_error_classes(self):
        # The original raised bare ValueError; callers persist the class name.
        from geak_evo.selection import advice
        self.assertFalse(hasattr(advice, "AdviceError"))
        with self.assertRaises(ValueError) as caught:
            advice._need(False, "x")
        self.assertIs(type(caught.exception), ValueError)


if __name__ == "__main__":
    unittest.main()
