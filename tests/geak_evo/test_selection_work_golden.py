"""Golden test of selection.work (drift test 8: advise/account/record_decision).

GOLDEN holds the SHA-256 of the canonical JSON output (or error text) of the
ORIGINAL pre-port functions (oct15 geak_evo/selection.py) on the synthetic
inputs below. The port must give byte-identical outputs.

To regenerate against another implementation, call ``outputs(api)`` with a
dict of its functions (see ``port_api``).
"""

import hashlib
import json
import math
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

JOB = {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1"}
LIMITS = {"engineer_active_seconds": 1000, "gpu_device_seconds": 1000, "model_usd": 100}
STRICT = {"weights": {"engineer_active_seconds": 0.5, "gpu_device_seconds": 0.25, "model_usd": 0.25},
          "c_ref": 0.05, "half_life": 0.25, "epsilon": 0.1, "beta": math.log(1.02)}
NATIVE_LIMITS = {"wall_seconds": 3600, "evaluations": 40}
OBSERVED = {"weights": {"wall_seconds": 0.5, "evaluations": 0.5}, "c_ref": 0.1, "half_life": 0.4,
            "epsilon": 0.01, "beta": 0.1, "gain_kind": "native_local_frontier_observed_log_v1"}


def _vector(limits, fraction):
    return {k: fraction * v for k, v in limits.items()}


def _opportunity(key, direction, limits, *, fraction=0.1, cost=0.05, sequence=None, complete=True,
                 island="island-1", engineer="engineer-1", opened=0, review=100, receipts=None, **extra):
    settled = sequence is not None
    row = dict(JOB, opportunity_id=key, direction_id=direction, island_id=island, engineer_id=engineer,
               question="question " + key, opened_at=opened, review_at=review,
               requested=_vector(limits, fraction), reserved=_vector(limits, fraction),
               receipts=([{"receipt_id": "r-" + key, "costs": _vector(limits, cost)}]
                         if receipts is None else receipts),
               settled=settled, completed_sequence=sequence, progress_complete=settled and complete)
    row.update(extra)
    return row


def _strict_records():
    return [
        _opportunity("a1", "a", LIMITS, sequence=0, cost=0.03),
        _opportunity("b1", "b", LIMITS, sequence=1, cost=0.07, island="island-2"),
        _opportunity("a2", "a", LIMITS, sequence=2, cost=0.02, opened=5),
        _opportunity("c1", "c", LIMITS, fraction=0.2, cost=0.04, opened=10),  # unsettled
        _opportunity("b2", "b", LIMITS, sequence=3, cost=0.01, complete=False),
    ]


def _strict_credit(key, source, gain):
    return dict(JOB, opportunity_id=key, credit_id="credit-" + key, source_id=source * 64,
                frontier_before_source_id="0" * 64, gain_log_lcb=gain,
                evidence_ref={"evidence_id": source * 64, "artifact": "strict/m.json"})


def _offer(direction, limits, weight=1.0, fraction=0.05, **extra):
    row = dict(JOB, direction_id=direction, island_id="island-1", engineer_id="engineer-1",
               priority_weight=weight, estimated_costs=_vector(limits, fraction))
    row.update(extra)
    return row


def _accept(ref, expected):
    return True


GAIN = "native_local_frontier_observed_log_v1"


def _observed_records():
    o = dict(progress_gain_kind=GAIN)
    return [
        _opportunity("i1", "d1", NATIVE_LIMITS, sequence=0, cost=0.02, progress_outcome_kind="baseline_initialization", **o),
        _opportunity("x1", "d1", NATIVE_LIMITS, sequence=1, cost=0.05, progress_outcome_kind="observed_comparison", **o),
        _opportunity("i2", "d2", NATIVE_LIMITS, sequence=2, cost=0.03, progress_outcome_kind="baseline_initialization", **o),
        _opportunity("u1", "d3", NATIVE_LIMITS, sequence=3, cost=0.04, complete=False, progress_outcome_kind="unknown", **o),
        _opportunity("p1", "d4", NATIVE_LIMITS, fraction=0.15, cost=0.01),  # still running
        _opportunity("x2", "d2", NATIVE_LIMITS, sequence=4, cost=0.06, progress_outcome_kind="observed_comparison", **o),
    ]


def _binding(source, revision=2):
    return {"snapshot_revision": 7, "direction_revision": revision, "parent_candidate_id": "c" * 64,
            "source_id": source * 64, "membership_ref": None,
            "clock_ref": {"path": "/campaign/clock.json", "sha256": "d" * 64},
            "reservation": _vector(NATIVE_LIMITS, 0.05)}


def _observed_offers():
    return [_offer(d, NATIVE_LIMITS, 1, offer_binding=_binding(s)) for d, s in
            (("d1", "a"), ("d2", "b"), ("d3", "e"), ("d4", "f"), ("d5", "9"))]


def _observed_credits():
    return [dict(JOB, opportunity_id="x1", credit_id="credit-x1", source_id="a" * 64, baseline_source_id="8" * 64,
                 gain_log_observed=0.05, gain_kind=GAIN,
                 evidence_ref={"evidence_id": "a" * 64, "artifact": "native_development_gain_proof.json"}),
            dict(JOB, opportunity_id="x2", credit_id="credit-x2", source_id="b" * 64, baseline_source_id="a" * 64,
                 gain_log_observed=0.01,
                 evidence_ref={"evidence_id": "b" * 64, "artifact": "native_development_gain_proof.json"})]


def _account(api, records, limits, now=20, commitments=()):
    return api["account_opportunities"](records, binding=JOB, limits=limits, deadline=1000, now=now,
                                        commitments=commitments)


def _strict_advice(api, **change):
    config = dict(STRICT, **change)
    return api["advise"]([_offer("a", LIMITS, 2.0), _offer("b", LIMITS, 1.0), _offer("c", LIMITS, 0.5),
                          _offer("new", LIMITS, 1.0)],
                         _account(api, _strict_records(), LIMITS),
                         [_strict_credit("a1", "2", 0.03), _strict_credit("a2", "3", 0.01),
                          _strict_credit("b1", "4", 0.02)],
                         config=config, verify_progress=_accept)


def _observed_advice(api):
    return api["advise"](_observed_offers(), _account(api, _observed_records(), NATIVE_LIMITS),
                         _observed_credits(), config=OBSERVED, verify_progress=_accept)


def _choice(direction, source, donors=(), island="island-1"):
    return {"direction_id": direction, "island_id": island, "engineer_id": "engineer-9",
            "working_source_id": source * 64, "donor_source_ids": [d * 64 for d in donors]}


CASES = {
    "account_strict": lambda api: _account(api, _strict_records(), LIMITS),
    "account_expired_commitments": lambda api: _account(
        api, _strict_records(), LIMITS, now=150,
        commitments=[{"commitment_id": "hold-1", "reserved": _vector(LIMITS, 0.1)},
                     {"commitment_id": "hold-2", "reserved": _vector(LIMITS, 0.05)}]),
    "account_unknown_receipt": lambda api: _account(api, [_opportunity(
        "z", "z", LIMITS, sequence=0, receipts=[{"receipt_id": "z1", "costs": {"model_usd": 2}},
                                                {"receipt_id": "z2", "costs": _vector(LIMITS, 0.01)}])], LIMITS),
    "account_observed": lambda api: _account(api, _observed_records(), NATIVE_LIMITS),
    "account_error_overrun": lambda api: _account(api, [_opportunity("o", "o", LIMITS, sequence=0, cost=0.5)], LIMITS),
    "advise_strict": lambda api: _strict_advice(api),
    "advise_strict_zero_beta": lambda api: _strict_advice(api, beta=0),
    "advise_strict_long_half_life": lambda api: _strict_advice(api, half_life=10.0, epsilon=1.0),
    "advise_strict_unknown_cost": lambda api: api["advise"](
        [_offer("a", LIMITS), _offer("z", LIMITS, estimated_costs={})],
        _account(api, _strict_records() + [_opportunity("z1", "z", LIMITS, sequence=9, receipts=[])], LIMITS),
        [], config=STRICT),
    "advise_observed": _observed_advice,
    "advise_observed_initialization_only": lambda api: api["advise"](
        _observed_offers()[1:2], _account(api, _observed_records()[2:3], NATIVE_LIMITS), [],
        config=OBSERVED, verify_progress=_accept),
    "advise_error_nonuniform_prior": lambda api: api["advise"](
        [_offer("d1", NATIVE_LIMITS, 1, offer_binding=_binding("a")),
         _offer("d2", NATIVE_LIMITS, 2, offer_binding=_binding("b"))],
        _account(api, [], NATIVE_LIMITS), [], config=OBSERVED),
    "advise_error_initialization_credit": lambda api: api["advise"](
        _observed_offers(), _account(api, _observed_records(), NATIVE_LIMITS),
        [dict(_observed_credits()[0], opportunity_id="i1")], config=OBSERVED, verify_progress=_accept),
    "advise_error_observed_in_strict": lambda api: api["advise"](
        [_offer("d1", NATIVE_LIMITS)], _account(api, _observed_records(), NATIVE_LIMITS), [],
        config=dict(STRICT, weights={"wall_seconds": 0.5, "evaluations": 0.5})),
    "decision_strict_override": lambda api: api["record_decision"](
        _strict_advice(api), dict(_choice("c", "7"), engineer_id="engineer-1"), reason="test the slow idea",
        override_reason="prerequisite", evidence_refs=[{"evidence_id": "5" * 64, "artifact": "a/b.json"}]),
    "decision_strict_top": lambda api: api["record_decision"](
        _strict_advice(api, beta=0), dict(_choice("a", "7"), engineer_id="engineer-1"), reason="best rate"),
    "decision_observed_match": lambda api: api["record_decision"](
        _observed_advice(api), _choice("d1", "a"), reason="continue", override_reason="noted"),
    "decision_observed_donors": lambda api: api["record_decision"](
        _observed_advice(api), _choice("d1", "a", donors=("b",)), reason="combine"),
    "decision_observed_override": lambda api: api["record_decision"](
        _observed_advice(api), _choice("d2", "b"), reason="second", override_reason="needs a repair first"),
    "decision_observed_other_source": lambda api: api["record_decision"](
        _observed_advice(api), _choice("d2", "a"), reason="older source"),
    "decision_error_missing_override": lambda api: api["record_decision"](
        _strict_advice(api), dict(_choice("c", "7"), engineer_id="engineer-1"), reason="slow idea"),
}

GOLDEN = {
    "account_error_overrun": "711c12c0f94855ca4aa58f72119d4a92608409102631e0a9fc7f1d631c289cf6",
    "account_expired_commitments": "245b3bf2e366486daabe015660cfb23320014bbe9ca910297e04a45ca693bc82",
    "account_observed": "230cbd16e0cf2f423c3ca675bda7c676218b0e829d5fd020d62852bc1e6ceadc",
    "account_strict": "290d36abfe2d04a2368a030b244724600230eaabb8c0b3e8bb1b3ae6444357b0",
    "account_unknown_receipt": "e1d0094b1ec7fdfc85e17c36320fe1c0955493905392138f7d8c8d6546a2927e",
    "advise_error_initialization_credit": "22672decc8d00561fcabef55fca5f9a50a073690cddcfaa0fa955bba54ee1a20",
    "advise_error_nonuniform_prior": "88ebfd9b2447d05a601be21eb13db6de44623abaf75b86bcddc02cd00e6376b7",
    "advise_error_observed_in_strict": "d3328ef2ddefa3e856b7692785e41ac5e390c2a246bf91ad1f869c3b204eedf2",
    "advise_observed": "b4a80566aad7669f1aab53bbc7c3c9a905c613351bf0f6ec210f32a5dbdffc64",
    "advise_observed_initialization_only": "63981e82571db8b05f72cbec69b612827d89cb6ee9354d677ec94747f6e4885c",
    "advise_strict": "f7d002cc53647ee5d45478dad16d560c9861f727508a039eb860752525bf91a3",
    "advise_strict_long_half_life": "66773fad00cf50b5c319c9e45469ff46763ac34f37a488c16620d41dc2142a41",
    "advise_strict_unknown_cost": "3aaa64c02972f16409a7d2305193e579119347033ce7b8a262b4c1bcb2c0e331",
    "advise_strict_zero_beta": "9370fa1f2c3d91a3a908693d540db32ae40cad2a9007b88619c8c47bc014d1c0",
    "decision_error_missing_override": "ae3b03731a921183581ed986f8f4287260221111b86e35037042680d83e60e91",
    "decision_observed_donors": "76568c73f418fd723094429a316bc2113a2b4468a453ed04b69934242643a19d",
    "decision_observed_match": "20a271c42c8dfa1b3f7cae81fcbd219cb303983d489cda01191fec9b796b5cda",
    "decision_observed_other_source": "432e0263b2d9012a17cdf98357c1d0bacce4b733821489c89b5e7e6bb6822306",
    "decision_observed_override": "ab0489ad889de0ffa7afa0657415f5e40ed79aa37780b1bcb0517e322f7c793f",
    "decision_strict_override": "28a4cc5053f0084540f47f50aaca45e5afd1dccff9a91cab3f591c3473228c32",
    "decision_strict_top": "a1fa1dccc8af62fd2c2c80d9af8b05c86d867d5df522b3e3f90b4904214722ea",
}


def outputs(api):
    """Map case name -> sha256 of canonical JSON of its output or error."""
    result = {}
    for name, case in CASES.items():
        try:
            value = {"ok": case(api)}
        except api["SelectionError"] as error:
            value = {"error": type(error).__name__, "message": str(error)}
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def port_api():
    from geak_evo.selection import work
    return {name: getattr(work, name) for name in
            ("account_opportunities", "advise", "record_decision", "SelectionError")}


class SelectionGoldenTest(unittest.TestCase):
    def test_outputs_match_original(self):
        actual = outputs(port_api())
        self.assertEqual(set(actual), set(GOLDEN))
        for name in sorted(GOLDEN):
            with self.subTest(case=name):
                self.assertEqual(actual[name], GOLDEN[name])


if __name__ == "__main__":
    unittest.main()
