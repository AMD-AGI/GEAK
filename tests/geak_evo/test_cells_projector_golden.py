"""Golden test of archive.cells.NativeCellProjector (drift test 8).

A synthetic campaign with one reviewed episode (repeated, incomplete-runtime
and failed trials), one unreviewed episode and one unsettled event is
projected with stub episode readers; a second run makes the review lookup
ambiguous. GOLDEN is the SHA-256 of the canonical JSON output of the ORIGINAL
pre-port module (oct15 bridge/native_cells.py). The scratch directory path
is replaced by "<dir>" before hashing.

Run from the repo root:  python3 -B tests/geak_evo/test_cells_projector_golden.py
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def _scenario(cells, D):
    """Project the campaign with cells module ``cells`` and files under ``D``."""
    JOB = {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1"}

    def put(name, value):
        data = json.dumps(value, sort_keys=True).encode()
        path = os.path.join(D, name)
        open(path, "wb").write(data)
        return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}

    cases = ["case-a", "case-b"]
    request = {"contract": {"timing_policy": {"iterations": 10}}, "benchmark_case_ids": cases,
               "benchmark_report_ids": {"case-a": "r1", "case-b": "r2"}, "correctness_case_ids": ["c"],
               "phases": ["benchmark"], "evaluator_sha256": "2" * 64,
               "binding": dict(JOB, candidate_source_id="src-sel")}
    host = {"runtime": {"image": "img", "config_digest": "cfg", "physical_devices": ["GPU-1"]}}
    req_ref, host_ref = put("request.json", request), put("host_receipt.json", host)
    bad_host_ref = put("host_bad.json", {"runtime": {"image": "img"}})
    fb = put("feedback.json", {"source_id": "src-fail", "operation_id": "op-f"})
    def verified(op, times, hostref=host_ref):
        return {"source_id": "src-sel", "operation_id": op, "per_case_times_ms": dict(zip(cases, times)),
                "measurement_valid": True, "artifact_index": [dict(req_ref, name="request.json"),
                                                             dict(hostref, name="host_receipt.json")]}
    trials = [{"verified": verified("op-1", [1.0, 2.0]), "feedback_ref": "fb1", "error": None},
              {"verified": verified("op-2", [1.5, 2.5]), "feedback_ref": "fb2", "error": None},
              {"verified": verified("op-1", [1.0, 2.0]), "feedback_ref": "fb1", "error": None},
              {"verified": verified("op-3", [1.0, 2.0], bad_host_ref), "feedback_ref": "fb3", "error": None},
              {"verified": None, "feedback_ref": fb, "error": "compile failed"}]
    coll = {"evidence_id": "ev-1"}
    history = [{"attempt_id": "a1", "settled": True, "collection_ref": coll, "transport_ref": "t1", "candidate_id": "c-term"},
               {"attempt_id": "a2", "settled": True, "collection_ref": {"evidence_id": "ev-2"}, "transport_ref": "t2", "candidate_id": "x"},
               {"attempt_id": "a3", "settled": False},
               {"review_kind": "research_review", "direction_revision": 3,
                "proposal": {"direction_id": "d1", "transport_ref": "t1", "collection_ref": coll}},
               {"review_kind": "research_review", "direction_revision": 4,
                "proposal": {"direction_id": "d1", "transport_ref": "t1", "collection_ref": coll}}]
    state = {"binding": JOB, "history": history,
             "opportunities": [{"opportunity_id": "a1", "direction_id": "d1"}, {"opportunity_id": "a2", "direction_id": "d2"}]}
    snapshot = {"campaign_id": "campaign", "revision": 7, "state": state, "frozen_cases": []}

    class Store:
        def get(self, name):
            return {"revision": 7, "note": {"evolution": json.loads(json.dumps(state))}}
        def evidence(self, eid):
            return {"candidate_id": "c-term"}, {"native_source_roles.json": json.dumps({"terminal": "c-term", "selected": "c-sel"})}
        def get_candidate(self, cid):
            return dict(JOB, source_id={"c-term": "src-term", "c-sel": "src-sel"}[cid])

    material = {"harvest": {"terminal": {"source_id": "src-term"}, "selected": {"source_id": "src-sel"}}, "native_trials": trials}
    out = {}
    for name, review_rev in (("ok", 3), ("ambiguous", 9)):
        proj = cells.NativeCellProjector(Store(), episode_material=lambda s, e: (None, material),
                                         episode_review=lambda st, s, e, d: {"revision": review_rev})
        try:
            out[name] = {"ok": proj(snapshot)}
        except ValueError as error:
            out[name] = {"error": str(error)}
    try:
        cells.NativeCellProjector(Store(), episode_material=None, episode_review=None)
    except ValueError as error:
        out["init"] = str(error)
    return out


GOLDEN = "aa1d00c243c3c03756f6a46dd2919b6938b40c83f8351d8c3ca6c96565b0ebf7"


def output_hash(module):
    with tempfile.TemporaryDirectory() as directory:
        directory = os.path.realpath(directory)
        text = json.dumps(_scenario(module, directory), sort_keys=True, separators=(",", ":"))
        text = text.replace(directory, "<dir>")
    return hashlib.sha256(text.encode()).hexdigest()


class CellProjectorGoldenTest(unittest.TestCase):
    def test_projection_matches_original(self):
        from geak_evo.archive import cells
        self.assertEqual(output_hash(cells), GOLDEN)


if __name__ == "__main__":
    unittest.main()
