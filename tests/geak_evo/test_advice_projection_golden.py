"""Golden test of selection.advice.NativeDevelopmentAdvice (drift test 8).

A synthetic four-episode campaign (baseline initialization, an observed
improvement with a failed side trial, a slower episode and an episode whose
native evidence is unavailable) is projected, recorded through a fake loop,
read back by ``inputs`` and replayed. GOLDEN is the SHA-256 of the canonical
JSON of everything returned or saved, produced by the ORIGINAL pre-port
module (oct15 bridge/native_advice.py) with the same stub episode_review.
The scratch directory path is replaced by "<dir>" before hashing.

Run from the repo root:  python3 -B tests/geak_evo/test_advice_projection_golden.py
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


def _episode_review(store, snapshot, event, direction):
    return {"revision": 1, "direction": direction, "attempt": event["attempt_id"]}


def _scenario(A, D):
    """Run the campaign against advice module ``A`` with files under ``D``."""
    JOB = {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1"}
    CASES = ["case-a", "case-b"]

    def put(name, value):
        data = json.dumps(value, sort_keys=True).encode()
        path = os.path.join(D, name)
        open(path, "wb").write(data)
        return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}

    HOST = put("host.json", {"runtime": {"image": "img", "config_digest": "cfg", "physical_devices": ["GPU-1"]}})
    def request(source):
        return put("req-" + source + ".json", {"contract": {"timing_policy": {"n": 10}}, "benchmark_case_ids": CASES,
            "benchmark_report_ids": {"case-a": "r1", "case-b": "r2"}, "correctness_case_ids": ["c"],
            "phases": ["benchmark"], "evaluator_sha256": "2" * 64, "binding": dict(JOB, candidate_source_id=source)})

    class Store:
        def __init__(self):
            self.revisions, self.blobs, self.candidates = {}, {}, {}
        def get_revision(self, name, revision):
            return copy.deepcopy(self.revisions[(name, revision)])
        def evidence(self, eid):
            return copy.deepcopy(self.blobs[eid])
        def archive_evidence(self, candidate_id, files):
            eid = "ev-%d" % len(self.blobs)
            self.blobs[eid] = ({"candidate_id": candidate_id}, dict(files))
            return eid
        def get_candidate(self, cid):
            return copy.deepcopy(self.candidates[cid])

    store = Store()
    for cid, src in (("c0", "src-0"), ("c1", "src-1"), ("c2", "src-2"), ("c3", "src-3"), ("c4", "src-4")):
        store.candidates[cid] = dict(JOB, source_id=src)

    episodes = {}  # attempt -> spec
    def episode(attempt, direction, opened, parent, selected, trials, fail=None):
        coll = {"evidence_id": "coll-" + attempt}
        event = {"attempt_id": attempt, "settled": True, "collection_ref": coll, "transport_ref": "t-" + attempt,
                 "candidate_id": "c" + attempt[-1], "packet_ref": {"evidence_id": "pk-" + attempt, "artifact": "packet.json"}}
        review = {"review_kind": "research_review", "direction_revision": 2,
                  "proposal": {"direction_id": direction, "transport_ref": "t-" + attempt, "collection_ref": coll}}
        opp = {"opportunity_id": attempt, "direction_id": direction, "opened_at": opened, "engineer_id": "e-" + direction}
        episodes[attempt] = dict(event=event, review=review, opp=opp, parent=parent, selected=selected, trials=trials, fail=fail)

    def trial(attempt, op, source, times, failed=False):
        fb = put("fb-" + op + ".json", {"source_id": source, "operation_id": op})
        if failed:
            return {"feedback_ref": fb, "verified": None, "error": "compile failed"}, op
        req = request(source)
        return {"feedback_ref": fb, "error": None, "verified": {"source_id": source, "operation_id": op, "turn_id": attempt,
                "per_case_times_ms": dict(zip(CASES, times)), "measurement_valid": True,
                "artifact_index": [dict(req, name="request.json"), dict(HOST, name="host_receipt.json")]}}, op

    episode("a1", "d1", 10, "c0", "src-1", [trial("a1", "op-1", "src-1", [2.0, 4.0]), trial("a1", "op-1b", "src-1", [2.2, 4.4])])
    episode("a2", "d2", 20, "c1", "src-2", [trial("a2", "op-2", "src-2", [1.0, 2.0]), trial("a2", "op-2f", "src-9", None, failed=True)])
    episode("a3", "d1", 30, "c1", "src-3", [trial("a3", "op-3", "src-3", [3.0, 5.0])])
    episode("a4", "d3", 40, "c2", "src-4", [], fail="native evidence is unavailable")

    state = {"binding": JOB, "history": [], "opportunities": []}
    revision = [0]
    def commit(new_state):
        revision[0] += 1
        store.revisions[("campaign", revision[0])] = {"note": {"evolution": copy.deepcopy(new_state)}}
        return {"campaign_id": "campaign", "revision": revision[0], "state": copy.deepcopy(new_state),
                "accounting": {"now": None}}
    for attempt in ("a1", "a2", "a3", "a4"):
        spec = episodes[attempt]
        frozen = commit(state)
        frozen["accounting"]["now"] = spec["opp"]["opened_at"] - 1
        store.blobs["pl-" + attempt] = ({"candidate_id": "c0"}, {"planner_context.json": json.dumps(frozen)})
        store.blobs["pk-" + attempt] = ({"candidate_id": "c0"}, {"packet.json": json.dumps({
            "opportunity": {"opportunity_id": attempt}, "planner_ref": {"evidence_id": "pl-" + attempt, "artifact": "planner_context.json"},
            "proposal": {"parent_candidate_id": spec["parent"]}})})
        state["opportunities"].append(dict(spec["opp"], progress_complete=False))
        state["history"] += [spec["event"], spec["review"]]

    class Evidence:
        def episode(self, snapshot, event):
            spec = episodes[event["attempt_id"]]
            if spec["fail"]:
                raise ValueError(spec["fail"])
            return None, {"native_receipt": {"invocation_id": event["attempt_id"],
                          "native_results": [{"feedback_ref": t["feedback_ref"], "operation_id": op} for t, op in spec["trials"]]},
                          "native_trials": [t for t, _ in spec["trials"]], "diagnostic_refs_not_revalidated": [],
                          "harvest": {"selected": {"source_id": spec["selected"]}}, "host_provenance": {"h": event["attempt_id"]}}

    class Loop:
        def __init__(self):
            self.store, self.current = store, commit(state)
        def snapshot(self):
            return copy.deepcopy(self.current)
        def record_progress_review(self, attempt, ref, verify_review):
            _, files = store.evidence(ref["evidence_id"])
            raw = files[ref["artifact"]]
            value = json.loads(raw)
            assert verify_review(ref, raw, {"opportunity_id": attempt, "complete": value["complete"]})
            new = copy.deepcopy(self.current["state"])
            for row in new["opportunities"]:
                if row["opportunity_id"] == attempt:
                    row.setdefault("progress_review_refs", []).append(ref)
                    row.update(progress_complete=value["complete"], progress_gain_kind=value["gain_kind"],
                               progress_outcome_kind=value["outcome_kind"])
            self.current = commit(new)

    loop = Loop()
    adv = A.NativeDevelopmentAdvice(loop, Evidence())
    out = {"projected": [[e["attempt_id"], v, p] for e, v, p in adv._project(loop.snapshot())]}
    final = adv.record_reviewed_progress(loop.snapshot())
    inputs = adv.inputs(final)
    out["progress"] = inputs["progress"]
    out["verify"] = [inputs["verify_progress"](r["evidence_ref"], {k: v for k, v in r.items() if k != "evidence_ref"})
                     for r in inputs["progress"]]
    again = adv.record_reviewed_progress(final)
    out["replay_revision"] = [final["revision"], again["revision"]]
    out["stored"] = sorted(json.dumps([k, v[0], {n: b.decode() if isinstance(b, bytes) else b for n, b in v[1].items()}], sort_keys=True)
                           for k, v in store.blobs.items() if k.startswith("ev-"))
    return out


GOLDEN = "2283976b1e989af75d5b6f0be67deb3e3c55abed3231a798e2c92743dedad05d"


def output_hash(module):
    """SHA-256 of the scenario output for ``module`` (episode_review stubbed)."""
    original = module.episode_review
    module.episode_review = _episode_review
    try:
        with tempfile.TemporaryDirectory() as directory:
            directory = os.path.realpath(directory)
            text = json.dumps(_scenario(module, directory), sort_keys=True, separators=(",", ":"))
            text = text.replace(directory, "<dir>")
    finally:
        module.episode_review = original
    return hashlib.sha256(text.encode()).hexdigest()


class AdviceProjectionGoldenTest(unittest.TestCase):
    def test_projection_matches_original(self):
        from geak_evo.selection import advice
        self.assertEqual(output_hash(advice), GOLDEN)


if __name__ == "__main__":
    unittest.main()
