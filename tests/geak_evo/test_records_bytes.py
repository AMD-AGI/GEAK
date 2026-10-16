"""Saved-record byte compatibility with the Oct-15 original (synthetic CPU fixture).

GOLDEN_* below were produced by running ``_scenario`` against the original
``geak_evo.evo_archive``, ``geak_evo.claims`` and ``geak_evo.research_state``
modules. The port must write the same bytes under the same evidence IDs and
must read the original's bytes to the same values.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.archive import claims, records
from geak_evo.archive.store import ResearchStore

CASES = {"job": ["short", "long"]}
NOTE = {"claim": "CPU fixture only", "scope": "synthetic job", "next_test": "inspect exact evidence"}
IDEA = {"idea_id": "Q1", "statement": "Test streaming summary", "status": "proposed"}


def _verifier(artifacts, expected):
    return {key: True for key in ("independent", "settled", "correct", "eligible")}


def _authorize(*_):
    return True


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _store(root):
    work = root / "work"
    work.mkdir()
    store = ResearchStore(root / "archive", allowed_epochs=[("job", "epoch")],
                          source_paths={"job": ["kernel.py"]}, verify_evidence=_verifier)

    def candidate(text):
        (work / "kernel.py").write_text(text)
        return store.candidate("job", "epoch", store.capture_source(work, ["kernel.py"]))
    return store, candidate("baseline"), candidate("donor")


def _quantity(value, **changes):
    result = {"value": value, "unit": "us", "denominator": "call", "population": "fixture-inputs",
              "aggregation": "mean", "measurement_kind": "CPU fixture timing"}
    result.update(changes)
    return result


def _write_records(store, a, donor):
    written = []

    def record(candidate, kind, **fields):
        ref = records.archive_record(store, candidate, dict(kind=kind, **fields), frozen_cases=CASES)
        written.append(ref)
        return ref
    ch1 = record(a, "evo_island_charter", island_id="I1", question="Can a different dataflow help?",
                 descriptor_spec={"ownership": ["row", "split"]})
    ch2 = record(donor, "evo_island_charter", island_id="I2", question="Does split ownership help?",
                 descriptor_spec={"ownership": ["split"]})
    assign = {"status": "unknown", "descriptors": {"ownership": None}, "evidence_refs": [],
              "reason": "Case evidence pending"}
    m1 = record(a, "evo_membership", island_id="I1", charter_ref=ch1, assignment=assign)
    m2 = record(donor, "evo_membership", island_id="I2", charter_ref=ch2, assignment=assign)
    m3 = record(a, "evo_membership", island_id="I2", charter_ref=ch2, assignment=dict(assign, status="tentative"))
    migration = record(a, "evo_decision", decision_type="migration", rationale="Explore ownership question",
                       evidence_refs=[m1, m3], details={"origin_ref": m1, "destination_ref": m3})
    for direction, tip, card in (("D1", a, {"idea": IDEA, "membership_refs": [m3], "decision_refs": [migration]}),
                                 ("D2", donor, {"idea": IDEA, "membership_refs": [m2]})):
        state = records.archive_checkpoint(store, direction, tip, card=card, note=NOTE, expected_revision=0,
                                           frozen_cases=CASES)
        written.append(state["note"]["archive_card_ref"])
    return written


def _write_claims(store, a):
    binding = {"job_id": "job", "case_id": "short", "source_id": store.get_candidate(a)["source_id"],
               "work_id": "work-1", "measurement_epoch_id": "epoch", "hardware_id": "CPU fixture",
               "scope": "complete_callable"}
    log = claims.ClaimLog(store, verify_quantity=_authorize)
    metrics = []

    def report(value):
        data = json.dumps({"metrics": {"latency": {"binding": binding, "quantity": _quantity(value)}}}).encode()
        ref = {"evidence_id": store.archive_evidence(a, {"metrics.json": data}), "artifact": "metrics.json"}
        metrics.append(ref)
        return log.append(a, {"kind": "reported_quantity", "statement": "The fixture reports this mean.",
                              "binding": dict(binding), "evidence": [ref], "depends_on": [], "assumptions": [],
                              "applicability": "CPU fixture only", "next_test": "Inspect the baseline.",
                              "quantity": _quantity(value),
                              "extraction": {"ref": ref, "pointer": "/metrics/latency"}}, binding)
    r1, r2 = report("0.3"), report("0.2")
    r3 = log.append(a, {"kind": "derived_quantity", "statement": "Declared arithmetic.", "binding": dict(binding),
                        "evidence": [], "depends_on": [r1, r2], "assumptions": ["Needs interpretation."],
                        "applicability": "Fixture", "next_test": "Check comparison.",
                        "quantity": _quantity("0.1", aggregation="difference", measurement_kind="derived"),
                        "derivation": {"operation": "difference", "operands": [r1, r2], "weights": []}}, binding)
    r4 = log.correct(a, [r2], scope=binding, reason="Wrong input set", consequence="Difference is stale")
    return metrics + [r1, r2, r3, r4]


def _objects(store, refs):
    result = []
    for ref in refs:
        envelope, artifacts = store.evidence(ref["evidence_id"])
        result.append({"owner": envelope["candidate_id"], "ref": ref, "bytes": artifacts[ref["artifact"]].decode()})
    return result


class SavedRecordBytesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store, self.a, self.donor = _store(Path(temp.name))

    def test_port_writes_the_original_bytes_and_ids(self):
        self.assertEqual([self.a, self.donor], GOLDEN_CANDIDATES)
        written = _write_records(self.store, self.a, self.donor)
        claim_refs = _write_claims(self.store, self.a)
        self.assertEqual(_objects(self.store, written + claim_refs), GOLDEN_OBJECTS)
        self.assertEqual(records.format_archive(records.read_archive(self.store, frozen_cases=CASES)),
                         GOLDEN_ARCHIVE_TEXT)

    def test_port_reads_original_bytes_identically(self):
        self.assertEqual([self.a, self.donor], GOLDEN_CANDIDATES)
        for item in GOLDEN_OBJECTS:
            identity = self.store.archive_evidence(item["owner"], {item["ref"]["artifact"]: item["bytes"].encode()})
            self.assertEqual(identity, item["ref"]["evidence_id"])
        record_refs = [item["ref"] for item in GOLDEN_OBJECTS if item["ref"]["artifact"] == "record.json"]
        read = [records.read_record(self.store, ref, frozen_cases=CASES) for ref in record_refs]
        self.assertEqual(_digest(read), GOLDEN_RECORDS_SHA256)
        head = GOLDEN_OBJECTS[-1]["ref"]
        view = claims.ClaimLog(self.store, head, verify_quantity=_authorize).view()
        self.assertEqual(_digest(view), GOLDEN_CLAIM_VIEW_SHA256)
        self.assertEqual(sorted(entry["state"] for entry in view["claims"].values()), ["active", "disputed", "stale"])


# Produced from the Oct-15 original; regenerate only when the saved format changes on purpose.
GOLDEN_CANDIDATES = ['f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
 '3e8abaa3dfa1462cfc67c9adf80e7d921c78d0799ab4009bb7431009291ade49']
GOLDEN_RECORDS_SHA256 = 'ce567bd0679014323628cd0b97bbb3aadb24be5890250fe8dc0c8210fc71653f'
GOLDEN_CLAIM_VIEW_SHA256 = '55595454b64b8bd4e1b8c26622f9963a130c3fe661c8896724d016cfc7847d2f'
GOLDEN_ARCHIVE_TEXT = (
    'ARCHIVE 2 directions / 2 distinct tip sources / 1 islands\n'
    '  ISLAND I2 (agent assignments; no inferred elites)\n'
    '    D1: tentative; Does split ownership help?\n'
    '    D2: unknown; Does split ownership help?\n'
    '  D1 r1 [active]\n'
    '    anchor unknown\n'
    '    tip    f70a78b1c902 / source 8650e09c76d4\n'
    '    idea Q1: Test streaming summary\n'
    '    engineer unknown; 0 direct evidence refs\n'
    '    novelty/roofline/elite: not established by archive metadata\n'
    '  D2 r1 [active]\n'
    '    anchor unknown\n'
    '    tip    3e8abaa3dfa1 / source e9c4050c4d1d\n'
    '    idea Q1: Test streaming summary\n'
    '    engineer unknown; 0 direct evidence refs\n'
    '    novelty/roofline/elite: not established by archive metadata'
)
GOLDEN_OBJECTS = [{'bytes': '{"binding":{"candidate_source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","descriptor_spec":{"ownership":["row","split"]},"island_id":"I1","kind":"evo_island_charter","question":"Can '
           'a different dataflow help?","schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'record.json',
          'evidence_id': '4b42cd1f306e42c919b59c5ac4e7c7620884f3925c518ddcb170d8bea236b48d'}},
 {'bytes': '{"binding":{"candidate_source_id":"e9c4050c4d1dff957941b164f0f5bc2002dbd890fcf5b39620e2a196eede0e2d","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"3e8abaa3dfa1462cfc67c9adf80e7d921c78d0799ab4009bb7431009291ade49","descriptor_spec":{"ownership":["split"]},"island_id":"I2","kind":"evo_island_charter","question":"Does '
           'split ownership help?","schema_version":1}',
  'owner': '3e8abaa3dfa1462cfc67c9adf80e7d921c78d0799ab4009bb7431009291ade49',
  'ref': {'artifact': 'record.json',
          'evidence_id': 'a5b2d2b19a46779c571ef3362f1f8e6dc777fb2ebd50684b6c2de43bf4c1c436'}},
 {'bytes': '{"assignment":{"descriptors":{"ownership":null},"evidence_refs":[],"reason":"Case evidence '
           'pending","status":"unknown"},"binding":{"candidate_source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","charter_ref":{"artifact":"record.json","evidence_id":"4b42cd1f306e42c919b59c5ac4e7c7620884f3925c518ddcb170d8bea236b48d"},"island_id":"I1","kind":"evo_membership","schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'record.json',
          'evidence_id': '167f2715edf2e8fb9b21214ea7ec5888ebe7b7fe67cd5fa05a9b6e1fe5bb2351'}},
 {'bytes': '{"assignment":{"descriptors":{"ownership":null},"evidence_refs":[],"reason":"Case evidence '
           'pending","status":"unknown"},"binding":{"candidate_source_id":"e9c4050c4d1dff957941b164f0f5bc2002dbd890fcf5b39620e2a196eede0e2d","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"3e8abaa3dfa1462cfc67c9adf80e7d921c78d0799ab4009bb7431009291ade49","charter_ref":{"artifact":"record.json","evidence_id":"a5b2d2b19a46779c571ef3362f1f8e6dc777fb2ebd50684b6c2de43bf4c1c436"},"island_id":"I2","kind":"evo_membership","schema_version":1}',
  'owner': '3e8abaa3dfa1462cfc67c9adf80e7d921c78d0799ab4009bb7431009291ade49',
  'ref': {'artifact': 'record.json',
          'evidence_id': 'e1becc7d7253eb62a8f036a8ea6abb68227b4f05ede2a20acea4775b7c528d24'}},
 {'bytes': '{"assignment":{"descriptors":{"ownership":null},"evidence_refs":[],"reason":"Case evidence '
           'pending","status":"tentative"},"binding":{"candidate_source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","charter_ref":{"artifact":"record.json","evidence_id":"a5b2d2b19a46779c571ef3362f1f8e6dc777fb2ebd50684b6c2de43bf4c1c436"},"island_id":"I2","kind":"evo_membership","schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'record.json',
          'evidence_id': 'c648d5606ff8cd810313d738b02afc7407ae482413b38b2f16aa6f55aa0af54c'}},
 {'bytes': '{"binding":{"candidate_source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","decision_type":"migration","details":{"destination_ref":{"artifact":"record.json","evidence_id":"c648d5606ff8cd810313d738b02afc7407ae482413b38b2f16aa6f55aa0af54c"},"origin_ref":{"artifact":"record.json","evidence_id":"167f2715edf2e8fb9b21214ea7ec5888ebe7b7fe67cd5fa05a9b6e1fe5bb2351"}},"evidence_refs":[{"artifact":"record.json","evidence_id":"167f2715edf2e8fb9b21214ea7ec5888ebe7b7fe67cd5fa05a9b6e1fe5bb2351"},{"artifact":"record.json","evidence_id":"c648d5606ff8cd810313d738b02afc7407ae482413b38b2f16aa6f55aa0af54c"}],"kind":"evo_decision","rationale":"Explore '
           'ownership question","schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'record.json',
          'evidence_id': '3fe545d5a5e858e3b7c6dc7ea9b3fd6de7597759179857842ead8b9457c0caee'}},
 {'bytes': '{"binding":{"candidate_source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","decision_refs":[{"artifact":"record.json","evidence_id":"3fe545d5a5e858e3b7c6dc7ea9b3fd6de7597759179857842ead8b9457c0caee"}],"direction_id":"D1","idea":{"idea_id":"Q1","statement":"Test '
           'streaming '
           'summary","status":"proposed"},"kind":"evo_archive_card","membership_refs":[{"artifact":"record.json","evidence_id":"c648d5606ff8cd810313d738b02afc7407ae482413b38b2f16aa6f55aa0af54c"}],"schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'archive_card.json',
          'evidence_id': '3acbe8aeddbd8eb590e6610ff9ca352cc3e4dedf7c92db835d76139d2b04ad4b'}},
 {'bytes': '{"binding":{"candidate_source_id":"e9c4050c4d1dff957941b164f0f5bc2002dbd890fcf5b39620e2a196eede0e2d","job_id":"job","measurement_epoch_id":"epoch"},"candidate_id":"3e8abaa3dfa1462cfc67c9adf80e7d921c78d0799ab4009bb7431009291ade49","direction_id":"D2","idea":{"idea_id":"Q1","statement":"Test '
           'streaming '
           'summary","status":"proposed"},"kind":"evo_archive_card","membership_refs":[{"artifact":"record.json","evidence_id":"e1becc7d7253eb62a8f036a8ea6abb68227b4f05ede2a20acea4775b7c528d24"}],"schema_version":1}',
  'owner': '3e8abaa3dfa1462cfc67c9adf80e7d921c78d0799ab4009bb7431009291ade49',
  'ref': {'artifact': 'archive_card.json',
          'evidence_id': '0223c480f6fa3e1953e278bff416770d360e79a8003a5ca3138e1c5d39474613'}},
 {'bytes': '{"metrics": {"latency": {"binding": {"job_id": "job", "case_id": "short", "source_id": '
           '"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f", "work_id": "work-1", '
           '"measurement_epoch_id": "epoch", "hardware_id": "CPU fixture", "scope": "complete_callable"}, '
           '"quantity": {"value": "0.3", "unit": "us", "denominator": "call", "population": '
           '"fixture-inputs", "aggregation": "mean", "measurement_kind": "CPU fixture timing"}}}}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'metrics.json',
          'evidence_id': 'bc0e80f50a388dc9518b6acd5821fdd4935d4505759e808993057052e68c9c2d'}},
 {'bytes': '{"metrics": {"latency": {"binding": {"job_id": "job", "case_id": "short", "source_id": '
           '"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f", "work_id": "work-1", '
           '"measurement_epoch_id": "epoch", "hardware_id": "CPU fixture", "scope": "complete_callable"}, '
           '"quantity": {"value": "0.2", "unit": "us", "denominator": "call", "population": '
           '"fixture-inputs", "aggregation": "mean", "measurement_kind": "CPU fixture timing"}}}}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'metrics.json',
          'evidence_id': '8a67047c7d1e9446253a110e39725d495d7228a2cafbd610183b7974d5ad3ebc'}},
 {'bytes': '{"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","kind":"claim","previous":null,"record":{"applicability":"CPU '
           'fixture only","assumptions":[],"binding":{"case_id":"short","hardware_id":"CPU '
           'fixture","job_id":"job","measurement_epoch_id":"epoch","scope":"complete_callable","source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","work_id":"work-1"},"depends_on":[],"evidence":[{"artifact":"metrics.json","evidence_id":"bc0e80f50a388dc9518b6acd5821fdd4935d4505759e808993057052e68c9c2d"}],"extraction":{"pointer":"/metrics/latency","ref":{"artifact":"metrics.json","evidence_id":"bc0e80f50a388dc9518b6acd5821fdd4935d4505759e808993057052e68c9c2d"}},"kind":"reported_quantity","next_test":"Inspect '
           'the baseline.","quantity":{"aggregation":"mean","denominator":"call","measurement_kind":"CPU '
           'fixture timing","population":"fixture-inputs","unit":"us","value":"0.3"},"statement":"The '
           'fixture reports this mean."},"schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'claim.json',
          'evidence_id': '624563d7d1ce8cf403c18ec88315069841ef4613fcfdbdb2275b1b734f379013'}},
 {'bytes': '{"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","kind":"claim","previous":{"artifact":"claim.json","evidence_id":"624563d7d1ce8cf403c18ec88315069841ef4613fcfdbdb2275b1b734f379013"},"record":{"applicability":"CPU '
           'fixture only","assumptions":[],"binding":{"case_id":"short","hardware_id":"CPU '
           'fixture","job_id":"job","measurement_epoch_id":"epoch","scope":"complete_callable","source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","work_id":"work-1"},"depends_on":[],"evidence":[{"artifact":"metrics.json","evidence_id":"8a67047c7d1e9446253a110e39725d495d7228a2cafbd610183b7974d5ad3ebc"}],"extraction":{"pointer":"/metrics/latency","ref":{"artifact":"metrics.json","evidence_id":"8a67047c7d1e9446253a110e39725d495d7228a2cafbd610183b7974d5ad3ebc"}},"kind":"reported_quantity","next_test":"Inspect '
           'the baseline.","quantity":{"aggregation":"mean","denominator":"call","measurement_kind":"CPU '
           'fixture timing","population":"fixture-inputs","unit":"us","value":"0.2"},"statement":"The '
           'fixture reports this mean."},"schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'claim.json',
          'evidence_id': 'f822a0cb99eae57d3408538bb8be81df6640517ab82e2b9932ffb83cc97403b4'}},
 {'bytes': '{"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","kind":"claim","previous":{"artifact":"claim.json","evidence_id":"f822a0cb99eae57d3408538bb8be81df6640517ab82e2b9932ffb83cc97403b4"},"record":{"applicability":"Fixture","assumptions":["Needs '
           'interpretation."],"binding":{"case_id":"short","hardware_id":"CPU '
           'fixture","job_id":"job","measurement_epoch_id":"epoch","scope":"complete_callable","source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","work_id":"work-1"},"depends_on":[{"artifact":"claim.json","evidence_id":"624563d7d1ce8cf403c18ec88315069841ef4613fcfdbdb2275b1b734f379013"},{"artifact":"claim.json","evidence_id":"f822a0cb99eae57d3408538bb8be81df6640517ab82e2b9932ffb83cc97403b4"}],"derivation":{"operands":[{"artifact":"claim.json","evidence_id":"624563d7d1ce8cf403c18ec88315069841ef4613fcfdbdb2275b1b734f379013"},{"artifact":"claim.json","evidence_id":"f822a0cb99eae57d3408538bb8be81df6640517ab82e2b9932ffb83cc97403b4"}],"operation":"difference","weights":[]},"evidence":[],"kind":"derived_quantity","next_test":"Check '
           'comparison.","quantity":{"aggregation":"difference","denominator":"call","measurement_kind":"derived","population":"fixture-inputs","unit":"us","value":"0.1"},"statement":"Declared '
           'arithmetic."},"schema_version":1}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'claim.json',
          'evidence_id': '7ad68cfc5fb7fc83efaa5df59409b4fadcaef9a331dec98e57794177ab015c39'}},
 {'bytes': '{"candidate_id":"f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317","consequence":"Difference '
           'is '
           'stale","kind":"claim_correction","previous":{"artifact":"claim.json","evidence_id":"7ad68cfc5fb7fc83efaa5df59409b4fadcaef9a331dec98e57794177ab015c39"},"reason":"Wrong '
           'input set","replacement":null,"schema_version":1,"scope":{"case_id":"short","hardware_id":"CPU '
           'fixture","job_id":"job","measurement_epoch_id":"epoch","scope":"complete_callable","source_id":"8650e09c76d4fa405c07d6b5bdc4c41396034b42eda6c288e230dbf0530c329f","work_id":"work-1"},"status":"disputed","targets":[{"artifact":"claim.json","evidence_id":"f822a0cb99eae57d3408538bb8be81df6640517ab82e2b9932ffb83cc97403b4"}]}',
  'owner': 'f70a78b1c902325f780e2986c8b63983d0d4c570478e615df19f2ef7631cb317',
  'ref': {'artifact': 'correction.json',
          'evidence_id': '6a1fc104785c18dcc31900ec08f502c4ddcea44865daebf9d3d78343b2e3a4ba'}}]


if __name__ == "__main__":
    unittest.main()
