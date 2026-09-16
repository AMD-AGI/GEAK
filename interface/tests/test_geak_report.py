"""Tests for the report driver (geak_report): persistence isolation + status.

These lock the two driver-level guarantees Astra's review asked for: two runs of
the same model must never overwrite or mix each other's exported artifacts, and a
run that captured nothing must report that rather than a healthy zero-call "ok".
"""

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                      # interface/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "e2e_workflow", "scripts"))     # ledger deps

import geak_report as R  # noqa: E402


def _row(mid, output):
    return {"message_id": mid, "agent_label": "engineer:compute", "role": "engineer",
            "sub_phase": "compute", "transcript": "agent-a.jsonl", "group_id": "agent-a.jsonl#0",
            "model": "claude-opus-4-8", "cost_usd": 1.0, "output": output}


def _run_export(base, name, count, persist_root, model="same-model"):
    """Persist a synthetic run of `count` calls; return _persist's destination."""
    ev = os.path.join(base, name)
    trace = os.path.join(ev, "reports", "trace")
    os.makedirs(trace, exist_ok=True)
    calls = os.path.join(trace, "llm_calls.jsonl")
    with open(calls, "w", encoding="utf-8") as fh:
        for i in range(count):
            fh.write(json.dumps(_row("%s_%d" % (name, i), "%s_output_%d" % (name, i))) + "\n")
    report = os.path.join(ev, "report")
    os.makedirs(report, exist_ok=True)
    with open(os.path.join(report, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(name)
    return R._persist(ev, calls, report, model, persist_root)


class TestPersistIsolation(unittest.TestCase):
    def test_two_runs_same_model_do_not_mix_or_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_test_") as tmp:
            shared = os.path.join(tmp, "shared")
            dst1, _ = _run_export(tmp, "first", 2, shared)
            dst2, _ = _run_export(tmp, "second", 1, shared)
            # distinct run directories under the same model
            self.assertNotEqual(dst1, dst2)
            self.assertTrue(dst1.startswith(os.path.join(shared, "same-model")))
            # the second run's export holds ONLY its own artifacts
            arts = sorted(os.listdir(os.path.join(dst2, "geak_llm_artifacts")))
            self.assertEqual(len(arts), 1)
            got = json.load(open(os.path.join(dst2, "geak_llm_artifacts", arts[0])))
            self.assertEqual(got["output"], "second_output_0")
            # the first run stays individually recoverable
            self.assertEqual(len(os.listdir(os.path.join(dst1, "geak_llm_artifacts"))), 2)

    def test_manifest_carries_run_id_and_counts_only_this_run(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_test_") as tmp:
            shared = os.path.join(tmp, "shared")
            dst, _ = _run_export(tmp, "solo", 3, shared)
            man = json.load(open(os.path.join(dst, "_manifest.json")))
            self.assertTrue(man["run_id"].startswith("run-"))
            self.assertEqual(man["per_call_artifacts"], 3)
            self.assertIn(man["run_id"], dst)

    def test_regenerating_the_same_run_replaces_it_in_place(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_test_") as tmp:
            shared = os.path.join(tmp, "shared")
            dst_a, _ = _run_export(tmp, "same", 2, shared)
            dst_b, _ = _run_export(tmp, "same", 2, shared)  # identical content
            self.assertEqual(dst_a, dst_b)                  # same run id -> same dir
            self.assertEqual(len(os.listdir(os.path.join(dst_b, "geak_llm_artifacts"))), 2)


class TestPersistAtomicRegen(unittest.TestCase):
    """Astra Finding 3: regenerating a run's export must not destroy the prior valid
    export if the new one fails to write. The old code rmtree'd the run dir BEFORE
    copying, so any mid-copy failure left a half-deleted export and nothing to fall
    back to. Regeneration now stages then atomically swaps, so a staging failure
    leaves the previous export exactly as it was."""

    def test_failed_regeneration_leaves_prior_export_intact(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_test_") as tmp:
            shared = os.path.join(tmp, "shared")
            dst, _ = _run_export(tmp, "same", 2, shared)          # a good export exists
            good = sorted(os.listdir(os.path.join(dst, "geak_llm_artifacts")))
            self.assertEqual(len(good), 2)

            def _boom(*a, **k):
                raise RuntimeError("disk full mid-copy")

            orig = R._write_per_call_artifacts
            R._write_per_call_artifacts = _boom
            try:
                with self.assertRaises(RuntimeError):
                    _run_export(tmp, "same", 2, shared)           # regenerate -> fails mid-stage
            finally:
                R._write_per_call_artifacts = orig

            # The prior export is untouched: same dir, same two artifacts, still readable.
            self.assertTrue(os.path.isdir(dst))
            self.assertEqual(sorted(os.listdir(os.path.join(dst, "geak_llm_artifacts"))), good)
            # No stage/retired debris is left behind under the model directory.
            model_dir = os.path.dirname(dst)
            debris = [d for d in os.listdir(model_dir) if ".stage-" in d or ".old-" in d]
            self.assertEqual(debris, [])


class TestNoCaptureStatus(unittest.TestCase):
    def test_empty_transcript_glob_reports_no_capture(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_test_") as tmp:
            res = R.run(eval_dir=os.path.join(tmp, "empty"),
                        transcripts=[os.path.join(tmp, "absent", "*.jsonl")],
                        model="missing")
            self.assertEqual(res["status"], "no-capture")


if __name__ == "__main__":
    unittest.main()
