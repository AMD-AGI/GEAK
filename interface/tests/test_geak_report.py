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
from pathlib import Path
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                      # interface/
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                        "e2e_workflow", "scripts")
sys.path.insert(0, _SCRIPTS)                                    # ledger deps
sys.path.insert(0, os.path.join(_SCRIPTS, "tests"))            # ledger test fixtures

import geak_report as R  # noqa: E402
import claude_trace_mirror as M  # noqa: E402
from test_llm_ledger import (asst_rec, ev, prompt_for, timeline,  # noqa: E402
                             user_rec, write_transcript)


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


class TestTranscriptScopeVisibility(unittest.TestCase):
    """Fix 3 (Astra re-review): however transcripts were selected, the choice is
    surfaced — persisted in the ledger meta AND rendered in the report — never
    a silent scope with a null saved value and no mention of a fallback."""

    def _one_call_transcript(self, path, eval_dir):
        write_transcript(path, [
            user_rec(prompt_for("director", "setup", eval_dir), 0),
            asst_rec(10, "msg_1", read=1000, out=10,
                     text="done", thinking="thinking"),
        ])

    def _meta(self, eval_dir):
        p = os.path.join(eval_dir, "reports", "trace", "token_stats.json")
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("meta", {})

    def _md(self, res):
        with open(res["md"], encoding="utf-8") as fh:
            return fh.read()

    def test_explicit_scope_is_persisted_and_rendered(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_test_") as tmp:
            ev = os.path.join(tmp, "run")
            tdir = os.path.join(tmp, "t"); os.makedirs(tdir)
            self._one_call_transcript(os.path.join(tdir, "a.jsonl"), ev)
            res = R.run(eval_dir=ev,
                        transcripts=[os.path.join(tdir, "*.jsonl")],
                        model="m")
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["transcript_scope"], "explicit")
            self.assertEqual(self._meta(ev).get("transcript_scope"), "explicit")
            self.assertIn("transcript scope", self._md(res))

    def test_substring_fallback_is_visible_not_silent(self):
        # No workflow record owns this eval-dir, so scope resolution cannot claim
        # a whole-run scope: the driver falls back to substring discovery, and
        # that fallback must be recorded in meta AND flagged in the markdown.
        with tempfile.TemporaryDirectory(prefix="geak_report_test_") as tmp:
            home = os.path.join(tmp, "home", ".claude")
            proj = os.path.join(home, "projects", "p"); os.makedirs(proj)
            ev = os.path.join(tmp, "run")
            # A transcript substring-discovery will find (mentions eval-dir), but
            # NO wf_*.json record naming it -> resolve_run_scope is 'unresolved'.
            self._one_call_transcript(os.path.join(proj, "drv.jsonl"), ev)
            old = os.environ.get("CLAUDE_CONFIG_DIR")
            os.environ["CLAUDE_CONFIG_DIR"] = home
            try:
                res = R.run(eval_dir=ev, model="m")   # no explicit transcripts
            finally:
                if old is None:
                    os.environ.pop("CLAUDE_CONFIG_DIR", None)
                else:
                    os.environ["CLAUDE_CONFIG_DIR"] = old
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["transcript_scope"], "substring-fallback")
            self.assertTrue(res.get("scope_warnings"))
            self.assertEqual(self._meta(ev).get("transcript_scope"),
                             "substring-fallback")
            self.assertIn("FALLBACK", self._md(res))


class TestMidRunReportBeforeReturn(unittest.TestCase):
    """P2 (Astra re-review): the report is emitted from INSIDE the dispatcher,
    before it returns — so the record has ``args.exp_root`` (+ kernel_path etc.)
    but NO ``result.eval_dir`` yet, and the report's ``--eval-dir`` (the lane's
    generated dir) is not named by any record. The mid-run report must still scope
    to the run's OWN dir by anchoring on the enclosing exp_root, never fall back to
    substring discovery and over-attribute concurrent sessions."""

    def _native_home(self, tmp, *, with_result):
        """Materialize the ACTUAL native mid-run kernel record shape and one real
        agent transcript. Returns (home, eval_dir, record_path, record)."""
        home = os.path.join(tmp, "home", ".claude")
        exp = os.path.join(tmp, "exp")
        eval_dir = os.path.join(exp, "team_task_x", "task")
        sess = os.path.join(home, "projects", "proj", "sess")
        wfdir = os.path.join(sess, "workflows")
        os.makedirs(wfdir)
        # args carries exp_root + kernel_path + workflow_dir (no eval_dir); result
        # is written ONLY once the dispatcher returns.
        record = {"runId": "wf_live", "timestamp": "2026-09-16T01:00:00Z",
                  "args": {"exp_root": exp,
                           "kernel_path": "/tasks/fused_moe_int4",
                           "workflow_dir": "/GEAK/kernel_workflow"}}
        if with_result:
            record["result"] = {"eval_dir": eval_dir}
        rp = os.path.join(wfdir, "wf_live.json")
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
        rundir = os.path.join(sess, "subagents", "workflows", "wf_live")
        os.makedirs(rundir)
        write_transcript(os.path.join(rundir, "agent-director.jsonl"), [
            user_rec(prompt_for("director", "setup", eval_dir), 9),
            asst_rec(10, "msg_1", read=1000, out=10, text="done", thinking="t"),
        ])
        # the timeline the dispatcher persists just before the report (nested=[])
        tl = os.path.join(eval_dir, "reports", "trace", "agent_timeline.json")
        os.makedirs(os.path.dirname(tl))
        doc = timeline([ev("Setup", "director:setup")], workflow="kernel_lane")
        doc["instance"] = eval_dir
        with open(tl, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        return home, eval_dir, rp, record

    def _scope_from(self, eval_dir, home):
        with mock.patch.object(M, "candidate_homes", return_value=[Path(home)]):
            return R.run(eval_dir=eval_dir)

    def _meta(self, eval_dir):
        p = os.path.join(eval_dir, "reports", "trace", "token_stats.json")
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("meta", {})

    def _md(self, res):
        with open(res["md"], encoding="utf-8") as fh:
            return fh.read()

    def test_before_return_anchors_on_exp_root_not_substring(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_p2_") as tmp:
            home, eval_dir, _rp, _rec = self._native_home(tmp, with_result=False)
            res = self._scope_from(eval_dir, home)
            self.assertEqual(res["status"], "ok")
            # NOT substring-fallback: it anchored on the enclosing exp_root. But the
            # run's OWN eval_dir is not yet on record, so ownership is INFERRED by
            # containment, not proven — scope is the weaker inferred variant and the
            # anchor/incompleteness is persisted (Astra r3: never promote containment
            # to complete ownership).
            self.assertEqual(res["transcript_scope"], "run-scoped-inferred")
            self.assertEqual(res.get("transcript_scope_anchor"), "exp_root-ancestor")
            self.assertTrue(res.get("scope_warnings"))
            meta = self._meta(eval_dir)
            self.assertEqual(meta.get("transcript_scope"), "run-scoped-inferred")
            self.assertEqual(meta.get("transcript_scope_anchor"), "exp_root-ancestor")
            # inferred scope is never billed as a complete/final usage record.
            self.assertFalse(meta.get("complete"))
            self.assertIn("INFERRED", self._md(res))

    def test_after_return_uses_the_runs_own_eval_dir(self):
        with tempfile.TemporaryDirectory(prefix="geak_report_p2_") as tmp:
            home, eval_dir, _rp, _rec = self._native_home(tmp, with_result=True)
            res = self._scope_from(eval_dir, home)
            self.assertEqual(res["transcript_scope"], "run-scoped")
            # own eval-dir on record -> no weaker exp_root anchor is reported
            self.assertNotIn("transcript_scope_anchor", res)

    def test_completed_sibling_never_owns_an_absent_target(self):
        """Astra r3 blocker: a COMPLETED sibling run (runB) that shares the target's
        exp_root but KNOWS its own, different eval_dir (runB) must never be promoted
        to the enclosing dispatcher of an absent target (runA). Reporting runA must
        NOT bill runB's completed transcript as runA's run-scoped-complete usage."""
        with tempfile.TemporaryDirectory(prefix="geak_report_sib_") as tmp:
            home = os.path.join(tmp, "home", ".claude")
            exp = os.path.join(tmp, "exp")
            target = os.path.join(exp, "runA", "task")   # requested; NO record
            other = os.path.join(exp, "runB", "task")    # completed sibling
            os.makedirs(target)
            os.makedirs(other)
            sess = os.path.join(home, "projects", "proj", "B")
            wfdir = os.path.join(sess, "workflows"); os.makedirs(wfdir)
            record = {"runId": "wf_B", "timestamp": "2026-09-16T01:00:00Z",
                      "status": "completed",
                      "args": {"exp_root": exp},
                      "result": {"eval_dir": other}}
            with open(os.path.join(wfdir, "wf_B.json"), "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            rundir = os.path.join(sess, "subagents", "workflows", "wf_B")
            os.makedirs(rundir)
            write_transcript(os.path.join(rundir, "agent-B.jsonl"), [
                user_rec(prompt_for("director", "setup", other), 9),
                asst_rec(10, "msg_from_B", read=1000, out=10, text="ok"),
            ])
            # Resolver must NOT select the sibling for the absent target.
            with mock.patch.object(M, "candidate_homes",
                                   return_value=[Path(home)]):
                info = M.resolve_run_scope([Path(home)], eval_dir=target)
            self.assertEqual(info["scope"], "unresolved")
            self.assertEqual(info["globs"], [])
            self.assertFalse(info["complete"])
            # And the driver never reports runA as run-scoped-complete off runB.
            res = self._scope_from(target, home)
            self.assertNotIn(res.get("transcript_scope"),
                             ("run-scoped", "run-scoped-inferred"))
            if res.get("status") == "ok":
                self.assertFalse(self._meta(target).get("complete"))


if __name__ == "__main__":
    unittest.main()
