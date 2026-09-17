"""Tests for run-scoped transcript discovery (claude_trace_mirror).

These lock the contamination fix: the report ledger must bill a run for the
transcripts THAT RUN OWNS and nothing else. The old path is substring discovery
in llm_ledger — any concurrent session whose transcript merely mentions the
eval-dir path gets attributed to the run, inflating call count and cost (a real
kernel run rendered 1,239 calls / $144 instead of its true 500 / $47). Scoping
to the run's own ``subagents/workflows/<runId>/agent-*.jsonl`` removes that class
of error by construction; that is what these tests pin.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                      # interface/

import claude_trace_mirror as M  # noqa: E402


def _home(root, session, run_id, eval_dir, *, agents=2, key="args",
          holder_key="eval_dir", exp_root=None, timestamp=None,
          enc="-home-aditysin-PROJECTS-GEAK"):
    """Materialize a minimal Claude home with one workflow record and its
    subagent transcripts. Returns the home Path.

        <home>/projects/<enc>/<session>/workflows/wf_<run_id>.json
        <home>/projects/<enc>/<session>/subagents/workflows/<run_id>/agent-N.jsonl

    ``exp_root`` (added under ``args`` alongside whatever ``key``/``holder_key``
    place) and ``timestamp`` let a record carry a DISTINCT experiment-root field
    and a recorded time — needed to exercise field provenance (an exp_root that
    equals another run's eval_dir) and the newest-wins tie-break honestly.
    """
    home = Path(root)
    sess = home / "projects" / enc / session
    (sess / "workflows").mkdir(parents=True, exist_ok=True)
    record = {"runId": run_id, key: {holder_key: eval_dir}}
    if exp_root is not None:
        record.setdefault("args", {})["exp_root"] = exp_root
    if timestamp is not None:
        record["timestamp"] = timestamp
    (sess / "workflows" / ("wf_%s.json" % run_id)).write_text(
        json.dumps(record), encoding="utf-8")
    run_dir = sess / "subagents" / "workflows" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for i in range(agents):
        (run_dir / ("agent-%d.jsonl" % i)).write_text("{}\n", encoding="utf-8")
        # a sibling .output that MUST NOT be swept in by the glob
        (run_dir / ("agent-%d.output" % i)).write_text("symlink-ish\n", encoding="utf-8")
    return home


class TestRunScopedGlobs(unittest.TestCase):
    def test_resolves_to_the_runs_own_subagent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/team_task_A/task"
            home = _home(tmp, "sess-A", "wf_aaa-111", ev, agents=3)
            globs = M.run_transcript_globs([home], eval_dir=ev)
            self.assertEqual(len(globs), 1)
            self.assertIn("subagents/workflows/wf_aaa-111/agent-*.jsonl", globs[0])
            import glob as G
            self.assertEqual(len(G.glob(globs[0])), 3)   # the 3 agent-*.jsonl

    def test_glob_pattern_excludes_output_symlinks(self):
        # agent-*.jsonl, NOT agent-* — the .output siblings must never match, or
        # every call would be counted twice (once via the transcript, once via
        # the symlink that points back at it).
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/team_task_B/task"
            home = _home(tmp, "sess-B", "wf_bbb-222", ev, agents=2)
            globs = M.run_transcript_globs([home], eval_dir=ev)
            import glob as G
            hits = G.glob(globs[0])
            self.assertTrue(all(h.endswith(".jsonl") for h in hits))
            self.assertFalse(any(h.endswith(".output") for h in hits))

    def test_foreign_session_is_not_swept_in(self):
        # A second session touched the same eval-dir (an interactive debugging
        # session) and has its own agent transcripts, but owns no workflow record
        # for this run. Scoping resolves the record's session ONLY, so the
        # foreign session's dir never appears in the globs.
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/team_task_C/task"
            home = _home(tmp, "sess-real", "wf_ccc-333", ev, agents=2)
            # foreign session: transcripts under a DIFFERENT runId, no record here
            foreign = (home / "projects" / "-home-aditysin-PROJECTS-GEAK"
                       / "sess-foreign" / "subagents" / "workflows" / "wf_zzz-999")
            foreign.mkdir(parents=True, exist_ok=True)
            (foreign / "agent-0.jsonl").write_text("{}\n", encoding="utf-8")
            globs = M.run_transcript_globs([home], eval_dir=ev)
            self.assertEqual(len(globs), 1)
            self.assertIn("sess-real", globs[0])
            self.assertNotIn("sess-foreign", globs[0])
            self.assertNotIn("wf_zzz-999", globs[0])

    def test_unresolvable_record_returns_empty_for_fallback(self):
        # No record names this eval-dir -> [] so the caller falls back to the
        # old substring discovery rather than silently emitting an empty ledger.
        with tempfile.TemporaryDirectory() as tmp:
            _home(tmp, "sess-D", "wf_ddd-444", "/runs/other/task")
            globs = M.run_transcript_globs([Path(tmp)], eval_dir="/runs/absent/task")
            self.assertEqual(globs, [])

    def test_nested_lanes_are_unioned_and_deduped(self):
        # A dispatcher run whose timeline nests lane eval-dirs: each lane's own
        # record resolves its own runId dir; the union is returned, deduped.
        with tempfile.TemporaryDirectory() as tmp:
            top = "/runs/exp/dispatch/task"
            laneA = "/runs/exp/dispatch/laneA/task"
            home = _home(tmp, "sess-top", "wf_top-000", top)
            _home(tmp, "sess-laneA", "wf_lane-a01", laneA)
            globs = M.run_transcript_globs(
                [home], eval_dir=top, nested_eval_dirs=[laneA, laneA])  # dup on purpose
            self.assertEqual(len(globs), 2)                     # deduped
            joined = "\n".join(globs)
            self.assertIn("wf_top-000", joined)
            self.assertIn("wf_lane-a01", joined)

    def test_result_holder_is_matched_too(self):
        # The eval-dir can live under result{} instead of args{} (a finished run).
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/team_task_E/task"
            home = _home(tmp, "sess-E", "wf_eee-555", ev, key="result")
            globs = M.run_transcript_globs([home], eval_dir=ev)
            self.assertEqual(len(globs), 1)
            self.assertIn("wf_eee-555", globs[0])


class TestResolveRunScopeCoverage(unittest.TestCase):
    """Astra re-review counterexamples: the scope resolver must report coverage
    honestly — never adopt a sibling, never omit the parent, never call a
    lane-only slice a whole run, never mark a run complete with a missing lane."""

    def test_sibling_sharing_exp_root_is_not_selected(self):
        # Two lanes genuinely share an exp_root (each record carries args.exp_root
        # == the shared root AND result.eval_dir == its own lane), and the sibling
        # B is recorded LATER. Resolving A by eval-dir must still pick A's own
        # runId dir: A's result.eval_dir is a SAME-FIELD exact match (rank 0),
        # while B relates to A only through the shared exp_root (ANCESTOR) — so
        # neither B's newer timestamp nor the shared root can steal the selection.
        with tempfile.TemporaryDirectory() as tmp:
            exp = "/runs/exp/shared"
            evA = exp + "/laneA/task"
            evB = exp + "/laneB/task"
            home = _home(tmp, "sess-A", "wf_A-111", evA, key="result",
                         exp_root=exp, timestamp="2026-09-16T01:00:00Z")
            # Sibling B: same exp_root, a different eval-dir, recorded LATER.
            _home(tmp, "sess-B", "wf_B-222", evB, key="result",
                  exp_root=exp, timestamp="2026-09-16T02:00:00Z")
            info = M.resolve_run_scope([home], eval_dir=evA)
            self.assertEqual(info["scope"], "run-scoped")
            self.assertEqual(len(info["globs"]), 1)
            self.assertIn("wf_A-111", info["globs"][0])
            self.assertNotIn("wf_B-222", info["globs"][0])

    def test_parent_is_included_not_omitted(self):
        # A dispatcher whose lane eval-dir is ACTUALLY CONTAINED in the parent's
        # (child = parent + "/laneA/task"): the returned globs must carry BOTH the
        # parent (top) runId dir and the lane's — the parent is never dropped in
        # favour of the lane alone, even though the lane record's eval-dir is a
        # CHILD of the top the parent record names exactly.
        with tempfile.TemporaryDirectory() as tmp:
            top = "/runs/exp/dispatch"
            lane = top + "/laneA/task"          # genuinely inside top
            home = _home(tmp, "sess-top", "wf_parent-0", top, key="result")
            _home(tmp, "sess-lane", "wf_lane-1", lane, key="result")
            info = M.resolve_run_scope([home], eval_dir=top, nested_eval_dirs=[lane])
            self.assertEqual(info["scope"], "run-scoped")
            self.assertTrue(info["complete"])
            joined = "\n".join(info["globs"])
            self.assertIn("wf_parent-0", joined)
            self.assertIn("wf_lane-1", joined)
            self.assertEqual(info["missing"], [])

    def test_child_exp_root_equal_to_parent_eval_dir_does_not_displace_parent(self):
        # FIELD PROVENANCE: the lane declares args.exp_root == the parent's own
        # eval-dir (a real dispatcher/lane shape) and is recorded LATER. Resolving
        # the parent by eval-dir must select the PARENT's dir (its result.eval_dir
        # is a same-field exact match, rank 0), never the child whose exp_root only
        # equals it through the OTHER field (rank 1). Both dirs end up in scope.
        with tempfile.TemporaryDirectory() as tmp:
            parent = "/runs/exp/e2e_run"
            child = parent + "/lane/task"
            home = _home(tmp, "sess-top", "wf_parent-0", parent, key="result",
                         timestamp="2026-09-16T01:00:00Z")
            _home(tmp, "sess-lane", "wf_lane-1", child, key="result",
                  exp_root=parent, timestamp="2026-09-16T02:00:00Z")
            info = M.resolve_run_scope([home], eval_dir=parent, nested_eval_dirs=[child])
            self.assertEqual(info["scope"], "run-scoped")
            self.assertTrue(info["complete"])
            joined = "\n".join(info["globs"])
            self.assertIn("wf_parent-0", joined)   # parent NOT displaced
            self.assertIn("wf_lane-1", joined)

    def test_child_exp_root_cannot_stand_in_for_a_missing_parent(self):
        # Same shape, but the parent's own record is ABSENT. The child's exact
        # exp_root == the requested eval-dir is an OTHER-FIELD match and must NOT
        # be accepted as the parent's identity: with no same-field owner for the
        # top, the run is unresolved and the caller falls back — the lane-only
        # slice is never billed as the whole run.
        with tempfile.TemporaryDirectory() as tmp:
            parent = "/runs/exp/e2e_run"
            child = parent + "/lane/task"
            home = _home(tmp, "sess-lane", "wf_only-lane", child, key="result",
                         exp_root=parent, timestamp="2026-09-16T02:00:00Z")
            info = M.resolve_run_scope([home], eval_dir=parent, nested_eval_dirs=[child])
            self.assertEqual(info["scope"], "unresolved")
            self.assertFalse(info["complete"])
            self.assertEqual(info["globs"], [])
            self.assertIn(parent, info["missing"])

    def test_missing_top_returns_unresolved_empty_for_fallback(self):
        # The top-level record is absent; only a lane is present. The resolver must
        # NOT bill the lane-only slice as the whole run — it returns unresolved +
        # empty globs so the caller falls back to substring discovery.
        with tempfile.TemporaryDirectory() as tmp:
            top = "/runs/exp/dispatch/task"          # no record owns this
            lane = "/runs/exp/dispatch/laneA/task"
            home = _home(tmp, "sess-lane", "wf_only-lane", lane)
            info = M.resolve_run_scope([home], eval_dir=top, nested_eval_dirs=[lane])
            self.assertEqual(info["scope"], "unresolved")
            self.assertFalse(info["complete"])
            self.assertEqual(info["globs"], [])
            self.assertIn(top, info["missing"])

    def test_missing_lane_downgrades_to_partial_not_complete(self):
        # Top resolves, but one declared lane has no owning record. The run must be
        # marked PARTIAL / incomplete with the lane tracked as missing — never
        # 'run-scoped' + complete while silently dropping the lane.
        with tempfile.TemporaryDirectory() as tmp:
            top = "/runs/exp/dispatch/task"
            lane_ok = "/runs/exp/dispatch/laneA/task"
            lane_gone = "/runs/exp/dispatch/laneB/task"   # no record
            home = _home(tmp, "sess-top", "wf_parent-0", top)
            _home(tmp, "sess-lane", "wf_lane-1", lane_ok)
            info = M.resolve_run_scope(
                [home], eval_dir=top, nested_eval_dirs=[lane_ok, lane_gone])
            self.assertEqual(info["scope"], "partial")
            self.assertFalse(info["complete"])
            self.assertIn(lane_gone, info["missing"])
            self.assertTrue(info["warnings"])            # a reason is surfaced
            # coverage still carries the resolved slices
            joined = "\n".join(info["globs"])
            self.assertIn("wf_parent-0", joined)
            self.assertIn("wf_lane-1", joined)

    def test_resolved_but_empty_dir_is_not_proof_of_coverage(self):
        # A record owns the eval-dir and has a runId, but the run's subagents dir
        # holds NO agent-*.jsonl files. A nonempty glob STRING is not proof files
        # exist: the top is treated as unresolved -> fallback.
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/empty/task"
            home = _home(tmp, "sess-empty", "wf_empty-9", ev, agents=0)
            info = M.resolve_run_scope([home], eval_dir=ev)
            self.assertEqual(info["scope"], "unresolved")
            self.assertEqual(info["globs"], [])
            self.assertIn(ev, info["missing"])

    def test_missing_timestamp_sorts_last_not_first(self):
        # _neg_ts orders newest-first under an ascending sort; a MISSING stamp must
        # land LAST among equal ranks. A single-char sentinel got this wrong (an
        # inverted real stamp can top the code-point range), so a record with no
        # timestamp used to sort ahead of real ones.
        present = "2026-09-17T00:00:00Z"
        keys = [(None, M._neg_ts(None)), (present, M._neg_ts(present))]
        ordered = [label for label, _ in sorted(keys, key=lambda kv: kv[1])]
        self.assertEqual(ordered, [present, None])   # present first, missing last

    def test_clean_whole_run_is_run_scoped_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/solo/task"
            home = _home(tmp, "sess-solo", "wf_solo-1", ev, agents=2)
            info = M.resolve_run_scope([home], eval_dir=ev)
            self.assertEqual(info["scope"], "run-scoped")
            self.assertTrue(info["complete"])
            self.assertEqual(info["missing"], [])
            self.assertEqual(info["resolved"], [ev])
            self.assertEqual(len(info["globs"]), 1)

    def test_completed_sibling_does_not_own_absent_target(self):
        # Astra r3 blocker: a COMPLETED sibling (runB) that shares the target's
        # exp_root but KNOWS its own, different eval_dir must never be promoted to
        # the enclosing dispatcher of an absent target (runA). A strict-ancestor
        # exp_root match is not ownership when the record's own eval_dir contradicts
        # the request. The sibling must be rejected -> the target stays unresolved.
        with tempfile.TemporaryDirectory() as tmp:
            exp = os.path.join(tmp, "exp")
            target = os.path.join(exp, "runA", "task")   # requested; NO record
            other = os.path.join(exp, "runB", "task")    # completed sibling
            os.makedirs(target)
            home = _home(tmp, "sess-B", "wf_B", other, key="result",
                         holder_key="eval_dir", exp_root=exp,
                         timestamp="2026-09-16T01:00:00Z")
            info = M.resolve_run_scope([home], eval_dir=target)
            self.assertEqual(info["scope"], "unresolved")
            self.assertEqual(info["globs"], [])
            self.assertFalse(info["complete"])

    def test_args_only_ancestor_is_inferred_not_complete(self):
        # The legitimate mid-run case: the enclosing dispatcher's record carries
        # ONLY args.exp_root (result.eval_dir is written at return). Containment
        # here is a real but WEAKER signal — scope is 'run-scoped-inferred',
        # incomplete, the anchor is surfaced, and a warning explains that ownership
        # is inferred, not proven. (Never promote containment to complete ownership.)
        with tempfile.TemporaryDirectory() as tmp:
            exp = os.path.join(tmp, "exp")
            target = os.path.join(exp, "team_task_x", "task")
            os.makedirs(target)
            home = _home(tmp, "sess-top", "wf_top", exp, key="args",
                         holder_key="exp_root", exp_root=exp,
                         timestamp="2026-09-16T01:00:00Z")
            info = M.resolve_run_scope([home], eval_dir=target)
            self.assertEqual(info["scope"], "run-scoped-inferred")
            self.assertFalse(info["complete"])
            self.assertEqual(info["top_anchor"], "exp_root-ancestor")
            self.assertTrue(info["inferred"])
            self.assertTrue(info["warnings"])
            self.assertEqual(len(info["globs"]), 1)
            self.assertIn("wf_top", info["globs"][0])


if __name__ == "__main__":
    unittest.main()
