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


def _journal_only(root, session, run_id, journal_lines, *, agents=1,
                  enc="-home-aditysin-PROJECTS-GEAK"):
    """A workflow invocation with NO record: still running, or killed before it
    returned. Only its journal and agent transcripts exist."""
    run_dir = Path(root) / "projects" / enc / session / "subagents" / "workflows" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "journal.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in journal_lines), encoding="utf-8")
    for i in range(agents):
        (run_dir / ("agent-%d.jsonl" % i)).write_text("{}\n", encoding="utf-8")
    return Path(root)


def _setup_result(eval_dir):
    return {"type": "result", "key": "k", "agentId": "a0",
            "result": {"eval_dir": eval_dir, "model_name": "m"}}


class TestOwnedInvocations(unittest.TestCase):
    """A run is every invocation whose OWN eval-dir is this one (2026-09-25).

    The gpt-oss-120b run of 2026-09-24 was an 18-hour original, killed before it could write
    its record, plus a 53-minute ``phases: final`` re-entry that did write one. Scoping to the
    newest record kept the re-entry alone: 103 of 2,270 calls.
    """

    def test_a_killed_original_and_its_re_entry_are_both_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/e2e_run"
            home = _home(tmp, "sess", "wf_final", ev, key="args")
            _journal_only(tmp, "sess", "wf_orig", [_setup_result(ev)], agents=3)
            info = M.resolve_run_scope([home], eval_dir=ev)
            self.assertEqual(len(info["globs"]), 2)
            self.assertEqual({i["run_id"]: i["evidence"] for i in info["invocations"]},
                             {"wf_final": "record", "wf_orig": "journal"})
            # One invocation's ownership rests on its journal alone: usable, never complete.
            self.assertEqual(info["scope"], "run-scoped-inferred")
            self.assertFalse(info["complete"])
            self.assertEqual(info["top_anchor"], "record+journal")
            self.assertTrue(any("wf_orig" in w for w in info["warnings"]))

    def test_two_recorded_invocations_are_a_complete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/e2e_run"
            home = _home(tmp, "sess", "wf_one", ev, key="args")
            _home(tmp, "sess", "wf_two", ev, key="result")
            info = M.resolve_run_scope([home], eval_dir=ev)
            self.assertEqual(info["scope"], "run-scoped")
            self.assertTrue(info["complete"])
            self.assertEqual(sorted(i["run_id"] for i in info["invocations"]), ["wf_one", "wf_two"])

    def test_a_mere_mention_of_the_path_is_not_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/e2e_run"
            _journal_only(tmp, "sess", "wf_reader", [
                {"type": "result", "result": {"note": "read %s/final_report.md" % ev}}])
            self.assertEqual(M.owned_invocations([Path(tmp)], ev), [])

    def test_a_child_eval_dir_does_not_own_its_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/e2e_run"
            _journal_only(tmp, "sess", "wf_lane",
                          [_setup_result(ev + "/kernels/_exp/team_x/task")])
            self.assertEqual(M.owned_invocations([Path(tmp)], ev), [])

    def test_an_invocation_without_transcripts_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/e2e_run"
            _journal_only(tmp, "sess", "wf_empty", [_setup_result(ev)], agents=0)
            self.assertEqual(M.owned_invocations([Path(tmp)], ev), [])

    def test_the_same_invocation_in_two_homes_is_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/e2e_run"
            live = _journal_only(os.path.join(tmp, "live"), "sess", "wf_orig", [_setup_result(ev)])
            mirror = _journal_only(os.path.join(tmp, "mirror"), "sess", "wf_orig", [_setup_result(ev)])
            found = M.owned_invocations([live, mirror], ev)
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0]["glob"].startswith(str(live)))


class TestTheMirrorPreservesEveryOwnedInvocation(unittest.TestCase):
    """What the report counts, the mirror must keep (2026-09-25).

    ``resolve_run_scope`` has counted every owning invocation since 824db57, but the
    mirror still copied the single invocation ``find_record`` returned. A run that
    resolved to two invocations LIVE therefore rebuilt from the mirror as one -- and
    called itself complete. The original's transcripts were lost with the live home,
    which is the one thing a durable mirror exists to prevent.
    """

    def _run(self, tmp):
        """A killed original (journal only) plus the re-entry that wrote the record."""
        ev = os.path.join(tmp, "eval")
        os.makedirs(ev, exist_ok=True)
        home = _home(tmp, "sess", "wf_final", ev, key="args", agents=2)
        _journal_only(tmp, "sess", "wf_orig", [_setup_result(ev)], agents=3)
        return ev, home

    def test_a_killed_original_is_mirrored_beside_its_re_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev, home = self._run(tmp)
            out = M.mirror_run_trace(ev, homes=[home])
            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["invocations"], 2)
            self.assertTrue(out["coverage"]["complete"])
            self.assertEqual(out["coverage"]["incomplete"], [])
            mirror = Path(ev) / M.MIRROR_DIRNAME
            for run_id, agents in (("wf_final", 2), ("wf_orig", 3)):
                got = sorted((mirror / "projects" / "-home-aditysin-PROJECTS-GEAK" / "sess"
                              / "subagents" / "workflows" / run_id).glob("agent-*.jsonl"))
                self.assertEqual(len(got), agents, run_id)

    def test_the_mirror_alone_still_resolves_both_invocations(self):
        """The durability claim itself: rebuild with the live home gone."""
        with tempfile.TemporaryDirectory() as tmp:
            ev, home = self._run(tmp)
            M.mirror_run_trace(ev, homes=[home])
            mirror = Path(ev) / M.MIRROR_DIRNAME
            info = M.resolve_run_scope([mirror], eval_dir=ev)
            self.assertEqual(
                sorted(i["run_id"] for i in info["invocations"]), ["wf_final", "wf_orig"])

    def test_a_session_shared_by_two_invocations_is_copied_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev, home = self._run(tmp)
            man = M.mirror_run_trace(ev, homes=[home])
            paths = [f["path"] for f in
                     json.loads((Path(ev) / M.MIRROR_DIRNAME / M.MANIFEST_NAME)
                                .read_text(encoding="utf-8"))["files"]]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(man["coverage"]["captured"], 2)

    def test_an_invocation_whose_transcripts_are_gone_is_named_not_dropped(self):
        """A nonempty subset must not certify the whole known scope as complete."""
        with tempfile.TemporaryDirectory() as tmp:
            ev, home = self._run(tmp)
            sites = M._owned_sites([home], ev)
            self.assertEqual(len(sites), 2)
            sites.append({"run_id": "wf_lost", "evidence": M.EVIDENCE_JOURNAL,
                          "record_path": None,
                          "session_dir": Path(tmp) / "projects" / "gone" / "sess"})
            man = M.mirror_invocations(sites, Path(ev) / M.MIRROR_DIRNAME)
            self.assertEqual(man["coverage"]["invocations"], 3)
            self.assertFalse(man["coverage"]["complete"])
            self.assertEqual(man["coverage"]["incomplete"], ["wf_lost"])
            lost = [e for e in man["invocations"] if e["run_id"] == "wf_lost"][0]
            self.assertEqual(lost["status"], "no_transcripts")


class TestOwnershipIsReadFromAStatedResult(unittest.TestCase):
    """A journal owns an eval-dir only where an agent RETURNED it as its own.

    The rule used to be a text match for ``"eval_dir": "<path>"`` anywhere in the
    file. A row can carry that text about something it is merely INSPECTING -- a
    comparison target, a diagnostic subject -- and the bytes are identical to an
    owner's, so the match billed unrelated invocations to this run (2026-09-25).
    """

    def test_an_eval_dir_nested_under_another_key_is_not_a_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/subject/task"
            home = _journal_only(tmp, "sess-x", "wf_looker", [
                {"type": "debug", "comparison_target": {"eval_dir": ev}}], agents=2)
            self.assertEqual(M.owned_invocations([home], ev), [])
            self.assertEqual(M._owned_sites([home], ev), [])

    def test_a_result_row_owns_only_its_own_direct_eval_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/subject/task"
            home = _journal_only(tmp, "sess-y", "wf_wrapper", [
                {"type": "result", "key": "k", "agentId": "a0",
                 "result": {"eval_dir": "/runs/exp/other/task",
                            "comparison_target": {"eval_dir": ev}}}], agents=2)
            self.assertEqual(M.owned_invocations([home], ev), [])

    def test_a_mention_is_reported_rather_than_passed_over_in_silence(self):
        """Evidence we refuse to act on is not the same as no evidence."""
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "eval")
            os.makedirs(ev, exist_ok=True)
            home = _home(tmp, "sess-own", "wf_top", ev, agents=2)
            _journal_only(tmp, "sess-x", "wf_looker", [
                {"type": "debug", "comparison_target": {"eval_dir": ev}}], agents=1)
            scope = M.resolve_run_scope([home], eval_dir=ev)
            self.assertEqual([i["run_id"] for i in scope["invocations"]], ["wf_top"])
            self.assertIn("wf_looker", " ".join(scope["warnings"]))
            self.assertIn("mention", " ".join(scope["warnings"]))

    def test_the_agent_that_declared_it_travels_with_the_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/live/task"
            home = _journal_only(tmp, "sess-live", "wf_live", [
                {"type": "result", "key": "baseline", "agentId": "a-77",
                 "result": {"eval_dir": ev}}], agents=2)
            scope = M.resolve_run_scope([home], eval_dir=ev)
            inv = scope["invocations"][0]
            self.assertEqual(inv["evidence"], M.EVIDENCE_JOURNAL)
            self.assertEqual((inv["owner_key"], inv["owner_agent"]), ("baseline", "a-77"))


class TestTheInventoryKeepsWhatThePublicViewFilters(unittest.TestCase):
    """One inventory, two views. The private one must not drop the holes."""

    def test_a_declared_owner_without_transcripts_stays_in_the_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/empty/task"
            home = _journal_only(tmp, "sess-e", "wf_empty",
                                 [_setup_result(ev)], agents=0)
            sites = M._owned_sites([home], ev)
            self.assertEqual([(s["run_id"], s["files_exist"]) for s in sites],
                             [("wf_empty", False)])
            # the public view still refuses to hand out an unusable glob
            self.assertEqual(M.owned_invocations([home], ev), [])

    def test_a_journal_only_owner_makes_the_run_partial_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "eval")
            os.makedirs(ev, exist_ok=True)
            home = _home(tmp, "sess", "wf_final", ev, agents=2)
            _journal_only(tmp, "sess", "wf_orig", [_setup_result(ev)], agents=0)
            scope = M.resolve_run_scope([home], eval_dir=ev)
            self.assertFalse(scope["complete"])
            self.assertEqual(scope["scope"], "partial")
            self.assertIn("wf_orig", " ".join(scope["warnings"]))


class TestTranscriptsAreCountedOnlyWhereThereIsEvidence(unittest.TestCase):
    """A file under the workflow dir is not automatically a transcript."""

    def _sites(self, tmp, ev, run_id, session="sess"):
        return M._owned_sites([Path(tmp)], ev)

    def test_a_journal_is_not_itself_a_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "eval")
            os.makedirs(ev, exist_ok=True)
            home = _journal_only(tmp, "sess", "wf_j", [_setup_result(ev)], agents=0)
            man = M.mirror_invocations(M._owned_sites([home], ev),
                                       Path(ev) / M.MIRROR_DIRNAME)
            entry = man["invocations"][0]
            self.assertEqual(entry["transcripts"], 0)
            self.assertEqual(entry["status"], "no_transcripts")
            self.assertFalse(man["coverage"]["complete"])

    def test_a_zero_byte_transcript_is_not_usable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "eval")
            os.makedirs(ev, exist_ok=True)
            home = _journal_only(tmp, "sess", "wf_z", [_setup_result(ev)], agents=1)
            blank = (Path(home) / "projects" / "-home-aditysin-PROJECTS-GEAK" / "sess"
                     / "subagents" / "workflows" / "wf_z" / "agent-0.jsonl")
            blank.write_text("", encoding="utf-8")
            man = M.mirror_invocations(M._owned_sites([home], ev),
                                       Path(ev) / M.MIRROR_DIRNAME)
            entry = man["invocations"][0]
            self.assertEqual(entry["transcripts"], 0)
            self.assertEqual([p.rsplit("/", 1)[-1] for p in entry["transcripts_empty"]],
                             ["agent-0.jsonl"])
            self.assertEqual(entry["status"], "unusable_transcripts")
            self.assertFalse(man["coverage"]["complete"])


class TestTheMirrorPreservesEveryInstanceTheReportCounts(unittest.TestCase):
    """What the resolver scopes and what the copier keeps must be one set.

    The resolver learned to union a run's nested lanes; the copier kept mirroring
    the top eval-dir alone. The report therefore counted lane calls that were
    never copied, and once the mirror was the only surviving source those calls
    were simply gone (2026-09-25).
    """

    def _run(self, tmp):
        ev = os.path.join(tmp, "eval")
        lane = os.path.join(ev, "kernels", "_exp", "lane1", "task")
        os.makedirs(os.path.join(ev, "reports", "trace"), exist_ok=True)
        with open(os.path.join(ev, "reports", "trace", "agent_timeline.json"),
                  "w", encoding="utf-8") as fh:
            json.dump({"events": [], "nested": [{"instance": lane, "nested": []}]}, fh)
        home = _home(tmp, "sess", "wf_top", ev, agents=2)
        _home(tmp, "sess-lane", "wf_lane", lane, agents=3)
        return ev, lane, home

    def test_a_nested_lanes_invocation_is_mirrored_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev, lane, home = self._run(tmp)
            out = M.mirror_run_trace(ev, homes=[home])
            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["invocations"], 2)
            got = sorted((Path(ev) / M.MIRROR_DIRNAME / "projects"
                          / "-home-aditysin-PROJECTS-GEAK" / "sess-lane" / "subagents"
                          / "workflows" / "wf_lane").glob("agent-*.jsonl"))
            self.assertEqual(len(got), 3)

    def test_the_mirror_alone_still_carries_the_lanes_calls(self):
        """The durability claim, proved with the live home out of the picture."""
        with tempfile.TemporaryDirectory() as tmp:
            ev, lane, home = self._run(tmp)
            M.mirror_run_trace(ev, homes=[home])
            mirror = Path(ev) / M.MIRROR_DIRNAME
            rebuilt = M.resolve_run_scope([mirror], eval_dir=ev, nested_eval_dirs=[lane])
            self.assertEqual(sorted(i["run_id"] for i in rebuilt["invocations"]), ["wf_top"])
            self.assertTrue(rebuilt["complete"])
            self.assertEqual(
                sorted(g.split("workflows/")[-1].split("/")[0] for g in rebuilt["globs"]),
                ["wf_lane", "wf_top"])

    def test_the_caller_can_still_pin_the_lanes_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev, lane, home = self._run(tmp)
            out = M.mirror_run_trace(ev, homes=[home], nested_eval_dirs=())
            self.assertEqual(out["invocations"], 1)


if __name__ == "__main__":
    unittest.main()


class TestOwnershipIsOneStrictRule(unittest.TestCase):
    """A journal MENTIONING a path is not a journal OWNING it (2026-09-25).

    ``owned_invocations`` always demanded an ``"eval_dir": "<path>"`` field, but
    ``resolve_run_scope`` used to re-admit the rejects through a second, looser
    anchor that accepted any whole-path mention — so a scope the strict rule had
    refused came back as ``run-scoped-inferred`` with ``owned=True``. These pin
    the single rule: mention, descendant, and record-contradicted journals are
    all out, and the only thing left that a live run needs (its own eval_dir
    field, written before its record exists) is still in.
    """

    def _scope(self, home, ev, **kw):
        return M.resolve_run_scope([home], eval_dir=ev, **kw)

    def test_a_journal_that_merely_mentions_the_eval_dir_owns_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/mention/task"
            home = _journal_only(tmp, "sess", "wf_mention", [
                {"type": "agent", "prompt": "write %s/final_report.md" % ev}])
            scope = self._scope(home, ev)
            self.assertEqual(scope["scope"], "unresolved")
            self.assertEqual(scope["globs"], [])
            self.assertEqual(scope["invocations"], [])

    def test_a_journal_naming_a_descendant_owns_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/desc/task"
            home = _journal_only(tmp, "sess", "wf_desc", [
                _setup_result(ev + "/child")])
            self.assertEqual(self._scope(home, ev)["scope"], "unresolved")

    def test_a_live_runs_own_eval_dir_field_still_owns_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/live/task"
            home = _journal_only(tmp, "sess", "wf_live", [_setup_result(ev)], agents=2)
            scope = self._scope(home, ev)
            self.assertEqual(scope["scope"], "run-scoped-inferred")
            self.assertEqual(scope["top_anchor"], "live-journal")
            self.assertEqual([i["run_id"] for i in scope["invocations"]], ["wf_live"])

    def test_a_journal_is_not_adopted_over_its_own_records_verdict(self):
        # wf_other's record assigns it to ANOTHER run; its journal happens to
        # carry this eval-dir too. The record decides, and the disagreement is
        # reported -- not silently resolved in the journal's favour and then
        # warned about as "no workflow record".
        with tempfile.TemporaryDirectory() as tmp:
            ev, mine = "/runs/exp/conflict/task", "/runs/exp/conflict/owner"
            home = _home(tmp, "sess-mine", "wf_mine", mine, agents=2)
            _home(tmp, "sess-other", "wf_other", "/runs/elsewhere/task", agents=2)
            journal = (home / "projects" / "-home-aditysin-PROJECTS-GEAK" / "sess-other"
                       / "subagents" / "workflows" / "wf_other" / "journal.jsonl")
            journal.write_text(json.dumps(_setup_result(mine)) + "\n", encoding="utf-8")
            scope = self._scope(home, mine)
            self.assertEqual([i["run_id"] for i in scope["invocations"]], ["wf_mine"])
            joined = " ".join(scope["warnings"])
            self.assertIn("wf_other", joined)
            self.assertIn("/runs/elsewhere/task", joined)
            self.assertNotIn("no workflow record", joined)


class TestCoverageCountsEveryOwnerNotJustTheUsableOnes(unittest.TestCase):
    """Complete means every owned invocation is accounted for (2026-09-25).

    Two holes let a strict subset certify the whole: nested lanes resolved by
    "newest record wins" instead of by ownership, so a lane's other invocations
    vanished; and an owner whose transcripts were missing was dropped from the
    owned set entirely, leaving the survivors to report ``complete=True``.
    """

    def test_a_lane_with_two_invocations_contributes_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, lane = "/runs/exp/top/task", "/runs/exp/top/lane"
            home = _home(tmp, "sess-top", "wf_top", top, agents=2)
            _home(tmp, "sess-l1", "wf_lane1", lane, agents=2, timestamp="2026-09-24T00:00:00Z")
            _home(tmp, "sess-l2", "wf_lane2", lane, agents=3, timestamp="2026-09-25T00:00:00Z")
            scope = M.resolve_run_scope([home], eval_dir=top, nested_eval_dirs=[lane])
            self.assertEqual(scope["scope"], "run-scoped")
            self.assertTrue(scope["complete"])
            self.assertEqual(
                sorted(g.split("workflows/")[-1].split("/")[0] for g in scope["globs"]),
                ["wf_lane1", "wf_lane2", "wf_top"])

    def test_an_owner_without_transcripts_makes_the_run_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = "/runs/exp/hole/task"
            home = _home(tmp, "sess-kept", "wf_kept", ev, agents=2)
            _home(tmp, "sess-gone", "wf_gone", ev, agents=0)
            scope = M.resolve_run_scope([home], eval_dir=ev)
            self.assertEqual(scope["scope"], "partial")
            self.assertFalse(scope["complete"])
            self.assertEqual(scope["missing"], [ev])
            self.assertIn("wf_gone", " ".join(scope["warnings"]))
            # the usable one is still scoped -- a hole is reported, not a blackout
            self.assertEqual([i["run_id"] for i in scope["invocations"]], ["wf_kept"])
