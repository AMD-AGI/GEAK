#!/usr/bin/env python3
"""Guard tests for the live tracker's wiring into the workflows.

The tracker is only automatic if its launch hook is actually present in both
workflow entry points and the report driver still emits the trace. Those are
three lines in files that are edited constantly for unrelated reasons, so a
refactor can silently switch tracking off and nothing else would fail.

These tests are deliberately about WIRING, not behaviour: they assert the hook
exists, is opt-out rather than opt-in, and cannot take the workflow down with
it. If one fails, tracking has been disconnected -- re-attach it rather than
relaxing the test.
"""

import os
import re
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "interface"))

E2E_JS = os.path.join(REPO, "e2e_workflow", "e2e_workflow.js")
KERNEL_JS = os.path.join(REPO, "kernel_workflow", "kernel_workflow.js")
REPORT_PY = os.path.join(REPO, "interface", "geak_report.py")
COLLECTOR = os.path.join(REPO, "interface", "geak_trace_collector.py")
RENDERER = os.path.join(REPO, "interface", "geak_trace_report.py")


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class LaunchHookPresenceTest(unittest.TestCase):
    """Both lanes must start the tracker themselves."""

    def test_modules_exist(self):
        for path in (COLLECTOR, RENDERER):
            self.assertTrue(os.path.exists(path), "%s is missing" % path)

    def test_e2e_lane_starts_the_tracker(self):
        src = read(E2E_JS)
        self.assertIn("geak_trace_collector.py", src,
                      "e2e workflow no longer launches the execution tracker")
        self.assertIn("--watch", src, "e2e tracker hook is not in live mode")

    def test_kernel_lane_starts_the_tracker(self):
        src = read(KERNEL_JS)
        self.assertIn("geak_trace_collector.py", src,
                      "kernel workflow no longer launches the execution tracker")
        self.assertIn("--watch", src, "kernel tracker hook is not in live mode")

    def test_both_lanes_resolve_by_exp_root_not_substring(self):
        """The hook must pass an identity the resolver can anchor on."""
        for path in (E2E_JS, KERNEL_JS):
            src = read(path)
            self.assertIn("--exp-root", src,
                          "%s tracker hook lost its run identity" % path)

    def test_hooks_are_opt_out_not_opt_in(self):
        """Tracking must be on by default; GEAK_LIVE_TRACE=0 turns it off."""
        for path in (E2E_JS, KERNEL_JS):
            src = read(path)
            self.assertIn("GEAK_LIVE_TRACE", src)
            # Default must be '1' (on), i.e. the env var only disables.
            self.assertRegex(
                src, r"GEAK_LIVE_TRACE\s*\|\|\s*'1'",
                "%s: tracker must default to ON" % path)

    def test_hooks_cannot_kill_the_workflow(self):
        """A tracker failure must never fail the run it observes."""
        for path in (E2E_JS, KERNEL_JS):
            src = read(path)
            # rindex: the first mention is the explanatory comment; the last is
            # the actual spawn argv, which is what must sit inside try/catch.
            idx = src.rindex("geak_trace_collector.py")
            window = src[max(0, idx - 1500):idx + 1200]
            self.assertIn("try {", window,
                          "%s: tracker launch is not wrapped in try/catch" % path)
            self.assertIn("catch", window,
                          "%s: tracker launch has no catch" % path)

    def test_hooks_handle_async_spawn_errors(self):
        """spawn ENOENT is asynchronous: try/catch cannot catch it."""
        for path in (E2E_JS, KERNEL_JS):
            src = read(path)
            idx = src.rindex("geak_trace_collector.py")
            window = src[max(0, idx - 1500):idx + 1600]
            self.assertIn("child.on('error'", window,
                          "%s: missing async spawn-error listener" % path)

    def test_hooks_write_per_run_output(self):
        """Two runs under one exp_root must not overwrite each other's trace."""
        for path in (E2E_JS, KERNEL_JS):
            src = read(path)
            self.assertIn("--out-dir", src,
                          "%s: tracker output is not per-run" % path)

    def test_kernel_hook_does_not_require_exp_root_arg(self):
        """kernel_lane defaults EXP_ROOT, so requiring the arg disabled tracking."""
        src = read(KERNEL_JS)
        idx = src.rindex("geak_trace_collector.py")
        window = src[max(0, idx - 1600):idx]
        self.assertIn("/exp", window,
                      "kernel hook must apply the same EXP_ROOT default as kernel_lane")

    def test_hooks_are_detached_and_do_not_block(self):
        for path in (E2E_JS, KERNEL_JS):
            src = read(path)
            idx = src.rindex("geak_trace_collector.py")
            window = src[max(0, idx - 1500):idx + 1200]
            self.assertIn("detached", window,
                          "%s: tracker must be detached" % path)
            self.assertIn("unref", window,
                          "%s: tracker must not hold the workflow open" % path)
            self.assertNotIn("execFileSync", window,
                             "%s: tracker must not block the workflow" % path)


class ReportDriverWiringTest(unittest.TestCase):
    """The end-of-run path must still emit the execution trace."""

    def test_driver_writes_the_execution_trace(self):
        src = read(REPORT_PY)
        self.assertIn("_write_execution_trace", src)
        self.assertIn("geak_trace_collector", src)
        self.assertIn("geak_trace_report", src)

    def test_trace_is_additive_and_cannot_break_the_ledger_report(self):
        """A trace failure must not lose the caller its ledger report."""
        src = read(REPORT_PY)
        start = src.index("def _write_execution_trace")
        body = src[start:src.index("\ndef ", start + 10)]
        self.assertIn("except Exception", body,
                      "trace writing must swallow its own failures")
        # The call site must come AFTER the ledger report is already built.
        call_idx = src.index("trace_info = _write_execution_trace(")
        self.assertGreater(call_idx, src.index("html_path, md_path = tree.write"),
                           "trace must be written after the ledger report exists")


class DecouplingTest(unittest.TestCase):
    """The collector must stay decoupled from GEAK workflow internals.

    Its inputs are runtime artifacts (journal/meta/transcripts), so adding roles,
    phases or rounds to a workflow must never require touching it. Importing
    workflow modules here would silently re-couple them.
    """

    def test_collector_does_not_import_workflow_code(self):
        """Coupling means importing or pathing into workflow code.

        Naming a workflow in a comment is documentation, not a dependency, so the
        guard targets real coupling: imports and path references.
        """
        src = read(COLLECTOR)
        for banned in ("import kernel_workflow", "from kernel_workflow",
                       "import kernel_lane", "from kernel_lane",
                       "import e2e_workflow", "from e2e_workflow",
                       "kernel_workflow/", "e2e_workflow/", "import roles"):
            self.assertNotIn(banned, src,
                             "collector must not depend on workflow internals (%s)"
                             % banned)

    def test_collector_hardcodes_no_role_names(self):
        """Role/phase vocabulary lives in the run's data, not in the collector."""
        src = read(COLLECTOR).lower()
        for role in ("tech_lead", "profile_engineer", "verify_engineer",
                     "integrator", "benchmark_engineer"):
            self.assertNotIn('"%s"' % role, src)
            self.assertNotIn("'%s'" % role, src)

    def test_renderer_label_coupling_is_confined_and_declared(self):
        """Label parsing is allowed, but only in the documented phase regex."""
        src = read(RENDERER)
        self.assertIn("_ROUND_RE", src)
        # The grouping must be presented as inferred wherever it is shown.
        self.assertIn("inferred", src.lower())

    def test_unparsed_labels_raise_a_warning_not_silence(self):
        import geak_trace_report as R
        agents = [{"agent_id": "a1", "ordinal": 0, "label": "totally_new_label",
                   "description": None, "spawn_depth": 1, "status": "completed",
                   "result_status": "returned_to_workflow", "result": None,
                   "result_preview": None, "result_truncated": False,
                   "transcript_status": "present", "first_ts_ms": 1,
                   "last_ts_ms": 2, "totals": {}, "calls": []},
                  {"agent_id": "a2", "ordinal": 1, "label": "eng r1_d0:compute",
                   "description": None, "spawn_depth": 1, "status": "completed",
                   "result_status": "returned_to_workflow", "result": None,
                   "result_preview": None, "result_truncated": False,
                   "transcript_status": "present", "first_ts_ms": 1,
                   "last_ts_ms": 2, "totals": {}, "calls": []},
                  {"agent_id": "a3", "ordinal": 2, "label": "mystery_step",
                   "description": None, "spawn_depth": 1, "status": "completed",
                   "result_status": "returned_to_workflow", "result": None,
                   "result_preview": None, "result_truncated": False,
                   "transcript_status": "present", "first_ts_ms": 1,
                   "last_ts_ms": 2, "totals": {}, "calls": []}]
        view = R.build_view({"schema": "geak.trace/1", "run": {}, "agents": agents,
                             "edges": [], "warnings": []})
        self.assertTrue(any("did not match any known phase pattern" in w
                            for w in view["warnings"]),
                        "unparsed labels must warn, not silently misfile")

    def test_all_agents_still_render_when_labels_are_unknown(self):
        """A label-format change must degrade grouping only, never lose agents."""
        import geak_trace_report as R
        agents = [{"agent_id": "a%d" % i, "ordinal": i, "label": "unknown_%d" % i,
                   "description": None, "spawn_depth": 1, "status": "completed",
                   "result_status": "returned_to_workflow", "result": None,
                   "result_preview": None, "result_truncated": False,
                   "transcript_status": "present", "first_ts_ms": 1,
                   "last_ts_ms": 2, "totals": {}, "calls": []} for i in range(5)]
        view = R.build_view({"schema": "geak.trace/1", "run": {}, "agents": agents,
                             "edges": [], "warnings": []})
        placed = [aid for p in view["phases"] for aid in p["agents"]]
        self.assertEqual(sorted(placed), sorted(a["agent_id"] for a in agents))
        self.assertEqual(view["totals"]["agents"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
