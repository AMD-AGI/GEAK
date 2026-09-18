#!/usr/bin/env python3
"""Tests for the execution-trace renderer (interface/geak_trace_report.py)."""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geak_trace_report as R  # noqa: E402


def _call(idx=0, out="hi", actions=(), kind="text", cost=0.5, out_tok=10,
          input_kind="new_input_since_previous_response", blocks=None,
          reasoning=None):
    return {
        "call_id": "m%d" % idx, "index": idx, "message_id": "m%d" % idx,
        "model": "claude-opus-4-8", "ts_ms": 1000 + idx, "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "cache_read_input_tokens": 2,
                  "cache_creation_input_tokens": 0, "cache_write_5m_tokens": 0,
                  "cache_write_1h_tokens": 0, "output_tokens": out_tok},
        "input": {"kind": input_kind, "note": "NOT the full API request.",
                  "blocks": blocks if blocks is not None else
                  [{"kind": "text", "text": "go", "truncated": False,
                    "bytes_total": 2}],
                  "source_uuids": []},
        "reasoning": reasoning or {"blocks": 0, "state": "not_captured", "text": ""},
        "output_text": out, "output_truncated": False, "output_bytes_total": len(out),
        "actions": list(actions), "output_kind": kind,
        "cost_usd": cost, "cost_breakdown": {},
    }


def _agent(aid="a1", label="director:setup", ordinal=0, calls=None,
           returned=True, first=1000, last=5000):
    calls = calls if calls is not None else [_call()]
    return {
        "agent_id": aid, "ordinal": ordinal, "label": label,
        "description": "d", "agent_type": "workflow-subagent", "spawn_depth": 1,
        "transcript_status": "present", "transcript": "/x",
        "status": "completed" if returned else "running_or_incomplete",
        "result_status": "returned_to_workflow" if returned else "pending_or_absent",
        "result": {"ok": True} if returned else None,
        "result_preview": '{"ok": true}' if returned else None,
        "result_truncated": False, "result_bytes_total": 12,
        "first_ts_ms": first, "last_ts_ms": last,
        "timing_provenance": "transcript_timestamps_estimated",
        "totals": {"calls": len(calls), "input_tokens": 1,
                   "cache_read_input_tokens": 2, "cache_creation_input_tokens": 0,
                   "output_tokens": sum(c["usage"]["output_tokens"] for c in calls),
                   "cost_usd": sum(c["cost_usd"] for c in calls),
                   "actions": sum(len(c["actions"]) for c in calls)},
        "calls": calls,
    }


def _trace(agents, warnings=()):
    return {
        "schema": "geak.trace/1",
        "run": {"run_id": "wf_test", "status": "complete", "origin_ts_ms": 1000,
                "end_ts_ms": 9000, "elapsed_ms_est": 8000,
                "timing_note": "offset, not a sum"},
        "agents": agents,
        "edges": [{"type": "orchestration", "from": "run:wf_test",
                   "to": "agent:%s" % a["agent_id"], "proven": True}
                  for a in agents],
        "warnings": list(warnings),
    }


class PhaseGroupingTest(unittest.TestCase):
    def test_engineer_round_tag_with_underscore_is_matched(self):
        """Regression: r1_d0 must land in Round 1, not a catch-all bucket."""
        self.assertIsNotNone(R._ROUND_RE.search("eng r1_d0:compute"))
        self.assertEqual(R._ROUND_RE.search("eng r1_d0:compute").group(1), "1")

    def test_round_variants_all_resolve(self):
        for label, want in [("verify r1_d0", "1"), ("reprofile r2", "2"),
                            ("tech_lead:plan r3", "3"), ("clock pre-r1", "1"),
                            ("integrate r10", "10")]:
            m = R._ROUND_RE.search(label)
            self.assertIsNotNone(m, label)
            self.assertEqual(m.group(1), want, label)

    def test_full_round_membership(self):
        agents = [
            _agent("a0", "director:setup", 0),
            _agent("a1", "tech_lead:plan r1", 1),
            _agent("a2", "eng r1_d0:compute", 2),
            _agent("a3", "verify r1_d0", 3),
            _agent("a4", "integrate r1", 4),
            _agent("a5", "tech_lead:report", 5),
        ]
        view = R.build_view(_trace(agents))
        phases = {p["name"]: p["agents"] for p in view["phases"]}
        self.assertEqual(phases["Setup"], ["a0"])
        self.assertEqual(phases["Round 1"], ["a1", "a2", "a3", "a4"])
        self.assertIn("a5", phases["Finalize"])

    def test_every_agent_appears_exactly_once(self):
        agents = [_agent("a%d" % i, "eng r%d_d0:x" % (i % 3), i) for i in range(9)]
        view = R.build_view(_trace(agents))
        placed = [aid for p in view["phases"] for aid in p["agents"]]
        self.assertEqual(sorted(placed), sorted(a["agent_id"] for a in agents))
        self.assertEqual(len(placed), len(set(placed)))

    def test_journal_order_preserved_within_phase(self):
        agents = [_agent("a1", "eng r1_d0", 0), _agent("a2", "eng r1_d1", 1),
                  _agent("a3", "verify r1_d0", 2)]
        view = R.build_view(_trace(agents))
        self.assertEqual(view["phases"][0]["agents"], ["a1", "a2", "a3"])


class TotalsTest(unittest.TestCase):
    def test_totals_sum_across_agents(self):
        agents = [_agent("a1", "x", 0, [_call(0, cost=1.0, out_tok=5)]),
                  _agent("a2", "y", 1, [_call(0, cost=2.0, out_tok=7)])]
        t = R.build_view(_trace(agents))["totals"]
        self.assertEqual(t["agents"], 2)
        self.assertEqual(t["calls"], 2)
        self.assertAlmostEqual(t["cost_usd"], 3.0)
        self.assertEqual(t["output_tokens"], 12)

    def test_timeline_offsets_are_relative_to_origin(self):
        agents = [_agent("a1", "x", 0, first=1000, last=3000),
                  _agent("a2", "y", 1, first=2000, last=9000)]
        view = R.build_view(_trace(agents))
        a1, a2 = view["agents"]
        self.assertEqual((a1["start_off"], a1["end_off"]), (0, 2000))
        self.assertEqual((a2["start_off"], a2["end_off"]), (1000, 8000))
        # Overlapping spans: the sum of durations exceeds the run elapsed.
        self.assertGreater(a1["dur"] + a2["dur"],
                           view["run"]["elapsed_ms_est"])

    def test_missing_timing_stays_none_not_zero(self):
        a = _agent("a1", "x", 0, first=None, last=None)
        view = R.build_view(_trace([a]))
        self.assertIsNone(view["agents"][0]["start_off"])
        self.assertIsNone(view["agents"][0]["dur"])


class RenderHtmlTest(unittest.TestCase):
    def test_renders_and_contains_all_three_views(self):
        out = R.render_html(R.build_view(_trace([_agent()])))
        for marker in ('data-v="tree"', 'data-v="timeline"', 'data-v="graph"'):
            self.assertIn(marker, out)

    def test_script_breakout_is_neutralized(self):
        payload = "</script><img src=x onerror=alert(1)>"
        call = _call(blocks=[{"kind": "text", "text": payload,
                              "truncated": False, "bytes_total": 9}])
        out = R.render_html(R.build_view(_trace([_agent(calls=[call])])))
        data = out.split("window.__TRACE__ = ", 1)[1].split("</script>", 1)[0]
        self.assertNotIn("</", data)
        self.assertIn("\\u003c", data)
        # Only the two real closing script tags exist.
        self.assertEqual(out.count("</script>"), 2)

    def test_label_injection_is_escaped_in_markup(self):
        out = R.render_html(R.build_view(_trace([_agent(label="<b>x</b>")])))
        self.assertNotIn("<b>x</b>", out.split("window.__TRACE__")[0])

    def test_input_disclaimer_present(self):
        out = R.render_html(R.build_view(_trace([_agent()])))
        self.assertIn("Not the full API request", out)

    def test_warnings_rendered(self):
        out = R.render_html(R.build_view(_trace([_agent()], ["timing is estimated"])))
        self.assertIn("timing is estimated", out)

    def test_inferred_label_shown_for_phases(self):
        out = R.render_html(R.build_view(_trace([_agent()])))
        self.assertIn("inferred", out.lower())


class RenderMarkdownTest(unittest.TestCase):
    def test_counts_agree_with_html_view(self):
        agents = [_agent("a1", "eng r1_d0", 0), _agent("a2", "verify r1_d0", 1)]
        view = R.build_view(_trace(agents))
        md = R.render_markdown(view)
        self.assertIn("| Agents | 2 (2 returned a result) |", md)
        self.assertIn("| API calls | 2 |", md)

    def test_phase_tree_and_timeline_sections_present(self):
        md = R.render_markdown(R.build_view(_trace([_agent()])))
        self.assertIn("## Phase tree", md)
        self.assertIn("## Timeline", md)
        self.assertIn("## Delegation graph", md)
        self.assertIn("never sum them", md.lower().replace("—", "—"))

    def test_no_result_agent_marked_in_markdown(self):
        md = R.render_markdown(R.build_view(_trace([_agent(returned=False)])))
        self.assertIn("no result", md)

    def test_agent_without_calls_is_explicit(self):
        a = _agent(calls=[])
        a["transcript_status"] = "missing"
        md = R.render_markdown(R.build_view(_trace([a])))
        self.assertIn("No transcript calls recorded", md)


class WriteReportsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-trace-report-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_writes_both_files(self):
        res = R.write_reports(_trace([_agent()]), self.dir)
        self.assertTrue(os.path.exists(res["html"]))
        self.assertTrue(os.path.exists(res["md"]))
        self.assertEqual(res["totals"]["agents"], 1)

    def test_empty_run_renders_without_crashing(self):
        res = R.write_reports(_trace([]), self.dir)
        with open(res["html"], encoding="utf-8") as fh:
            self.assertIn("<!doctype html>", fh.read())

    def test_html_is_valid_json_payload(self):
        res = R.write_reports(_trace([_agent()]), self.dir)
        with open(res["html"], encoding="utf-8") as fh:
            body = fh.read()
        raw = body.split("window.__TRACE__ = ", 1)[1].rsplit(";</script>", 1)[0]
        self.assertEqual(json.loads(raw)["run"]["run_id"], "wf_test")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RecordedLinkageRenderTest(unittest.TestCase):
    """Astra R7 #6: validated transfer/spawn edges must actually be rendered."""

    def _trace_with_links(self):
        t = _trace([_agent("p1", "tech_lead:plan r1", 0),
                    _agent("c1", "eng r1_d0:compute", 1)])
        t["run"]["linkage"] = {"present": True, "complete": True}
        t["edges"] += [
            {"type": "result_supplied_to_dispatch", "from": "agent:p1",
             "to": "agent:c1", "event_id": "t1", "proven": True,
             "producer_result_ref": "result.directions[0]",
             "consumer_input_ref": "dispatch.prompt#offset=120",
             "forwarding": "literal", "transformation": None,
             "transformation_known": False},
            {"type": "agent_spawn", "from": "agent:p1", "to": "agent:c1",
             "spawn_event_id": "s1", "spawn_tool_call_id": "toolu_9",
             "proven": True, "return_status": "returned",
             "attempts": [{"attempt_id": "a1", "status": "returned"}]},
        ]
        return t

    def test_html_renders_both_recorded_edge_types(self):
        out = R.render_html(R.build_view(self._trace_with_links()))
        self.assertIn("Recorded linkage", out)
        self.assertIn("result supplied", out)
        self.assertIn("spawn", out)

    def test_html_carries_the_inspectable_references(self):
        view = R.build_view(self._trace_with_links())
        payload = json.loads(R.render_html(view)
                             .split("window.__TRACE__ = ", 1)[1]
                             .rsplit(";</script>", 1)[0])
        kinds = {e["type"] for e in payload["edges"]}
        self.assertIn("result_supplied_to_dispatch", kinds)
        self.assertIn("agent_spawn", kinds)
        transfer = next(e for e in payload["edges"]
                        if e["type"] == "result_supplied_to_dispatch")
        self.assertEqual(transfer["producer_result_ref"], "result.directions[0]")
        self.assertEqual(payload["linkage"]["present"], True)

    def test_markdown_lists_recorded_edges_with_references(self):
        md = R.render_markdown(R.build_view(self._trace_with_links()))
        self.assertIn("## Recorded linkage", md)
        self.assertIn("result.directions[0]", md)
        self.assertIn("toolu_9", md)
        self.assertIn("a1=returned", md)

    def test_absent_linkage_says_not_recorded_not_none_exist(self):
        t = _trace([_agent()])
        t["run"]["linkage"] = {"present": False,
                               "note": "No recorded linkage events for this run."}
        md = R.render_markdown(R.build_view(t))
        self.assertIn("No recorded linkage events", md)

    def test_incomplete_coverage_is_surfaced(self):
        t = self._trace_with_links()
        t["run"]["linkage"]["complete"] = False
        md = R.render_markdown(R.build_view(t))
        self.assertIn("INCOMPLETE", md)
