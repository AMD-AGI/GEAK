#!/usr/bin/env python3
"""Tests for the explicit linkage contracts (interface/geak_trace_events.py).

These fixtures establish that the TRACKER consumes recorded linkage events
correctly. They do NOT establish that the native runtime emits them -- that is
a separate question about runtime instrumentation, and nothing here should be
read as evidence for it.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geak_trace_events as E  # noqa: E402


def supplied(**kw):
    base = {"type": E.RESULT_SUPPLIED, "producer_invocation_id": "p1",
            "producer_result_ref": "result.directions[0]",
            "consumer_invocation_id": "c1",
            "consumer_input_ref": "dispatch.prompt#offset=120"}
    base.update(kw)
    return base


def spawn(**kw):
    base = {"type": E.SPAWN, "parent_invocation_id": "p1",
            "spawn_event_id": "s1", "child_invocation_id": "c1",
            "spawn_tool_call_id": "toolu_1"}
    base.update(kw)
    return base


def ret(**kw):
    base = {"type": E.SPAWN_RETURN, "spawn_event_id": "s1",
            "child_invocation_id": "c1", "status": "returned"}
    base.update(kw)
    return base


class ValidationTest(unittest.TestCase):
    def test_complete_result_supplied_event_validates(self):
        kind, norm = E.validate(supplied())
        self.assertEqual(kind, E.RESULT_SUPPLIED)
        self.assertEqual(norm["forwarding"], E.FORWARD_UNKNOWN)

    def test_every_required_field_is_required(self):
        for field in ("producer_invocation_id", "producer_result_ref",
                      "consumer_invocation_id", "consumer_input_ref"):
            ev = supplied()
            del ev[field]
            with self.assertRaises(E.EventError, msg=field):
                E.validate(ev)

    def test_partial_event_is_rejected_not_downgraded(self):
        with self.assertRaises(E.EventError):
            E.validate({"type": E.RESULT_SUPPLIED, "producer_invocation_id": "p1"})

    def test_transformed_without_description_is_marked_unknown(self):
        _, norm = E.validate(supplied(forwarding=E.FORWARD_TRANSFORMED))
        self.assertEqual(norm["forwarding"], E.FORWARD_TRANSFORMED)
        self.assertFalse(norm["transformation_known"])

    def test_transformed_with_description_is_known(self):
        _, norm = E.validate(supplied(forwarding=E.FORWARD_TRANSFORMED,
                                      transformation="json path + truncation"))
        self.assertTrue(norm["transformation_known"])

    def test_literal_forwarding_is_preserved(self):
        _, norm = E.validate(supplied(forwarding=E.FORWARD_LITERAL))
        self.assertEqual(norm["forwarding"], E.FORWARD_LITERAL)

    def test_invalid_forwarding_rejected(self):
        with self.assertRaises(E.EventError):
            E.validate(supplied(forwarding="probably"))

    def test_spawn_requires_parent_event_and_child(self):
        for field in ("parent_invocation_id", "spawn_event_id", "child_invocation_id"):
            ev = spawn()
            del ev[field]
            with self.assertRaises(E.EventError, msg=field):
                E.validate(ev)

    def test_unknown_event_type_rejected(self):
        with self.assertRaises(E.EventError):
            E.validate({"type": "vibes"})

    def test_invalid_return_status_rejected(self):
        with self.assertRaises(E.EventError):
            E.validate(ret(status="probably-fine"))


class EdgeBuildTest(unittest.TestCase):
    def _val(self, events):
        out = []
        for ev in events:
            kind, norm = E.validate(ev)
            norm["_kind"] = kind
            out.append(norm)
        return out

    def test_result_supplied_edge_is_proven_and_carries_refs(self):
        edges, unresolved, stats = E.build_edges(
            self._val([supplied(forwarding=E.FORWARD_LITERAL)]), {"p1", "c1"})
        self.assertEqual(stats["result_supplied_edges"], 1)
        edge = edges[0]
        self.assertTrue(edge["proven"])
        self.assertEqual(edge["provenance"], "recorded_event")
        self.assertEqual(edge["producer_result_ref"], "result.directions[0]")
        self.assertEqual(edge["consumer_input_ref"], "dispatch.prompt#offset=120")

    def test_unknown_invocation_is_unresolved_not_a_new_node(self):
        edges, unresolved, stats = E.build_edges(
            self._val([supplied(consumer_invocation_id="ghost")]), {"p1"})
        self.assertEqual(edges, [])
        self.assertEqual(stats["unresolved"], 1)
        self.assertIn("not present", unresolved[0]["reason"])

    def test_spawn_and_return_join_into_one_edge(self):
        edges, _, stats = E.build_edges(self._val([spawn(), ret()]), {"p1", "c1"})
        self.assertEqual(stats["spawn_edges"], 1)
        edge = edges[0]
        self.assertEqual(edge["from"], "agent:p1")
        self.assertEqual(edge["to"], "agent:c1")
        self.assertEqual(edge["return_status"], "returned")
        self.assertEqual(edge["spawn_tool_call_id"], "toolu_1")

    def test_spawn_without_return_is_flagged_unmatched(self):
        edges, unresolved, stats = E.build_edges(self._val([spawn()]), {"p1", "c1"})
        self.assertEqual(edges[0]["return_status"], "unmatched")
        self.assertEqual(stats["spawns_without_return"], 1)
        self.assertTrue(any("no matching return" in u["reason"] for u in unresolved))

    def test_retries_keep_separate_attempt_identities(self):
        events = self._val([spawn(),
                            ret(attempt_id="a1", status="error", error="boom"),
                            ret(attempt_id="a2", status="returned")])
        edges, _, _ = E.build_edges(events, {"p1", "c1"})
        attempts = edges[0]["attempts"]
        self.assertEqual([a["attempt_id"] for a in attempts], ["a1", "a2"])
        self.assertEqual([a["status"] for a in attempts], ["error", "returned"])

    def test_parallel_children_produce_distinct_edges(self):
        events = self._val([spawn(spawn_event_id="s1", child_invocation_id="c1"),
                            spawn(spawn_event_id="s2", child_invocation_id="c2"),
                            ret(spawn_event_id="s1", child_invocation_id="c1"),
                            ret(spawn_event_id="s2", child_invocation_id="c2")])
        edges, _, stats = E.build_edges(events, {"p1", "c1", "c2"})
        self.assertEqual(stats["spawn_edges"], 2)
        self.assertEqual({e["to"] for e in edges}, {"agent:c1", "agent:c2"})

    def test_duplicate_replayed_spawn_is_idempotent(self):
        edges, _, stats = E.build_edges(self._val([spawn(), spawn()]), {"p1", "c1"})
        self.assertEqual(stats["spawn_edges"], 1)

    def test_conflicting_duplicate_spawn_is_unresolved_not_overwritten(self):
        events = self._val([spawn(), spawn(child_invocation_id="other")])
        edges, unresolved, _ = E.build_edges(events, {"p1", "c1", "other"})
        self.assertEqual(len(edges), 1)
        self.assertTrue(any("conflicting child" in u["reason"] for u in unresolved))

    def test_orphan_return_is_reported(self):
        edges, unresolved, stats = E.build_edges(self._val([ret()]), {"p1", "c1"})
        self.assertEqual(stats["orphan_returns"], 1)
        self.assertTrue(any("no spawn event" in u["reason"] for u in unresolved))

    def test_no_edge_is_ever_built_without_an_event(self):
        edges, _, stats = E.build_edges([], {"p1", "c1"})
        self.assertEqual(edges, [])
        self.assertEqual(stats["result_supplied_edges"], 0)
        self.assertEqual(stats["spawn_edges"], 0)


class ReadAndAttachTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-events-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, "events.jsonl")

    def _write(self, rows, raw_extra=None):
        with open(self.path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            if raw_extra:
                fh.write(raw_extra)

    def _trace(self, agent_ids=("p1", "c1")):
        return {"schema": "geak.trace/1", "run": {"run_id": "wf_e"},
                "agents": [{"agent_id": a} for a in agent_ids],
                "edges": [], "warnings": []}

    def test_malformed_line_is_reported_not_skipped(self):
        self._write([supplied()], raw_extra='{"type": "agent_spawn"\n')
        events, problems = E.read_events(self.path)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(problems), 1)

    def test_absent_events_file_is_explicit_about_meaning(self):
        trace = E.attach(self._trace(), os.path.join(self.dir, "nope.jsonl"))
        link = trace["run"]["linkage"]
        self.assertFalse(link["present"])
        self.assertIn("not that they did not occur", link["note"])
        self.assertEqual(trace["edges"], [])

    def test_attach_adds_proven_edges_and_coverage(self):
        self._write([supplied(forwarding=E.FORWARD_LITERAL), spawn(), ret()])
        trace = E.attach(self._trace(), self.path)
        kinds = {e["type"] for e in trace["edges"]}
        self.assertEqual(kinds, {"result_supplied_to_dispatch", "agent_spawn"})
        self.assertTrue(trace["run"]["linkage"]["complete"])

    def test_incomplete_linkage_warns(self):
        self._write([supplied(consumer_invocation_id="ghost")])
        trace = E.attach(self._trace(), self.path)
        self.assertFalse(trace["run"]["linkage"]["complete"])
        self.assertTrue(any("INCOMPLETE" in w for w in trace["warnings"]))

    def test_historical_run_gets_no_synthesised_links(self):
        """The existing flat runs must stay flat."""
        trace = E.attach(self._trace(), None)
        self.assertEqual(trace["edges"], [])
        self.assertFalse(trace["run"]["linkage"]["present"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
