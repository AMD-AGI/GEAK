#!/usr/bin/env python3
"""Tests for the live GEAK execution tracker (interface/geak_trace_collector.py).

These cover the acceptance cases the design review called out: streamed-record
merging, input-window attribution, honest markers for unavailable data, tool
action/result joining, partial and missing artifacts, and injection inertness.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geak_trace_collector as C  # noqa: E402


def _rec(**kw):
    return json.dumps(kw)


def _asst(mid, blocks, ts="2026-09-16T19:39:03.338Z", usage=None,
          block_index=0, stop="end_turn", model="claude-opus-4-8", rid="req1"):
    return _rec(type="assistant", timestamp=ts, requestId=rid,
                apiBlockIndex=block_index, uuid="u-" + mid + str(block_index),
                message={"id": mid, "model": model, "stop_reason": stop,
                         "content": blocks, "usage": usage or {}})


def _user(blocks, ts="2026-09-16T19:39:01.000Z", uuid="uu1"):
    return _rec(type="user", timestamp=ts, uuid=uuid,
                message={"content": blocks})


class TraceCollectorTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-trace-test-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _write(self, name, lines):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    def _journal(self, entries):
        lines = [_rec(type="launched")]
        for aid, label in entries:
            lines.append(_rec(type="started", key="k-" + aid, agentId=aid,
                              label=label, phase="x-lane"))
        return self._write("journal.jsonl", lines)

    # ---- streaming / dedup -------------------------------------------------

    def test_streamed_records_one_call_with_all_blocks(self):
        """One response flushed as several records is ONE call, not three."""
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "thinking", "thinking": "", "signature": "sig"}],
                  block_index=0, usage={"output_tokens": 1}),
            _asst("m1", [{"type": "text", "text": "hello"}],
                  block_index=1, usage={"output_tokens": 5}),
            _asst("m1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                          "input": {"command": "ls"}}],
                  block_index=2, usage={"output_tokens": 9}),
        ])
        calls = C.build_agent_calls(path)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["output_text"], "hello")
        self.assertEqual([a["name"] for a in call["actions"]], ["Bash"])
        # usage comes from the largest-output flush, never summed
        self.assertEqual(call["usage"]["output_tokens"], 9)
        self.assertEqual(call["output_kind"], "mixed")

    def test_longest_text_wins_never_concatenates_prefix(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "text", "text": "par"}], block_index=0),
            _asst("m1", [{"type": "text", "text": "partial answer"}], block_index=0),
        ])
        calls = C.build_agent_calls(path)
        self.assertEqual(calls[0]["output_text"], "partial answer")

    # ---- input window ------------------------------------------------------

    def test_input_attaches_to_next_distinct_call_only(self):
        """User text + tool results map to the NEXT distinct call, not later ones."""
        path = self._write("agent-a1.jsonl", [
            _user([{"type": "text", "text": "do the thing"}]),
            _asst("m1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                          "input": {"command": "ls"}}], stop="tool_use"),
            _user([{"type": "tool_result", "tool_use_id": "t1",
                    "content": "file.txt"}]),
            _asst("m2", [{"type": "text", "text": "done"}]),
        ])
        calls = C.build_agent_calls(path)
        self.assertEqual(len(calls), 2)
        first, second = calls
        self.assertEqual([b["kind"] for b in first["input"]["blocks"]], ["text"])
        self.assertEqual([b["kind"] for b in second["input"]["blocks"]], ["tool_result"])
        self.assertEqual(second["input"]["blocks"][0]["tool_use_id"], "t1")

    def test_streamed_continuation_does_not_reset_input_window(self):
        """A later record for the SAME id must merge, not re-take the window."""
        path = self._write("agent-a1.jsonl", [
            _user([{"type": "text", "text": "first"}]),
            _asst("m1", [{"type": "text", "text": "a"}], block_index=0),
            _asst("m1", [{"type": "text", "text": "b"}], block_index=1),
            _user([{"type": "text", "text": "second"}]),
            _asst("m2", [{"type": "text", "text": "c"}]),
        ])
        calls = C.build_agent_calls(path)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["input"]["blocks"][0]["text"], "first")
        self.assertEqual(calls[1]["input"]["blocks"][0]["text"], "second")

    def test_call_with_no_new_input_is_marked_not_invented(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "text", "text": "hi"}]),
        ])
        calls = C.build_agent_calls(path)
        self.assertEqual(calls[0]["input"]["kind"], "none_recorded")
        self.assertEqual(calls[0]["input"]["blocks"], [])

    def test_input_never_claims_to_be_the_full_api_request(self):
        path = self._write("agent-a1.jsonl", [
            _user([{"type": "text", "text": "x"}]),
            _asst("m1", [{"type": "text", "text": "y"}]),
        ])
        note = C.build_agent_calls(path)[0]["input"]["note"].lower()
        self.assertIn("not the full api request", note)

    # ---- honest markers ----------------------------------------------------

    def test_empty_signed_reasoning_is_recorded_unreadable_not_absent(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "thinking", "thinking": "", "signature": "AAA"},
                         {"type": "text", "text": "ok"}]),
        ])
        r = C.build_agent_calls(path)[0]["reasoning"]
        self.assertEqual(r["state"], C.RECORDED_UNREADABLE)
        self.assertEqual(r["blocks"], 1)
        self.assertEqual(r["text"], "")

    def test_signature_is_never_serialized(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "thinking", "thinking": "", "signature": "SECRETSIG"}]),
        ])
        self.assertNotIn("SECRETSIG", json.dumps(C.build_agent_calls(path)))

    def test_no_reasoning_block_is_not_captured(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "text", "text": "ok"}]),
        ])
        self.assertEqual(C.build_agent_calls(path)[0]["reasoning"]["state"],
                         C.NOT_CAPTURED)

    def test_readable_reasoning_is_kept(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "thinking", "thinking": "step one"}]),
        ])
        r = C.build_agent_calls(path)[0]["reasoning"]
        self.assertEqual(r["state"], C.TEXT)
        self.assertIn("step one", r["text"])

    # ---- output kinds ------------------------------------------------------

    def test_tool_only_turn_shows_actions_not_empty(self):
        """The 'blank output' defect: a tool-only turn must show what it DID."""
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "tool_use", "id": "t1", "name": "Edit",
                          "input": {"file_path": "/x"}}], stop="tool_use"),
        ])
        call = C.build_agent_calls(path)[0]
        self.assertEqual(call["output_kind"], "actions")
        self.assertEqual(call["actions"][0]["name"], "Edit")
        self.assertNotEqual(call["actions"], [])

    def test_output_kind_is_independent_of_stop_reason(self):
        """Output kind describes observable content, not lifecycle status."""
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "text", "text": "final"}], stop="max_tokens"),
        ])
        call = C.build_agent_calls(path)[0]
        self.assertEqual(call["output_kind"], "text")
        self.assertEqual(call["stop_reason"], "max_tokens")

    # ---- tool action <-> result joining -----------------------------------

    def test_actions_join_results_by_tool_use_id(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                          "input": {"command": "ls"}}], stop="tool_use"),
            _user([{"type": "tool_result", "tool_use_id": "t1",
                    "content": "out.txt"}]),
        ])
        act = C.build_agent_calls(path)[0]["actions"][0]
        self.assertEqual(act["result"]["status"], "ok")
        self.assertIn("out.txt", act["result"]["preview"])

    def test_error_result_status_preserved(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                          "input": {}}], stop="tool_use"),
            _user([{"type": "tool_result", "tool_use_id": "t1",
                    "is_error": True, "content": "boom"}]),
        ])
        self.assertEqual(
            C.build_agent_calls(path)[0]["actions"][0]["result"]["status"], "error")

    def test_unmatched_action_is_missing_not_fabricated(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "tool_use", "id": "t9", "name": "Bash",
                          "input": {}}], stop="tool_use"),
        ])
        res = C.build_agent_calls(path)[0]["actions"][0]["result"]
        self.assertEqual(res["status"], "missing")

    def test_tool_results_do_not_become_extra_calls(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "tool_use", "id": "t1", "name": "B", "input": {}}],
                  stop="tool_use"),
            _user([{"type": "tool_result", "tool_use_id": "t1", "content": "a"},
                   {"type": "tool_result", "tool_use_id": "t2", "content": "b"}]),
        ])
        self.assertEqual(len(C.build_agent_calls(path)), 1)

    def test_binary_blocks_are_placeheld_not_inlined(self):
        path = self._write("agent-a1.jsonl", [
            _asst("m1", [{"type": "tool_use", "id": "t1", "name": "R", "input": {}}],
                  stop="tool_use"),
            _user([{"type": "tool_result", "tool_use_id": "t1",
                    "content": [{"type": "image", "source": {"data": "BASE64PAYLOAD"}}]}]),
        ])
        prev = C.build_agent_calls(path)[0]["actions"][0]["result"]["preview"]
        self.assertNotIn("BASE64PAYLOAD", prev)
        self.assertIn("omitted non-text blocks", prev)

    # ---- bounded previews / redaction / injection --------------------------

    def test_preview_is_byte_capped_and_flags_original_length(self):
        shown, trunc, total = C.preview("x" * 10000, cap=100)
        self.assertTrue(trunc)
        self.assertEqual(total, 10000)
        self.assertLessEqual(len(shown.encode("utf-8")), 100)

    def test_preview_truncation_does_not_split_a_codepoint(self):
        shown, trunc, _ = C.preview("é" * 500, cap=101)
        self.assertTrue(trunc)
        shown.encode("utf-8").decode("utf-8")  # must not raise

    def test_credentials_are_redacted_before_persisting(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        shown, _, _ = C.preview("key is %s here" % secret)
        self.assertNotIn(secret, shown)
        self.assertIn("REDACTED", shown)

    def test_literal_html_is_preserved_as_data_not_executed(self):
        """Renderers must escape; the collector must not silently mangle."""
        payload = "</script><img src=x onerror=alert(1)>"
        path = self._write("agent-a1.jsonl", [
            _user([{"type": "text", "text": payload}]),
            _asst("m1", [{"type": "text", "text": "ok"}]),
        ])
        blk = C.build_agent_calls(path)[0]["input"]["blocks"][0]
        self.assertEqual(blk["text"], payload)
        # It round-trips through JSON as inert data.
        self.assertEqual(json.loads(json.dumps(blk))["text"], payload)

    # ---- tolerant reading --------------------------------------------------

    def test_partial_trailing_line_is_skipped_not_fatal(self):
        path = os.path.join(self.dir, "agent-a1.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "ok"}]) + "\n")
            fh.write('{"type": "assistant", "message": {"id": "m2"')  # mid-flush
        calls = C.build_agent_calls(path)
        self.assertEqual(len(calls), 1)

    def test_missing_transcript_returns_no_calls(self):
        self.assertEqual(C.build_agent_calls(os.path.join(self.dir, "nope.jsonl")), [])

    # ---- run-level graph ---------------------------------------------------

    def test_edges_are_workflow_to_agent_and_back_only(self):
        self._journal([("a1", "director:setup"), ("a2", "eng r1_d0")])
        with open(os.path.join(self.dir, "journal.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(_rec(type="result", key="k-a1", agentId="a1",
                          result={"ok": True}) + "\n")
        self._write("agent-a1.jsonl", [_asst("m1", [{"type": "text", "text": "x"}])])
        trace = C.build_trace(self.dir)
        kinds = sorted({e["type"] for e in trace["edges"]})
        self.assertEqual(kinds, ["orchestration", "return"])
        self.assertTrue(all(e["proven"] for e in trace["edges"]))
        # exactly one return edge: only a1 returned
        self.assertEqual(sum(1 for e in trace["edges"] if e["type"] == "return"), 1)

    def test_no_agent_to_agent_edges_are_invented(self):
        self._journal([("a1", "tech_lead:plan r1"), ("a2", "eng r1_d0:compute")])
        trace = C.build_trace(self.dir)
        agent_nodes = {"agent:a1", "agent:a2"}
        for e in trace["edges"]:
            self.assertFalse(e["from"] in agent_nodes and e["to"] in agent_nodes,
                             "must not fabricate agent-to-agent spawn edges")

    def test_started_without_result_is_pending_not_completed(self):
        self._journal([("a1", "eng r1_d0")])
        self._write("agent-a1.jsonl", [_asst("m1", [{"type": "text", "text": "x"}])])
        trace = C.build_trace(self.dir)
        agent = trace["agents"][0]
        self.assertEqual(agent["result_status"], "pending_or_absent")
        self.assertNotEqual(agent["status"], "completed")
        # With no run record on disk, completion CANNOT be established.
        self.assertEqual(trace["run"]["status"], "unknown")

    def test_returned_result_is_preserved_verbatim(self):
        self._journal([("a1", "verify r1_d1")])
        payload = {"status": "pass", "verified_geomean": 1.23}
        with open(os.path.join(self.dir, "journal.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(_rec(type="result", key="k-a1", agentId="a1",
                          result=payload) + "\n")
        trace = C.build_trace(self.dir)
        self.assertEqual(trace["agents"][0]["result"], payload)
        self.assertEqual(trace["agents"][0]["result_status"], "returned_to_workflow")

    def test_journal_order_is_preserved_as_ordinal(self):
        self._journal([("a3", "third"), ("a1", "first"), ("a2", "second")])
        trace = C.build_trace(self.dir)
        self.assertEqual([a["agent_id"] for a in trace["agents"]], ["a3", "a1", "a2"])
        self.assertEqual([a["ordinal"] for a in trace["agents"]], [0, 1, 2])

    def test_timing_is_labelled_estimated_and_warned(self):
        self._journal([("a1", "x")])
        self._write("agent-a1.jsonl", [_asst("m1", [{"type": "text", "text": "x"}])])
        trace = C.build_trace(self.dir)
        self.assertEqual(trace["agents"][0]["timing_provenance"],
                         "transcript_timestamps_estimated")
        self.assertTrue(any("ESTIMATES" in w for w in trace["warnings"]))
        self.assertIn("must not be summed", trace["run"]["timing_note"])

    def test_missing_transcript_does_not_fabricate_timing(self):
        self._journal([("a1", "x")])
        trace = C.build_trace(self.dir)
        agent = trace["agents"][0]
        self.assertIsNone(agent["first_ts_ms"])
        self.assertEqual(agent["transcript_status"], "missing")
        self.assertTrue(any("no transcript" in w for w in trace["warnings"]))

    def test_empty_directory_is_reported_not_crashed(self):
        trace = C.build_trace(self.dir)
        self.assertEqual(trace["agents"], [])
        self.assertEqual(trace["edges"], [])
        self.assertTrue(trace["warnings"])

    def test_interleaved_agents_keep_separate_sequences(self):
        self._journal([("a1", "eng r1"), ("a2", "eng r2")])
        self._write("agent-a1.jsonl", [_asst("m1", [{"type": "text", "text": "one"}])])
        self._write("agent-a2.jsonl", [_asst("m2", [{"type": "text", "text": "two"}])])
        trace = C.build_trace(self.dir)
        by_id = {a["agent_id"]: a for a in trace["agents"]}
        self.assertEqual(by_id["a1"]["calls"][0]["output_text"], "one")
        self.assertEqual(by_id["a2"]["calls"][0]["output_text"], "two")

    def test_repeated_labels_stay_distinct_agents(self):
        """Repeated rounds must not collapse into one group."""
        self._journal([("a1", "verify r1_d0"), ("a2", "verify r1_d0")])
        trace = C.build_trace(self.dir)
        self.assertEqual(len(trace["agents"]), 2)
        self.assertEqual({a["agent_id"] for a in trace["agents"]}, {"a1", "a2"})

    # ---- durability --------------------------------------------------------

    def test_write_trace_is_atomic_and_leaves_no_tmp(self):
        self._journal([("a1", "x")])
        out = os.path.join(self.dir, "out", "geak_trace.json")
        C.write_trace(C.build_trace(self.dir), out)
        self.assertTrue(os.path.exists(out))
        self.assertFalse(os.path.exists(out + ".tmp"))
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["schema"], C.SCHEMA)

    def test_recollection_is_idempotent(self):
        self._journal([("a1", "x")])
        self._write("agent-a1.jsonl", [_asst("m1", [{"type": "text", "text": "x"}])])
        out = os.path.join(self.dir, "t.json")
        a = C.collect_once(self.dir, out)
        b = C.collect_once(self.dir, out)
        for t in (a, b):
            t["run"].pop("collected_at_ms")
        self.assertEqual(json.dumps(a, default=str), json.dumps(b, default=str))

    def test_appending_new_agent_is_picked_up_on_next_pass(self):
        """Live growth: a second pass sees work that did not exist in the first."""
        self._journal([("a1", "x")])
        out = os.path.join(self.dir, "t.json")
        first = C.collect_once(self.dir, out)
        self.assertEqual(first["run"]["agents_started"], 1)
        with open(os.path.join(self.dir, "journal.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(_rec(type="started", key="k-a2", agentId="a2",
                          label="y", phase="x-lane") + "\n")
        second = C.collect_once(self.dir, out)
        self.assertEqual(second["run"]["agents_started"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class LifecycleTest(unittest.TestCase):
    """Completion must come from the run record, never from agent quiescence."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-life-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        # Lay out <session>/subagents/workflows/<runId> so the run record resolves.
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_x")
        os.makedirs(self.wf)
        self.rec_dir = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(self.rec_dir)

    def _journal(self, *lines):
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _record(self, status):
        with open(os.path.join(self.rec_dir, "wf_x.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_x", "status": status}, fh)

    def test_all_returned_is_not_complete_without_record(self):
        """The sequential-workflow trap: the gap before the next dispatch."""
        self._journal(_rec(type="launched"),
                      _rec(type="started", key="k", agentId="a1", label="x", phase="P"),
                      _rec(type="result", key="k", agentId="a1", result={"ok": 1}))
        trace = C.build_trace(self.wf)
        self.assertNotEqual(trace["run"]["status"], "complete")
        self.assertTrue(any("NOT evidence" in w for w in trace["warnings"]))

    def test_all_returned_is_not_complete_while_record_is_running(self):
        self._record("running")
        self._journal(_rec(type="launched"),
                      _rec(type="started", key="k", agentId="a1", label="x", phase="P"),
                      _rec(type="result", key="k", agentId="a1", result={"ok": 1}))
        trace = C.build_trace(self.wf)
        self.assertEqual(trace["run"]["status"], "live")
        self.assertEqual(trace["run"]["status_provenance"], "run_record")

    def test_record_completed_marks_complete(self):
        self._record("completed")
        self._journal(_rec(type="launched"),
                      _rec(type="started", key="k", agentId="a1", label="x", phase="P"),
                      _rec(type="result", key="k", agentId="a1", result={"ok": 1}))
        trace = C.build_trace(self.wf)
        self.assertEqual(trace["run"]["status"], "complete")
        self.assertIn("completed", trace["run"]["status_reason"])

    def test_failed_run_is_terminal_but_not_called_complete_falsely(self):
        self._record("failed")
        self._journal(_rec(type="launched"))
        trace = C.build_trace(self.wf)
        self.assertEqual(trace["run"]["status"], "complete")
        self.assertIn("failed", trace["run"]["status_reason"])

    def test_pending_agent_with_deadline_yields_partial_not_complete(self):
        """A watch deadline must never be reported as completion."""
        self._record("running")
        self._journal(_rec(type="launched"),
                      _rec(type="started", key="k", agentId="a1", label="x", phase="P"))
        out = os.path.join(self.dir, "t.json")
        trace = C.watch(self.wf, out, interval=0.01, max_seconds=0)
        self.assertEqual(trace["run"]["status"], "partial")
        self.assertIn("deadline", trace["run"]["status_reason"])
        self.assertEqual(trace["run"]["agents_returned"], 0)

    def test_watch_keeps_observing_across_a_dispatch_gap(self):
        """The regression: it must not exit when agent 1 returns."""
        self._record("running")
        self._journal(_rec(type="launched"),
                      _rec(type="started", key="k", agentId="a1", label="x", phase="P"),
                      _rec(type="result", key="k", agentId="a1", result={"ok": 1}))
        out = os.path.join(self.dir, "t.json")
        trace = C.watch(self.wf, out, interval=0.01, max_seconds=0.05)
        # It ran to the deadline instead of declaring victory at agent 1.
        self.assertEqual(trace["run"]["status"], "partial")

    def test_status_file_is_written_for_visibility(self):
        self._record("completed")
        self._journal(_rec(type="launched"))
        out = os.path.join(self.dir, "t.json")
        sp = os.path.join(self.dir, "t.status.json")
        C.collect_once(self.wf, out, status_path=sp)
        with open(sp, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["state"], "complete")

    def test_staging_file_is_unique_per_writer(self):
        trace = C.build_trace(self.wf)
        out = os.path.join(self.dir, "u.json")
        C.write_trace(trace, out)
        leftovers = [f for f in os.listdir(self.dir) if ".tmp" in f]
        self.assertEqual(leftovers, [])


class ReasoningMergeTest(unittest.TestCase):
    """Reasoning merges by block identity, not by counting blocks."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-think-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _write(self, lines):
        path = os.path.join(self.dir, "agent-a1.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    def test_empty_block_later_filled_becomes_readable(self):
        """The regression: same block, empty then filled, must upgrade to text."""
        path = self._write([
            _asst("m1", [{"type": "thinking", "thinking": "", "signature": "s"}],
                  block_index=0, usage={"output_tokens": 1}),
            _asst("m1", [{"type": "thinking", "thinking": "real reasoning",
                          "signature": "s"}], block_index=0,
                  usage={"output_tokens": 2}),
        ])
        r = C.build_agent_calls(path)[0]["reasoning"]
        self.assertEqual(r["state"], C.TEXT)
        self.assertIn("real reasoning", r["text"])

    def test_separately_streamed_blocks_accumulate(self):
        path = self._write([
            _asst("m1", [{"type": "thinking", "thinking": "first"}], block_index=0),
            _asst("m1", [{"type": "thinking", "thinking": "second"}], block_index=1),
        ])
        r = C.build_agent_calls(path)[0]["reasoning"]
        self.assertEqual(r["blocks"], 2)
        self.assertIn("first", r["text"])
        self.assertIn("second", r["text"])

    def test_still_unreadable_when_every_block_is_empty(self):
        path = self._write([
            _asst("m1", [{"type": "thinking", "thinking": "", "signature": "s"}]),
        ])
        self.assertEqual(C.build_agent_calls(path)[0]["reasoning"]["state"],
                         C.RECORDED_UNREADABLE)


class UsageCoverageTest(unittest.TestCase):
    """A record with no usage is UNKNOWN, never a silent zero."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-usage-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_missing_usage_is_flagged_not_zeroed(self):
        path = os.path.join(self.dir, "agent-a1.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "x"}], usage=None) + "\n")
        call = C.build_agent_calls(path)[0]
        self.assertFalse(call["usage_known"])

    def test_present_usage_is_marked_known(self):
        path = os.path.join(self.dir, "agent-a1.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "x"}],
                           usage={"output_tokens": 5}) + "\n")
        self.assertTrue(C.build_agent_calls(path)[0]["usage_known"])


class OwnershipResolverTest(unittest.TestCase):
    """Ownership must not be 'whichever matching record is newest'."""

    def test_no_identity_given_is_unresolved(self):
        wf, info = C.resolve_workflow_dir()
        self.assertIsNone(wf)
        self.assertIn("ownership", info["error"])

    def test_resolver_uses_ownership_not_ranking(self):
        """A ranking helper would adopt a sibling that merely contains the path."""
        import inspect
        src = inspect.getsource(C.resolve_workflow_dir)
        self.assertIn("_owns", src)
        self.assertNotIn("mirror.find_record(", src)

    def test_launch_path_requires_a_live_record(self):
        """The prospective path must not adopt an already-completed run."""
        import inspect
        src = inspect.getsource(C._await_workflow_dir)
        self.assertIn("require_live=True", src)

    def test_any_run_cli_lifts_require_live_not_just_the_time_filter(self):
        """--any-run must actually reach a completed record through main()."""
        import inspect
        src = inspect.getsource(C.main)
        self.assertIn("require_live=args.prospective", src)
        self.assertIn("prospective=args.prospective", src)


class FinalRenderTest(unittest.TestCase):
    """The end-of-run HTML must be built from what was TRACKED during the run."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-render-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_r")
        os.makedirs(self.wf)
        self.rec_dir = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(self.rec_dir)
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_rec(type="launched") + "\n")
            fh.write(_rec(type="started", key="k", agentId="a1",
                          label="director:setup", phase="Setup") + "\n")
            fh.write(_rec(type="result", key="k", agentId="a1",
                          result={"ok": True}) + "\n")
        with open(os.path.join(self.wf, "agent-a1.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "hello"}],
                           usage={"output_tokens": 3}) + "\n")

    def _record(self, status, eval_dir=None):
        doc = {"runId": "wf_r", "status": status}
        if eval_dir:
            doc["result"] = {"eval_dir": eval_dir}
        with open(os.path.join(self.rec_dir, "wf_r.json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

    def test_watch_renders_html_from_tracked_trace_on_completion(self):
        self._record("completed")
        out = os.path.join(self.dir, "geak_trace_wf_r.json")
        C.watch(self.wf, out, interval=0.01, max_seconds=5)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "geak_execution_trace_wf_r.html")))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "geak_execution_trace_wf_r.md")))

    def test_partial_run_still_renders_what_was_captured(self):
        """A run that never reported terminal still yields a usable report."""
        self._record("running")
        out = os.path.join(self.dir, "geak_trace_wf_r.json")
        C.watch(self.wf, out, interval=0.01, max_seconds=0)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "geak_execution_trace_wf_r.html")))

    def test_final_render_also_lands_in_the_runs_report_dir(self):
        eval_dir = os.path.join(self.dir, "run")
        os.makedirs(eval_dir)
        self._record("completed", eval_dir=eval_dir)
        out = os.path.join(self.dir, "geak_trace_wf_r.json")
        C.watch(self.wf, out, interval=0.01, max_seconds=5)
        report_dir = os.path.join(eval_dir, "report")
        self.assertTrue(os.path.exists(os.path.join(report_dir, "geak_execution_trace.html")))
        # The tracked trace is persisted there too, so the report can be rebuilt
        # later even if the transcripts are gone.
        self.assertTrue(os.path.exists(os.path.join(report_dir, "geak_trace.json")))

    def test_render_can_be_disabled(self):
        self._record("completed")
        out = os.path.join(self.dir, "geak_trace_wf_r.json")
        C.watch(self.wf, out, interval=0.01, max_seconds=5, render=False)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "geak_execution_trace_wf_r.html")))
        self.assertTrue(os.path.exists(out))  # tracking still happened

    def test_rendered_html_contains_the_tracked_call(self):
        self._record("completed")
        out = os.path.join(self.dir, "geak_trace_wf_r.json")
        C.watch(self.wf, out, interval=0.01, max_seconds=5)
        with open(os.path.join(self.dir, "geak_execution_trace_wf_r.html"), encoding="utf-8") as fh:
            body = fh.read()
        self.assertIn("director:setup", body)

    def test_eval_dir_of_run_reads_the_record(self):
        self._record("completed", eval_dir="/some/eval")
        self.assertEqual(C.eval_dir_of_run(self.wf), "/some/eval")

    def test_eval_dir_absent_is_none_not_guessed(self):
        self._record("completed")
        self.assertIsNone(C.eval_dir_of_run(self.wf))


class PerRunReportPathTest(unittest.TestCase):
    """Two runs sharing an output directory must not overwrite each other."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-perrun-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _trace(self, run_id):
        return {"schema": C.SCHEMA,
                "run": {"run_id": run_id, "status": "complete"},
                "agents": [], "edges": [], "warnings": []}

    def test_two_runs_produce_distinct_report_files(self):
        for rid in ("wf_one", "wf_two"):
            out = os.path.join(self.dir, "geak_trace_%s.json" % rid)
            C.write_trace(self._trace(rid), out)
            C.publish_final(self._trace(rid), self.dir, out, render=True)
        names = sorted(f for f in os.listdir(self.dir) if f.endswith(".html"))
        self.assertEqual(names, ["geak_execution_trace_wf_one.html",
                                 "geak_execution_trace_wf_two.html"])


class UnknownUsagePricingTest(unittest.TestCase):
    """A call with no usage must be unknown-cost, not $0."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-unk-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_missing_usage_is_not_priced_as_zero(self):
        path = os.path.join(self.dir, "agent-a1.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "x"}], usage=None) + "\n")
        rates, fns = C._load_cost_support()
        if rates is None:
            self.skipTest("ledger pricing unavailable")
        call = C.build_agent_calls(path, rates, fns)[0]
        self.assertFalse(call["usage_known"])
        self.assertIsNone(call["cost_usd"])

    def test_known_usage_is_still_priced(self):
        path = os.path.join(self.dir, "agent-a1.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "x"}],
                           usage={"output_tokens": 100}) + "\n")
        rates, fns = C._load_cost_support()
        if rates is None:
            self.skipTest("ledger pricing unavailable")
        call = C.build_agent_calls(path, rates, fns)[0]
        self.assertTrue(call["usage_known"])
        self.assertIsNotNone(call["cost_usd"])


class RetentionTest(unittest.TestCase):
    """Captured history must survive a source that shrinks or disappears."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-retain-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_k")
        os.makedirs(self.wf)
        self.rec = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(self.rec)
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_rec(type="launched") + "\n")
            fh.write(_rec(type="started", key="k", agentId="a1",
                          label="eng", phase="P") + "\n")
        self.tp = os.path.join(self.wf, "agent-a1.jsonl")
        with open(self.tp, "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "captured work"}],
                           usage={"output_tokens": 5}) + "\n")
        self._record("running")

    def _record(self, status):
        with open(os.path.join(self.rec, "wf_k.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_k", "status": status}, fh)

    def test_removed_transcript_does_not_erase_captured_calls(self):
        out = os.path.join(self.dir, "t.json")
        first = C.collect_once(self.wf, out)
        self.assertEqual(sum(len(a["calls"]) for a in first["agents"]), 1)
        os.remove(self.tp)
        self._record("completed")
        second = C.collect_once(self.wf, out, previous=first)
        self.assertEqual(sum(len(a["calls"]) for a in second["agents"]), 1)
        self.assertEqual(second["run"]["capture_retention"]["retained_calls"], 1)

    def test_retained_capture_is_labelled_not_passed_off_as_fresh(self):
        out = os.path.join(self.dir, "t.json")
        first = C.collect_once(self.wf, out)
        os.remove(self.tp)
        second = C.collect_once(self.wf, out, previous=first)
        agent = second["agents"][0]
        self.assertEqual(agent["transcript_status"], "missing_source_retained_capture")
        self.assertIn("retained", agent["capture_note"])
        self.assertTrue(any("retained history" in w for w in second["warnings"]))

    def test_retention_reads_previous_from_disk_when_not_passed(self):
        out = os.path.join(self.dir, "t.json")
        C.collect_once(self.wf, out)
        os.remove(self.tp)
        again = C.collect_once(self.wf, out)  # previous loaded from out_path
        self.assertEqual(sum(len(a["calls"]) for a in again["agents"]), 1)

    def test_growing_transcript_still_takes_the_fresh_read(self):
        out = os.path.join(self.dir, "t.json")
        first = C.collect_once(self.wf, out)
        with open(self.tp, "a", encoding="utf-8") as fh:
            fh.write(_asst("m2", [{"type": "text", "text": "more"}],
                           usage={"output_tokens": 7}) + "\n")
        second = C.collect_once(self.wf, out, previous=first)
        self.assertEqual(sum(len(a["calls"]) for a in second["agents"]), 2)
        self.assertIsNone(second["run"].get("capture_retention"))


class ProspectiveIdentityTest(unittest.TestCase):
    """The observer must track ITS launch's run, not a neighbour under the root."""

    def _fake_mirror(self, records):
        import types

        class _P:
            def __init__(self):
                self.parent = types.SimpleNamespace(
                    parent=types.SimpleNamespace(name="sess"))

        def owns(rec, ed, er):
            return bool(er) and er.rstrip("/") == rec["args"].get("exp_root")

        return types.SimpleNamespace(
            candidate_homes=lambda: [],
            iter_records=lambda h: [(_P(), r) for r in records],
            _owns=owns,
            record_paths_typed=lambda r: [("exp_root", r["args"]["exp_root"])])

    def _rec(self, rid, status, start_ms, exp="/exp/shared", args=None):
        return {"runId": rid, "status": status, "startTime": start_ms,
                "args": args if args is not None else {"exp_root": exp},
                "result": {}}

    def test_neighbour_makes_attachment_unresolved_not_guessed(self):
        """A recent neighbour must NOT be adopted via a time window."""
        import unittest.mock as mock
        now = int(time.time() * 1000)
        recs = [self._rec("wf_neighbour", "running", now - 90_000),
                self._rec("wf_mine", "running", now - 1_000)]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(exp_root="/exp/shared",
                                              require_live=True, prospective=True)
        self.assertIsNone(wf, "must refuse rather than pick one on timing")
        self.assertEqual(info["ambiguous"], 2)
        self.assertIn("requires a supported invocation identity", info["error"])

    def test_explicit_run_id_is_proof_and_resolves(self):
        import unittest.mock as mock
        now = int(time.time() * 1000)
        recs = [self._rec("wf_neighbour", "running", now - 90_000),
                self._rec("wf_mine", "running", now - 1_000)]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(exp_root="/exp/shared",
                                              require_live=True, prospective=True,
                                              run_id="wf_mine")
        self.assertEqual(info["run_id"], "wf_mine")
        self.assertEqual(info["identity"], "explicit-run-id")

    def test_sole_owning_record_is_NOT_enough_for_prospective(self):
        """Astra R7: one candidate is not proof it is THIS launch."""
        import unittest.mock as mock
        now = int(time.time() * 1000)
        recs = [self._rec("wf_only", "running", now - 90_000)]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(exp_root="/exp/shared",
                                              require_live=True, prospective=True)
        self.assertIsNone(wf, "a sole owning record must not be adopted")
        self.assertIn("requires a supported invocation identity", info["error"])
        self.assertIn("integration_gap", info)

    def test_args_fingerprint_identifies_this_launch(self):
        """The runtime-supplied join: the record carries the hook's own args."""
        import unittest.mock as mock
        now = int(time.time() * 1000)
        mine = {"exp_root": "/exp/shared", "kernel_path": "/k", "deadline_epoch": 111}
        other = {"exp_root": "/exp/shared", "kernel_path": "/k", "deadline_epoch": 222}
        recs = [self._rec("wf_other", "running", now - 90_000, args=other),
                self._rec("wf_mine", "running", now - 1_000, args=mine)]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(exp_root="/exp/shared",
                                              require_live=True, prospective=True,
                                              identity_args=mine)
        self.assertEqual(info["run_id"], "wf_mine")
        self.assertEqual(info["identity"], "args-fingerprint")
        self.assertEqual(info["args_mismatch_records"], ["wf_other"])

    def test_fingerprint_with_no_match_stays_unresolved(self):
        import unittest.mock as mock
        now = int(time.time() * 1000)
        recs = [self._rec("wf_other", "running", now - 1_000,
                          args={"exp_root": "/exp/shared", "deadline_epoch": 9})]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(
                exp_root="/exp/shared", require_live=True, prospective=True,
                identity_args={"exp_root": "/exp/shared", "deadline_epoch": 1})
        self.assertIsNone(wf)
        self.assertIn("rejected on args fingerprint", info["error"])

    def test_identical_launches_are_ambiguous_not_picked(self):
        import unittest.mock as mock
        now = int(time.time() * 1000)
        same = {"exp_root": "/exp/shared", "deadline_epoch": 5}
        recs = [self._rec("wf_a", "running", now - 2_000, args=same),
                self._rec("wf_b", "running", now - 1_000, args=same)]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(exp_root="/exp/shared",
                                              require_live=True, prospective=True,
                                              identity_args=same)
        self.assertIsNone(wf)
        self.assertEqual(info["ambiguous"], 2)

    def test_fingerprint_is_order_independent(self):
        a = C.args_fingerprint({"b": 2, "a": 1})
        b = C.args_fingerprint({"a": 1, "b": 2})
        self.assertEqual(a, b)
        self.assertNotEqual(a, C.args_fingerprint({"a": 1, "b": 3}))

    def test_no_time_window_is_used_at_all(self):
        """A record with no usable timestamp must not be admitted by timing."""
        import inspect
        src = inspect.getsource(C.resolve_workflow_dir)
        self.assertNotIn("not_before_ms", src)
        self.assertIn("A time window cannot establish it", src)

    def test_two_concurrent_live_runs_refuse_rather_than_guess(self):
        import unittest.mock as mock
        now = int(time.time() * 1000)
        recs = [self._rec("wf_a", "running", now - 3_000),
                self._rec("wf_b", "running", now - 2_000)]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(exp_root="/exp/shared",
                                              require_live=True, prospective=True)
        self.assertIsNone(wf)
        self.assertEqual(info["ambiguous"], 2)

    def test_reporting_on_a_past_run_can_opt_out_of_prospective(self):
        """--any-run: no not_before filter, so an old run resolves normally."""
        import unittest.mock as mock
        now = int(time.time() * 1000)
        recs = [self._rec("wf_old", "completed", now - 3600_000)]
        with mock.patch.dict(sys.modules,
                             {"claude_trace_mirror": self._fake_mirror(recs)}):
            wf, info = C.resolve_workflow_dir(exp_root="/exp/shared")
        self.assertEqual(info["run_id"], "wf_old")

    def test_resolver_provenance_reaches_the_trace(self):
        d = tempfile.mkdtemp(prefix="geak-prov-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wf = os.path.join(d, "sess", "subagents", "workflows", "wf_p")
        os.makedirs(wf)
        trace = C.build_trace(wf, resolver_info={"ambiguous": 2, "run_id": "wf_p",
                                                 "skipped_started_before_observer": 1})
        self.assertTrue(any("ambiguous" in w for w in trace["warnings"]))
        self.assertTrue(any("pre-existing runs" in w for w in trace["warnings"]))
        self.assertEqual(trace["run"]["resolver"]["ambiguous"], 2)


class MergeInvariantTest(unittest.TestCase):
    """Retention must be monotonic, run-scoped, and keep the graph consistent."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-merge-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_d")
        os.makedirs(self.wf)
        rd = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(rd)
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_rec(type="launched") + "\n")
            fh.write(_rec(type="started", key="k", agentId="a1",
                          label="eng", phase="P") + "\n")
            fh.write(_rec(type="result", key="k", agentId="a1",
                          result={"ok": 1}) + "\n")
        self._transcript("captured work at length", 100)
        with open(os.path.join(rd, "wf_d.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_d", "status": "completed"}, fh)

    def _transcript(self, text, tokens):
        with open(os.path.join(self.wf, "agent-a1.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": text}],
                           usage={"output_tokens": tokens}) + "\n")

    def test_shorter_reread_of_same_call_does_not_lose_content(self):
        out = os.path.join(self.dir, "t.json")
        first = C.collect_once(self.wf, out)
        self._transcript("p", 1)  # truncated re-read of the SAME call id
        second = C.collect_once(self.wf, out, previous=first)
        call = second["agents"][0]["calls"][0]
        self.assertEqual(call["usage"]["output_tokens"], 100)
        self.assertEqual(call["output_text"], "captured work at length")

    def test_previous_trace_from_another_run_is_not_imported(self):
        other = {"schema": C.SCHEMA, "run": {"run_id": "wf_unrelated"},
                 "agents": [{"agent_id": "zz", "ordinal": 0, "label": "other",
                             "calls": [], "totals": {}, "result_status": "x"}],
                 "edges": [], "warnings": []}
        merged = C.merge_traces(other, C.build_trace(self.wf))
        self.assertNotIn("zz", {a["agent_id"] for a in merged["agents"]})
        self.assertTrue(any("belongs to run wf_unrelated" in w
                            for w in merged["warnings"]))

    def test_journal_loss_keeps_edges_and_consistent_headers(self):
        out = os.path.join(self.dir, "t.json")
        first = C.collect_once(self.wf, out)
        os.remove(os.path.join(self.wf, "journal.jsonl"))
        second = C.collect_once(self.wf, out, previous=first)
        self.assertEqual(len(second["edges"]), 2)
        self.assertEqual(second["run"]["agents_started"], 1)
        self.assertEqual(second["run"]["agents_returned"], 1)
        self.assertIsNotNone(second["run"]["origin_ts_ms"])

    def test_report_driver_retains_across_source_loss(self):
        """Astra's surviving R3 reproduction, through the real driver call."""
        import unittest.mock as mock
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import geak_report as G
        eval_dir = os.path.join(self.dir, "eval")
        report_dir = os.path.join(eval_dir, "report")
        os.makedirs(report_dir)
        with mock.patch.object(C, "resolve_workflow_dir",
                               return_value=(self.wf, {"run_id": "wf_d"})):
            first = G._write_execution_trace(eval_dir, report_dir)
            self.assertEqual(first["calls"], 1)
            os.remove(os.path.join(self.wf, "agent-a1.jsonl"))
            second = G._write_execution_trace(eval_dir, report_dir)
        self.assertEqual(second["calls"], 1)


class MirrorTest(unittest.TestCase):
    """Durable mirroring: the run's own sources, copied so a rebuild survives."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-mirror-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_m")
        os.makedirs(self.wf)
        self.recdir = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(self.recdir)
        self.dest = os.path.join(self.dir, "mirror")
        self._journal(["a1"])
        self._transcript("a1", [_asst("m1", [{"type": "text", "text": "one"}],
                                      usage={"output_tokens": 5})])
        with open(os.path.join(self.wf, "agent-a1.meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"description": "d", "spawnDepth": 1}, fh)
        self._record("running")

    def _journal(self, agents, results=()):
        lines = [_rec(type="launched")]
        for a in agents:
            lines.append(_rec(type="started", key="k" + a, agentId=a,
                              label="eng " + a, phase="P"))
        for a in results:
            lines.append(_rec(type="result", key="k" + a, agentId=a, result={"ok": a}))
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _transcript(self, aid, lines):
        with open(os.path.join(self.wf, "agent-%s.jsonl" % aid), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _record(self, status):
        with open(os.path.join(self.recdir, "wf_m.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_m", "status": status}, fh)

    def test_first_pass_copies_every_run_owned_artifact(self):
        man = C.mirror_sources(self.wf, self.dest)
        for name in ("journal.jsonl", "agent-a1.jsonl", "agent-a1.meta.json"):
            self.assertIn(name, man["files"])
            self.assertTrue(os.path.exists(os.path.join(self.dest, name)))
        self.assertTrue(man["complete"])

    def test_second_pass_copies_nothing_when_unchanged(self):
        C.mirror_sources(self.wf, self.dest)
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["bytes_copied"], 0)
        self.assertTrue(all(f["action"] == "unchanged"
                            for n, f in man["files"].items() if n.endswith(".jsonl")))

    def test_append_only_growth_copies_just_the_delta(self):
        C.mirror_sources(self.wf, self.dest)
        with open(os.path.join(self.wf, "agent-a1.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(_asst("m2", [{"type": "text", "text": "two"}],
                           usage={"output_tokens": 6}) + "\n")
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"], "appended")
        self.assertGreater(man["bytes_copied"], 0)
        # and the mirrored copy really does contain both calls
        calls = C.build_agent_calls(os.path.join(self.dest, "agent-a1.jsonl"))
        self.assertEqual(len(calls), 2)

    def test_shrinking_source_never_truncates_the_mirror(self):
        """The mirror is the surviving record; a prune must not propagate."""
        C.mirror_sources(self.wf, self.dest)
        before = os.path.getsize(os.path.join(self.dest, "agent-a1.jsonl"))
        with open(os.path.join(self.wf, "agent-a1.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("")
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"],
                         "source_shrank_mirror_retained")
        self.assertEqual(os.path.getsize(os.path.join(self.dest, "agent-a1.jsonl")), before)
        self.assertGreaterEqual(man["retained"], 1)

    def test_removed_source_keeps_the_mirrored_copy(self):
        C.mirror_sources(self.wf, self.dest)
        os.remove(os.path.join(self.wf, "agent-a1.jsonl"))
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"],
                         "source_missing_mirror_retained")
        self.assertTrue(os.path.exists(os.path.join(self.dest, "agent-a1.jsonl")))

    def test_rewritten_source_preserves_the_previous_capture(self):
        C.mirror_sources(self.wf, self.dest)
        # Same path, longer, but different content at the append boundary.
        self._transcript("a1", [_asst("zz", [{"type": "text", "text": "x" * 200}],
                                      usage={"output_tokens": 9}),
                                _asst("yy", [{"type": "text", "text": "y" * 200}],
                                      usage={"output_tokens": 9})])
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"],
                         "rewritten_previous_kept")
        self.assertTrue(os.path.exists(
            os.path.join(self.dest, "agent-a1.jsonl.gen1")))

    def test_budget_is_reported_not_silently_truncated(self):
        man = C.mirror_sources(self.wf, self.dest, max_bytes=1)
        self.assertGreater(man["skipped_budget"], 0)
        self.assertFalse(man["complete"])

    def test_mirror_is_self_contained_for_rebuild(self):
        """A rebuild off the mirror must keep the run's real lifecycle status."""
        self._journal(["a1"], results=["a1"])
        self._record("completed")
        C.mirror_sources(self.wf, self.dest)
        trace = C.build_trace(self.dest)
        self.assertEqual(len(trace["agents"]), 1)
        self.assertEqual(trace["agents"][0]["totals"]["calls"], 1)
        self.assertEqual(trace["run"]["status"], "complete")
        self.assertEqual(trace["run"]["record_status"], "completed")

    def test_manifest_is_written_for_auditability(self):
        C.mirror_sources(self.wf, self.dest)
        with open(os.path.join(self.dest, "mirror_manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        self.assertIn("files", man)
        self.assertIn("complete", man)

    def test_collect_once_records_mirror_coverage_in_the_trace(self):
        out = os.path.join(self.dir, "t.json")
        trace = C.collect_once(self.wf, out, mirror_dir=self.dest)
        self.assertTrue(trace["run"]["mirror"]["complete"])
        self.assertEqual(trace["run"]["mirror"]["dest"], os.path.abspath(self.dest))

    def test_incomplete_mirror_warns_in_the_trace(self):
        out = os.path.join(self.dir, "t.json")
        import unittest.mock as mock
        real = C.mirror_sources
        with mock.patch.object(C, "mirror_sources",
                               lambda wf, d, **k: dict(real(wf, d, max_bytes=1))):
            trace = C.collect_once(self.wf, out, mirror_dir=self.dest)
        self.assertFalse(trace["run"]["mirror"]["complete"])
        self.assertTrue(any("INCOMPLETE" in w for w in trace["warnings"]))


class IdentityMergeRegressionTest(unittest.TestCase):
    """The three R5 retention counterexamples: merge by identity, not by score."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-idmerge-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_x")
        os.makedirs(self.wf)
        rd = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(rd)
        with open(os.path.join(rd, "wf_x.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_x", "status": "running"}, fh)
        self.out = os.path.join(self.dir, "t.json")

    def _journal(self, with_result=True):
        lines = [_rec(type="launched"),
                 _rec(type="started", key="k", agentId="a1", label="eng", phase="P")]
        if with_result:
            lines.append(_rec(type="result", key="k", agentId="a1", result={"ok": 1}))
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _transcript(self, blocks, tool_result=None):
        lines = [_asst("m1", blocks, stop="tool_use", usage={"output_tokens": 10})]
        if tool_result:
            lines.append(_user([{"type": "tool_result", "tool_use_id": tool_result,
                                 "content": "out"}]))
        with open(os.path.join(self.wf, "agent-a1.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def test_longer_text_does_not_drop_a_captured_tool_action(self):
        self._journal()
        self._transcript([{"type": "tool_use", "id": "t1", "name": "Bash",
                           "input": {"c": "ls"}}], tool_result="t1")
        first = C.collect_once(self.wf, self.out)
        self.assertEqual(len(first["agents"][0]["calls"][0]["actions"]), 1)
        # Re-read with LONGER text but no tool block at all.
        self._transcript([{"type": "text", "text": "a much longer response than before"}])
        second = C.collect_once(self.wf, self.out, previous=first)
        call = second["agents"][0]["calls"][0]
        self.assertEqual(len(call["actions"]), 1, "captured action was dropped")
        self.assertIn("retained", (call.get("capture_note") or "").lower())

    def test_equal_action_count_does_not_lose_a_captured_result(self):
        self._journal()
        self._transcript([{"type": "tool_use", "id": "t1", "name": "Bash",
                           "input": {"c": "ls"}}], tool_result="t1")
        first = C.collect_once(self.wf, self.out)
        self.assertEqual(first["agents"][0]["calls"][0]["actions"][0]["result"]["status"], "ok")
        # Same single action, but the recorded tool_result is gone from the source.
        self._transcript([{"type": "tool_use", "id": "t1", "name": "Bash",
                           "input": {"c": "ls"}}])
        second = C.collect_once(self.wf, self.out, previous=first)
        action = second["agents"][0]["calls"][0]["actions"][0]
        self.assertEqual(action["result"]["status"], "ok",
                         "a recorded result was replaced by 'missing'")
        self.assertTrue(action.get("retained_from_earlier_capture"))

    def test_partial_journal_loss_keeps_return_consistent_with_the_graph(self):
        self._journal(with_result=True)
        self._transcript([{"type": "text", "text": "x"}])
        first = C.collect_once(self.wf, self.out)
        self.assertEqual(first["run"]["agents_returned"], 1)
        self._journal(with_result=False)  # ONLY the result event removed
        second = C.collect_once(self.wf, self.out, previous=first)
        agent = second["agents"][0]
        returns = [e for e in second["edges"] if e["type"] == "return"]
        self.assertEqual(len(returns), 1)
        self.assertEqual(agent["result_status"], "returned_to_workflow",
                         "graph proves a return but the agent says pending")
        self.assertEqual(second["run"]["agents_returned"], 1)
        self.assertTrue(any("retained from earlier passes" in w
                            for w in second["warnings"]))


class RebuildIdentityTest(unittest.TestCase):
    """A rebuild from a mirror must keep the ORIGINAL run id."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-rebuildid-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_real")
        os.makedirs(self.wf)
        rd = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(rd)
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_rec(type="launched") + "\n")
            fh.write(_rec(type="started", key="k", agentId="a1",
                          label="eng", phase="P") + "\n")
        with open(os.path.join(self.wf, "agent-a1.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_asst("m1", [{"type": "text", "text": "x"}],
                           usage={"output_tokens": 3}) + "\n")
        with open(os.path.join(rd, "wf_real.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_real", "status": "completed"}, fh)

    def test_mirror_rebuild_keeps_the_original_run_id(self):
        dest = os.path.join(self.dir, "geak_trace_sources_wf_real")
        C.mirror_sources(self.wf, dest)
        trace = C.build_trace(dest)
        self.assertEqual(trace["run"]["run_id"], "wf_real",
                         "the mirror folder name must not become the run id")
        self.assertEqual(trace["run"]["source_dir_name"], "geak_trace_sources_wf_real")
        self.assertTrue(any("identity restored" in w for w in trace["warnings"]))

    def test_graph_run_node_uses_the_real_run_id(self):
        dest = os.path.join(self.dir, "geak_trace_sources_wf_real")
        C.mirror_sources(self.wf, dest)
        trace = C.build_trace(dest)
        froms = {e["from"] for e in trace["edges"] if e["type"] == "orchestration"}
        self.assertEqual(froms, {"run:wf_real"})

    def test_rebuilt_trace_reconciles_with_its_original_capture(self):
        """Identity preservation is what lets the retention guard accept it."""
        out = os.path.join(self.dir, "t.json")
        original = C.collect_once(self.wf, out)
        dest = os.path.join(self.dir, "geak_trace_sources_wf_real")
        C.mirror_sources(self.wf, dest)
        rebuilt = C.build_trace(dest)
        merged = C.merge_traces(original, rebuilt)
        self.assertFalse(any("belongs to run" in w for w in merged["warnings"]),
                         "rebuilt trace was rejected as a different run")


class MirrorExactnessTest(unittest.TestCase):
    """Astra R7: append eligibility must be verified by content, and each
    superseded generation must be retained separately and reconciled."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-exact-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_e")
        os.makedirs(self.wf)
        rd = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(rd)
        with open(os.path.join(rd, "wf_e.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_e", "status": "running"}, fh)
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_rec(type="launched") + "\n")
            fh.write(_rec(type="started", key="k", agentId="a1",
                          label="eng", phase="P") + "\n")
        self.src = os.path.join(self.wf, "agent-a1.jsonl")
        self.dest = os.path.join(self.dir, "mirror")

    def _write(self, mids, filler="y"):
        with open(self.src, "w", encoding="utf-8") as fh:
            for m in mids:
                fh.write(_asst(m, [{"type": "text", "text": filler * 40}],
                               usage={"output_tokens": 5}) + "\n")

    def test_same_size_rewrite_is_detected_not_called_unchanged(self):
        self._write(["m1"])
        C.mirror_sources(self.wf, self.dest)
        self._write(["m2"])  # same byte length, different content
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"],
                         "rewritten_same_size_previous_kept")
        self.assertTrue(os.path.exists(os.path.join(self.dest, "agent-a1.jsonl.gen1")))

    def test_changed_early_bytes_are_not_reported_as_append(self):
        """A boundary sample cannot establish whole-prefix identity."""
        self._write(["m1"], filler="a")
        C.mirror_sources(self.wf, self.dest)
        # rewrite the EARLY content, then append a new call
        self._write(["m1", "m2"], filler="b")
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"],
                         "rewritten_previous_kept")
        mirrored = C.build_agent_calls(os.path.join(self.dest, "agent-a1.jsonl"))
        source = C.build_agent_calls(self.src)
        self.assertEqual([c["output_text"] for c in mirrored],
                         [c["output_text"] for c in source])

    def test_each_generation_is_kept_separately(self):
        for filler in ("a", "b", "c"):
            self._write(["m1"], filler=filler)
            C.mirror_sources(self.wf, self.dest)
        self.assertTrue(os.path.exists(os.path.join(self.dest, "agent-a1.jsonl.gen1")))
        self.assertTrue(os.path.exists(os.path.join(self.dest, "agent-a1.jsonl.gen2")))

    def test_rebuild_reconciles_calls_across_generations(self):
        self._write(["m1"])
        C.mirror_sources(self.wf, self.dest)
        self._write(["m2"])          # replaces m1 entirely, same size
        C.mirror_sources(self.wf, self.dest)
        trace = C.build_trace(self.dest)
        ids = {c["call_id"] for a in trace["agents"] for c in a["calls"]}
        self.assertEqual(ids, {"m1", "m2"}, "a retained generation was ignored")

    def test_rewrite_budget_uses_the_actual_copy_size(self):
        self._write(["m1"])
        C.mirror_sources(self.wf, self.dest)
        before = os.path.getsize(os.path.join(self.dest, "agent-a1.jsonl"))
        self._write(["m1", "m2"], filler="z")  # rewrite + growth
        man = C.mirror_sources(self.wf, self.dest, max_bytes=1)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"], "skipped_budget")
        self.assertEqual(man["bytes_copied"], 0)
        self.assertFalse(man["complete"])
        self.assertEqual(os.path.getsize(os.path.join(self.dest, "agent-a1.jsonl")),
                         before)

    def test_divergent_shrink_keeps_both(self):
        self._write(["m1", "m2"])
        C.mirror_sources(self.wf, self.dest)
        self._write(["m3"])  # shorter AND different
        man = C.mirror_sources(self.wf, self.dest)
        self.assertEqual(man["files"]["agent-a1.jsonl"]["action"],
                         "source_shrank_divergent_both_kept")
        trace = C.build_trace(self.dest)
        ids = {c["call_id"] for a in trace["agents"] for c in a["calls"]}
        self.assertEqual(ids, {"m1", "m2", "m3"})

    def test_generations_are_not_themselves_mirrored_as_sources(self):
        self._write(["m1"])
        C.mirror_sources(self.wf, self.dest)
        self._write(["m2"])
        man = C.mirror_sources(self.wf, self.dest)
        self.assertFalse(any(".gen" in n for n in man["files"]))


class RetentionPayloadTest(unittest.TestCase):
    """Astra R7 #5: retained payloads must not shrink or collapse."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-payload-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_p")
        os.makedirs(self.wf)
        rd = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(rd)
        with open(os.path.join(rd, "wf_p.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_p", "status": "running"}, fh)
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(_rec(type="launched") + "\n")
            fh.write(_rec(type="started", key="k", agentId="a1",
                          label="eng", phase="P") + "\n")
        self.out = os.path.join(self.dir, "t.json")

    def _tr(self, result_text, input_text="previously captured input"):
        lines = [_user([{"type": "text", "text": input_text}]),
                 _asst("m1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                               "input": {"c": "ls"}}], stop="tool_use",
                       usage={"output_tokens": 10}),
                 _user([{"type": "tool_result", "tool_use_id": "t1",
                         "content": result_text}])]
        with open(os.path.join(self.wf, "agent-a1.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def test_shrinking_but_ok_tool_result_is_not_lost(self):
        self._tr("captured tool output")
        first = C.collect_once(self.wf, self.out)
        self._tr("x")  # same ok status, much shorter payload
        second = C.collect_once(self.wf, self.out, previous=first)
        res = second["agents"][0]["calls"][0]["actions"][0]["result"]
        self.assertIn("captured tool output", res["preview"])

    def test_same_count_input_payload_is_not_lost(self):
        self._tr("out", input_text="previously captured input")
        first = C.collect_once(self.wf, self.out)
        self._tr("out", input_text="x")  # one block either way, shorter text
        second = C.collect_once(self.wf, self.out, previous=first)
        blocks = second["agents"][0]["calls"][0]["input"]["blocks"]
        self.assertTrue(any("previously captured input" in (b.get("text") or "")
                            for b in blocks))

    def test_distinct_transfer_edges_do_not_collapse(self):
        prev = {"schema": C.SCHEMA, "run": {"run_id": "wf_p"}, "agents": [],
                "edges": [
                    {"type": "result_supplied_to_dispatch", "from": "agent:p",
                     "to": "agent:c", "event_id": "t1"},
                    {"type": "result_supplied_to_dispatch", "from": "agent:p",
                     "to": "agent:c", "event_id": "t2"}],
                "warnings": []}
        cur = {"schema": C.SCHEMA, "run": {"run_id": "wf_p"},
               "agents": [{"agent_id": "x", "calls": [], "totals": {},
                           "result_status": "pending_or_absent"}],
               "edges": [{"type": "result_supplied_to_dispatch", "from": "agent:p",
                          "to": "agent:c", "event_id": "t1"}],
               "warnings": []}
        merged = C.merge_traces(prev, cur)
        ids = {e.get("event_id") for e in merged["edges"]}
        self.assertEqual(ids, {"t1", "t2"}, "distinct transfers collapsed")

    def test_spawn_edge_keeps_its_attempts_when_return_disappears(self):
        prev = {"schema": C.SCHEMA, "run": {"run_id": "wf_p"}, "agents": [],
                "edges": [{"type": "agent_spawn", "from": "agent:p", "to": "agent:c",
                           "spawn_event_id": "s1", "return_status": "returned",
                           "attempts": [{"attempt_id": "a1", "status": "returned"}]}],
                "warnings": []}
        cur = {"schema": C.SCHEMA, "run": {"run_id": "wf_p"},
               "agents": [{"agent_id": "x", "calls": [], "totals": {},
                           "result_status": "pending_or_absent"}],
               "edges": [{"type": "agent_spawn", "from": "agent:p", "to": "agent:c",
                          "spawn_event_id": "s1", "return_status": "unmatched",
                          "attempts": []}],
               "warnings": []}
        merged = C.merge_traces(prev, cur)
        edge = next(e for e in merged["edges"] if e["type"] == "agent_spawn")
        self.assertEqual(edge["return_status"], "returned")
        self.assertEqual(len(edge["attempts"]), 1)


class WorkflowTimelinePhaseTest(unittest.TestCase):
    """A nested lane's phases survive via the workflow's own recorded timeline."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="geak-tl-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.wf = os.path.join(self.dir, "sess", "subagents", "workflows", "wf_t")
        os.makedirs(self.wf)
        self.rd = os.path.join(self.dir, "sess", "workflows")
        os.makedirs(self.rd)
        lines = [_rec(type="launched")]
        for aid, label in (("a1", "director:setup"), ("a2", "tech_lead:analyze"),
                           ("a3", "benchmark_engineer")):
            # The parent journal collapses a nested lane to ONE phase.
            lines.append(_rec(type="started", key="k" + aid, agentId=aid,
                              label=label, phase="▸ kernel-lane"))
        with open(os.path.join(self.wf, "journal.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _record(self, events):
        doc = {"runId": "wf_t", "status": "completed",
               "result": {"llm_timeline": {"schema": "geak.agent_timeline/1",
                                           "workflow": "kernel_lane",
                                           "events": events, "nested": []}}}
        with open(os.path.join(self.rd, "wf_t.json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

    def test_recorded_timeline_supplies_the_real_phases(self):
        self._record([{"seq": 0, "phase": "Setup", "label": "director:setup"},
                      {"seq": 1, "phase": "Analyze", "label": "tech_lead:analyze"},
                      {"seq": 2, "phase": "Benchmark", "label": "benchmark_engineer"}])
        trace = C.build_trace(self.wf)
        phases = [a.get("timeline_phase") for a in trace["agents"]]
        self.assertEqual(phases, ["Setup", "Analyze", "Benchmark"])
        self.assertTrue(all(a.get("phase_provenance") == "workflow_timeline"
                            for a in trace["agents"]))
        self.assertTrue(any("workflow's OWN recorded timeline" in w
                            for w in trace["warnings"]))

    def test_repeated_labels_join_in_dispatch_order(self):
        with open(os.path.join(self.wf, "journal.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(_rec(type="started", key="k4", agentId="a4",
                          label="director:setup", phase="x") + "\n")
        self._record([{"seq": 0, "phase": "Setup", "label": "director:setup"},
                      {"seq": 1, "phase": "Analyze", "label": "tech_lead:analyze"},
                      {"seq": 2, "phase": "Benchmark", "label": "benchmark_engineer"},
                      {"seq": 3, "phase": "Finalize", "label": "director:setup"}])
        trace = C.build_trace(self.wf)
        by_id = {a["agent_id"]: a for a in trace["agents"]}
        self.assertEqual(by_id["a1"]["timeline_phase"], "Setup")
        self.assertEqual(by_id["a4"]["timeline_phase"], "Finalize")

    def test_agents_without_a_timeline_entry_are_reported(self):
        self._record([{"seq": 0, "phase": "Setup", "label": "director:setup"}])
        trace = C.build_trace(self.wf)
        self.assertTrue(any("no timeline entry" in w for w in trace["warnings"]))

    def test_absent_timeline_falls_back_without_inventing(self):
        with open(os.path.join(self.rd, "wf_t.json"), "w", encoding="utf-8") as fh:
            json.dump({"runId": "wf_t", "status": "completed"}, fh)
        trace = C.build_trace(self.wf)
        self.assertTrue(all(a.get("timeline_phase") is None for a in trace["agents"]))

    def test_timeline_survives_a_mirror_rebuild(self):
        self._record([{"seq": 0, "phase": "Setup", "label": "director:setup"},
                      {"seq": 1, "phase": "Analyze", "label": "tech_lead:analyze"},
                      {"seq": 2, "phase": "Benchmark", "label": "benchmark_engineer"}])
        dest = os.path.join(self.dir, "mirror")
        C.mirror_sources(self.wf, dest)
        trace = C.build_trace(dest)
        self.assertEqual([a.get("timeline_phase") for a in trace["agents"]],
                         ["Setup", "Analyze", "Benchmark"])
