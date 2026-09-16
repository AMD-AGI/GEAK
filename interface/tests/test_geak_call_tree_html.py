"""Tests for the role-execution-tree report renderer (geak_call_tree_html).

These lock the two things the renderer decides that the ledger does not:
how flat call rows fold into agent nodes, and how those nodes nest into a
delegation tree by role rank + execution order. Everything else (cost, tokens)
is summed straight from ``llm_calls.jsonl`` fields.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geak_call_tree_html as R  # noqa: E402


def call(role, sub_phase, ts_ms, label=None, model="claude-opus-5",
         inp=0, cr=0, cw5=0, cw1=0, out=0, cost=None, breakdown=None,
         prompt="", output="", thinking="", duration_ms=1000.0):
    """One llm_calls.jsonl-shaped row."""
    if breakdown is None:
        breakdown = {"cache_write": 0.0, "cache_read": 0.0,
                     "uncached_input": 0.0, "router": 0.0, "output": 0.0}
    if cost is None:
        cost = sum(breakdown.values())
    return {
        "role": role, "sub_phase": sub_phase, "phase": "P",
        "agent_label": label if label is not None else "%s:%s" % (role, sub_phase),
        "attribution": "", "model": model,
        "ts_ms": ts_ms, "ts": "2026-01-01T00:00:%02dZ" % (ts_ms % 60),
        "duration_ms": duration_ms,
        "input_tokens": inp, "cache_read_input_tokens": cr,
        "cache_write_5m_tokens": cw5, "cache_write_1h_tokens": cw1,
        "output_tokens": out, "cost_usd": cost, "cost_breakdown": breakdown,
        "prompt": prompt, "output": output, "thinking": thinking,
    }


class TestAgentize(unittest.TestCase):
    def test_calls_fold_into_one_node_per_conversation(self):
        rows = [call("engineer", "d1", 10, label="engineer:d1"),
                call("engineer", "d1", 20, label="engineer:d1"),
                call("verify", "d1", 30, label="verify:d1")]
        nodes = R.agentize(rows)
        self.assertEqual(len(nodes), 2)
        eng = next(n for n in nodes if n["role"] == "engineer")
        self.assertEqual(eng["calls"], 2)

    def test_node_aggregates_tokens_cost_and_first_ts(self):
        bd = {"cache_write": 1.0, "cache_read": 2.0, "uncached_input": 0.5,
              "router": 0.0, "output": 0.25}
        rows = [call("engineer", "d1", 50, out=100, breakdown=dict(bd)),
                call("engineer", "d1", 20, out=40, breakdown=dict(bd))]
        node = R.agentize(rows)[0]
        self.assertEqual(node["calls"], 2)
        self.assertEqual(node["tokens"]["output"], 140)
        self.assertEqual(node["ts_ms"], 20)  # earliest wins
        self.assertAlmostEqual(node["cost"]["cache_read"], 4.0)
        self.assertAlmostEqual(node["cost_usd"], 2 * sum(bd.values()))

    def test_output_and_thinking_concatenated(self):
        rows = [call("engineer", "d1", 10, output="first", thinking="plan-a"),
                call("engineer", "d1", 20, output="second", thinking="")]
        d = R.node_detail(R.agentize(rows)[0])
        self.assertIn("first", d["output"])
        self.assertIn("second", d["output"])
        self.assertEqual(d["thinking"], "plan-a")

    def test_agent_attempt_expands_to_individual_api_calls(self):
        # An agent attempt is not one call: 3 API responses stay distinct.
        rows = [call("engineer", "d1", 10, out=5),
                call("engineer", "d1", 20, out=7),
                call("engineer", "d1", 30, out=9)]
        d = R.node_detail(R.agentize(rows)[0])
        self.assertEqual(d["calls"], 1 if False else 3)  # 3 API calls under 1 agent
        self.assertEqual(len(d["api_calls"]), 3)
        self.assertEqual([c["tokens"]["output"] for c in d["api_calls"]], [5, 7, 9])

    def test_input_cost_is_labelled_subtotal_of_input_leaves(self):
        bd = {"cache_write": 1.0, "cache_read": 2.0, "uncached_input": 0.5,
              "router": 0.0, "output": 0.25}
        d = R.node_detail(R.agentize([call("engineer", "d1", 10, breakdown=bd)])[0])
        # subtotal is the three input leaves, NOT including output or router
        self.assertAlmostEqual(d["input_cost_subtotal"], 3.5)
        self.assertNotAlmostEqual(d["input_cost_subtotal"], d["cost_usd"])


class TestBuildTree(unittest.TestCase):
    def _tree(self):
        rows = [
            call("director", "", 10),
            call("tech_lead", "", 20),
            call("engineer", "d1", 30),
            call("verify", "d1", 40),
            call("engineer", "d2", 50),
            call("profiler", "", 60),   # rank 2: nests back under tech_lead
        ]
        return R.build_tree(R.agentize(rows))

    def test_director_is_top_level_under_root(self):
        root = self._tree()
        self.assertEqual(len(root["children"]), 1)
        self.assertEqual(root["children"][0]["role"], "director")

    def test_tech_lead_nests_under_director(self):
        director = self._tree()["children"][0]
        self.assertEqual([c["role"] for c in director["children"]], ["tech_lead"])

    def test_engineers_and_verify_are_siblings_under_tech_lead(self):
        tl = self._tree()["children"][0]["children"][0]
        roles = [c["role"] for c in tl["children"]]
        self.assertEqual(roles, ["engineer", "verify", "engineer", "profiler"])

    def test_unknown_role_becomes_leaf_in_order(self):
        rows = [call("director", "", 10), call("mystery_scope", "", 20)]
        root = R.build_tree(R.agentize(rows))
        # unknown rank (LEAF) > director → nests under director
        director = root["children"][0]
        self.assertEqual(director["children"][0]["role"], "mystery_scope")

    def test_ordering_is_by_first_timestamp_not_input_order(self):
        rows = [call("engineer", "d2", 90), call("director", "", 10),
                call("tech_lead", "", 20)]
        root = R.build_tree(R.agentize(rows))
        self.assertEqual(root["children"][0]["role"], "director")


class TestRunTotals(unittest.TestCase):
    def test_totals_sum_across_nodes_and_split_by_model(self):
        rows = [
            call("director", "", 10, model="claude-opus-5", out=10,
                 breakdown={"cache_write": 1.0, "cache_read": 0.0,
                            "uncached_input": 0.0, "router": 0.0, "output": 0.0}),
            call("engineer", "d1", 20, model="claude-sonnet-5", out=20,
                 breakdown={"cache_write": 0.0, "cache_read": 2.0,
                            "uncached_input": 0.0, "router": 0.0, "output": 0.5}),
        ]
        nodes = R.agentize(rows)
        total, per_model = R.run_totals(nodes)
        self.assertEqual(total["calls"], 2)
        self.assertAlmostEqual(total["cost_usd"], 1.0 + 2.5)
        self.assertAlmostEqual(total["cost"]["cache_read"], 2.0)
        self.assertEqual(total["tokens"]["output"], 30)
        self.assertEqual(set(per_model), {"claude-opus-5", "claude-sonnet-5"})


class TestCompleteness(unittest.TestCase):
    def test_three_distinct_signals(self):
        rows = [call("engineer", "d1", 10), call("verify", "d1", 20)]
        rows[0]["stop_reason"] = "end_turn"
        rows[1]["stop_reason"] = "tool_use"
        c = R.completeness(rows, meta={"complete": True, "warnings": [],
                                       "attribution_mode": "recorded"})
        self.assertTrue(c["workflow_completion"]["complete"])
        self.assertEqual(c["capture_completeness"]["api_calls"], 2)
        self.assertEqual(c["capture_completeness"]["incomplete_output"], 0)
        self.assertIn("child-scope", c["cost_coverage"])

    def test_missing_terminal_stop_is_flagged_incomplete(self):
        rows = [call("engineer", "d1", 10), call("engineer", "d1", 20)]
        rows[0]["stop_reason"] = "end_turn"
        rows[1]["stop_reason"] = None   # captured mid-flight
        c = R.completeness(rows)
        self.assertEqual(c["capture_completeness"]["incomplete_output"], 1)
        self.assertFalse(c["capture_completeness"]["complete"])
        self.assertIsNone(c["workflow_completion"]["complete"])  # no meta -> unknown

    def test_completeness_surfaces_in_outputs(self):
        rows = [call("engineer", "d1", 10)]
        rows[0]["stop_reason"] = None
        html, md = R.render(rows, "M", meta={"complete": False,
                                             "warnings": ["no agent_timeline.json"]})
        self.assertIn("capture", md.lower())
        self.assertIn("mid-flight", html.lower())


class TestRender(unittest.TestCase):
    def _rows(self):
        return [call("director", "", 10, output="dir-out", prompt="dir-prompt"),
                call("tech_lead", "", 20),
                call("engineer", "d1", 30, output="eng-out")]

    def test_html_is_self_contained_and_embeds_tree(self):
        html, _ = R.render(self._rows(), "MODELX")
        self.assertIn("<!doctype html>", html.lower())
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)   # no external deps
        self.assertIn("director", html)
        self.assertIn("MODELX", html)
        # embedded JSON payload parses
        start = html.index('type="application/json">') + len('type="application/json">')
        end = html.index("</script>", start)
        payload = json.loads(html[start:end].replace("<\\/", "</"))
        self.assertEqual(payload["model"], "MODELX")
        self.assertEqual(payload["tree"]["children"][0]["title"], "director")

    def test_markdown_has_tree_and_details(self):
        _, md = R.render(self._rows(), "MODELX")
        self.assertIn("# GEAK run report — MODELX", md)
        self.assertIn("## Role view (organizational)", md)
        self.assertIn("director", md)
        self.assertIn("engineer:d1", md)

    def test_write_emits_both_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "llm_calls.jsonl")
            with open(calls, "w") as fh:
                for r in self._rows():
                    fh.write(json.dumps(r) + "\n")
            hp, mp = R.write(calls, os.path.join(d, "report"), "MODELX")
            self.assertTrue(os.path.isfile(hp))
            self.assertTrue(os.path.isfile(mp))
            self.assertTrue(hp.endswith("geak_run_report_MODELX.html"))

    def test_empty_ledger_renders_without_error(self):
        html, md = R.render([], "EMPTY")
        self.assertIn("EMPTY", html)
        self.assertIn("No agent calls found", html)
        self.assertIn("# GEAK run report — EMPTY", md)

    def test_captured_text_is_escaped_not_executable(self):
        # A prompt/output containing HTML/script must not survive as live markup.
        evil = "<script>alert('x')</script><img src=x onerror=alert(1)>"
        rows = [call("engineer", "d1", 10, prompt=evil, output=evil)]
        html, _ = R.render(rows, "SAFE")
        # The raw payload lives only inside the JSON <script type=application/json>;
        # it must never appear as a live <script>/<img> tag in the document body.
        start = html.index('type="application/json">')
        body_before = html[:start]
        self.assertNotIn("<script>alert", body_before)
        self.assertNotIn("onerror=alert", body_before)

    def test_organizational_view_is_labelled_not_a_spawn_tree(self):
        html, md = R.render([call("director", "", 10)], "M")
        self.assertIn("organizational", html.lower())
        self.assertIn("not a literal spawn tree", md.lower())


class TestAcceptanceFixtures(unittest.TestCase):
    """Astra's fixture list: duplicate usage, multi-block, retry/resume, missing
    terminal usage, nested workflows — node totals must agree with the summed rows."""

    def test_node_totals_agree_with_summed_rows(self):
        bd = {"cache_write": 0.5, "cache_read": 0.5, "uncached_input": 0.1,
              "router": 0.0, "output": 0.2}
        rows = [call("engineer", "d1", 10, out=3, breakdown=dict(bd)),
                call("engineer", "d1", 20, out=4, breakdown=dict(bd)),
                call("verify", "d1", 30, out=5, breakdown=dict(bd))]
        nodes = R.agentize(rows)
        total, _ = R.run_totals(nodes)
        self.assertAlmostEqual(total["cost_usd"], 3 * sum(bd.values()))
        self.assertEqual(total["tokens"]["output"], 12)
        # per-node own totals sum to the run total (no double counting)
        self.assertAlmostEqual(sum(R.node_detail(n)["cost_usd"] for n in nodes),
                               total["cost_usd"])

    def test_missing_optional_fields_do_not_crash(self):
        # A row missing tokens/cost/model/ts still renders.
        bare = {"role": "engineer", "sub_phase": "d1", "agent_label": "engineer:d1"}
        html, md = R.render([bare], "BARE")
        self.assertIn("engineer", html)
        self.assertIn("engineer", md)


if __name__ == "__main__":
    unittest.main()
