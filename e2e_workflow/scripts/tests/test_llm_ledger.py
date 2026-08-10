#!/usr/bin/env python3
"""Unit tests for llm_ledger.py -- the run's per-API-call token + time ledger.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_llm_ledger.py

This script is the only thing that will ever say what a GEAK run cost, and nothing downstream can
sanity-check it -- a wrong number here does not fail loudly, it just becomes the number everyone
quotes. So the tests pin the four places it could quietly lie:

  - calls_of / dedupe          : the SAME response is written to a transcript twice (identical
                                 message.id). Counting both inflates every figure in the report by
                                 roughly a factor of two, which is the single most dangerous bug
                                 this file can have.
  - cost_of                    : four token classes at four different prices, plus the 5-minute vs
                                 1-hour cache-write split. Mixing them up misprices the run.
  - attribute                  : phase attribution. `op_benchmarker:bakeoff` runs in BOTH HeadKernel
                                 and Milestone and `director:setup` runs in BOTH the e2e and the
                                 kernel layer, so prompt text alone cannot place them -- these pin
                                 that the recorded timeline resolves them and that its absence is
                                 reported as "inferred" rather than guessed silently.
  - build / soft failure       : an unreadable or missing transcript must yield an INCOMPLETE ledger
                                 with a stated reason, never an exception -- this runs at the end of
                                 a multi-hour GPU run.

Plus a structural guard on the three instrumented workflow JS files: there is no node/deno on the
CI runner, so bracket balance is the one machine-checkable property of that edit, and it is exactly
the class of mistake that wrapping a call in a helper introduces.

Stdlib only, no GPU, no network -- everything is driven from synthetic transcripts in tempdirs.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEAK_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = _load("llm_ledger_under_test", "llm_ledger.py")


# --------------------------------------------------------------------------- #
# Synthetic transcript builders
# --------------------------------------------------------------------------- #
def _ts(sec):
    """A deterministic ISO timestamp `sec` seconds into 2026-08-10T00:00:00Z."""
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return "2026-08-10T%02d:%02d:%02d.000Z" % (h, m, s)


def user_rec(text, sec):
    return {"type": "user", "timestamp": _ts(sec), "message": {"role": "user", "content": text}}


def asst_rec(sec, mid, inp=0, read=0, write5=0, write1h=0, out=0, model="claude-opus-4-8"):
    return {
        "type": "assistant", "timestamp": _ts(sec), "requestId": "req_" + mid,
        "message": {
            "id": mid, "model": model, "role": "assistant", "stop_reason": "tool_use",
            "usage": {
                "input_tokens": inp,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write5 + write1h,
                "cache_creation": {"ephemeral_5m_input_tokens": write5,
                                   "ephemeral_1h_input_tokens": write1h},
                "output_tokens": out, "service_tier": "standard",
            },
        },
    }


def prompt_for(role, subphase, eval_dir, extra=""):
    """Byte-shaped like roleAgent() in the workflows: the line the ledger keys on."""
    return ("You are the %s. PHASE=%s.\nFirst Read /wf/roles/%s.md and follow its instructions.\n\n"
            "## Inputs\n- EVAL_DIR: %s\n%s" % (role, subphase, role, eval_dir, extra))


def write_transcript(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def timeline(events, workflow="e2e_workflow", nested=None):
    return {"schema": "geak.agent_timeline/1", "workflow": workflow,
            "events": [dict(seq=i, **e) for i, e in enumerate(events)],
            "nested": nested or []}


def ev(phase, label, attempt=1, ok=True):
    return {"phase": phase, "label": label, "attempt": attempt, "ok": ok}


class LedgerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="llm_ledger_test_")
        self.eval_dir = os.path.join(self.tmp, "exp", "e2e_run")
        self.tdir = os.path.join(self.tmp, "transcripts")
        os.makedirs(self.eval_dir)
        os.makedirs(self.tdir)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def glob(self):
        return [os.path.join(self.tdir, "*.jsonl")]

    def put_timeline(self, doc, sub=None):
        d = os.path.join(sub or self.eval_dir, "reports", "trace")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "agent_timeline.json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

    def build(self, **kw):
        return L.build(self.eval_dir, self.glob(), **kw)


# --------------------------------------------------------------------------- #
class TestDedupe(LedgerTestBase):
    def test_repeated_message_id_counted_once(self):
        """The same response flushed twice must not double the run's cost."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 0),
            asst_rec(1, "msg_1", read=1000, out=10),
            asst_rec(1, "msg_1", read=1000, out=10),   # duplicate flush of the SAME call
            asst_rec(2, "msg_2", read=2000, out=20),
        ])
        rows, _, agg, _ = self.build()
        self.assertEqual(len(rows), 2)
        self.assertEqual(agg["total"]["cache_read_input_tokens"], 3000)
        self.assertEqual(agg["total"]["output_tokens"], 30)

    def test_partial_flush_keeps_the_larger_output(self):
        """A truncated first flush must not undercount the final response."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 0),
            asst_rec(1, "msg_1", read=1000, out=5),
            asst_rec(1, "msg_1", read=1000, out=900),
        ])
        rows, _, _, _ = self.build()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["output_tokens"], 900)


class TestCost(LedgerTestBase):
    def test_each_token_class_is_priced_separately(self):
        row = {"model": "claude-opus-4-8", "input_tokens": 1_000_000,
               "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 2_000_000,
               "cache_write_5m_tokens": 1_000_000, "cache_write_1h_tokens": 1_000_000,
               "output_tokens": 1_000_000}
        # 5.00 fresh + 0.50 read + 6.25 5m-write + 10.00 1h-write + 25.00 out
        self.assertAlmostEqual(L.cost_of(row, L.DEFAULT_RATES), 46.75, places=6)

    def test_list_cost_prices_every_input_token_fresh(self):
        row = {"model": "m", "input_tokens": 0, "cache_read_input_tokens": 1_000_000,
               "cache_creation_input_tokens": 1_000_000, "cache_write_5m_tokens": 1_000_000,
               "cache_write_1h_tokens": 0, "output_tokens": 0}
        # The no-reuse counterfactual must NOT include the storage surcharge, which
        # only exists because reuse is switched on.
        self.assertAlmostEqual(L.list_cost_of(row, L.DEFAULT_RATES), 10.00, places=6)

    def test_missing_cache_split_defaults_to_five_minute(self):
        """Older CLIs omit the 5m/1h breakdown; everything stored is a 5m write."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 0),
            {"type": "assistant", "timestamp": _ts(1), "requestId": "r",
             "message": {"id": "m", "model": "x",
                         "usage": {"cache_creation_input_tokens": 4000}}},
        ])
        rows, _, _, _ = self.build()
        self.assertEqual(rows[0]["cache_write_5m_tokens"], 4000)
        self.assertEqual(rows[0]["cache_write_1h_tokens"], 0)


class TestPhaseAttribution(LedgerTestBase):
    def test_timeline_resolves_bakeoff_running_in_two_phases(self):
        """`bakeoff` runs in HeadKernel AND Milestone -- the recorded order decides."""
        self.put_timeline(timeline([
            ev("HeadKernel", "op_benchmarker:bakeoff:h0"),
            ev("Milestone", "op_benchmarker:bakeoff:k1"),
        ]))
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("op_benchmarker", "bakeoff", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        write_transcript(os.path.join(self.tdir, "b.jsonl"), [
            user_rec(prompt_for("op_benchmarker", "bakeoff", self.eval_dir), 100),
            asst_rec(101, "m2", read=200, out=2),
        ])
        rows, _, agg, meta = self.build()
        self.assertEqual(meta["attribution_mode"], "timeline")
        by_ts = {r["ts_ms"]: r for r in rows}
        first, second = sorted(by_ts)
        self.assertEqual(by_ts[first]["phase"], "HeadKernel")
        self.assertEqual(by_ts[second]["phase"], "Milestone")
        self.assertEqual(by_ts[first]["agent_label"], "op_benchmarker:bakeoff:h0")
        self.assertIn("HeadKernel", agg["by_phase"])
        self.assertIn("Milestone", agg["by_phase"])

    def test_without_a_timeline_it_says_inferred_rather_than_guessing(self):
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("op_benchmarker", "bakeoff", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        rows, _, _, meta = self.build()
        self.assertEqual(meta["attribution_mode"], "inferred")
        self.assertFalse(meta["complete"])
        self.assertTrue(any("agent_timeline" in w for w in meta["warnings"]))
        # It must not invent a phase name that looks authoritative.
        self.assertTrue(rows[0]["phase"].startswith(L.UNATTRIBUTED))

    def test_director_setup_is_split_between_the_two_layers(self):
        """`director:setup` exists in BOTH workflows; the kernel eval dir decides."""
        kdir = os.path.join(self.eval_dir, "kernels", "_exp", "team_k1")
        self.put_timeline(timeline(
            [ev("Setup", "director:setup")],
            nested=[timeline([ev("Setup", "director:setup")], workflow="kernel_lane")],
        ))
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("director", "setup", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        write_transcript(os.path.join(self.tdir, "b.jsonl"), [
            user_rec(prompt_for("director", "setup", kdir), 100),
            asst_rec(101, "m2", read=100, out=1),
        ])
        rows, _, _, _ = self.build()
        phases = {r["ts_ms"]: r["phase"] for r in rows}
        first, second = sorted(phases)
        self.assertEqual(phases[first], "Setup")
        self.assertEqual(phases[second], "kernel/Setup")

    def test_nested_kernel_phases_are_prefixed(self):
        """A nested phase must be distinguishable from the e2e phase it runs inside."""
        self.put_timeline(timeline([ev("Milestone", "system_architect:plan_milestone")], nested=[
            timeline([ev("Optimize", "tech_lead:plan_round")], workflow="kernel_lane")]))
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("tech_lead", "plan_round", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        rows, _, _, _ = self.build()
        self.assertEqual(rows[0]["phase"], "kernel/Optimize")

    def test_retries_consume_successive_timeline_slots(self):
        self.put_timeline(timeline([
            ev("Profile", "profiler:baseline", attempt=1, ok=False),
            ev("Profile", "profiler:baseline", attempt=2, ok=True),
        ]))
        for i, name in enumerate(("a.jsonl", "b.jsonl")):
            write_transcript(os.path.join(self.tdir, name), [
                user_rec(prompt_for("profiler", "baseline", self.eval_dir), i * 100),
                asst_rec(i * 100 + 1, "m%d" % i, read=100, out=1),
            ])
        _, agent_rows, agg, _ = self.build()
        self.assertEqual(agg["by_phase"]["Profile"]["calls"], 2)
        self.assertEqual([a["attempt"] for a in agent_rows if a["api_calls"]], [1, 2])


class TestAgentRows(LedgerTestBase):
    def test_attempt_without_a_transcript_is_reported_with_zero_calls(self):
        """An attempt that never answered must still appear, or a retry storm is invisible.

        Two attempts were recorded but only one conversation exists, so one slot is
        left over. Which one is left over follows the documented positional rule
        (conversations fill slots in order), so this asserts the leftover EXISTS and
        is labelled honestly, not which of the two it happened to be -- the ledger
        cannot know that, and pretending otherwise would be the bug.
        """
        self.put_timeline(timeline([
            ev("Setup", "director:setup", attempt=1, ok=False),
            ev("Setup", "director:setup", attempt=2, ok=True),
        ]))
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("director", "setup", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        _, agent_rows, agg, _ = self.build()
        silent = [a for a in agent_rows if a["api_calls"] == 0]
        self.assertEqual(len(silent), 1)
        self.assertIn("no transcript", silent[0]["attribution"])
        self.assertEqual(silent[0]["phase"], "Setup")
        # The workflow's own count is what makes the retry visible at all.
        self.assertEqual(agg["total"]["agents"], 2)
        self.assertEqual(agg["total"]["agent_attempts_failed"], 1)
        self.assertEqual(agg["total"]["conversations"], 1)

    def test_span_and_llm_time_are_separate(self):
        """Span includes the tool work between calls; llm_ms is only model time."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 0),
            asst_rec(2, "m1", read=100, out=1),      # 2s waiting on the model
            user_rec("tool result", 60),             # 58s of benchmarking
            asst_rec(63, "m2", read=100, out=1),     # 3s waiting on the model
        ])
        _, agent_rows, _, _ = self.build()
        row = agent_rows[0]
        self.assertEqual(row["span_ms"], 61_000)
        self.assertEqual(row["llm_ms"], 5_000)


class TestSoftFailure(LedgerTestBase):
    def test_unreadable_transcript_does_not_raise(self):
        with open(os.path.join(self.tdir, "bad.jsonl"), "wb") as fh:
            fh.write(b"\x00\x01 not json at all\n{\"type\": \"assistant\"\n")
        write_transcript(os.path.join(self.tdir, "good.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        rows, _, _, meta = self.build()
        self.assertEqual(len(rows), 1)          # the good file still counts
        self.assertTrue(meta["eval_dir"])

    def test_no_transcripts_yields_an_incomplete_ledger_not_a_crash(self):
        rows, agent_rows, agg, meta = self.build()
        self.assertEqual(rows, [])
        self.assertFalse(meta["complete"])
        self.assertTrue(any("no transcripts" in w for w in meta["warnings"]))
        self.assertEqual(agg["total"]["calls"], 0)

    def test_main_returns_zero_even_when_everything_is_missing(self):
        """The CLI is called at the end of a real run; it must never fail it."""
        rc = L.main(["--eval-dir", os.path.join(self.tmp, "does", "not", "exist"), "--quiet"])
        self.assertEqual(rc, 0)


class TestDiscovery(LedgerTestBase):
    def test_only_transcripts_mentioning_this_eval_dir_are_used(self):
        roots = os.path.join(self.tmp, "claude")
        proj = os.path.join(roots, "projects", "p")
        os.makedirs(proj)
        write_transcript(os.path.join(proj, "mine.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1)])
        write_transcript(os.path.join(proj, "someone_elses.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", "/other/run"), 0),
            asst_rec(1, "m2", read=999, out=9)])
        found = L.discover_transcripts(self.eval_dir, None, [roots])
        self.assertEqual([os.path.basename(f) for f in found], ["mine.jsonl"])

    def test_needle_straddling_a_read_chunk_is_still_found(self):
        p = os.path.join(self.tdir, "big.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("x" * (1024 * 1024 - 5) + self.eval_dir + "\n")
        self.assertTrue(L._mentions(p, self.eval_dir))


class TestOutputs(LedgerTestBase):
    def _one_run(self):
        self.put_timeline(timeline([ev("Setup", "director:setup"),
                                    ev("Profile", "profiler:baseline")]))
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("director", "setup", self.eval_dir), 0),
            asst_rec(1, "m1", inp=10, read=1000, write5=500, out=100),
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 10),
            asst_rec(12, "m2", inp=5, read=2000, write1h=100, out=50),
        ])
        return self.build()

    def test_writes_every_output_file(self):
        rows, agent_rows, agg, meta = self._one_run()
        out = L.write_outputs(self.eval_dir, rows, agent_rows, agg, meta)
        for name in ("llm_calls.jsonl", "agent_calls.jsonl", "token_stats.json", "token_stats.md"):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
        with open(os.path.join(out, "llm_calls.jsonl"), encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh]
        self.assertEqual(len(lines), 2)
        self.assertEqual({r["phase"] for r in lines}, {"Setup", "Profile"})

    def test_markdown_carries_every_table(self):
        rows, agent_rows, agg, meta = self._one_run()
        md = L.render_md(agg, meta)
        for heading in ("## Run totals", "## Tokens by phase", "## Time by phase",
                        "## By role", "## Ten most expensive agents"):
            self.assertIn(heading, md)
        self.assertIn("Setup", md)
        self.assertIn("Prices used", md)

    def test_phase_totals_sum_to_the_run_total(self):
        """The headline and the breakdown must never disagree."""
        _, _, agg, _ = self._one_run()
        for field in ("calls", "output_tokens", "cache_read_input_tokens"):
            self.assertEqual(sum(p[field] for p in agg["by_phase"].values()),
                             agg["total"][field], field)
        self.assertAlmostEqual(sum(p["cost"] for p in agg["by_phase"].values()),
                               agg["total"]["cost"], places=9)

    def test_rates_can_be_overridden(self):
        rows, _, _, _ = self.build(rates={"_default": dict(
            L.DEFAULT_RATES["_default"], output=0.0, input=0.0, cache_read=0.0,
            cache_write_5m=0.0, cache_write_1h=0.0)})
        self.assertEqual(sum(r["cost_usd"] for r in rows), 0.0)


# --------------------------------------------------------------------------- #
# Structural guard on the instrumented JS. There is no node/deno on the CI
# runner, so this is the one property of that edit a machine can check here --
# and an unbalanced paren is exactly what wrapping a call in a helper risks.
# --------------------------------------------------------------------------- #
def js_balanced(src):
    i, n, stack, tmpl, line = 0, len(src), [], [], 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "/" and i + 1 < n and not tmpl:
            if src[i + 1] == "/":
                j = src.find("\n", i)
                i = n if j < 0 else j
                continue
            if src[i + 1] == "*":
                j = src.find("*/", i + 2)
                if j < 0:
                    return False, "unterminated /* at line %d" % line
                line += src.count("\n", i, j)
                i = j + 2
                continue
        if c in "'\"":
            q, i = c, i + 1
            while i < n and src[i] != q:
                if src[i] == "\\":
                    i += 1
                elif src[i] == "\n":
                    break
                i += 1
            i += 1
            continue
        if c == "`":
            tmpl.append(0)
            i += 1
            while i < n and tmpl:
                ch = src[i]
                if ch == "\n":
                    line += 1
                elif ch == "\\":
                    i += 1
                elif ch == "`" and tmpl[-1] == 0:
                    tmpl.pop()
                elif ch == "$" and i + 1 < n and src[i + 1] == "{":
                    tmpl[-1] += 1
                    i += 1
                elif ch == "}" and tmpl[-1] > 0:
                    tmpl[-1] -= 1
                elif ch == "`" and tmpl[-1] > 0:
                    tmpl.append(0)
                i += 1
            continue
        if c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack or stack[-1][0] != {")": "(", "]": "[", "}": "{"}[c]:
                return False, "unbalanced '%s' at line %d" % (c, line)
            stack.pop()
        i += 1
    if stack:
        return False, "unclosed '%s' opened at line %d" % (stack[-1][0], stack[-1][1])
    return True, "balanced"


class TestInstrumentedWorkflowsAreWellFormed(unittest.TestCase):
    FILES = ("e2e_workflow/e2e_workflow.js",
             "kernel_workflow/kernel_lane.js",
             "kernel_workflow/kernel_workflow.js")

    def test_brackets_balance(self):
        for rel in self.FILES:
            path = os.path.join(GEAK_ROOT, rel)
            if not os.path.isfile(path):
                self.skipTest("%s not present" % rel)
            with open(path, encoding="utf-8") as fh:
                ok, why = js_balanced(fh.read())
            self.assertTrue(ok, "%s: %s" % (rel, why))

    def test_each_workflow_records_a_timeline(self):
        """Every LLM chokepoint must feed the ledger, or a whole layer goes uncounted."""
        for rel in self.FILES:
            path = os.path.join(GEAK_ROOT, rel)
            if not os.path.isfile(path):
                self.skipTest("%s not present" % rel)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            self.assertIn("const LLM_STATS", src, rel)
            self.assertIn("tlAgent(", src, rel)

    def test_the_feature_is_opt_out(self):
        path = os.path.join(GEAK_ROOT, "e2e_workflow", "e2e_workflow.js")
        if not os.path.isfile(path):
            self.skipTest("e2e_workflow.js not present")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("A.llm_stats", src)
        self.assertIn("if (EVAL_DIR && LLM_STATS)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
