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
import glob as _glob
import importlib.util
import json
import os
import shutil
import subprocess
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


def asst_rec(sec, mid, inp=0, read=0, write5=0, write1h=0, out=0, model="claude-opus-4-8",
             text=None, thinking=None):
    """A synthetic assistant record.

    ``text``/``thinking`` inject typed content blocks so the output-capture path
    can be exercised; when both are None the message carries no content, matching
    the older fixtures that only cared about token counts.
    """
    message = {
        "id": mid, "model": model, "role": "assistant", "stop_reason": "tool_use",
        "usage": {
            "input_tokens": inp,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write5 + write1h,
            "cache_creation": {"ephemeral_5m_input_tokens": write5,
                               "ephemeral_1h_input_tokens": write1h},
            "output_tokens": out, "service_tier": "standard",
        },
    }
    if text is not None or thinking is not None:
        blocks = []
        if thinking is not None:
            blocks.append({"type": "thinking", "thinking": thinking})
        if text is not None:
            blocks.append({"type": "text", "text": text})
        message["content"] = blocks
    return {
        "type": "assistant", "timestamp": _ts(sec), "requestId": "req_" + mid,
        "message": message,
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
        # It must not invent a phase name that looks authoritative -- but it should
        # still say WHICH agent the spend belongs to. `~` marks the distinction.
        self.assertEqual(rows[0]["phase"], "~op_benchmarker:bakeoff")

    def test_free_form_labels_still_resolve_via_recorded_identity(self):
        """Found on the first real run: labels are display strings, not identities.

        GEAK's call sites label agents for humans -- 'architect:strategize' when the role
        is system_architect, 'bakeoff <op name>', 'eng r1_d0:memory'. Parsing those as
        role:sub_phase left most of a 12-hour run's spend unattributed. The workflow now
        records role and sub_phase from the prompt, and THAT is the join key.
        """
        self.put_timeline({
            "schema": "geak.agent_timeline/1", "workflow": "e2e_workflow", "nested": [],
            "events": [
                {"seq": 0, "phase": "Strategize", "label": "architect:strategize",
                 "role": "system_architect", "sub_phase": "strategize", "attempt": 1, "ok": True},
                {"seq": 1, "phase": "HeadKernel", "label": "bakeoff mlp.gate_up_proj fp8 GEMM",
                 "role": "op_benchmarker", "sub_phase": "bakeoff", "attempt": 1, "ok": True},
            ],
        })
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("system_architect", "strategize", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        write_transcript(os.path.join(self.tdir, "b.jsonl"), [
            user_rec(prompt_for("op_benchmarker", "bakeoff", self.eval_dir), 100),
            asst_rec(101, "m2", read=200, out=2),
        ])
        rows, _, agg, _ = self.build()
        self.assertEqual({r["phase"] for r in rows}, {"Strategize", "HeadKernel"})
        self.assertNotIn("~system_architect:strategize", agg["by_phase"])

    def test_headerless_kernel_lane_agents_are_not_dumped_on_the_driver(self):
        """Found on the second measured run: the biggest cost centre read as "(driver)".

        Two kernel_lane.js agents skip the `You are the X. PHASE=Y.` header -- the
        optimization engineers (:683) and the round-winner commit (:817). Without a pattern
        for them they never open a conversation, so their calls stayed attached to the
        preceding driver group. On the Qwen3-14B runs that was 948 calls / $108 -- 41% of the
        run -- filed as though it were not GEAK's spend at all. Specialty is kept as the
        sub_phase so the memory lane stays distinguishable from the compute lane.
        """
        eng = ("You are Engineer r2_d1 (specialty=memory) for round 2.\n"
               "First create YOUR private workspace, then optimize.\n- EVAL_DIR: %s\n" % self.eval_dir)
        commit = ("You are the TechLead committing round 2's winning patch into the canonical "
                  "workspace.\n- EVAL_DIR: %s\n" % self.eval_dir)
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(eng, 0), asst_rec(1, "m1", read=100, out=1),
            user_rec(commit, 100), asst_rec(101, "m2", read=200, out=2),
        ])
        rows, _, agg, _ = self.build()
        self.assertEqual({r["phase"] for r in rows}, {"~engineer:memory", "~tech_lead:commit"})
        self.assertNotIn("(driver)", agg["by_phase"])

    def test_a_quoted_prompt_in_a_tool_result_does_not_open_a_conversation(self):
        """The two headerless patterns are anchored, so quoting one cannot split a group.

        A tech_lead reviewing a round quotes its engineers' prompts back. An unanchored
        pattern would treat each quote as a new agent and re-file that spend.
        """
        quoted = ("Here is what the lane dispatched:\n\n"
                  "  You are Engineer r1_d0 (specialty=compute) for round 1.\n")
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("tech_lead", "analyze", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
            user_rec(quoted, 10),
            asst_rec(11, "m2", read=100, out=1),
        ])
        rows, _, _, _ = self.build()
        self.assertEqual({r["phase"] for r in rows}, {"~tech_lead:analyze"})

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


class TestRunWindow(LedgerTestBase):
    """Found against a live run: the eval-dir filter finds a transcript but does not date it.

    A session that launches a run keeps ONE long transcript, and when a human drove it
    interactively that transcript also holds everything else they did that day -- all of it
    mentioning the eval-dir path. Counted naively it dwarfed the run: 248 of 386 calls and
    95M of 115M input tokens came from before the run started, and the reported wall clock
    read 24h for an 84-minute run.
    """

    def _session_with_prior_history(self):
        # One long driver transcript: unrelated work, THEN the run's agents.
        write_transcript(os.path.join(self.tdir, "driver.jsonl"), [
            user_rec("unrelated work mentioning %s" % self.eval_dir, 0),
            asst_rec(10, "old1", read=9_000_000, out=5000),
            asst_rec(20, "old2", read=9_000_000, out=5000),
            user_rec(prompt_for("director", "setup", self.eval_dir), 1000),
            asst_rec(1010, "run1", read=1000, out=10),
        ])

    def test_calls_before_the_first_role_agent_are_excluded(self):
        self._session_with_prior_history()
        rows, _, agg, meta = self.build()
        self.assertEqual(len(rows), 1)
        self.assertEqual(agg["total"]["cache_read_input_tokens"], 1000)
        self.assertEqual(meta["calls_excluded_outside_window"], 2)
        self.assertTrue(any("outside the run window" in w for w in meta["warnings"]))

    def test_wall_clock_reflects_the_run_not_the_session(self):
        self._session_with_prior_history()
        _, _, agg, _ = self.build()
        # Session spans ~17 minutes; the run itself is one call.
        self.assertLess(agg["total"]["wall_ms"], 60_000)

    def test_since_overrides_the_default_window(self):
        self._session_with_prior_history()
        rows, _, _, _ = self.build(since_ms=L._iso_to_ms(_ts(0)))
        self.assertEqual(len(rows), 3)   # caller asked for everything

    def test_driver_work_during_the_run_is_kept_but_labelled(self):
        """In-window driver calls are real; they just are not GEAK's."""
        write_transcript(os.path.join(self.tdir, "driver.jsonl"), [
            user_rec(prompt_for("director", "setup", self.eval_dir), 0),
            asst_rec(10, "run1", read=1000, out=10),
        ])
        write_transcript(os.path.join(self.tdir, "human.jsonl"), [
            user_rec("a human checking on %s mid-run" % self.eval_dir, 20),
            asst_rec(30, "human1", read=500_000, out=100),
        ])
        _, _, agg, _ = self.build()
        self.assertIn(L.DRIVER, agg["by_phase"])
        self.assertEqual(agg["by_phase"][L.DRIVER]["calls"], 1)
        self.assertEqual(agg["by_role"]["director"]["calls"], 1)


class TestOwnedScopeWindow(LedgerTestBase):
    """Fix 4 (Astra re-review): when the caller has ALREADY established the
    transcripts as this run's own (run-scoped discovery), the window's inferred
    lower bound must be lifted — an early OWNED call is real run work, not the
    pre-run session history the inferred fence is meant to drop."""

    def _owned_early_then_late(self):
        # Both calls are the run's own; the first precedes the first role-agent
        # user turn only because setup work runs before the director is prompted.
        write_transcript(os.path.join(self.tdir, "own.jsonl"), [
            asst_rec(10, "msg_owned_early", read=1000, out=10),
            user_rec(prompt_for("director", "setup", self.eval_dir), 15),
            asst_rec(20, "msg_director", read=2000, out=20),
        ])

    def test_owned_scope_keeps_the_early_owned_call(self):
        self._owned_early_then_late()
        rows, _, agg, meta = self.build(owned_scope=True)
        # Both owned calls counted; nothing dropped as "before the run".
        self.assertEqual(len(rows), 2)
        self.assertEqual(agg["total"]["cache_read_input_tokens"], 3000)
        self.assertEqual(meta.get("calls_excluded_outside_window", 0), 0)

    def test_unowned_scope_still_drops_the_early_call(self):
        # Same transcript, but WITHOUT the owned-scope trust: the inferred fence
        # still fires (substring discovery cannot vouch for the early call).
        self._owned_early_then_late()
        rows, _, _, meta = self.build()
        self.assertEqual(len(rows), 1)
        self.assertEqual(meta["calls_excluded_outside_window"], 1)

    def test_owned_scope_records_the_scope_in_meta(self):
        self._owned_early_then_late()
        _, _, _, meta = self.build(owned_scope=True, scope="run-scoped")
        self.assertEqual(meta["transcript_scope"], "run-scoped")

    def test_scope_warnings_mark_the_run_incomplete(self):
        # A partial-coverage caller passes a warning; it joins the ledger's own
        # and flips complete -> False so the report never claims full coverage.
        self._owned_early_then_late()
        _, _, _, meta = self.build(
            owned_scope=True, scope="partial",
            scope_warnings=["lane laneB could not be established"])
        self.assertFalse(meta["complete"])
        self.assertTrue(any("laneB" in w for w in meta["warnings"]))


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


class TestCostBreakdown(unittest.TestCase):
    """The per-bucket split a report shows must reconcile to the single total."""

    def _row(self, model="claude-opus-4-8"):
        return {"model": model, "input_tokens": 1_000_000,
                "cache_read_input_tokens": 2_000_000, "cache_creation_input_tokens": 3_000_000,
                "cache_write_5m_tokens": 2_000_000, "cache_write_1h_tokens": 1_000_000,
                "output_tokens": 500_000}

    def test_buckets_sum_to_cost_of(self):
        row = self._row()
        bd = L.cost_breakdown(row, L.DEFAULT_RATES)
        self.assertAlmostEqual(sum(bd.values()), L.cost_of(row, L.DEFAULT_RATES), places=9)

    def test_each_bucket_is_priced_from_its_own_tokens(self):
        # Opus: in 5, read 0.5, 5m-write 6.25, 1h-write 10, out 25 (per M).
        bd = L.cost_breakdown(self._row(), L.DEFAULT_RATES)
        self.assertAlmostEqual(bd["uncached_input"], 5.00, places=6)   # 1M * 5
        self.assertAlmostEqual(bd["cache_read"], 1.00, places=6)       # 2M * 0.5
        self.assertAlmostEqual(bd["cache_write"], 22.50, places=6)     # 2M*6.25 + 1M*10
        self.assertAlmostEqual(bd["output"], 12.50, places=6)          # 0.5M * 25
        self.assertEqual(bd["router"], 0.0)                            # static router: no LLM call

    def test_router_bucket_is_labelled_zero_not_absent(self):
        """The static deterministic router spends nothing, but the line must exist
        so a future dynamic router has a place to report its cost."""
        self.assertIn("router", L.cost_breakdown(self._row(), L.DEFAULT_RATES))

    def test_uncached_input_is_the_fresh_bucket_only(self):
        """'uncached-context' is Anthropic's input_tokens — cache is already netted
        out of it, so it must not be re-derived from the total."""
        row = dict(self._row(), input_tokens=0)
        self.assertEqual(L.cost_breakdown(row, L.DEFAULT_RATES)["uncached_input"], 0.0)

    def test_breakdown_respects_per_model_rates(self):
        opus = L.cost_breakdown(self._row("claude-opus-4-8"), L.DEFAULT_RATES)
        rates = dict(L.DEFAULT_RATES, **{"claude-sonnet-5": dict(
            L.DEFAULT_RATES["_default"], input=2.0, output=10.0, cache_read=0.2,
            cache_write_5m=2.5, cache_write_1h=4.0)})
        sonnet = L.cost_breakdown(self._row("claude-sonnet-5"), rates)
        self.assertLess(sonnet["output"], opus["output"])             # 10 vs 25 per M
        self.assertAlmostEqual(sonnet["cache_read"], 0.40, places=6)  # 2M * 0.2


class TestOutputCapture(LedgerTestBase):
    """Output (thinking + response) must be recorded, not just the input prompt."""

    def _run_with_content(self):
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("director", "setup", self.eval_dir), 0),
            asst_rec(1, "m1", inp=10, read=1000, write5=500, out=100,
                     thinking="weighing tile sizes", text="I'll raise BLOCK_SIZE_M."),
        ])
        return self.build()

    def test_output_and_thinking_are_captured(self):
        rows, _, _, _ = self._run_with_content()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["output"], "I'll raise BLOCK_SIZE_M.")
        self.assertEqual(rows[0]["thinking"], "weighing tile sizes")

    def test_per_call_prompt_is_populated(self):
        rows, _, _, _ = self._run_with_content()
        self.assertIn("You are the director", rows[0]["prompt"])

    def test_each_row_carries_a_cost_breakdown_summing_to_cost_usd(self):
        rows, _, _, _ = self._run_with_content()
        r = rows[0]
        self.assertAlmostEqual(sum(r["cost_breakdown"].values()), r["cost_usd"], places=6)

    def test_content_absent_leaves_empty_strings_not_crash(self):
        """Older transcripts (no content blocks) must still ledger cleanly."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("director", "setup", self.eval_dir), 0),
            asst_rec(1, "m1", inp=10, out=100),
        ])
        rows, _, _, _ = self.build()
        self.assertEqual(rows[0]["output"], "")
        self.assertEqual(rows[0]["thinking"], "")

    def test_partial_flush_keeps_the_larger_output_with_its_text(self):
        """When a duplicate message id is merged to the larger token count, the
        captured text must travel with the kept copy."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("director", "setup", self.eval_dir), 0),
            asst_rec(1, "m1", out=10, text="partial"),
            asst_rec(2, "m1", out=100, text="the full answer"),
        ])
        rows, _, _, _ = self.build()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["output"], "the full answer")

    def _blockrec(self, sec, mid, block, index, out):
        """One assistant record carrying a single content block, with the
        transcript's ``apiBlockIndex`` so blocks of one response merge by block."""
        return {"type": "assistant", "timestamp": _ts(sec), "apiBlockIndex": index,
                "requestId": "req_" + mid,
                "message": {"id": mid, "model": "claude-opus-4-8", "stop_reason": "tool_use",
                            "content": [block],
                            "usage": {"input_tokens": 5, "output_tokens": out}}}

    def test_earlier_content_blocks_survive_a_tool_use_final_record(self):
        """Astra P1 #2: one response emits thinking, text, then a tool_use record
        that carries the largest output count. Keeping only the largest-usage
        record dropped the earlier thinking/text; merging by block keeps them."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("engineer", "compute", self.eval_dir), 0),
            self._blockrec(1, "msg_mb", {"type": "thinking", "thinking": "Earlier reasoning"}, 0, 2),
            self._blockrec(2, "msg_mb", {"type": "text", "text": "Earlier response text"}, 1, 2),
            self._blockrec(3, "msg_mb", {"type": "tool_use", "id": "t1", "name": "Bash",
                                         "input": {"command": "true"}}, 2, 100),
        ])
        rows, _, _, _ = self.build()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["output_tokens"], 100)          # usage from the largest record
        self.assertEqual(rows[0]["output"], "Earlier response text")   # text block recovered
        self.assertEqual(rows[0]["thinking"], "Earlier reasoning")     # thinking block recovered

    def test_cumulative_flush_does_not_duplicate_text(self):
        """A block re-flushed as it grows must be kept once (the longest), not
        concatenated onto its own prefix."""
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("engineer", "compute", self.eval_dir), 0),
            self._blockrec(1, "msg_c", {"type": "text", "text": "Hello"}, 0, 3),
            self._blockrec(2, "msg_c", {"type": "text", "text": "Hello world, done."}, 0, 6),
        ])
        rows, _, _, _ = self.build()
        self.assertEqual(rows[0]["output"], "Hello world, done.")


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
# A '/' starts a regex only where a value cannot already stand -- i.e. right after an
# operator, an opener, a separator, or nothing at all. After a name, ')' , ']' or a
# literal it is division. `None` covers the start of file / start of a fresh expression.
_REGEX_OK = set("([{,;=:!?&|+-*%^~<>") | {None}
# ...and after a keyword that expects a value, e.g. `return /re/`, the char before '/' is a
# letter yet it is still a regex. This lookbehind is why the fallback handles the common
# keyword-context case; it is NOT a general parser (node --check is the authority above).
_VALUE_KEYWORDS = frozenset((
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "do", "else", "yield", "await", "case", "throw"))


def _prev_word(src, i):
    """The identifier immediately before position i, skipping whitespace. '' if none."""
    j = i - 1
    while j >= 0 and src[j] in " \t\r\n":
        j -= 1
    end = j + 1
    while j >= 0 and (src[j].isalnum() or src[j] in "_$"):
        j -= 1
    return src[j + 1:end]


def _find_node():
    """A node runtime if one is discoverable -- the syntax AUTHORITY. None on a bare CI box."""
    n = shutil.which("node")
    if n:
        return n
    cands = sorted(_glob.glob(os.path.expanduser(
        "~/.cursor-server/bin/linux-x64/*/node")), reverse=True)
    return cands[0] if cands else None


def js_balanced(src):
    i, n, stack, tmpl, line, last_sig = 0, len(src), [], [], 1, None
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r":
            i += 1
            continue  # whitespace never changes last_sig (comments don't either)
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
            if last_sig in _REGEX_OK or _prev_word(src, i) in _VALUE_KEYWORDS:
                # Regex literal: consume to the closing unescaped '/', treating
                # '/' inside a [...] char-class as literal.
                i += 1
                in_class = False
                while i < n:
                    ch = src[i]
                    if ch == "\\":
                        i += 2
                        continue
                    if ch == "\n":
                        break  # unterminated regex -- leave it; not our failure mode
                    if ch == "[":
                        in_class = True
                    elif ch == "]":
                        in_class = False
                    elif ch == "/" and not in_class:
                        i += 1
                        break
                    i += 1
                last_sig = "/"  # a regex is a value; a following '/' is division
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
            last_sig = c
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
            last_sig = "`"
            continue
        if c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack or stack[-1][0] != {")": "(", "]": "[", "}": "{"}[c]:
                return False, "unbalanced '%s' at line %d" % (c, line)
            stack.pop()
        last_sig = c
        i += 1
    if stack:
        return False, "unclosed '%s' opened at line %d" % (stack[-1][0], stack[-1][1])
    return True, "balanced"


class TestInstrumentedWorkflowsAreWellFormed(unittest.TestCase):
    FILES = ("e2e_workflow/e2e_workflow.js",
             "kernel_workflow/kernel_lane.js",
             "kernel_workflow/kernel_workflow.js")

    def test_brackets_balance(self):
        node = _find_node()
        for rel in self.FILES:
            path = os.path.join(GEAK_ROOT, rel)
            if not os.path.isfile(path):
                self.skipTest("%s not present" % rel)
            if node:
                # node --check is the syntax AUTHORITY wherever a runtime exists; js_balanced
                # is only the no-node fallback (its known limits are pinned separately below).
                r = subprocess.run([node, "--check", path], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, "%s: %s" % (rel, r.stderr.strip()))
            else:
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

    def test_opt_out_is_forwarded_to_child_lanes(self):
        """A parent llm_stats opt-out must reach the nested lane, or the "workflow-wide"
        opt-out silently reverts to the child's default -- the funnels drop it otherwise."""
        for rel in ("e2e_workflow/e2e_workflow.js", "kernel_workflow/kernel_workflow.js"):
            path = os.path.join(GEAK_ROOT, rel)
            if not os.path.isfile(path):
                self.skipTest("%s not present" % rel)
            with open(path, encoding="utf-8") as fh:
                self.assertIn("llm_stats: String(A.llm_stats)", fh.read(), rel)


class TestDispatchOrderAmbiguity(LedgerTestBase):
    """The timeline is recorded at DISPATCH; the parser fills its slots positionally by
    first-response order. When two same-key agents run concurrently (both attempt 1) the
    ledger cannot know which transcript is which, so the join is a guess -- it must SAY so
    (attribution="inferred") while still assigning the phase, not silently claim "timeline"."""

    def _two_concurrent_bakeoffs(self):
        # Same key op_benchmarker:bakeoff, two distinct attempt-1 dispatches -> ambiguous.
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

    def test_concurrent_same_key_agents_are_marked_inferred_not_timeline(self):
        self._two_concurrent_bakeoffs()
        rows, agent_rows, _, meta = self.build()
        # The run still HAS a timeline, so the run-level mode stays "timeline"...
        self.assertEqual(meta["attribution_mode"], "timeline")
        # ...but each individually-guessed row admits the guess.
        matched = [a for a in agent_rows if a["api_calls"] > 0]
        self.assertEqual(len(matched), 2)
        self.assertTrue(all(a["attribution"] == "inferred" for a in matched),
                        [a["attribution"] for a in matched])
        # The phase is still assigned -- honesty about the join must not cost attribution.
        self.assertEqual({r["phase"] for r in rows}, {"HeadKernel", "Milestone"})

    def test_a_lone_dispatch_with_retries_stays_timeline(self):
        """attempts 1,2,3 of ONE agent are sequential, not concurrent -- unambiguous."""
        self.put_timeline(timeline([
            ev("Profile", "profiler:baseline", attempt=1, ok=False),
            ev("Profile", "profiler:baseline", attempt=2, ok=True),
        ]))
        for i, name in enumerate(("a.jsonl", "b.jsonl")):
            write_transcript(os.path.join(self.tdir, name), [
                user_rec(prompt_for("profiler", "baseline", self.eval_dir), i * 100),
                asst_rec(i * 100 + 1, "m%d" % i, read=100, out=1),
            ])
        _, agent_rows, _, _ = self.build()
        matched = [a for a in agent_rows if a["api_calls"] > 0]
        self.assertTrue(all(a["attribution"] == "timeline" for a in matched),
                        [a["attribution"] for a in matched])
        # Both attempts have transcripts -> the retry outcomes are carried faithfully.
        by_attempt = {a["attempt"]: a for a in matched}
        self.assertIs(by_attempt[1]["ok"], False)
        self.assertIs(by_attempt[2]["ok"], True)

    def test_incomplete_retry_mapping_is_inferred_not_a_guessed_outcome(self):
        """Regression (Astra Finding 1): timeline records TWO attempts (1 failed, 2 ok) but
        only the SECOND produced a transcript. The lone conversation is slotted positionally
        onto attempt-1 -- an outcome it did not have. The join must ADMIT the guess
        (`inferred`) and must NOT report the recorded attempt-1 `ok=False` (nor a hardcoded
        `ok=True`) as if the mapping were known: the outcome is unknown, so `ok` is None.
        The retry itself stays visible via the timeline-only leftover row."""
        self.put_timeline(timeline([
            ev("Profile", "profiler:baseline", attempt=1, ok=False),
            ev("Profile", "profiler:baseline", attempt=2, ok=True),
        ]))
        # Only ONE transcript (cf. test_retries_consume_successive_timeline_slots which has two).
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(prompt_for("profiler", "baseline", self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ])
        _, agent_rows, agg, _ = self.build()
        matched = [a for a in agent_rows if a["api_calls"] > 0]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["attribution"], "inferred")
        self.assertIsNone(matched[0]["ok"])            # not the guessed attempt's outcome, not True
        # The other recorded attempt is still surfaced, with its own outcome intact.
        silent = [a for a in agent_rows if a["api_calls"] == 0]
        self.assertEqual(len(silent), 1)
        self.assertIn("no transcript", silent[0]["attribution"])
        self.assertEqual(agg["total"]["agent_attempts_failed"], 1)


class TestInstanceDedup(LedgerTestBase):
    """A nested kernel timeline is reachable both through the parent's merge and through the
    glob. Dedup must key on the run's stable `instance`, so the SAME run reached twice is
    collapsed while two DISTINCT lanes that share a shape are BOTH kept."""

    @staticmethod
    def _kernel_node(instance):
        node = timeline([ev("Setup", "director:setup")], workflow="kernel_lane")
        node["instance"] = instance
        return node

    def _count_kernel_setup(self):
        loaded = L.load_timeline(self.eval_dir)
        return sum(1 for e in loaded["events"]
                   if e["workflow"] == "kernel_lane" and e["key"] == "director:setup")

    def test_same_instance_reached_twice_is_deduped(self):
        node = self._kernel_node("run-1")
        # Parent merge carries it, AND it shows up standalone through the glob.
        self.put_timeline(timeline([ev("Setup", "director:setup")], nested=[node]))
        self.put_timeline(node, sub=os.path.join(self.eval_dir, "kernels", "_exp", "team_k1"))
        self.assertEqual(self._count_kernel_setup(), 1)

    def test_two_distinct_lanes_of_the_same_shape_are_both_kept(self):
        self.put_timeline(timeline([ev("Setup", "director:setup")],
                                    nested=[self._kernel_node("run-1"),
                                            self._kernel_node("run-2")]))
        self.assertEqual(self._count_kernel_setup(), 2)

    def test_legacy_timelines_without_instance_fall_back_to_shape(self):
        """Pre-`instance` timelines keep the old shape fingerprint -- two identical shapes
        collapse. Documented so the fallback is a known limitation, not a silent regression."""
        legacy = timeline([ev("Setup", "director:setup")], workflow="kernel_lane")  # no instance
        self.put_timeline(timeline([ev("Setup", "director:setup")],
                                    nested=[legacy, dict(legacy)]))
        self.assertEqual(self._count_kernel_setup(), 1)


class TestJsBalancedFallbackLimits(unittest.TestCase):
    """js_balanced is a heuristic used ONLY when no node runtime is discoverable (node --check
    is the authority in test_brackets_balance). These fixtures pin what the fallback does and does
    NOT handle, so nobody mistakes it for a sound JavaScript parser and over-trusts a green run."""

    def test_regex_after_a_value_keyword_is_not_read_as_division(self):
        # Without the keyword lookbehind the char before '/' is a letter, so the '/' would be
        # read as division and the char-class brackets would look unbalanced. `return`/`typeof`
        # are the cases that actually occur in the instrumented files' neighbourhood.
        self.assertEqual(js_balanced("function f() { return /[)]/; }")[0], True)
        self.assertEqual(js_balanced("const r = typeof x === 'string' ? /[(]/ : 0;")[0], True)

    def test_division_after_a_value_is_still_division(self):
        self.assertEqual(js_balanced("const a = (b) / c / d;")[0], True)
        self.assertEqual(js_balanced("const q = arr[0] / 2;")[0], True)

    def test_documented_limitation_not_every_context_is_covered(self):
        # The lookbehind covers keyword contexts, not the full grammar. A regex opened right
        # after a `)` that closes an `if (...)` head is valid JS but the heuristic reads the
        # '/' as division -- documented here so the limit is explicit, not a surprise. node
        # --check (preferred whenever available) does not share this blind spot.
        ok, _ = js_balanced("if (x) /[)]/.test(y);")
        self.assertFalse(ok)


# --------------------------------------------------------------------------- #
# Claude Code's workflow frame, and the runtime's per-agent metadata
# --------------------------------------------------------------------------- #
def framed(task, kind="computed task"):
    """Byte-shaped like Claude Code's workflow frame: one header line, then the task with every
    line indented two spaces."""
    return ("[Workflow harness — %s] The task text below was computed at runtime by a workflow "
            "script. The harness indents every line of the computed text. The computed task text "
            "follows:\n" % kind) + "\n".join("  " + line for line in task.split("\n"))


class TestHarnessFrameAndAgentMeta(LedgerTestBase):
    """Found on the 2026-09-24 gpt-oss-120b run: $242.61 of $457.08 read as "(driver)".

    The runtime now wraps every agent's prompt in a frame with each line indented, so the
    anchored engineer/commit patterns never matched; and a run killed before writing its timeline
    had no other source of names. The runtime's own agent-<id>.meta.json has both the label the
    script gave the agent and the phase it ran in.
    """

    def agent_file(self, run_id, agent_id, records, description=None, phase=None):
        d = os.path.join(self.tmp, "home", "projects", "slug", "sess", "subagents", "workflows", run_id)
        os.makedirs(d, exist_ok=True)
        write_transcript(os.path.join(d, "agent-%s.jsonl" % agent_id), records)
        if description is not None or phase is not None:
            with open(os.path.join(d, "agent-%s.meta.json" % agent_id), "w", encoding="utf-8") as fh:
                json.dump({"agentType": "workflow-subagent", "description": description or "",
                           "workflowPhase": phase or ""}, fh)
        return os.path.join(d, "agent-*.jsonl")

    def test_a_framed_engineer_prompt_is_recognised(self):
        eng = "You are Engineer r2_d0 (specialty=algorithm) for round 2.\n- EVAL_DIR: %s" % self.eval_dir
        write_transcript(os.path.join(self.tdir, "a.jsonl"), [
            user_rec(framed("optimize inference", kind="user request"), 0),
            user_rec(framed(eng), 1),
            asst_rec(2, "m1", read=100, out=1),
        ])
        rows, _, agg, _ = self.build()
        self.assertEqual([(r["role"], r["sub_phase"]) for r in rows], [("engineer", "algorithm")])
        self.assertNotIn("(driver)", agg["by_role"])

    def test_unwrapping_removes_only_the_frames_indent(self):
        self.assertEqual(L._unwrap_harness(framed("a\n  b\nc")), "a\n  b\nc")
        self.assertEqual(L._unwrap_harness("You are the x. PHASE=y."), "You are the x. PHASE=y.")

    def test_metadata_names_an_agent_no_pattern_recognises(self):
        g = self.agent_file("wf_run1", "a1", [
            user_rec(framed("Explore a ground-up kernel.\n- EVAL_DIR: %s" % self.eval_dir), 0),
            asst_rec(1, "m1", read=100, out=1),
        ], description="deep r1_d0:deep_explore", phase="▸ kernel-lane #4")
        rows, agent_rows, _, _ = L.build(self.eval_dir, [g], owned_scope=True)
        r = rows[0]
        self.assertEqual((r["role"], r["sub_phase"]), ("engineer", "deep_explore"))
        self.assertEqual(r["agent_label"], "deep r1_d0:deep_explore")
        self.assertEqual(r["phase"], "kernel-lane #4")
        self.assertEqual(r["attribution"], "agent-meta")
        self.assertEqual(r["workflow_run"], "wf_run1")
        self.assertEqual(agent_rows[0]["workflow_run"], "wf_run1")
        # The computed task stands in as the prompt snippet when no header named the agent.
        self.assertIn("Explore a ground-up kernel.", r["prompt"])

    def test_metadata_label_and_phase_win_over_a_derived_key(self):
        g = self.agent_file("wf_run1", "a1", [
            user_rec(framed(prompt_for("kernel_extractor", "extract_op", self.eval_dir)), 0),
            asst_rec(1, "m1", read=100, out=1),
        ], description="extract_op fused_moe_a16w4_decode", phase="HeadKernel")
        rows, _, _, _ = L.build(self.eval_dir, [g], owned_scope=True)
        self.assertEqual((rows[0]["role"], rows[0]["sub_phase"]), ("kernel_extractor", "extract_op"))
        self.assertEqual(rows[0]["agent_label"], "extract_op fused_moe_a16w4_decode")
        self.assertEqual(rows[0]["phase"], "HeadKernel")

    def test_a_single_agent_file_is_not_split_by_a_quoted_role_header(self):
        """ROLE_RE is a search, so a tool result quoting another role's prompt used to open a new
        conversation. A file with its own metadata is ONE agent, whatever it quotes."""
        g = self.agent_file("wf_run1", "a1", [
            user_rec(framed(prompt_for("director", "setup", self.eval_dir)), 0),
            asst_rec(1, "m1", read=100, out=1),
            user_rec("tool result: You are the tech_lead. PHASE=analyze.", 10),
            asst_rec(11, "m2", read=100, out=1),
        ], description="director:setup", phase="Setup")
        rows, _, _, _ = L.build(self.eval_dir, [g], owned_scope=True)
        self.assertEqual({r["group_id"] for r in rows}, {"agent-a1.jsonl#0"})
        self.assertEqual({r["role"] for r in rows}, {"director"})

    def test_a_partial_timeline_does_not_cap_the_agent_count(self):
        """A re-entered run's timeline records only the last invocation's attempts."""
        self.put_timeline(timeline([ev("Validate", "director:validate")]))
        for i, (role, sub) in enumerate((("director", "validate"), ("profiler", "baseline"),
                                         ("config_tuner", "sweep"))):
            write_transcript(os.path.join(self.tdir, "t%d.jsonl" % i), [
                user_rec(prompt_for(role, sub, self.eval_dir), i * 10),
                asst_rec(i * 10 + 1, "m%d" % i, read=100, out=1),
            ])
        _, _, agg, _ = self.build()
        self.assertEqual(agg["total"]["agents"], 3)

    def test_role_of_label(self):
        cases = {
            "eng r2_d0:algorithm": ("engineer", "algorithm"),
            "deep r1_d0:deep_explore": ("engineer", "deep_explore"),
            "commit r2": ("tech_lead", "commit"),
            "director:setup": ("director", "setup"),
            "tech_lead:plan r1": ("tech_lead", "plan"),
            "persist-e2e-checkpoint:config/e2e_validation.json": ("persist-e2e-checkpoint", "config"),
            "extract_op fused_moe (ck_tile stage1)": ("extract_op", ""),
            "verify r1_d0 (recovered)": ("verify", ""),
            "": ("", ""),
        }
        for label, want in cases.items():
            self.assertEqual(L.role_of_label(label), want, label)

    def test_workflow_run_is_read_from_the_path_only(self):
        self.assertEqual(L.workflow_run_of("/h/p/s/sess/subagents/workflows/wf_x/agent-1.jsonl"), "wf_x")
        self.assertIsNone(L.workflow_run_of("/h/p/s/sess.jsonl"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
