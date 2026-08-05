#!/usr/bin/env python3
"""Unit tests for gsm8k_eval.py's scoring logic (stdlib only; no pytest needed).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_gsm8k_eval.py

Covers the pure answer-extraction/normalisation path that decides whether a generation counts as
correct. These functions carry the accuracy number that gates a GEAK win on the quality side, so a
silent regression here would move exact_match without any kernel changing.

`requests` / `datasets` are stubbed at import time so this job stays dependency-free instead of
pulling in ~100MB of wheels. ask() and main() are driven against those stubs rather than a live
server: what is asserted there is the REQUEST SHAPE (greedy decoding must stay greedy) and the
SUBSET SELECTION (a fixed seed must pick the identical problems for baseline and candidate) -- if
either drifts, the two sides of an A/B are no longer comparable and every accuracy delta GEAK
reports becomes noise.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub(name, **attrs):
    """Install a placeholder module so a module-level import of it succeeds."""
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stub("requests", post=lambda *a, **k: None)
_stub("datasets", load_dataset=lambda *a, **k: None)
g = _load("gsm8k_eval", "gsm8k_eval.py")


# --------------------------------------------------------------------------- #
# _norm
# --------------------------------------------------------------------------- #
class TestNorm(unittest.TestCase):
    def test_strips_thousands_separators(self):
        self.assertEqual(g._norm("1,234"), "1234")
        self.assertEqual(g._norm("1,234,567"), "1234567")

    def test_strips_currency(self):
        self.assertEqual(g._norm("$5"), "5")
        self.assertEqual(g._norm("$1,200"), "1200")

    def test_strips_trailing_period(self):
        self.assertEqual(g._norm("7."), "7")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(g._norm(" 8 "), "8")

    def test_negative_is_preserved(self):
        self.assertEqual(g._norm("-42"), "-42")


# --------------------------------------------------------------------------- #
# gold_answer
# --------------------------------------------------------------------------- #
class TestGoldAnswer(unittest.TestCase):
    def test_extracts_marked_answer(self):
        self.assertEqual(g.gold_answer("Janet sells them.\n#### 18"), "18")

    def test_tolerates_missing_space_after_marker(self):
        self.assertEqual(g.gold_answer("####18"), "18")

    def test_normalises_separators(self):
        self.assertEqual(g.gold_answer("#### 1,000"), "1000")

    def test_negative_gold(self):
        self.assertEqual(g.gold_answer("#### -7"), "-7")

    def test_absent_marker_is_none(self):
        self.assertIsNone(g.gold_answer("There is no marked answer here."))


# --------------------------------------------------------------------------- #
# extract_pred -- strict "#### N" first, flexible last-number fallback
# --------------------------------------------------------------------------- #
class TestExtractPred(unittest.TestCase):
    def test_strict_match(self):
        self.assertEqual(g.extract_pred("...so the total is\n#### 18"), "18")

    def test_strict_takes_the_last_marker(self):
        # A reasoning model can emit the marker more than once; the final one is the answer.
        self.assertEqual(g.extract_pred("#### 1\nwait, recompute\n#### 2"), "2")

    def test_strict_beats_a_later_bare_number(self):
        # Guards the ordering: a trailing number must not override an explicit marker.
        self.assertEqual(g.extract_pred("#### 7\nconfidence 99"), "7")

    def test_flexible_fallback_to_last_number(self):
        self.assertEqual(g.extract_pred("The answer is 42."), "42")

    def test_flexible_fallback_handles_currency(self):
        self.assertEqual(g.extract_pred("she earns $3,000 a month"), "3000")

    def test_no_number_at_all_is_none(self):
        self.assertIsNone(g.extract_pred("no numbers here"))

    def test_empty_text_is_none(self):
        self.assertIsNone(g.extract_pred(""))


# --------------------------------------------------------------------------- #
# build_fewshot
# --------------------------------------------------------------------------- #
TRAIN = [
    {"question": "  Q1?  ", "answer": "  reasoning one #### 1  "},
    {"question": "Q2?", "answer": "reasoning two #### 2"},
    {"question": "Q3?", "answer": "reasoning three #### 3"},
]


class TestBuildFewshot(unittest.TestCase):
    def test_uses_exactly_k_shots(self):
        self.assertEqual(g.build_fewshot(TRAIN, 2).count("Question: "), 2)

    def test_shots_are_blank_line_separated(self):
        self.assertEqual(len(g.build_fewshot(TRAIN, 3).split("\n\n")), 3)

    def test_question_and_answer_are_stripped(self):
        shot = g.build_fewshot(TRAIN, 1)
        self.assertIn("Question: Q1?", shot)
        self.assertIn("Answer: reasoning one #### 1", shot)

    def test_each_shot_carries_the_format_instruction(self):
        # The instruction is what makes the strict "#### N" extraction work at eval time.
        self.assertEqual(g.build_fewshot(TRAIN, 2).count("formatted as: #### [number]"), 2)

    def test_zero_shots_is_empty(self):
        self.assertEqual(g.build_fewshot(TRAIN, 0), "")


# --------------------------------------------------------------------------- #
# ask -- request shape + reasoning/content merge
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, message, raise_exc=None):
        self._message = message
        self._raise = raise_exc

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def json(self):
        return {"choices": [{"message": self._message}]}


class _RecordingPost:
    """Stands in for requests.post, capturing the call and returning a canned message."""

    def __init__(self, message, raise_exc=None):
        self.message = message
        self.raise_exc = raise_exc
        self.url = None
        self.payload = None
        self.timeout = None

    def __call__(self, url, json=None, timeout=None):
        self.url = url
        self.payload = json
        self.timeout = timeout
        return _FakeResponse(self.message, self.raise_exc)


class TestAsk(unittest.TestCase):
    def setUp(self):
        self._orig_post = g.requests.post

    def tearDown(self):
        g.requests.post = self._orig_post

    def _call(self, message, **kw):
        post = _RecordingPost(message)
        g.requests.post = post
        text = g.ask("http://host:30000/v1", "my-model", "the prompt", 128, **kw)
        return text, post

    def test_posts_to_chat_completions_without_double_slash(self):
        _, post = self._call({"content": "#### 4"})
        self.assertEqual(post.url, "http://host:30000/v1/chat/completions")

    def test_trailing_slash_on_base_url_is_stripped(self):
        post = _RecordingPost({"content": "x"})
        g.requests.post = post
        g.ask("http://host:30000/v1/", "m", "p", 8)
        self.assertEqual(post.url, "http://host:30000/v1/chat/completions")

    def test_decoding_is_greedy_and_seeded(self):
        # The whole eval is only apples-to-apples if both servers decode greedily from
        # the same seed; a non-zero temperature here would silently make it a lottery.
        _, post = self._call({"content": "#### 4"})
        self.assertEqual(post.payload["temperature"], 0.0)
        self.assertEqual(post.payload["top_p"], 1.0)
        self.assertEqual(post.payload["seed"], 0)

    def test_model_and_prompt_are_forwarded(self):
        _, post = self._call({"content": "#### 4"})
        self.assertEqual(post.payload["model"], "my-model")
        self.assertEqual(post.payload["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertEqual(post.payload["max_tokens"], 128)

    def test_default_timeout_is_forwarded(self):
        _, post = self._call({"content": "x"})
        self.assertEqual(post.timeout, 1800)

    def test_explicit_timeout_overrides_the_default(self):
        _, post = self._call({"content": "x"}, timeout=5)
        self.assertEqual(post.timeout, 5)

    def test_reasoning_precedes_content(self):
        # extract_pred's flexible fallback takes the LAST number, so the final answer
        # must land at the end -- reasoning first, content last.
        text, _ = self._call({"reasoning_content": "thinking 11", "content": "#### 4"})
        self.assertEqual(text, "thinking 11\n#### 4")

    def test_null_content_does_not_crash(self):
        # A reasoning model truncated mid-CoT returns content=None; this used to TypeError.
        text, _ = self._call({"reasoning_content": "partial 7", "content": None})
        self.assertEqual(text, "partial 7")

    def test_null_reasoning_does_not_crash(self):
        text, _ = self._call({"reasoning_content": None, "content": "#### 9"})
        self.assertEqual(text, "#### 9")

    def test_both_null_yields_empty_string(self):
        text, _ = self._call({"reasoning_content": None, "content": None})
        self.assertEqual(text, "")

    def test_http_error_propagates(self):
        g.requests.post = _RecordingPost({"content": "x"}, raise_exc=RuntimeError("503"))
        with self.assertRaises(RuntimeError):
            g.ask("http://h/v1", "m", "p", 8)


# --------------------------------------------------------------------------- #
# main -- subset selection, scoring, summary/artifact shape
# --------------------------------------------------------------------------- #
def _dataset(n_test):
    """A gsm8k-shaped dataset whose i-th gold answer is i, so scoring is checkable by hand."""
    return {
        "train": [
            {"question": f"train q{i}", "answer": f"reasoning #### {i}"} for i in range(8)
        ],
        "test": [
            {"question": f"test q{i}", "answer": f"reasoning #### {i}"} for i in range(n_test)
        ],
    }


class _MainHarness(unittest.TestCase):
    """Runs main() against stubbed dataset/ask and captures what it printed and wrote."""

    def setUp(self):
        self._orig_argv = sys.argv
        self._orig_load = g.load_dataset
        self._orig_ask = g.ask
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        sys.argv = self._orig_argv
        g.load_dataset = self._orig_load
        g.ask = self._orig_ask

    def run_main(self, n_test=6, answer=lambda i: None, out=None, **flags):
        """answer(i) -> generated text for test item i; returning None means 'correct'."""
        data = _dataset(n_test)
        g.load_dataset = lambda *a, **k: data
        seen = []

        def fake_ask(base_url, model, prompt, max_tokens, timeout=1800):
            # Recover which item this is from the question echoed in the prompt.
            i = int(prompt.rsplit("Question: test q", 1)[1].split("\n", 1)[0])
            seen.append(i)
            text = answer(i)
            if isinstance(text, Exception):
                raise text
            return f"#### {i}" if text is None else text

        g.ask = fake_ask
        argv = ["gsm8k_eval.py", "--base-url", "http://h/v1", "--model", "m"]
        for k, v in flags.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        if out is not None:
            argv += ["--out", out]
        sys.argv = argv
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            g.main()
        lines = buf.getvalue().strip().splitlines()
        return json.loads(lines[0]), lines[-1], seen


class TestMainScoring(_MainHarness):
    def test_all_correct_is_one(self):
        summary, last, _ = self.run_main(n_test=6, limit=6)
        self.assertEqual(summary["exact_match"], 1.0)
        self.assertEqual(summary["correct"], 6)
        self.assertEqual(last, "GSM8K_EXACT_MATCH=1.0000")

    def test_all_wrong_is_zero(self):
        summary, last, _ = self.run_main(n_test=6, limit=6, answer=lambda i: "#### 999")
        self.assertEqual(summary["exact_match"], 0.0)
        self.assertEqual(last, "GSM8K_EXACT_MATCH=0.0000")

    def test_partial_score(self):
        summary, _, _ = self.run_main(
            n_test=4, limit=4, answer=lambda i: "#### 999" if i % 2 else None
        )
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["exact_match"], 0.5)

    def test_unparseable_generation_scores_wrong_not_crash(self):
        summary, _, _ = self.run_main(n_test=3, limit=3, answer=lambda i: "no digits at all")
        self.assertEqual(summary["correct"], 0)

    def test_request_failure_is_counted_wrong_and_does_not_abort(self):
        # One dead request must not take the whole eval down with it.
        summary, _, _ = self.run_main(
            n_test=4, limit=4, answer=lambda i: ConnectionError("boom") if i == 1 else None
        )
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["n"], 4)

    def test_empty_subset_scores_zero_rather_than_dividing_by_zero(self):
        summary, _, _ = self.run_main(n_test=5, limit=0)
        self.assertEqual(summary["exact_match"], 0.0)
        self.assertEqual(summary["n"], 0)

    def test_progress_is_reported_every_25(self):
        err = io.StringIO()
        data = _dataset(30)
        g.load_dataset = lambda *a, **k: data
        g.ask = lambda *a, **k: "#### 0"
        sys.argv = ["gsm8k_eval.py", "--base-url", "http://h/v1", "--model", "m", "--limit", "25"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            g.main()
        self.assertIn("[gsm8k] 25/25", err.getvalue())


class TestMainSubsetSelection(_MainHarness):
    def test_same_seed_picks_the_same_problems(self):
        # This is the property that makes baseline vs candidate comparable at all.
        _, _, first = self.run_main(n_test=40, limit=8, seed=7)
        _, _, second = self.run_main(n_test=40, limit=8, seed=7)
        self.assertEqual(sorted(first), sorted(second))

    def test_different_seed_picks_a_different_subset(self):
        _, _, a = self.run_main(n_test=40, limit=8, seed=1)
        _, _, b = self.run_main(n_test=40, limit=8, seed=2)
        self.assertNotEqual(sorted(a), sorted(b))

    def test_limit_caps_the_subset(self):
        summary, _, seen = self.run_main(n_test=40, limit=5)
        self.assertEqual(summary["n"], 5)
        self.assertEqual(len(seen), 5)

    def test_limit_above_dataset_size_uses_everything(self):
        summary, _, _ = self.run_main(n_test=6, limit=999)
        self.assertEqual(summary["n"], 6)


class TestMainSummary(_MainHarness):
    def test_summary_records_the_run_parameters(self):
        summary, _, _ = self.run_main(n_test=10, limit=4, fewshot=3, seed=11)
        self.assertEqual(summary["task"], "gsm8k")
        self.assertEqual(summary["fewshot"], 3)
        self.assertEqual(summary["seed"], 11)
        self.assertEqual(summary["limit"], 4)
        self.assertTrue(summary["greedy"])
        self.assertEqual(summary["base_url"], "http://h/v1")
        self.assertIsInstance(summary["elapsed_s"], float)

    def test_out_file_holds_summary_and_per_item_results(self):
        out = os.path.join(self._tmp.name, "nested", "gsm8k.json")
        summary, _, _ = self.run_main(n_test=6, limit=3, out=out)
        with open(out) as fh:
            blob = json.load(fh)
        self.assertEqual(blob["summary"], summary)
        self.assertEqual(len(blob["results"]), 3)
        self.assertEqual(set(blob["results"][0]), {"idx", "gold", "pred", "ok"})

    def test_out_directory_is_created(self):
        out = os.path.join(self._tmp.name, "a", "b", "c.json")
        self.run_main(n_test=4, limit=2, out=out)
        self.assertTrue(os.path.exists(out))

    def test_no_out_flag_writes_nothing(self):
        before = os.listdir(self._tmp.name)
        self.run_main(n_test=4, limit=2)
        self.assertEqual(os.listdir(self._tmp.name), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
