#!/usr/bin/env python3
"""Unit tests for gsm8k_eval.py's scoring logic (stdlib only; no pytest needed).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_gsm8k_eval.py

Covers the pure answer-extraction/normalisation path that decides whether a generation counts as
correct. These functions carry the accuracy number that gates a GEAK win on the quality side, so a
silent regression here would move exact_match without any kernel changing.

`requests` / `datasets` are stubbed at import time: they are module-level imports in gsm8k_eval but
are used only by ask()/main(), which need a live server and the HF dataset and are therefore out of
scope for L0. Stubbing keeps this job dependency-free instead of pulling in ~100MB of wheels.
"""
import importlib.util
import os
import sys
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
