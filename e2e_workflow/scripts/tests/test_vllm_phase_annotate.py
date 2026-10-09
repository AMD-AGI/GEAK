#!/usr/bin/env python3
"""Unit tests for vllm_phase_annotate.py -- the vLLM->sglang step-annotation dialect bridge.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v

This module exists to make a vLLM trace carry `step[EXTEND bs=N toks=M]` / `step[DECODE bs=N]`
spans so `sglang_step_modes.parse_step` can phase-tag device events. Two properties
are load-bearing and both fail silently at 8-GPU scale:

  1. THE NAME MUST MATCH THE PARSER'S REGEX, byte for byte. A name that is one space or one
     field off produces a trace that looks annotated, parses to zero step spans, and yields a
     phase-blind semantics table -- with nothing anywhere reporting an error. Every naming
     test below asserts against the REAL regex imported from semantic_kernel_mapping, not
     against a copy, so the two cannot drift apart.
  2. A MISLABELLED PHASE IS WORSE THAN NO LABEL. Every fusion candidate downstream inherits
     the phase attribution. So an undeterminable batch must return None (-> no annotation ->
     "unresolved" downstream), never a guess.

The hook itself is exercised without importing vllm: `_step_name` takes the runner and the
scheduler output as plain objects, so a stub with the right duck-type is a faithful test.
"""
import os
import re
import sys
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import vllm_phase_annotate as vpa                                    # noqa: E402
from sglang_step_modes import parse_step                             # noqa: E402


class _SchedulerOutput:
    def __init__(self, per_req):
        self.num_scheduled_tokens = dict(per_req)
        self.total_num_scheduled_tokens = sum(per_req.values())


class _Runner:
    """Duck-types the bits of GPUModelRunner that _step_name reads."""

    def __init__(self, uniform_decode_query_len=1, probe="real"):
        self.uniform_decode_query_len = uniform_decode_query_len
        if probe == "real":
            self._is_uniform_decode = self._real_probe
        elif probe == "raising":
            self._is_uniform_decode = self._raising_probe
        # probe == "absent" -> attribute simply not set

    @staticmethod
    def _real_probe(max_toks, query_len, total, num_reqs):
        return (max_toks == query_len) and (total == max_toks * num_reqs)

    @staticmethod
    def _raising_probe(*_a, **_kw):
        raise TypeError("signature changed in this release")


class TestStepNameDialect(unittest.TestCase):
    def test_decode_name_matches_the_parsers_regex_and_fields(self):
        name = vpa._step_name(_Runner(), _SchedulerOutput({"a": 1, "b": 1, "c": 1}))
        self.assertEqual(name, "step[DECODE bs=3]")
        step = parse_step(name)
        self.assertIsNotNone(step, "name must parse with the REAL step parser")
        self.assertEqual(step["mode"], "DECODE")
        self.assertEqual(step["batch_size"], 3)

    def test_prefill_name_matches_the_parsers_regex_and_fields(self):
        name = vpa._step_name(_Runner(), _SchedulerOutput({"a": 4096, "b": 2048}))
        self.assertEqual(name, "step[EXTEND bs=2 toks=6144]")
        step = parse_step(name)
        self.assertIsNotNone(step)
        self.assertEqual((step["mode"], step["batch_size"], step["tokens"]),
                         ("EXTEND", 2, 6144))

    def test_round_trips_through_the_parsers_own_name_builder(self):
        # _step_span_name() reconstructs the annotation from a parsed span; if our emitted
        # name does not survive that round trip, the CPU/GPU window pairing in
        # _collect_step_spans (which matches spans BY NAME) silently finds no partner.
        import semantic_kernel_mapping as skm
        for per_req, span in (
                ({"a": 1, "b": 1}, (0, 1, "D", 0, 2, "x", "y")),
                ({"a": 1024}, (0, 1, "P", 1024, 1, "x", "y"))):
            emitted = vpa._step_name(_Runner(), _SchedulerOutput(per_req))
            self.assertEqual(emitted, skm._step_span_name(span))


class TestPhaseClassification(unittest.TestCase):
    def test_mixed_chunked_prefill_and_decode_batch_is_extend(self):
        # Continuous batching runs a prefill chunk alongside decodes in ONE step. It is not a
        # uniform decode, so it must be EXTEND -- calling it DECODE would put prefill-shaped
        # GEMMs into the decode table.
        name = vpa._step_name(_Runner(), _SchedulerOutput({"p": 2048, "d1": 1, "d2": 1}))
        self.assertEqual(name, "step[EXTEND bs=3 toks=2050]")

    def test_speculative_decode_of_query_len_gt_1_is_still_decode(self):
        # With MTP/EAGLE a decode step schedules 1+num_spec_tokens per request. A naive
        # `max_toks == 1` test would misfile every speculative decode step as prefill.
        runner = _Runner(uniform_decode_query_len=4)
        name = vpa._step_name(runner, _SchedulerOutput({"a": 4, "b": 4}))
        self.assertEqual(name, "step[DECODE bs=2]")

    def test_ragged_batch_at_query_len_is_extend_not_decode(self):
        # Same max as query_len but NOT uniform across requests -> vLLM does not treat it as
        # a uniform decode batch, and neither do we.
        runner = _Runner(uniform_decode_query_len=4)
        name = vpa._step_name(runner, _SchedulerOutput({"a": 4, "b": 1}))
        self.assertEqual(name, "step[EXTEND bs=2 toks=5]")

    def test_single_request_decode(self):
        self.assertEqual(vpa._step_name(_Runner(), _SchedulerOutput({"a": 1})),
                         "step[DECODE bs=1]")


class TestFallbacksAndRefusals(unittest.TestCase):
    def test_empty_batch_returns_none_rather_than_a_zero_width_span(self):
        self.assertIsNone(vpa._step_name(_Runner(), _SchedulerOutput({})))

    def test_scheduler_output_without_the_fields_returns_none(self):
        self.assertIsNone(vpa._step_name(_Runner(), types.SimpleNamespace()))

    def test_arithmetic_fallback_matches_the_probe_when_the_probe_is_absent(self):
        # The probe is a private staticmethod that has moved across releases; when it is gone
        # the inline arithmetic must reach the SAME verdict, or the dialect flips meaning on
        # a version bump.
        for per_req, qlen in (({"a": 1, "b": 1}, 1), ({"a": 4, "b": 4}, 4),
                              ({"p": 512, "d": 1}, 1), ({"a": 4, "b": 1}, 4)):
            with_probe = vpa._step_name(_Runner(qlen, probe="real"),
                                        _SchedulerOutput(per_req))
            without = vpa._step_name(_Runner(qlen, probe="absent"),
                                     _SchedulerOutput(per_req))
            self.assertEqual(with_probe, without, "disagreement on %r" % (per_req,))

    def test_a_raising_probe_falls_back_instead_of_propagating(self):
        name = vpa._step_name(_Runner(1, probe="raising"),
                              _SchedulerOutput({"a": 1, "b": 1}))
        self.assertEqual(name, "step[DECODE bs=2]")

    def test_missing_uniform_decode_query_len_defaults_to_one(self):
        runner = types.SimpleNamespace()          # no attrs at all
        self.assertEqual(vpa._step_name(runner, _SchedulerOutput({"a": 1})),
                         "step[DECODE bs=1]")

    def test_zero_query_len_is_coerced_to_one(self):
        # `or 1` guards a build that reports 0; without it every decode step would be EXTEND.
        self.assertEqual(vpa._step_name(_Runner(0, probe="absent"),
                                        _SchedulerOutput({"a": 1})),
                         "step[DECODE bs=1]")


class TestArming(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("GEAK_VLLM_PHASE_ANNOTATE")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop("GEAK_VLLM_PHASE_ANNOTATE", None)
        else:
            os.environ["GEAK_VLLM_PHASE_ANNOTATE"] = self._saved

    def test_install_is_a_no_op_when_unarmed(self):
        # The overlay ships on the capture PYTHONPATH; leaving it unarmed must cost nothing,
        # so an accidental carry-over into an A/B run cannot perturb the measurement.
        os.environ.pop("GEAK_VLLM_PHASE_ANNOTATE", None)
        self.assertFalse(vpa.install())

    def test_armed_but_no_vllm_installed_degrades_to_false(self):
        os.environ["GEAK_VLLM_PHASE_ANNOTATE"] = "1"
        self.assertFalse(vpa.install())          # no vllm on this host -> warn, don't raise

    def test_arming_accepts_the_documented_truthy_spellings(self):
        for value, expected in (("1", True), ("true", True), ("True", True),
                                ("0", False), ("", False), ("yes", False)):
            os.environ["GEAK_VLLM_PHASE_ANNOTATE"] = value
            self.assertEqual(vpa._armed(), expected, value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
