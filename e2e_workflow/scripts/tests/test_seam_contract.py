#!/usr/bin/env python3
"""Regression tests for the live-seam binding contract."""
import contextlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "seam_contract", os.path.join(SCRIPTS_DIR, "seam_contract.py"))
sc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sc)


def _descriptor(fn, seam="fixture:entry", evidence=True):
    runtime = {"inplace_params": [], "returns_none": False}
    if evidence:
        runtime["hidden_context"] = []
    return sc._describe_signature(
        inspect.signature(fn), seam, fn.__name__, None,
        {"seam_runtime_evidence": runtime})


class TestBindingCompatibility(unittest.TestCase):
    def test_identical_positional_only_signature_is_bindable(self):
        def entry(a, b, /, c=None):
            return a

        desc = _descriptor(entry)
        verdict = sc.check_binding(desc, desc)
        self.assertTrue(verdict["bindable"], verdict["mismatches"])

    def test_rendered_entry_preserves_positional_only_marker(self):
        def entry(a, b, /, c=None):
            return a

        namespace = {}
        exec(sc.render_entry(_descriptor(entry)), namespace)
        self.assertEqual(str(inspect.signature(namespace["entry"])), "(a, b, /, c=None)")

    def test_missing_hidden_context_evidence_fails_closed(self):
        def entry(x):
            return x

        desc = _descriptor(entry, evidence=False)
        verdict = sc.check_binding(desc, _descriptor(entry))
        self.assertFalse(verdict["bindable"])
        self.assertIn("hidden_context_inputs", verdict["codes"])

    def test_null_hidden_context_is_not_explicit_purity_evidence(self):
        def entry(x):
            return x

        desc = sc._describe_signature(
            inspect.signature(entry), "fixture:entry", "entry", None,
            {"seam_runtime_evidence": {"hidden_context": None}})
        self.assertEqual(desc["hidden_context_evidence"], "unknown")
        self.assertFalse(sc.check_binding(desc, _descriptor(entry))["bindable"])

    def test_dropped_optional_parameter_is_rejected(self):
        def live(a, optional=None):
            return a

        def candidate(a):
            return a

        verdict = sc.check_binding(_descriptor(live), _descriptor(candidate))
        self.assertFalse(verdict["bindable"])
        self.assertIn("optional_param_dropped", verdict["codes"])

    # ---- C6 case #2: POSITIONAL_OR_KEYWORD live param declared POSITIONAL_ONLY on candidate.
    # The reviewer's own second false-positive. live f(a, b) accepts a keyword call f(a=.., b=..);
    # a candidate f(a, b, /) rejects that call, so it CANNOT be rebound at a keyword call site even
    # though required names + arity line up. The representative-call bind simulation must catch it.
    # Descriptors carry hidden_context=[] (via _descriptor) so the strict hidden_context gate does
    # not mask the param-kind logic under test.
    def test_positional_or_keyword_bound_as_positional_only_is_rejected(self):
        def live(a, b):
            return a

        def candidate(a, b, /):
            return a

        verdict = sc.check_binding(_descriptor(live), _descriptor(candidate))
        self.assertFalse(verdict["bindable"], verdict["mismatches"])
        self.assertIn("param_kind_mismatch", verdict["codes"])
        # It is specifically the keyword-call surface (not arity/name) that fails.
        self.assertNotIn("arity_mismatch", verdict["codes"])
        self.assertNotIn("param_name_mismatch", verdict["codes"])

    def test_reordered_positional_parameters_are_accepted(self):
        # live f(a, b) vs candidate f(b, a): every representative live call (positional AND keyword)
        # still binds against the candidate, so this is harmless and PASSES.
        def live(a, b):
            return a

        def candidate(b, a):
            return b

        verdict = sc.check_binding(_descriptor(live), _descriptor(candidate))
        self.assertTrue(verdict["bindable"], verdict["mismatches"])

    def test_candidate_varargs_varkw_swallows_live_signature(self):
        # A candidate that accepts (*args, **kwargs) can absorb every representative live call, so the
        # varargs/varkw candidate binds. Exercises the varargs/varkw short-circuits in check_binding.
        def live(a, b):
            return a

        def candidate(*args, **kwargs):
            return args

        verdict = sc.check_binding(_descriptor(live), _descriptor(candidate))
        self.assertTrue(verdict["bindable"], verdict["mismatches"])

    def test_live_varargs_not_absorbed_by_fixed_candidate_is_rejected(self):
        # live f(a, *args) accepts an extra positional; the representative-call simulation appends one
        # (include_optional leg), and a fixed-arity candidate f(a) rejects it -> param_kind_mismatch.
        def live(a, *args):
            return a

        def candidate(a):
            return a

        verdict = sc.check_binding(_descriptor(live), _descriptor(candidate))
        self.assertFalse(verdict["bindable"], verdict["mismatches"])
        self.assertIn("param_kind_mismatch", verdict["codes"])

    def test_live_varkw_not_absorbed_by_fixed_candidate_is_rejected(self):
        # live f(a, **kwargs) accepts an extra keyword; the representative-call simulation adds one, and
        # a fixed candidate f(a) rejects it -> param_kind_mismatch via the bind simulation.
        def live(a, **kwargs):
            return a

        def candidate(a):
            return a

        verdict = sc.check_binding(_descriptor(live), _descriptor(candidate))
        self.assertFalse(verdict["bindable"], verdict["mismatches"])
        self.assertIn("param_kind_mismatch", verdict["codes"])

    def test_positional_to_keyword_only_is_rejected_deliberate_safe_direction(self):
        # DOCUMENTED CURRENT BEHAVIOR: live f(a, b) vs candidate f(a, *, b). The reviewer flagged this
        # as harmless (a keyword-only candidate param can still receive the live positional value by
        # name), but the revised code intentionally FAILS CLOSED: a representative live call passes b
        # positionally, which the keyword-only candidate rejects -> param_kind_mismatch. This is a
        # deliberate safe-direction over-rejection (better to reject a bindable candidate than to admit
        # an unbindable one); this test pins that intended behavior, not an accidental one.
        def live(a, b):
            return a

        def candidate(a, *, b):
            return a

        verdict = sc.check_binding(_descriptor(live), _descriptor(candidate))
        self.assertFalse(verdict["bindable"], verdict["mismatches"])
        self.assertIn("param_kind_mismatch", verdict["codes"])

    # ---- C7 edge: hidden_context that is neither a list nor None. Only None was covered; a dict or a
    # str must ALSO be treated as unknown (not "declared") and therefore fail closed as not bindable.
    def test_hidden_context_dict_is_unknown_and_fails_closed(self):
        def entry(x):
            return x

        desc = sc._describe_signature(
            inspect.signature(entry), "fixture:entry", "entry", None,
            {"seam_runtime_evidence": {"inplace_params": [], "returns_none": False,
                                       "hidden_context": {"forward_ctx": "layer"}}})
        self.assertEqual(desc["hidden_context_evidence"], "unknown")
        verdict = sc.check_binding(desc, _descriptor(entry))
        self.assertFalse(verdict["bindable"])
        self.assertIn("hidden_context_inputs", verdict["codes"])

    def test_hidden_context_str_is_unknown_and_fails_closed(self):
        def entry(x):
            return x

        desc = sc._describe_signature(
            inspect.signature(entry), "fixture:entry", "entry", None,
            {"seam_runtime_evidence": {"inplace_params": [], "returns_none": False,
                                       "hidden_context": "forward_ctx"}})
        self.assertEqual(desc["hidden_context_evidence"], "unknown")
        verdict = sc.check_binding(desc, _descriptor(entry))
        self.assertFalse(verdict["bindable"])
        self.assertIn("hidden_context_inputs", verdict["codes"])


class TestCliBinding(unittest.TestCase):
    def test_target_override_updates_descriptor_identity_and_checks_rendered_entry(self):
        module = types.ModuleType("seam_contract_fixture")

        def entry(a, /, b=None):
            return a

        module.entry = entry
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        task = tempfile.mkdtemp(prefix="seam_contract_")
        self.addCleanup(shutil.rmtree, task, True)
        with open(os.path.join(task, "meta.json"), "w") as fh:
            json.dump({
                "target_callable": "wrong.module:entry",
                "seam_runtime_evidence": {
                    "inplace_params": [],
                    "returns_none": False,
                    "hidden_context": [],
                },
            }, fh)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = sc.main([
                "--task-dir", task,
                "--target-spec", "seam_contract_fixture:entry",
                "--mode", "binding",
                "--json",
            ])
        result = json.loads(output.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(result["binding_descriptor"]["seam"], "seam_contract_fixture:entry")
        self.assertEqual(result["binding_check"]["candidate"], "<rendered_entry>")
        self.assertTrue(result["binding_check"]["bindable"])

    def _install_fixture_module(self, name, **members):
        module = types.ModuleType(name)
        for attr, fn in members.items():
            setattr(module, attr, fn)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def _write_meta(self, **overrides):
        task = tempfile.mkdtemp(prefix="seam_contract_")
        self.addCleanup(shutil.rmtree, task, True)
        meta = {
            "seam_runtime_evidence": {
                "inplace_params": [], "returns_none": False, "hidden_context": [],
            },
        }
        meta.update(overrides)
        with open(os.path.join(task, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        return task

    # ---- C10 NEGATIVE: --baseline-spec must NEVER leak into the binding descriptor. The descriptor is
    # built from the deployment TARGET; the baseline only steers baseline_validation. (The positive
    # direction -- --target-spec drives the descriptor -- is covered above.)
    def test_baseline_spec_does_not_leak_into_binding_descriptor(self):
        def entry(a, /, b=None):
            return a

        def baseline(x, y, z):  # deliberately a DIFFERENT callable / different arity
            return x

        self._install_fixture_module("seam_c10_target", entry=entry)
        self._install_fixture_module("seam_c10_baseline", baseline=baseline)
        task = self._write_meta(
            target_callable="wrong.module:entry", baseline_callable="wrong.module:baseline")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sc.main([
                "--task-dir", task,
                "--target-spec", "seam_c10_target:entry",
                "--baseline-spec", "seam_c10_baseline:baseline",
                "--mode", "both",
                "--json",
            ])
        result = json.loads(output.getvalue())
        # The descriptor is the TARGET seam, never the baseline.
        self.assertEqual(result["binding_descriptor"]["seam"], "seam_c10_target:entry")
        self.assertNotEqual(result["binding_descriptor"]["seam"], "seam_c10_baseline:baseline")
        # Descriptor is built from the target's real signature, not the baseline's (x, y, z).
        self.assertTrue(result["binding_descriptor"]["signature"].endswith("(a, /, b=None)"))
        # --baseline-spec only steers baseline_validation.
        self.assertEqual(
            result["baseline_validation"]["baseline_callable"], "seam_c10_baseline:baseline")
        self.assertEqual(result["baseline_validation"]["target_callable"], "seam_c10_target:entry")

    # ---- C2: '--mode both' with NO --candidate must still emit a binding_check against the rendered
    # entry (candidate == "<rendered_entry>"). The existing coverage used '--mode binding'.
    def test_mode_both_without_candidate_checks_rendered_entry(self):
        def entry(a, /, b=None):
            return a

        self._install_fixture_module("seam_c2_target", entry=entry)
        task = self._write_meta(target_callable="seam_c2_target:entry")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sc.main([
                "--task-dir", task,
                "--target-spec", "seam_c2_target:entry",
                "--mode", "both",
                "--json",
            ])
        result = json.loads(output.getvalue())
        self.assertIn("binding_check", result)
        self.assertEqual(result["binding_check"]["candidate"], "<rendered_entry>")
        self.assertTrue(result["binding_check"]["bindable"], result["binding_check"]["mismatches"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
