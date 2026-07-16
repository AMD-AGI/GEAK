#!/usr/bin/env python3
"""Unit tests for probe_mbuckets_guard (stdlib only).
Run: python3 -m unittest e2e_workflow.scripts.tests.test_probe_mbuckets_guard -v
  or: python3 e2e_workflow/scripts/tests/test_probe_mbuckets_guard.py
"""
import importlib.util, os, unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guard = _load("probe_mbuckets_guard", "probe_mbuckets_guard.py")


class TestIsSynthesizedFallback(unittest.TestCase):
    def test_exact_synth_pair_is_fallback(self):
        # [1, CONC] 是 workflow 传入的合成猜测
        self.assertTrue(guard.is_synthesized_fallback([1, 64], 64))

    def test_conc_only_is_fallback(self):
        self.assertTrue(guard.is_synthesized_fallback([64], 64))

    def test_empty_is_fallback(self):
        self.assertTrue(guard.is_synthesized_fallback([], 64))

    def test_measured_with_extra_M_is_not_fallback(self):
        # 手动验证的标准答案 decode=[64, 512] —— 含 conc 之外的实测 512
        self.assertFalse(guard.is_synthesized_fallback([64, 512], 64))

    def test_measured_single_nonconc_is_not_fallback(self):
        self.assertFalse(guard.is_synthesized_fallback([512], 64))


class TestClassifyMbuckets(unittest.TestCase):
    def test_missing_key(self):
        self.assertEqual(guard.classify_mbuckets({}, 64), "missing")

    def test_synthesized(self):
        self.assertEqual(
            guard.classify_mbuckets({"decode_m_buckets": [1, 64]}, 64),
            "synthesized_fallback")

    def test_measured(self):
        self.assertEqual(
            guard.classify_mbuckets({"decode_m_buckets": [64, 512]}, 64),
            "measured")


if __name__ == "__main__":
    unittest.main()
