#!/usr/bin/env python3
"""Unit tests for probe_to_mbuckets.py -- probe shapes -> meta.json m_bucket lists (stdlib only).

This adapter is what replaces the INFERRED decode M ("M ~= WORKLOAD.conc") with the M values the
probe actually measured on a live server. Everything downstream -- the extracted unittest's timing
cases, attribute_weights.attribute_gemm's per-bucket weights -- consumes the two lists it emits, so a
mistake here silently retargets the whole optimization at shapes production never runs. The
behaviours pinned below are the ones that would do that quietly:

  - kernel selection : match by label OR target substring, and on a tie take the busiest kernel, so a
                       model with several captured GEMMs cannot bucket the wrong one
  - the decode/prefill split at conc * decode_max_mult, including the boundary M == threshold
  - the count-share floor : long-tail one-off prefill chunk shapes are dropped, and the floor is
                            evaluated against the kernel's OWN total, not the probe-wide total
  - the two failure paths return (None, reason) rather than an empty-but-plausible bucket list, and
    main() turns those into exit code 2

Run: python3 -m pytest e2e_workflow/scripts/tests/test_probe_to_mbuckets.py -v
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ptm = _load("probe_to_mbuckets", "probe_to_mbuckets.py")


def _case(m, count, extra_dims=(1024,)):
    """One probe case whose activation M is `m` (dims[0][0] is what the adapter reads)."""
    return {"dims": [[m, *extra_dims], [4096, 1024]], "dtypes": ["torch.bfloat16"], "count": count}


def _kernel(label="matmul_ogs", cases=(), status="captured", target="triton_kernels.matmul_ogs"):
    return {"label": label, "target": target, "probe_status": status,
            "probe_total_calls": sum(c.get("count", 0) for c in cases), "cases": list(cases)}


def _probe(*kernels):
    return {"schema": "per-shape-probe-v1", "kernels": list(kernels)}


class ExtractSelection(unittest.TestCase):
    """WHICH captured kernel the buckets are derived from."""

    def test_matches_on_label_or_target_substring(self):
        probe = _probe(_kernel(label="matmul_ogs", cases=[_case(64, 100)]))
        for match in ("matmul_ogs", "MATMUL_OGS", "triton_kernels"):
            res, err = ptm.extract(probe, conc=64, kernel_match=match,
                                   decode_max_mult=8.0, min_count_share=0.01)
            self.assertIsNone(err, match)
            self.assertEqual(res["decode_m_buckets"], [64])

    def test_empty_match_takes_any_captured_kernel(self):
        res, err = ptm.extract(_probe(_kernel(cases=[_case(32, 10)])), conc=64, kernel_match="",
                               decode_max_mult=8.0, min_count_share=0.01)
        self.assertIsNone(err)
        self.assertEqual(res["decode_m_buckets"], [32])

    def test_uncaptured_kernels_are_never_selected(self):
        probe = _probe(_kernel(cases=[_case(64, 10)], status="unhookable"))
        res, err = ptm.extract(probe, 64, "matmul_ogs", 8.0, 0.01)
        self.assertIsNone(res)
        self.assertIn("no captured kernel matching 'matmul_ogs'", err)

    def test_no_name_match_is_an_error_not_a_silent_other_kernel(self):
        probe = _probe(_kernel(label="unified_attention", target="aiter:unified_attention",
                               cases=[_case(64, 10)]))
        res, err = ptm.extract(probe, 64, "matmul_ogs", 8.0, 0.01)
        self.assertIsNone(res)
        self.assertIn("matmul_ogs", err)

    def test_several_matches_take_the_busiest(self):
        """Two captured GEMMs: the one carrying the traffic decides the buckets."""
        quiet = _kernel(label="matmul_ogs_aux", cases=[_case(7, 5)])
        busy = _kernel(label="matmul_ogs_main", cases=[_case(64, 5000)])
        res, err = ptm.extract(_probe(quiet, busy), 64, "matmul_ogs", 8.0, 0.01)
        self.assertIsNone(err)
        self.assertEqual(res["decode_m_buckets"], [64])
        self.assertIn("kernel=matmul_ogs_main", res["notes"])


class ExtractSplit(unittest.TestCase):
    """decode vs prefill, and which M survive the count-share floor."""

    def test_split_at_conc_times_mult_inclusive(self):
        """M == conc*mult is DECODE (<=), one above is prefill -- the boundary the doc promises."""
        probe = _probe(_kernel(cases=[_case(512, 100), _case(513, 100)]))
        res, _ = ptm.extract(probe, conc=64, kernel_match="", decode_max_mult=8.0,
                             min_count_share=0.01)
        self.assertEqual(res["decode_m_buckets"], [512])
        self.assertEqual(res["prefill_m_buckets"], [513])

    def test_realistic_moe_distribution(self):
        """conc=64 top_k=4: decode M in {1,64,256}, prefill chunks at 2048/4096."""
        probe = _probe(_kernel(cases=[
            _case(256, 9000), _case(64, 800), _case(1, 200),
            _case(4096, 60), _case(2048, 40),
        ]))
        res, err = ptm.extract(probe, conc=64, kernel_match="", decode_max_mult=8.0,
                               min_count_share=0.001)
        self.assertIsNone(err)
        self.assertEqual(res["decode_m_buckets"], [1, 64, 256])
        self.assertEqual(res["prefill_m_buckets"], [2048, 4096])

    def test_counts_for_one_m_are_summed_across_cases(self):
        """Same M under different secondary dims is ONE bucket whose share is the sum."""
        probe = _probe(_kernel(cases=[
            _case(64, 6, extra_dims=(1024,)), _case(64, 6, extra_dims=(2048,)),
            _case(9999, 88),
        ]))
        res, _ = ptm.extract(probe, 64, "", 8.0, min_count_share=0.11)
        # 12/100 = 12% clears an 11% floor only because the two cases were summed.
        self.assertEqual(res["decode_m_buckets"], [64])
        self.assertIn("distinct_M=2", res["notes"])

    def test_long_tail_below_share_floor_is_dropped(self):
        probe = _probe(_kernel(cases=[_case(64, 1000), _case(4096, 500), _case(4097, 1)]))
        res, _ = ptm.extract(probe, 64, "", 8.0, min_count_share=0.01)
        self.assertEqual(res["prefill_m_buckets"], [4096])
        self.assertIn("kept_prefill=1", res["notes"])

    def test_share_floor_is_relative_to_this_kernel_only(self):
        """A busier OTHER kernel must not push this kernel's shapes under the floor."""
        target = _kernel(label="matmul_ogs", cases=[_case(64, 10), _case(4096, 10)])
        noisy = _kernel(label="unified_attention", target="aiter:unified_attention",
                        cases=[_case(1, 10_000_000)])
        res, _ = ptm.extract(_probe(target, noisy), 64, "matmul_ogs", 8.0, 0.4)
        self.assertEqual(res["decode_m_buckets"], [64])
        self.assertEqual(res["prefill_m_buckets"], [4096])
        self.assertIn("total_calls=20", res["notes"])

    def test_dropping_everything_yields_empty_lists_not_an_error(self):
        probe = _probe(_kernel(cases=[_case(64, 1), _case(65, 1)]))
        res, err = ptm.extract(probe, 64, "", 8.0, min_count_share=0.9)
        self.assertIsNone(err)
        self.assertEqual((res["decode_m_buckets"], res["prefill_m_buckets"]), ([], []))

    def test_cases_without_usable_dims_are_skipped(self):
        """dims=[] / dims=[[]] / a non-int M are what a graph-hidden call records -- never a bucket."""
        probe = _probe(_kernel(cases=[
            {"dims": [], "count": 500},
            {"dims": [[]], "count": 500},
            {"count": 500},
            {"dims": [["dyn", 1024]], "count": 500},
            _case(64, 10),
        ]))
        res, err = ptm.extract(probe, 64, "", 8.0, 0.01)
        self.assertIsNone(err)
        self.assertEqual(res["decode_m_buckets"], [64])
        self.assertIn("total_calls=10", res["notes"])

    def test_all_dims_empty_is_an_error_not_empty_buckets(self):
        """The graph-capture failure mode: report it, do not hand back plausible empty lists."""
        probe = _probe(_kernel(cases=[{"dims": [], "count": 500}]))
        res, err = ptm.extract(probe, 64, "", 8.0, 0.01)
        self.assertIsNone(res)
        self.assertIn("no real activation shapes", err)

    def test_missing_count_defaults_to_zero_share(self):
        probe = _probe(_kernel(cases=[{"dims": [[64, 1024]]}, _case(4096, 3)]))
        res, err = ptm.extract(probe, 64, "", 8.0, 0.01)
        self.assertIsNone(err)
        self.assertEqual(res["decode_m_buckets"], [])
        self.assertEqual(res["prefill_m_buckets"], [4096])


class Main(unittest.TestCase):
    """CLI: stdout JSON, --out, and the exit code the extractor branches on."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="probe_to_mbuckets_test_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self._argv = sys.argv
        self.addCleanup(setattr, sys, "argv", self._argv)

    def _write(self, probe):
        path = os.path.join(self.tmp, "probe.json")
        with open(path, "w") as fh:
            json.dump(probe, fh)
        return path

    def _run(self, *extra):
        sys.argv = ["probe_to_mbuckets.py", *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            ptm.main()
        return out.getvalue(), err.getvalue()

    def test_prints_json_to_stdout(self):
        probe = self._write(_probe(_kernel(cases=[_case(64, 100), _case(4096, 100)])))
        out, _ = self._run("--probe", probe, "--conc", "64")
        parsed = json.loads(out)
        self.assertEqual(parsed["decode_m_buckets"], [64])
        self.assertEqual(parsed["prefill_m_buckets"], [4096])

    def test_out_flag_writes_the_same_json_it_prints(self):
        probe = self._write(_probe(_kernel(cases=[_case(64, 100)])))
        dest = os.path.join(self.tmp, "buckets.json")
        out, _ = self._run("--probe", probe, "--conc", "64", "--out", dest,
                           "--kernel-match", "matmul_ogs", "--decode-max-mult", "4",
                           "--min-count-share", "0.5")
        with open(dest) as fh:
            self.assertEqual(fh.read(), out.rstrip("\n"))

    def test_extraction_failure_exits_2_with_the_reason_on_stderr(self):
        probe = self._write(_probe(_kernel(cases=[_case(64, 1)], status="unhookable")))
        with self.assertRaises(SystemExit) as cm:
            self._run("--probe", probe, "--conc", "64")
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
