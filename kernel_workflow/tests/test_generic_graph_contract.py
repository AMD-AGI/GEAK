"""Generic (operator-neutral) graph-contract runner: gate logic, CLI contract, optional GPU smoke."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "generic_graph_contract.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import generic_graph_contract as ggc  # noqa: E402


def _passing(**extra):
    result = {
        "activation": {"status": "pass"},
        "launch_count": {"min": 2, "max": 2},
        "accuracy_results": [{"status": "pass"}],
        "replay_results": [{"status": "pass"}],
        "switch_test": {"status": "pass"},
        "speedup_gate": "pass",
    }
    result.update(extra)
    return result


class TestClaimLogic(unittest.TestCase):
    def test_complete_evidence_has_no_blockers(self):
        self.assertEqual(ggc.claim_blockers(_passing(), launch_target=2), [])

    def test_each_missing_or_failed_piece_blocks(self):
        cases = {
            "activation_not_run": {"activation": None},
            "activation_marker_missing": {"activation": {"status": "fail"}},
            "launch_target_not_met": {"launch_count": {"min": 2, "max": 3}},
            "accuracy_failed": {"accuracy_results": [{"status": "fail"}]},
            "liveness_failed": {"replay_results": [{"status": "pass"}, {"status": "fail"}]},
            "frozen_baseline_missing": {"speedup_gate": ggc.FROZEN_BASELINE_MISSING},
            "speedup_gate_failed": {"speedup_gate": "fail"},
            "timing_not_run": {"speedup_gate": None},
            "switch_test_not_run": {"switch_test": None},
            "switch_test_failed": {"switch_test": {"status": "fail"}},
            "target_activation_marker_missing": {"target_activation": {"status": "fail"}},
        }
        for blocker, patch in cases.items():
            result = _passing(**patch)
            for key in ("activation", "switch_test"):
                if key in patch and patch[key] is None:
                    del result[key]
            self.assertIn(blocker, ggc.claim_blockers(result, 2), blocker)

    def test_ranks_disagreeing_on_launches_block(self):
        self.assertIn("launch_target_not_met",
                      ggc.claim_blockers(_passing(launch_count={"min": 1, "max": 2}), 2))

    def test_capacity_fallback_only_where_allowed(self):
        allowed = _passing(mtpr_activation=[{"status": "fail", "fallback_allowed": True,
                                             "case": "128_uniform_mtpr128"}])
        self.assertEqual(ggc.claim_blockers(allowed, 2), [])
        forbidden = _passing(mtpr_activation=[{"status": "fail", "fallback_allowed": False,
                                               "case": "2048_uniform_mtpr2048"}])
        self.assertIn("activation_marker_missing:2048_uniform_mtpr2048",
                      ggc.claim_blockers(forbidden, 2))

    def test_switch_failure_is_advisory_only_when_declared(self):
        advisory = _passing(switch_test={"status": "fail", "advisory": True})
        self.assertEqual(ggc.claim_blockers(advisory, 2), [])

    def test_determinism_blocks_only_when_required(self):
        loose = _passing(determinism=[{"guard": "64@cap64", "bit_exact": False}])
        self.assertEqual(ggc.claim_blockers(loose, 2), [])
        self.assertIn("not_deterministic", ggc.claim_blockers(loose, 2, require_deterministic=True))

    def test_fewer_launches_than_the_target_pass(self):
        self.assertEqual(ggc.claim_blockers(_passing(launch_count={"min": 1, "max": 1}), 2), [])

    def test_shared_operator_may_fall_back_only_when_its_bucket_allows(self):
        allowed = _passing(activation={"status": "fail", "fallback_allowed": True})
        self.assertEqual(ggc.claim_blockers(allowed, 2), [])

    def test_guard_parsing(self):
        self.assertEqual(ggc.parse_guard("64_uniform_mtpr64"), (64, "uniform", 64))
        self.assertEqual(ggc.parse_guard("8192_rank-mixed-skew"), (8192, "rank-mixed-skew", None))
        self.assertEqual(ggc.parse_guard("64_rank-mixed-skew_mtpr64"), (64, "rank-mixed-skew", 64))
        for bad in ("uniform_64", "64", "64_uniform_mtprX"):
            with self.assertRaises(ValueError):
                ggc.parse_guard(bad)

    def test_fallback_is_per_bucket_and_never_at_the_target(self):
        self.assertTrue(ggc.fallback_allowed(8192, 0, [8192, 16384], 64))
        self.assertFalse(ggc.fallback_allowed(64, 1024, [64], 64))
        self.assertTrue(ggc.fallback_allowed(128, 1024, [], None))
        self.assertFalse(ggc.fallback_allowed(2048, 1024, [], None))

    def test_ragged_tokens_differ_per_rank_and_stay_positive(self):
        counts = [ggc.ragged_tokens(64, r, 8) for r in range(8)]
        self.assertEqual(counts[0], 64)
        self.assertEqual(len(set(counts)), 8)
        self.assertTrue(all(c >= 1 for c in counts))
        self.assertEqual(ggc.ragged_tokens(1, 7, 8), 1)

    def test_activation_needs_every_rank(self):
        self.assertEqual(ggc.activation_summary("m", 4, [1, 1, 1, 1])["status"], "pass")
        self.assertEqual(ggc.activation_summary("m", 4, [1, 0, 1, 1])["status"], "fail")
        self.assertEqual(ggc.activation_summary("m", 4, [1, 1, 1])["status"], "fail")

    def test_speedup_gates_on_frozen_baseline_not_switch_off(self):
        s = ggc.speedup_summary(10.0, [8.0, 8.0, 8.0], [8.0, 8.0, 8.0], 1.03, 0.05)
        self.assertAlmostEqual(s["absolute_speedup"], 1.25)
        self.assertAlmostEqual(s["incremental_switch_speedup"], 1.0)
        self.assertEqual(s["speedup_gate"], "pass")
        self.assertIn("denominator_mismatch", s)  # switch-off 8.0 vs frozen 10.0
        missing = ggc.speedup_summary(None, [8.0], [9.0], 1.03, 0.05)
        self.assertEqual(missing["speedup_gate"], ggc.FROZEN_BASELINE_MISSING)
        with self.assertRaises(ValueError):
            ggc.speedup_summary(10.0, [8.0], [], 1.0, 0.05)


class TestCliAndHelpers(unittest.TestCase):
    def test_runtime_file_is_an_alias_for_the_adapter(self):
        args = ggc.build_parser().parse_args(
            ["--candidate-tree", "/t", "--runtime-file", "/a.py", "--accuracy-cases", "8",
             "--json-output", "/o.json"])
        self.assertEqual(args.adapter, "/a.py")

    def test_claim_needs_a_path_marker_but_self_test_does_not(self):
        base = ["--candidate-tree", "/t", "--adapter", "/a.py", "--accuracy-cases", "8",
                "--json-output", "/o.json"]
        with self.assertRaisesRegex(ValueError, "MEGA_MEASUREMENT_GUIDE"):
            ggc.validate_args(ggc.build_parser().parse_args(base))
        self.assertEqual(ggc.validate_args(ggc.build_parser().parse_args(base + ["--self-test"])),
                         [8])
        with self.assertRaises(ValueError):
            ggc.validate_args(ggc.build_parser().parse_args(
                base + ["--path-marker", "m", "--activation-env", "NOEQUALS"]))

    def test_env_is_scoped(self):
        os.environ.pop("GGC_TEST_SWITCH", None)
        with ggc.scoped_env(ggc.parse_env(["GGC_TEST_SWITCH=1"])):
            self.assertEqual(os.environ["GGC_TEST_SWITCH"], "1")
        self.assertNotIn("GGC_TEST_SWITCH", os.environ)

    def test_adapter_protocol_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            partial = os.path.join(d, "a.py")
            with open(partial, "w") as f:
                f.write("def setup():\n    pass\n")
            with self.assertRaisesRegex(RuntimeError, "lacks build, make_inputs, forward, reference"):
                ggc.load_adapter(partial)
            with self.assertRaisesRegex(FileNotFoundError, "COMMANDMENT"):
                ggc.load_adapter(os.path.join(d, "missing.py"))

    def test_watchdog_names_the_stuck_phase(self):
        fired = []
        dog = ggc.Watchdog(0.2, lambda phase, idle: fired.append(phase)).start()
        dog.mark("replay 8_uniform")
        time.sleep(0.6)
        dog.stop()
        self.assertEqual(fired, ["replay 8_uniform"])
        quiet = []
        dog = ggc.Watchdog(0.4, lambda phase, idle: quiet.append(phase)).start()
        for i in range(6):
            dog.mark(f"step {i}")
            time.sleep(0.1)
        dog.stop()
        self.assertEqual(quiet, [])


TOY_ADAPTER = '''
import os
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size(), torch.device("cuda", rank)


class Op:
    def __init__(self):
        g = torch.Generator(device="cuda").manual_seed(7)
        self.w = torch.randn(64, 64, device="cuda", generator=g) / 8
        self.on = os.environ.get("GGC_TOY_ON") == "1"
        self.scale = 1.5 if os.environ.get("GGC_TOY_WRONG") == "1" else 1.0

    def __call__(self, x):
        if self.on and not torch.cuda.is_current_stream_capturing():
            print(f"[toy] path=ON rank={dist.get_rank()}", flush=True)
        return torch.relu(x @ self.w) * self.scale


def build(capacity):
    return Op()


def make_inputs(tokens, route, iteration):
    g = torch.Generator(device="cuda").manual_seed(1000 * iteration + tokens + dist.get_rank())
    return (torch.randn(tokens, 64, device="cuda", generator=g),)


def forward(op, inputs):
    return op(*inputs)


def reference(inputs):
    g = torch.Generator(device="cuda").manual_seed(7)
    w = torch.randn(64, 64, device="cuda", generator=g) / 8
    return torch.relu(inputs[0].double() @ w.double()).float()


def cleanup():
    dist.destroy_process_group()
'''


@unittest.skipUnless(os.environ.get("GEAK_GPU_SMOKE") == "1" and shutil.which("torchrun"),
                     "set GEAK_GPU_SMOKE=1 to run the 2-rank on-card smoke")
class TestGpuSmoke(unittest.TestCase):
    def _run(self, d, *extra):
        out = os.path.join(d, "out.json")
        cmd = ["torchrun", "--standalone", "--nproc_per_node=2", TOOL, "--candidate-tree", d,
               "--runtime-file", os.path.join(d, "adapter.py"), "--accuracy-cases", "16,64",
               "--routes", "uniform,other", "--replays", "8", "--mtpr-cases", "16",
               "--path-marker", "path=ON", "--control-env", "GGC_TOY_ON=0",
               "--frozen-baseline-ms", "1000", "--launch-target", "8", "--json-output", out, *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        with open(out) as f:
            return proc.returncode, json.load(f)

    def test_claim_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "adapter.py"), "w") as f:
                f.write(TOY_ADAPTER)
            rc, good = self._run(d, "--activation-env", "GGC_TOY_ON=1")
            self.assertEqual(rc, 0, good.get("error"))
            self.assertTrue(good["claim_complete"])
            self.assertEqual(len(good["accuracy_results"]), 3)
            self.assertEqual(len(good["replay_results"]), 6)
            self.assertEqual(good["switch_test"]["status"], "pass")
            self.assertGreater(good["switch_test"]["steps"], 0)
            self.assertTrue(all(row["bit_exact"] for row in good["determinism"]))
            rc, mtpr = self._run(d, "--activation-env", "GGC_TOY_ON=1",
                                 "--target-guard", "16_other_mtpr16")
            self.assertEqual(rc, 0, mtpr.get("error"))
            self.assertEqual(mtpr["target"]["capacity"], 16)
            self.assertEqual(mtpr["target_activation"]["status"], "pass")
            self.assertEqual(mtpr["paired_readings"][0]["guard"], "16_other_mtpr16")
            rc, off = self._run(d)
            self.assertIn("activation_marker_missing", off["claim_blockers"])
            rc, wrong = self._run(d, "--activation-env", "GGC_TOY_ON=1",
                                  "--activation-env", "GGC_TOY_WRONG=1")
            self.assertIn("accuracy_failed", wrong["claim_blockers"])
            self.assertFalse(wrong["claim_complete"])


if __name__ == "__main__":
    unittest.main()
