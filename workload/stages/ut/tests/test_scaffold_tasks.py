#!/usr/bin/env python3
"""scaffold_tasks: the UT stage installs exactly what its request names, and nothing else.

Run:  python3 -I workload/stages/ut/tests/test_scaffold_tasks.py   (stdlib only, CPU only)
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scaffold_tasks", SLOT / "kernel-ut/scripts/scaffold_tasks.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScaffoldTasksTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.req = self.root / "request"
        run = self.req / "tasks/op/run"
        run.mkdir(parents=True)
        (run / "ut_case_plan.json").write_text(json.dumps({"cases": []}))
        (run / "server_calls.json").write_text(json.dumps({"calls": []}))
        (run / "server_variants.json").write_text(json.dumps({"variants": []}))
        (self.req / "selection_targets.json").write_text(json.dumps({"targets": [{"id": "k", "overall_pct": 7}]}))
        files = {str(p.relative_to(self.req)): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in self.req.rglob("*") if p.is_file()}
        self.doc = {
            "contract": S.CONTRACT, "files": files,
            "tasks": [{"label": "op",
                       "requested_kernel": {"entity_kind": "gpu_kernel", "short_name": "op",
                                            "selected_identities": ["k"],
                                            "required_companions": [{"kernel": "c", "why": "chain"}],
                                            "target_callable": "pkg.m:op"},
                       "scaffold": {"family": "custom", "target_callable": "pkg.m:op"},
                       "manifest": {"cases": [{"sig": "DECODE_4", "regime": "DECODE",
                                               "axes": {"m": 4}, "effective_calls": 12.5}],
                                    "replay_sigs": []},
                       "server_preimports": ["sglang.srt.utils.common"],
                       "server_evidence": {"server_calls": "run/server_calls.json",
                                           "compile_variants": "run/server_variants.json"},
                       "expected_launches": {"DECODE": ["k"]},
                       "compile_variant_waivers": None, "provenance": {"from": "anywhere"}}],
            "blocked_identities": [{"kernel": "b", "blocker": "no source"}]}
        self.path = self.req / "ut_request.json"
        self.path.write_text(json.dumps(self.doc))
        self.out = self.root / "out"

    def test_installs_the_request_without_any_profile_evidence(self):
        req, root, problems = S.check_request(str(self.path))
        self.assertEqual(problems, [])
        plan = S.install(req, root, str(self.out))
        tdir = self.out / "tasks/op"
        rk = json.loads((tdir / "run/requested_kernel.json").read_text())
        self.assertEqual(rk["task_dir"], str(tdir))
        self.assertEqual(rk["server_preimports"], ["sglang.srt.utils.common"])
        self.assertTrue((tdir / "run/server_calls.json").exists())
        self.assertTrue((tdir / "compile_watch.py").exists())       # replays the server calls
        self.assertEqual(plan["tasks"][0]["scaffold"]["status"], "skipped")   # custom family
        self.assertEqual(plan["tasks"][0]["required_companions"], ["c"])
        self.assertEqual(plan["blocked_identities"], [{"kernel": "b", "blocker": "no source"}])
        # A file that merely sits beside the request is not an input.
        self.assertFalse((self.out / "selection_targets.json").exists())
        self.assertIsNone(plan["selection_targets"])
        (tdir / "meta.json").write_text("{}")
        S.stamp_meta(str(tdir), self.doc["tasks"][0])
        meta = json.loads((tdir / "meta.json").read_text())
        self.assertEqual(meta["expected_launches"], {"DECODE": ["k"]})
        self.assertEqual(meta["provenance"], {"from": "anywhere"})

    def test_a_mono_layer_request_binds_the_package_and_applies_its_overrides(self):
        """A mono_layer task filled from references/mono_models.json (Muse Spark target): the
        whole package binds, and an oracle the model's example does not cover leaves the frozen
        contract through a contract override with its why."""
        models = json.loads((SLOT / "kernel-ut/references/mono_models.json").read_text())["models"]
        m = next(x for x in models if x["name"] == "muse_spark_target")
        task = self.doc["tasks"][0]
        task["label"] = m["label"]
        task["scaffold"] = {
            "family": "mono_layer", "target_callable": m["target_callable"],
            "bind_module": m["bind_module"],
            "bind_file": "kernel_src/%s/__init__.py" % m["bind_module"].replace(".", "/"),
            "attrs": m["attrs"], "axes": dict(m["axes"], world=8),
            "phase_regimes": {"DECODE": {"cuda_graph": True}},
            "contract_overrides": [{"path": "oracles", "value": {"split": {"path": "split"}},
                                    "why": "native: not implemented by this example yet"}]}
        task["manifest"]["cases"] = [{"sig": "DECODE_8_c1024", "regime": "DECODE",
                                      "axes": {"rows": 8, "context": 1024}, "effective_calls": 3}]
        self.path.write_text(json.dumps(self.doc))
        req, root, problems = S.check_request(str(self.path))
        self.assertEqual(problems, [])
        tdir = self.out / "tasks" / m["label"]
        sc = S.install(req, root, str(self.out))["tasks"][0]["scaffold"]
        self.assertEqual(sc["status"], "ok", sc.get("output"))
        self.assertIn("--bind-module", sc["command"])
        meta = json.loads((tdir / "meta.json").read_text())
        self.assertEqual(meta["candidate_bind"]["module"], m["bind_module"])
        frozen = json.loads((tdir / "run/ut_contract.json").read_text())
        self.assertEqual(sorted(frozen["oracles"]), ["split"])
        self.assertTrue((tdir / "rank_harness.py").exists())

        task["scaffold"]["contract_overrides"][0]["why"] = " "
        self.path.write_text(json.dumps(self.doc))
        _, _, problems = S.check_request(str(self.path))
        self.assertTrue(any("override" in p for p in problems), problems)

    def test_launch_waivers_and_controls_need_their_reasons(self):
        task = self.doc["tasks"][0]
        for key, bad in (("launch_waivers", [{"pattern": "_f0_", "why": " "}]),
                         ("launch_waivers", [{"pattern": "(", "why": "x"}]),
                         ("negative_controls", [{"leaf": "kernel_src/x.py", "category": "kv_slot",
                                                 "find": "a", "replace": "b", "expect": ""}])):
            with self.subTest(key=key, bad=bad):
                task[key] = bad
                self.path.write_text(json.dumps(self.doc))
                _, _, problems = S.check_request(str(self.path))
                self.assertTrue(any(key.replace("_", " ")[:-1] in p for p in problems), problems)
                task.pop(key)

    def test_share_evidence_for_roofline_is_carried_only_when_passed(self):
        selection = self.root / "profile_selection.json"
        selection.write_text(json.dumps({"targets": [{"id": "k", "overall_pct": 12.5}]}))
        req, root, _ = S.check_request(str(self.path))
        plan = S.install(req, root, str(self.out), selection=str(selection))
        carried = self.out / "selection_targets.json"
        self.assertEqual(json.loads(carried.read_text())["targets"][0]["overall_pct"], 12.5)
        self.assertEqual(plan["selection_targets"]["source"], str(selection))
        self.assertEqual(plan["selection_targets"]["sha256"],
                         hashlib.sha256(selection.read_bytes()).hexdigest())

    def test_a_changed_request_file_is_refused(self):
        (self.req / "tasks/op/run/server_calls.json").write_text("{}")
        _, _, problems = S.check_request(str(self.path))
        self.assertTrue(any("changed" in p for p in problems), problems)

    def test_a_case_without_weight_is_refused(self):
        for bad in [-1, True, float('nan'), float('inf'), -float('inf')]:
            with self.subTest(weight=bad):
                self.doc["tasks"][0]["manifest"]["cases"][0]["effective_calls"] = bad
                self.path.write_text(json.dumps(self.doc))
                _, _, problems = S.check_request(str(self.path))
                self.assertTrue(any("effective_calls" in p for p in problems), problems)

    def test_finalize_without_an_installed_request_is_blocked(self):
        out = self.root / "empty"
        out.mkdir()
        env = dict(os.environ, OUT=str(out), STAGE="ut", STAGE_SKILL="kernel-ut")
        subprocess.run([sys.executable, str(SLOT / "finalize.py")], env=env,
                       capture_output=True, text=True)
        result = json.loads((out / "result.json").read_text())
        self.assertEqual(result["status"], "blocked")
        self.assertIn("ut_request.json", result["notes"])


if __name__ == "__main__":
    unittest.main()
