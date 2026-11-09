"""CPU regressions for the vLLM observation -> sampling contract.

The same entry failures occurred in the Nov 2 DeepSeek-V4.1-Flash experiment
and the Nov 5 Qwen3-8B-FP8 AgentX run: SGLang logs/CLI/RPCs were assumed.
"""
import datetime as dt
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared"
if not SHARED.exists():
    SHARED = ROOT.parents[2] / "scripts"
sys.path[:0] = [str(ROOT / "scripts"), str(SHARED)]
import observed_workload as ow
import serving_probe as probe

OBSERVER = ROOT / "scripts/vllm_observer"
if not OBSERVER.exists():
    OBSERVER = SHARED / "adapters/iteration_env"
spec = importlib.util.spec_from_file_location("test_geak_vllm_observer", OBSERVER / "geak_vllm_observer.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class VllmProfileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(probe, "FRAMEWORK", "vllm").start()
        patch.object(probe, "SERVED_MODEL", "served-model").start()
        patch.object(probe, "ITERATION_LOG_DIR", None).start()
        patch.object(probe, "CACHE_SALT", None).start()
        patch.object(probe, "ID_HI", 150000).start()

    def test_observer_preserves_return_and_records_geometry_without_payloads(self):
        result = object()
        class Worker:
            rank = 0
            def annotate_profile(self, output):
                return result
        cached = types.SimpleNamespace(req_ids=["secret-id"], num_computed_tokens=[400],
                                       is_context_phase=lambda rid: False)
        output = types.SimpleNamespace(total_num_scheduled_tokens=206,
            scheduled_new_reqs=[types.SimpleNamespace(req_id="prompt-id", num_computed_tokens=1000)],
            scheduled_cached_reqs=cached, num_scheduled_tokens={"prompt-id": 200, "secret-id": 6})
        with patch.dict(os.environ, {"GEAK_ITERATION_LOG_DIR": str(self.root)}):
            observer.patch_worker(types.SimpleNamespace(Worker=Worker))
            observer.patch_worker(types.SimpleNamespace(Worker=Worker))
            self.assertIs(Worker().annotate_profile(output), result)
        text = next(self.root.glob("rank0_pid*.jsonl")).read_text()
        self.assertEqual(len(text.splitlines()), 1)
        self.assertNotIn("secret-id", text)
        row = json.loads(text)
        self.assertEqual(row["scheduled_tokens"], 206)
        self.assertEqual(row["requests"], [{"phase": "PREFILL", "query": 200, "computed": 1000},
                                            {"phase": "DECODE", "query": 6, "computed": 400}])

    def test_observer_budget_failure_cannot_be_parsed_as_complete(self):
        class Worker:
            rank = 0
            def annotate_profile(self, output):
                return 7
        output = types.SimpleNamespace(total_num_scheduled_tokens=1)
        with patch.dict(os.environ, {"GEAK_ITERATION_LOG_DIR": str(self.root),
                                     "GEAK_ITERATION_LOG_MAX_BYTES": "0"}):
            observer.patch_worker(types.SimpleNamespace(Worker=Worker))
            self.assertEqual(Worker().annotate_profile(output), 7)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            ow.parse_vllm_iterations([str(next(self.root.glob("*.jsonl")))])

    def test_sitecustomize_chains_existing_overlay_without_importing_vllm(self):
        overlay = self.root / "overlay"
        overlay.mkdir()
        (overlay / "sitecustomize.py").write_text("import sys\nsys.geak_test_overlay = 'loaded'\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(OBSERVER), str(overlay)]),
                   GEAK_ITERATION_LOG_DIR=str(self.root / "iterations"))
        run = subprocess.run([sys.executable, "-c",
            "import sys; assert sys.geak_test_overlay == 'loaded'; assert 'vllm' not in sys.modules"],
            cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_lazy_worker_import_installs_once_and_keeps_original_call(self):
        """Exercise the actual import hook, as a spawned serving worker reaches it."""
        package = self.root / "vllm/v1/worker"
        package.mkdir(parents=True)
        for directory in (package, package.parent, package.parent.parent):
            (directory / "__init__.py").write_text("")
        (package / "gpu_worker.py").write_text(
            "class Worker:\n    rank = 0\n"
            "    def annotate_profile(self, output):\n        return 'original-result'\n")
        with patch.dict(sys.modules), patch.object(sys, "path", [str(self.root), *sys.path]), \
                patch.object(sys, "meta_path", list(sys.meta_path)), \
                patch.dict(os.environ, {"GEAK_ITERATION_LOG_DIR": str(self.root / "iterations")}):
            for name in list(sys.modules):
                if name == "vllm" or name.startswith("vllm."):
                    del sys.modules[name]
            observer.install()
            observer.install()
            module = importlib.import_module(observer.MODULE)
            observer.install()  # An already-loaded worker must not gain a second wrapper.
            self.assertTrue(module.Worker.annotate_profile._geak_iteration_observer)
            empty = types.SimpleNamespace(total_num_scheduled_tokens=0)
            self.assertEqual(module.Worker().annotate_profile(empty), "original-result")
            output = types.SimpleNamespace(total_num_scheduled_tokens=1,
                scheduled_new_reqs=[types.SimpleNamespace(req_id="private", num_computed_tokens=0)],
                scheduled_cached_reqs=types.SimpleNamespace(req_ids=[], num_computed_tokens=[]),
                num_scheduled_tokens={"private": 1})
            self.assertEqual(module.Worker().annotate_profile(output), "original-result")
        lines = next((self.root / "iterations").glob("*.jsonl")).read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["scheduled_tokens"], 1)

    def test_unsupported_worker_reports_missing_evidence_without_breaking_serving(self):
        class Worker:
            def execute_model(self):
                return 42
        with patch.object(sys, "stderr", new_callable=io.StringIO) as err:
            observer.patch_worker(types.SimpleNamespace(Worker=Worker))
        self.assertIn("observation unavailable", err.getvalue())
        self.assertEqual(Worker().execute_model(), 42)
        self.assertFalse(hasattr(Worker, "annotate_profile"))

    def rows(self):
        return [
            {"time": 100, "rank": 0, "scheduled_tokens": 206,
             "requests": [{"phase": "PREFILL", "query": 200, "computed": 1000},
                          {"phase": "DECODE", "query": 6, "computed": 400}]},
            {"time": 101, "rank": 0, "scheduled_tokens": 12,
             "requests": [{"phase": "DECODE", "query": 6, "computed": 1100},
                          {"phase": "DECODE", "query": 6, "computed": 406}]},
            {"time": 102, "rank": 1, "scheduled_tokens": 1,
             "requests": [{"phase": "DECODE", "query": 1, "computed": 10}]}]

    def test_cli_counts_real_iterations_and_marks_cache_and_graph_unknown(self):
        log, out = self.root / "steps.jsonl", self.root / "observed.json"
        log.write_text("\n".join(json.dumps(r) for r in self.rows()))
        run = subprocess.run([sys.executable, str(ROOT / "scripts/observed_workload.py"),
            "--log", str(log), "--log-format", "vllm-iterations", "--accept-len", "3.51",
            "--out", str(out)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        got = json.loads(out.read_text())
        self.assertEqual(got["counts"], {"prefill_batches": 1, "decode_rows_logged": 1, "decode_iterations": 1})
        self.assertEqual(got["raw_prefill_rows"][0]["new_token"], 206)
        self.assertEqual(got["mixed_batches"], 1)
        self.assertIsNone(got["prefill"]["prefix_cache_hit_pct"])
        self.assertIsNone(got["decode"]["cuda_graph_all"])

    def test_iteration_window_rank_and_conservation(self):
        log = self.root / "steps.jsonl"
        log.write_text("\n".join(json.dumps(r) for r in self.rows()))
        window = tuple(dt.datetime.fromtimestamp(t, dt.timezone.utc).replace(tzinfo=None) for t in (101, 102))
        pre, dec = ow.parse_vllm_iterations([str(log)], window=window)
        self.assertEqual((len(pre), len(dec)), (0, 1))
        self.assertEqual(dec[0]["token"], 1518)
        bad = self.rows()[0]
        bad["scheduled_tokens"] = 205
        log.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "conserve"):
            ow.parse_vllm_iterations([str(log)])

    def test_aggregate_log_reports_missing_geometry_instead_of_guessing(self):
        log = self.root / "server.log"
        log.write_text("(APIServer pid=123) INFO [loggers.py:320] Engine 000: Avg prompt throughput: 12 tokens/s\n")
        run = subprocess.run([sys.executable, str(ROOT / "scripts/observed_workload.py"),
            "--log", str(log), "--out", str(self.root / "observed.json")], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("vllm-iterations", run.stderr)
        self.assertFalse((self.root / "observed.json").exists())

    def test_generate_uses_openai_token_endpoint(self):
        with patch.object(probe, "post", return_value="{}") as post:
            journal = []
            probe.generate("http://server", [1, 2, 3], 8, journal, "case")
        args = post.call_args.args
        self.assertEqual(args[1], "/v1/completions")
        self.assertEqual(args[2], {"model": "served-model", "prompt": [1, 2, 3],
                                   "max_tokens": 8, "temperature": 0.0, "ignore_eos": True})
        self.assertEqual(journal[0]["event"], "request_done")

    def test_prefill_uses_step_token_total_not_running_request_multiplier(self):
        point = {"phase": "PREFILL", "new_tokens": 4244, "running": 8,
                 "cached_tokens": 558533}
        args = types.SimpleNamespace(seed=1234, request_timeout=30)
        journal = []
        with patch.object(probe, "post", return_value="{}") as post:
            scope = probe.drive_prefill("http://server", point, args, journal)
        post.assert_called_once()
        body = post.call_args.args[2]
        self.assertEqual(len(body["prompt"]), 4244)
        self.assertEqual(body["max_tokens"], 1)
        self.assertEqual(scope["planned_running"], 8)
        self.assertEqual(scope["prompt_requests"], 1)
        self.assertFalse(scope["reproduces_request_mix"])
        self.assertFalse(scope["reproduces_cached_context"])
        self.assertEqual(journal[0]["event"], "request_geometry")

    def test_prefill_request_error_is_not_a_successful_capture(self):
        args = types.SimpleNamespace(seed=1234, request_timeout=30)
        journal = []
        with patch.object(probe, "post", side_effect=RuntimeError("HTTP 400")):
            with self.assertRaisesRegex(RuntimeError, "prefill request failed"):
                probe.drive_prefill("http://server", {"new_tokens": 4244, "running": 8}, args, journal)
        self.assertEqual(journal[-1]["event"], "request_failed")

    def test_cli_isolates_prefix_cache_across_points_and_same_seed_retries(self):
        """A short prompt is a prefix of the long one; every invocation must recompute it."""
        plan = self.root / "points.json"
        plan.write_text(json.dumps({"points": [
            {"point_id": "long", "phase": "PREFILL", "new_tokens": 16384, "running": 8},
            {"point_id": "short", "phase": "PREFILL", "new_tokens": 4244, "running": 8}]}))
        cache, bodies, computed = {}, [], []
        def cached_server(base, endpoint, body, **kwargs):
            salt, prompt = body.get("cache_salt"), body["prompt"]
            matched = 0
            for old in cache.get(salt, []):
                common = 0
                for a, b in zip(old, prompt):
                    if a != b:
                        break
                    common += 1
                matched = max(matched, (common // 16) * 16)
            computed.append(len(prompt) - matched)
            cache.setdefault(salt, []).append(prompt)
            bodies.append(body)
            return "{}"
        with patch.object(probe, "post", side_effect=cached_server), \
                patch.object(probe, "start_profile"), patch.object(probe, "stop_profile"), \
                patch.object(probe, "traces_written", return_value=[{"path": "test.trace.json", "bytes": 1}]), \
                patch.object(probe, "verify_trace", return_value={"verified": True, "reason": "test"}), \
                patch.object(sys, "stdout", new_callable=io.StringIO):
            for index, pid in enumerate(("long", "short", "short")):
                out = self.root / (str(index) + ".json")
                argv = ["serving_probe.py", "--framework", "vllm", "--served-model", "model",
                        "--points", str(plan), "--point-id", pid, "--seed", "1234",
                        "--trace-dir", str(self.root / "traces"), "--out", str(out)]
                with patch.object(sys, "argv", argv):
                    self.assertEqual(probe.main(), 0)
                result = json.loads(out.read_text())
                self.assertEqual(result["journal"][0]["cache_salt"], bodies[-1]["cache_salt"])
                self.assertFalse(result["capture_scope"]["reproduces_cached_context"])
        self.assertEqual(computed, [16384, 4244, 4244])
        self.assertEqual(bodies[0]["prompt"][:4244], bodies[1]["prompt"])
        self.assertEqual(bodies[1]["prompt"], bodies[2]["prompt"])
        self.assertEqual(len({body["cache_salt"] for body in bodies}), 3)

    def test_wrong_prefill_shape_stays_partial(self):
        trace = self.root / "cached.trace.json"
        trace.write_text(json.dumps({"traceEvents": [
            {"name": "execute_4_context_1(sq4sk4244)_generation_0(sq0sk0)",
             "cat": "gpu_user_annotation", "ph": "X", "ts": 0, "dur": 10, "pid": 1, "tid": 1},
            {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 1, "pid": 1, "tid": 1}]}))
        result = probe.verify_trace([{"path": str(trace)}], "PREFILL", "new_tokens", 4244, [])
        self.assertEqual(result["verified"], "partial")
        self.assertFalse(result["exact_shape_present"])

    def test_sglang_request_does_not_receive_vllm_cache_salt(self):
        with patch.object(probe, "FRAMEWORK", "sglang"), patch.object(probe, "CACHE_SALT", "salt"), \
                patch.object(probe, "post", return_value="{}") as post:
            probe.generate("http://server", [1, 2], 1, [], "sglang")
        self.assertEqual(post.call_args.args[1], "/generate")
        self.assertNotIn("cache_salt", post.call_args.args[2])

    def test_profile_rpc_uses_vllm_fields(self):
        with patch.object(probe, "post", return_value="{}") as post:
            probe.start_profile("http://server", "/traces/de_2", 8, [], 30)
        self.assertEqual(post.call_args.args[1:3],
                         ("/start_profile", {"profile_prefix": "de_2", "max_iterations": 8}))

    def test_vllm_metrics_use_generation_counter(self):
        metrics = b'vllm:num_requests_running{engine="0"} 2\nvllm:generation_tokens_total{engine="0"} 50\nvllm:prompt_tokens_total{engine="0"} 200\n'
        with patch.object(probe.urllib.request, "urlopen", return_value=io.BytesIO(metrics)):
            got, err = probe.scrape_metrics("http://server")
        self.assertIsNone(err)
        self.assertEqual(got["num_running_reqs"], 2)
        self.assertEqual(got["decode_tokens"], 50)

    def test_trace_prefix_does_not_pick_another_point(self):
        target = self.root / "de_1"
        target.mkdir()
        (self.root / "de_1_rank0.trace.json").write_text('{}')
        (self.root / "de_10_rank0.trace.json").write_text('{}')
        got = probe.traces_written(str(target), 0)
        self.assertEqual([Path(r["path"]).name for r in got], ["de_1_rank0.trace.json"])

    @unittest.skipUnless((ROOT / "scripts/probe_server.py").exists(), "main uses its existing serving adapter")
    def test_vllm_launcher_and_init_preserve_backend(self):
        import probe_server
        fields = {k: None for k in ("torch_profiler_dump_cuda_time_total", "detailed_trace_annotation",
                                    "torch_profiler_record_shapes", "torch_profiler_with_stack")}
        config = types.ModuleType("vllm.config")
        config.ProfilerConfig = types.SimpleNamespace(model_fields=fields)
        cfg = {"framework": "vllm", "model": "model", "tp": 2, "port": 8001,
               "args": ["--max-num-seqs", "8"], "profiler_config": {"max_iterations": 7}}
        with patch.dict(sys.modules, {"vllm.config": config}), patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}):
            cmd, env = probe_server.command(cfg, "as_declared", str(self.root / "traces"))
        self.assertEqual(cmd[:3], ["vllm", "serve", "model"])
        self.assertNotIn("--model-path", cmd)
        self.assertNotIn("--enable-profile-cuda-graph", cmd)
        prof = json.loads(cmd[cmd.index("--profiler-config") + 1])
        self.assertTrue(prof["detailed_trace_annotation"])
        self.assertFalse(prof["torch_profiler_dump_cuda_time_total"])
        self.assertEqual(env["GEAK_ITERATION_LOG_DIR"], str(self.root / "iterations"))
        cfg["args"] += ["--profiler-config=" + json.dumps({"torch_profiler_dir": "/old", "max_iterations": 11})]
        with patch.dict(sys.modules, {"vllm.config": config}), patch.dict(os.environ, {"GPU_ARCHS": "gfx950"}):
            cmd, _ = probe_server.command(cfg, "as_declared", str(self.root / "traces"))
        self.assertEqual(cmd.count("--profiler-config"), 1)
        prof = json.loads(cmd[cmd.index("--profiler-config") + 1])
        self.assertEqual(prof["max_iterations"], 11)
        self.assertEqual(prof["torch_profiler_dir"], str(self.root / "traces"))
        out = self.root / "serving.json"
        run = subprocess.run([sys.executable, str(ROOT / "scripts/probe_server.py"),
                              "init", "--framework", "vllm", "--out", str(out)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(out.read_text())["launcher"], ["vllm", "serve"])

    @unittest.skipUnless((ROOT / "scripts/point_census.py").exists(), "main uses parse_profile")
    def test_multi_stream_iteration_and_phase_transition_count_once(self):
        import point_census
        name = "execute_206_context_1(sq200sk1200)_generation_1(sq6sk406)"
        dec = "execute_12_context_0(sq0sk0)_generation_2(sq12sk1518)"
        events = [("gpu_user_annotation", name, 100, 20, 0, 2, {"External id": 1}),
                  ("gpu_user_annotation", name, 105, 5, 0, 4, {"External id": 1}),
                  ("gpu_user_annotation", dec, 130, 20, 0, 2, {"External id": 9})]
        gpu, cpu = point_census.step_spans(events)
        self.assertEqual(len(gpu), 2)
        wins, _, err = point_census.instance_windows(gpu, cpu, "PREFILL")
        self.assertIsNone(err)
        self.assertEqual([(w["start"], w["end"], w["shape"]) for w in wins], [(100, 130, 206)])


if __name__ == "__main__":
    unittest.main()
