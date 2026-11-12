"""CPU regressions for the vLLM observation -> sampling contract.

The same entry failures occurred in the Nov 2 DeepSeek-V4.1-Flash experiment
and the Nov 5 Qwen3-8B-FP8 AgentX run: SGLang logs/CLI/RPCs were assumed.
"""
import datetime as dt
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

    def test_mixed_step_context_counts_context_requests_only(self):
        """2 context + 3 resident decode requests: the decode KV is not prefill context."""
        row = {"time": 100, "rank": 0, "scheduled_tokens": 4012,
               "requests": [{"phase": "PREFILL", "query": 2000, "computed": 30000},
                            {"phase": "PREFILL", "query": 2000, "computed": 50000},
                            {"phase": "DECODE", "query": 4, "computed": 900000},
                            {"phase": "DECODE", "query": 4, "computed": 1000000},
                            {"phase": "DECODE", "query": 4, "computed": 1100000}]}
        log = self.root / "mixed.jsonl"
        log.write_text(json.dumps(row))
        pre, dec = ow.parse_vllm_iterations([str(log)], accept_len=2.88)
        self.assertEqual((len(pre), len(dec)), (1, 0))
        p = pre[0]
        self.assertEqual(p["new_token"], 4012)                 # whole step M, mixed tokens included
        self.assertEqual(p["new_seq"], 2)
        self.assertEqual(p["cached_token"], 80000)             # context requests only
        self.assertEqual(p["mixed_decode_requests"], 3)
        self.assertEqual(p["mixed_decode_context_tokens"], 3000000)
        self.assertEqual(p["context_query_tokens"], 4000)
        sampling = ROOT.parent / "multi_config_sampling.py"
        if not sampling.exists():
            return
        spec = importlib.util.spec_from_file_location("test_mc_sampling", sampling)
        mcs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcs)
        got = mcs.observations({"raw_prefill_rows": [dict(p, ts=p["ts"].isoformat())]})
        self.assertEqual(len(got), 1)
        # per context request: (80000 cached + 4012 scheduled) / 2 -- not ~1.5M
        self.assertEqual(got[0]["context"], (80000 + 4012) / 2)
        self.assertEqual(round(got[0]["context"] * got[0]["sequences"] - got[0]["shape"]), 80000)

    def test_crosscheck_counts_decode_requests_of_mixed_steps(self):
        """Completed tokens include mixed-step generation; route A must too."""
        rows = [{"time": 100, "rank": 0, "scheduled_tokens": 2008,
                 "requests": [{"phase": "PREFILL", "query": 2000, "computed": 0},
                              {"phase": "DECODE", "query": 4, "computed": 500},
                              {"phase": "DECODE", "query": 4, "computed": 600}]},
                {"time": 101, "rank": 0, "scheduled_tokens": 8,
                 "requests": [{"phase": "DECODE", "query": 4, "computed": 504},
                              {"phase": "DECODE", "query": 4, "computed": 604}]}]
        log = self.root / "mixed_check.jsonl"
        log.write_text("\n".join(json.dumps(r) for r in rows))
        out = self.root / "observed.json"
        run = subprocess.run([sys.executable, str(ROOT / "scripts/observed_workload.py"),
            "--log", str(log), "--log-format", "vllm-iterations", "--accept-len", "2",
            "--completed-output-tokens", "8", "--out", str(out)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        check = json.loads(out.read_text())["decode"]["crosscheck"]
        self.assertEqual(check["route_a_pure_decode_request_steps"], 2)
        self.assertEqual(check["route_a_mixed_step_decode_request_steps"], 2)
        self.assertEqual(check["route_a_request_steps_from_log_structure"], 4)
        self.assertEqual(check["disagreement_pct"], 0.0)   # was 50% without the mixed term

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

    def test_mixed_prefill_probe_matches_total_query_tokens(self):
        """Ported from 49b3f37e: a mixed step's M is every query token in it, once."""
        for decode_requests in (1, 6):
            with self.subTest(decode_requests=decode_requests):
                trace = self.root / "mixed.trace.json"
                trace.write_text(json.dumps({"traceEvents": [
                    {"name": "execute_206_context_1(sq200sk1200)_generation_%d(sq6sk406)"
                             % decode_requests,
                     "cat": "gpu_user_annotation", "ph": "X", "ts": 0, "dur": 10, "pid": 1, "tid": 1},
                    {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 1, "pid": 1, "tid": 1},
                ]}))
                paths = [{"path": str(trace), "bytes": trace.stat().st_size}]
                result = probe.verify_trace(paths, "PREFILL", "new_tokens", 206, [])
                self.assertIs(result["verified"], True, result)
                self.assertEqual(result["step_spans_by_phase_shape"], {"PREFILL": {206: 1}})
                wrong = probe.verify_trace(paths, "PREFILL", "new_tokens", 200, [])
                self.assertIsNot(wrong["verified"], True)

    def test_cache_salt_only_for_vllm_prefill(self):
        a, b = probe.cache_salt_for("vllm", "PREFILL"), probe.cache_salt_for("vllm", "PREFILL")
        self.assertTrue(a and b and a != b)
        self.assertIsNone(probe.cache_salt_for("vllm", "DECODE"))
        self.assertIsNone(probe.cache_salt_for("sglang", "PREFILL"))

    def test_sglang_request_does_not_receive_vllm_cache_salt(self):
        with patch.object(probe, "FRAMEWORK", "sglang"), patch.object(probe, "CACHE_SALT", "salt"), \
                patch.object(probe, "post", return_value="{}") as post:
            probe.generate("http://server", [1, 2], 1, [], "sglang")
        self.assertEqual(post.call_args.args[1], "/generate")
        self.assertNotIn("cache_salt", post.call_args.args[2])

    def test_prefill_invocations_cannot_hit_an_earlier_invocations_prefix_cache(self):
        """Point A, then point B (same zero cached prefix, smaller M), then B's retry. Each
        invocation's measured requests must compute their full M; without the salt B's
        repeats (seeded independently of the point) and B's retry (same seed) hit A's / the
        first attempt's cache and compute 16 tokens."""
        plan = self.root / "points.json"
        plan.write_text(json.dumps({"points": [
            {"point_id": "mc_a", "phase": "PREFILL", "new_tokens": 512, "running": 1,
             "cached_tokens": 0},
            {"point_id": "mc_b", "phase": "PREFILL", "new_tokens": 256, "running": 1,
             "cached_tokens": 0}]}))
        cache, calls = {}, []

        def cached_server(base, endpoint, body, **kwargs):
            salt, prompt = body.get("cache_salt"), body["prompt"]
            matched = 0
            for old in cache.get(salt, []):
                common = 0
                for x, y in zip(old, prompt):
                    if x != y:
                        break
                    common += 1
                matched = max(matched, min((common // 16) * 16, len(prompt) - 1))
            cache.setdefault(salt, []).append(prompt)
            calls.append((salt, len(prompt), len(prompt) - matched))
            return "{}"

        salts = []
        with patch.object(probe, "post", side_effect=cached_server), \
                patch.object(probe, "start_profile"), patch.object(probe, "stop_profile"), \
                patch.object(probe, "server_up", return_value=True), \
                patch.object(probe, "wait_drained"), patch.object(probe, "wait_traces_settled"), \
                patch.object(probe, "traces_written", return_value=[{"path": "t.json", "bytes": 1}]), \
                patch.object(probe, "verify_trace", return_value={"verified": True, "reason": "test"}), \
                patch.object(sys, "stdout", new_callable=io.StringIO):
            for index, pid in enumerate(("mc_a", "mc_b", "mc_b")):
                out = self.root / ("%d.json" % index)
                argv = ["serving_probe.py", "--framework", "vllm", "--served-model", "model",
                        "--points", str(plan), "--point-id", pid, "--seed", "1234",
                        "--num-steps", "4", "--trace-dir", str(self.root / "traces"),
                        "--out", str(out)]
                before = len(calls)
                with patch.object(sys, "argv", argv):
                    self.assertEqual(probe.main(), 0)
                mine = calls[before:]
                salt = json.loads(out.read_text())["journal"][0]["cache_salt"]
                salts.append(salt)
                self.assertTrue(salt)
                self.assertEqual({c[0] for c in mine}, {salt})   # one salt per invocation
                # shape warmup + (num_steps + 1) measured repeats, each computing all of M
                want = 512 if pid == "mc_a" else 256
                self.assertEqual(len(mine), 1 + 5)
                self.assertEqual([c[2] for c in mine[1:]], [want] * 5)
        self.assertEqual(len(set(salts)), 3)

    def test_vllm_entry_defaults_preimports_and_stack_frames_do_not(self):
        import build_task_plan as B
        tdir = self.root / "tasks" / "t"
        (tdir / "run").mkdir(parents=True)
        (self.root / "evidence").mkdir()
        want = ["vllm.platforms", "vllm._custom_ops", "vllm._aiter_ops"]
        kda = {"kernels": ["chunk_kda_fwd_kernel"],
               "target_callable": "vllm.model_executor.layers.fla.ops.kda:chunk_kda"}
        se = B.server_evidence(str(self.root), kda, str(tdir))
        self.assertEqual(se["server_preimports"], want)
        # an AITER / Triton entry launched from vLLM code keeps a bare import environment
        cv = {"variants": [{"stack": ["vllm.model_executor.layers.fused_moe:fused_experts",
                                      "aiter.ops.triton.moe:moe_kernel"]}]}
        self.assertEqual(B.frameworks({"target_callable": "aiter.ops.triton.moe:moe"}, cv), [])
        self.assertEqual(B.frameworks(kda, cv), ["vllm"])
        # sglang keeps its stack-wide default
        cv["variants"].append({"stack": ["sglang.srt.layers.attention:fwd"]})
        self.assertEqual(B.frameworks({"target_callable": "aiter.x:y"}, cv), ["sglang"])
        # the task's own declaration always wins
        se = B.server_evidence(str(self.root), dict(kda, server_preimports=["vllm.platforms"]),
                               str(tdir))
        self.assertEqual(se["server_preimports"], ["vllm.platforms"])

    def test_profile_rpc_uses_vllm_fields(self):
        with patch.object(probe, "post", return_value="{}") as post:
            probe.start_profile("http://server", "/traces/de_2", 8, [], 30)
        self.assertEqual(post.call_args.args[1:3],
                         ("/start_profile", {"profile_prefix": "de_2", "max_iterations": 8}))

    def test_vllm_prefill_window_is_closed_by_stop_not_by_empty_worker_steps(self):
        # vLLM counts empty worker steps toward max_iterations: a window of num_steps held
        # one prefill on a KV-offload MI355X server. PREFILL arms with a runaway cap only and
        # is closed by /stop_profile after its num_steps + 1 requests; DECODE keeps num_steps.
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"points": [
            {"point_id": "pf", "phase": "PREFILL", "new_tokens": 64, "running": 1,
             "cached_tokens": 0}]}))
        order = []
        with patch.object(probe, "post", side_effect=lambda *a, **k: order.append(a[1]) or "{}"), \
                patch.object(probe, "start_profile",
                             side_effect=lambda *a, **k: order.append(("arm", a[2], k))) as arm, \
                patch.object(probe, "stop_profile", side_effect=lambda *a, **k: order.append("stop")), \
                patch.object(probe, "server_up", return_value=True), \
                patch.object(probe, "wait_drained"), patch.object(probe, "wait_traces_settled"), \
                patch.object(probe, "traces_written", return_value=[{"path": "t.json", "bytes": 1}]), \
                patch.object(probe, "verify_trace", return_value={"verified": True, "reason": "test"}), \
                patch.object(sys, "stdout", new_callable=io.StringIO):
            argv = ["serving_probe.py", "--framework", "vllm", "--served-model", "model",
                    "--points", str(plan), "--point-id", "pf", "--num-steps", "4",
                    "--trace-dir", str(self.root / "traces"), "--out", str(self.root / "pf.json")]
            with patch.object(sys, "argv", argv):
                self.assertEqual(probe.main(), 0)
        self.assertEqual(arm.call_args.kwargs["max_iterations"], probe.vllm_prefill_iteration_cap(4))
        self.assertGreater(probe.vllm_prefill_iteration_cap(4), 5 * 8)
        armed = next(i for i, x in enumerate(order) if isinstance(x, tuple))
        # all 5 measured requests are sent inside the window, then it is stopped explicitly
        self.assertEqual(order[armed + 1:], ["/v1/completions"] * 5 + ["stop"])
        with patch.object(probe, "post", return_value="{}") as post:
            probe.start_profile("http://server", "/traces/pf", 4, [], 30, max_iterations=320)
        self.assertEqual(post.call_args.args[2], {"profile_prefix": "pf", "max_iterations": 320})

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

    def test_vllm_frontend_trace_is_not_linked_as_a_rank(self):
        target = self.root / "pf"
        target.mkdir()
        for name in ("pf_dp0_pp0_tp0_dcp0_ep0_rank0.17943.pt.trace.json.gz",
                     "pf_dp0_pp0_tp1_dcp1_ep1_rank1.17943.pt.trace.json.gz",
                     "pf_crsuse2-m2m-v3-258_4520.async_llm.17943.pt.trace.json.gz"):
            (self.root / name).write_text("{}")
        got = sorted(Path(r["path"]).name for r in probe.traces_written(str(target), 0))
        self.assertEqual(got, ["pf_dp0_pp0_tp0_dcp0_ep0_rank0.17943.pt.trace.json.gz",
                               "pf_dp0_pp0_tp1_dcp1_ep1_rank1.17943.pt.trace.json.gz"])

    @unittest.skipUnless((ROOT / "scripts/probe_server.py").exists(), "main uses its existing serving adapter")
    def test_vllm_launcher_and_init_preserve_backend(self):
        import probe_server
        fields = {k: None for k in ("torch_profiler_dump_cuda_time_total", "detailed_trace_annotation",
                                    "torch_profiler_record_shapes", "torch_profiler_with_stack",
                                    "ignore_frontend")}
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
        self.assertTrue(prof["ignore_frontend"])
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
