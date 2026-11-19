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

    def test_frontend_trace_is_not_a_rank(self):
        # vLLM 0.33 (MiniMax-M3.1-Flash image) also traces the API server per window.
        # Linked as a 5th file at TP=4, verify scanned it (0 kernels) and every census
        # failed: "ranks disagree" for prefill, "fewer than 2 decode anchors" for decode.
        target = self.root / "pt_a"
        target.mkdir()
        rank = "pt_a_dp0_pp0_tp0_dcp0_ep0_rank0.1795.pt.trace.json.gz"
        (self.root / rank).write_text("{}")
        (self.root / "pt_a_host_376.async_llm.1795.pt.trace.json.gz").write_text("{}" * 99)
        got = probe.traces_written(str(target), 0)
        self.assertEqual([Path(r["path"]).name for r in got], [rank])

    def test_eager_window_prefix_is_valid_and_kept_apart(self):
        # vLLM 0.33 rejects "pt_a@eager" (HTTP 400): the eager evidence window never armed.
        body = {}
        with patch.object(probe, "post", lambda base, path, b, timeout: body.update(b) or ""):
            probe.start_profile("http://x", str(self.root / "pt_a@eager"), 8, [], 5)
        self.assertEqual(body["profile_prefix"], "pt_a-eager")
        graph, eager = self.root / "pt_a", self.root / "pt_a@eager"
        graph.mkdir()
        eager.mkdir()
        (self.root / "pt_a_rank0.1.pt.trace.json.gz").write_text("{}")
        (self.root / "pt_a-eager_rank0.2.pt.trace.json.gz").write_text("{}")
        self.assertEqual([Path(r["path"]).name for r in probe.traces_written(str(graph), 0)],
                         ["pt_a_rank0.1.pt.trace.json.gz"])
        self.assertEqual([Path(r["path"]).name for r in probe.traces_written(str(eager), 0)],
                         ["pt_a-eager_rank0.2.pt.trace.json.gz"])

    @unittest.skipUnless((ROOT / "scripts/probe_server.py").exists(), "main uses its existing serving adapter")
    def test_vllm_server_info_from_engine_log(self):
        import probe_server
        log = self.root / "server.log"
        log.write_text("INFO [scheduler.py:346] Chunked prefill is enabled with max_num_batched_tokens=32768.\n"
                       "speculative_config=SpeculativeConfig(method='dspark'), compilation_config={"
                       "'cudagraph_capture_sizes': [1, 2, 4, 8, 2048], "
                       "'decode_cudagraph_capture_request_sizes': [1, 2, 3, 16], 'x': 1}\n")
        info = probe_server.vllm_server_info(str(log))
        self.assertEqual(info["server_args"], {"chunked_prefill_size": 32768, "cuda_graph_bs": [1, 2, 3, 16]})
        # without the V2 request sizes, spec-decode token capture sizes are not batch sizes
        log.write_text(log.read_text().replace("'decode_cudagraph_capture_request_sizes': [1, 2, 3, 16], ", ""))
        info = probe_server.vllm_server_info(str(log))
        self.assertNotIn("cuda_graph_bs", info["server_args"])
        self.assertTrue(info["notes"])
        log.write_text("nothing stated\n")
        self.assertIsNone(probe_server.vllm_server_info(str(log)))

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


    @unittest.skipUnless((ROOT / "scripts/point_census.py").exists(), "main uses parse_profile")
    def test_copies_are_billed_only_where_no_kernel_runs(self):
        import point_census
        copy = "Memcpy HtoD (Host -> Device)"
        # two overlapping copy rows under a kernel, one exposed copy, a memset half exposed
        dev = [(0, 10, "gemm"), (2, 3, copy), (3, 3, copy), (12, 2, copy),
               (15, 5, "norm"), (18, 4, "Memset (Device)")]
        got, n, coll = point_census.kernels_in(dev, [d[0] for d in dev], 0, 100)
        self.assertEqual(got, {"gemm": 10, "norm": 5, copy: 2, "Memset (Device)": 2})
        self.assertEqual((n, coll), (6, {}))


    def test_profiler_traces_threads_started_after_the_window_opens(self):
        # --enable-dbo runs each prefill micro-batch in a new thread: no host ops without this
        started = []
        class TorchProfilerWrapper:
            def __init__(self):
                self._profiler_kwargs = {"record_shapes": True}
            def _start(self):
                started.append(dict(self._profiler_kwargs))
        module = types.SimpleNamespace(TorchProfilerWrapper=TorchProfilerWrapper)
        prof = types.ModuleType("torch._C._profiler")
        prof._ExperimentalConfig = lambda **kw: kw
        with patch.dict(sys.modules, {"torch": types.ModuleType("torch"),
                                      "torch._C": types.ModuleType("torch._C"),
                                      "torch._C._profiler": prof}):
            observer.patch_profiler(module)
            observer.patch_profiler(module)
        TorchProfilerWrapper()._start()
        self.assertEqual(started[0]["experimental_config"], {"profile_all_threads": True})
        with patch.dict(os.environ, {"GEAK_PROFILE_ALL_THREADS": "1"}, clear=True):
            self.assertEqual(set(observer.wanted()), {observer.PROFILER_MODULE})

    def test_frameless_prefill_kernels_request_a_serial_window(self):
        import run_points
        shapes = self.root / "shapes"
        shapes.mkdir()
        bare = {"resolved_by": "external_id", "host_op": "akr::moe", "py_frames": [],
                "graph_replayed": False}
        framed = dict(bare, py_frames=["vllm/models/x.py(1): forward"])
        kern = lambda *samples: {"variants": [{"samples": list(samples)}]}
        (shapes / "pf.json").write_text(json.dumps({"kernels": {
            "moe_prefill": kern(bare), "norm": kern(framed), "unwanted": kern(bare)}}))
        (shapes / "de.json").write_text(json.dumps({"kernels": {"graphed": kern(
            dict(bare, graph_replayed=True))}}))
        phase = {"pf": "PREFILL", "de": "DECODE"}
        wanted = {"moe_prefill", "norm", "graphed"}
        self.assertEqual(run_points.frameless_prefill_kernels(str(shapes), phase, wanted),
                         ["moe_prefill"])
        (shapes / "pf2@serial.json").write_text(json.dumps({"kernels": {"moe_prefill": kern(framed)}}))
        self.assertEqual(run_points.frameless_prefill_kernels(str(shapes), phase, wanted), [])
        cmd = ["vllm", "serve", "m", "--enable-dbo", "--dbo-prefill-token-threshold", "16384"]
        self.assertEqual(run_points.flag_value(cmd, "--dbo-prefill-token-threshold"), "16384")
        self.assertEqual(run_points.flag_value(["--x=3"], "--x"), "3")


    def test_hip_runtime_kernels_have_no_source_to_select(self):
        # DBO prefill on vLLM 0.33: __amd_rocclr_streamOpsWait.kd was 4.6% and was selected
        import build_ranking
        for name in ("__amd_rocclr_streamOpsWait.kd", "__amd_rocclr_copyBuffer.kd",
                     "Memcpy HtoD (Host -> Device)"):
            self.assertEqual(build_ranking.kind(name), "memory")
        self.assertEqual(build_ranking.kind("kernel_gemm1_mxfp4_h6144_i768.kd"), "compute")


if __name__ == "__main__":
    unittest.main()
