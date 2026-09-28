import os
import sys
import tempfile
import json
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import run_semantic_shape_capture as capture


class RunSemanticShapeCaptureTest(unittest.TestCase):
    def test_required_phases_are_inferred_from_capture_plan(self):
        plan = {
            "target_buckets": [
                {"phase": "decode"},
                {"phase": "prefill"},
                {"phase": "decode"},
            ]
        }
        self.assertEqual(
            capture._required_phases(plan), ["decode", "prefill"])

    def test_required_phase_contract_wins_when_a_table_is_still_missing(self):
        plan = {
            "phase_coverage": {"required_phases": ["prefill", "decode"]},
            "target_buckets": [{"phase": "prefill"}],
        }
        self.assertEqual(
            capture._required_phases(plan), ["decode", "prefill"])

class OwnProcessGroupTeardownTest(unittest.TestCase):
    """Regression tests for B6."""

    def test_b6_teardown_group_kills_recorded_pgid_never_pattern_kills(self):
        with mock.patch.object(capture, "_docker") as docker:
            capture._stop_service("container", "/out/server.pgid", 8935)
        command = docker.call_args[0][1]
        self.assertNotIn("pkill", command)
        self.assertNotIn("pgrep", command)
        self.assertIn("/out/server.pgid", command)
        # The leading '-' is what makes this a process-group kill.
        self.assertIn('kill -TERM -"$pgid"', command)
        self.assertIn('kill -KILL -"$pgid"', command)
        # Liveness is polled on the port: the server leaves unreaped
        # children behind, so `kill -0` on the group never clears.
        self.assertIn("/dev/tcp/127.0.0.1/8935", command)

    def test_b6_teardown_is_a_noop_when_no_pgid_was_recorded(self):
        with mock.patch.object(capture, "_docker") as docker:
            capture._stop_service("container", "/out/server.pgid", 8935)
        command = docker.call_args[0][1]
        self.assertIn('[ -s "$pgid_file" ] || exit 0', command)

    def test_b6_busy_port_is_reported_not_killed(self):
        completed = mock.Mock(stdout=b"BUSY\n")
        with mock.patch.object(capture.subprocess, "run",
                               return_value=completed):
            with self.assertRaises(RuntimeError) as caught:
                capture._assert_port_free("container", 8935)
        self.assertIn("8935", str(caught.exception))
        self.assertIn("did not start", str(caught.exception))

    def test_b6_free_port_passes(self):
        completed = mock.Mock(stdout=b"FREE\n")
        with mock.patch.object(capture.subprocess, "run",
                               return_value=completed):
                capture._assert_port_free("container", 8935)

    def test_runtime_patch_is_restored(self):
        with mock.patch.object(capture, "_docker") as docker:
            capture._restore("container", "/runtime/model_runner.py",
                             "/runtime/geak_capture.py")
        command = docker.call_args[0][1]
        self.assertIn("model_runner.py.geak_semantics_bak", command)
        self.assertIn("geak_capture.py.geak_semantics_bak", command)
        self.assertIn("rm -f", command)


class DecodeProbeTest(unittest.TestCase):
    """Graph-construction shape capture regression tests."""

    def test_b4_narrowed_phases_are_reported(self):
        notes = capture._warn_narrowed_phases(
            ["prefill"], {"target_buckets": [{"phase": "prefill"}]},
            "plan.target_buckets")
        self.assertTrue(notes)
        joined = " ".join(notes)
        self.assertIn("decode", joined)
        self.assertIn("graph construction", joined)

    def test_b4_full_phase_set_warns_about_nothing(self):
        self.assertEqual(
            capture._warn_narrowed_phases(["prefill", "decode"], {}, "cli"), [])

    def test_observed_phases_reads_the_shape_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "shape.jsonl")
            with open(log, "w") as fh:
                fh.write(json.dumps({"context": {"phase": "EXTEND"}}) + "\n")
                fh.write(json.dumps({"context": {"phase": "DECODE"}}) + "\n")
                fh.write("not json\n")
            # EXTEND is an alias of prefill; the audit must compare like for like
            self.assertEqual(capture._observed_phases(log),
                             {"prefill", "decode"})

    def test_shape_capture_exports_graph_trace_without_workload_profiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_path = os.path.join(tmp, "setup.json")
            plan_path = os.path.join(tmp, "plan.json")
            out_dir = os.path.join(tmp, "out")
            with open(setup_path, "w") as fh:
                json.dump({
                    "container": "c",
                    "model": "/model",
                    "benchmark_repository": "/repo",
                    "benchmark": "/repo/bench.sh",
                    "port": 30001,
                    "tensor_parallel_size": 8,
                    "gpu_ids": "0,1,2,3,4,5,6,7",
                    "mem_fraction": 0.8,
                    "bench_client": "inferencex",
                    "inferencex_path": "/repo",
                    "workload": {
                        "concurrency": 4,
                        "input_length": 8192,
                        "output_length": 1024,
                    },
                    "extra_server_args": "--disable-radix-cache",
                    "extra_env": "SGLANG_USE_AITER=1",
                }, fh)
            with open(plan_path, "w") as fh:
                json.dump({
                    "target_buckets": [{"phase": "decode"}],
                    "capture_targets": [{"representative_layer_id": 2}],
                }, fh)

            def fake_docker(container, command, stdout=None):
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "shape.jsonl"), "w") as fh:
                    fh.write(json.dumps({
                        "phase": "decode", "layer_id": 2}) + "\n")
                with open(os.path.join(
                        out_dir, "graph_capture-TP-0.trace.json"), "w") as fh:
                    json.dump({"traceEvents": []}, fh)
                with open(os.path.join(
                        out_dir, "operator_schema_manifest.json"), "w") as fh:
                    json.dump({"schemas": []}, fh)
            with mock.patch.object(capture, "_assert_port_free"), \
                    mock.patch.object(capture, "_deploy"), \
                    mock.patch.object(capture, "_restore") as restore, \
                    mock.patch.object(capture, "_stop_service"), \
                    mock.patch.object(capture, "_docker",
                                      side_effect=fake_docker) as docker:
                result = capture.capture(
                    setup_path, plan_path, out_dir, phases=["decode"])

            command = docker.call_args[0][1]
            self.assertIn("--enable-profile-cuda-graph", command)
            self.assertNotIn("--disable-cuda-graph", command)
            self.assertIn("GEAK_SEMANTICS_REQUIRE_PROFILER=0", command)
            self.assertIn("GEAK_SEMANTICS_LAYER_SCOPES=1", command)
            self.assertIn(
                "GEAK_SEMANTICS_OPERATOR_SCHEMA_MANIFEST=", command)
            self.assertIn("export PROFILE=0", command)
            self.assertIn("export GPU=0,1,2,3,4,5,6,7", command)
            self.assertIn(
                "export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", command)
            self.assertIn("export NUM_PROMPTS=20", command)
            self.assertIn("export MEM_FRACTION=0.8", command)
            self.assertIn("export OUT_DIR=", command)
            self.assertEqual(result["shape_capture_execution"], "graph_capture")
            self.assertTrue(result["all_main_layer_scopes"])
            self.assertEqual(
                result["capture_trace"],
                os.path.join(out_dir, "graph_capture-TP-0.trace.json"))
            self.assertEqual(
                result["operator_schema_manifest"],
                os.path.join(out_dir, "operator_schema_manifest.json"))
            self.assertTrue(os.path.exists(result["operator_probe_plan"]))
            self.assertNotIn("clean_traces_by_phase", result)
            self.assertNotIn("capture_traces_by_phase", result)
            restore.assert_called_once_with(
                "c", capture.DEFAULT_MODEL_RUNNER,
                capture.DEFAULT_RUNTIME_MODULE)

    def test_shape_capture_rejects_disable_cuda_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup = os.path.join(tmp, "setup.json")
            plan = os.path.join(tmp, "plan.json")
            with open(setup, "w") as fh:
                json.dump({
                    "container": "c", "model": "/model",
                    "benchmark": "/repo/bench.sh", "port": 30001,
                    "tensor_parallel_size": 1,
                    "workload": {"concurrency": 1, "input_length": 1,
                                 "output_length": 1},
                    "extra_server_args": "--disable-cuda-graph",
                }, fh)
            with open(plan, "w") as fh:
                json.dump({
                    "target_buckets": [{"phase": "decode"}],
                    "capture_targets": [{"representative_layer_id": 0}],
                }, fh)
            with mock.patch.object(capture, "_assert_port_free"), \
                    mock.patch.object(capture, "_deploy"):
                with self.assertRaisesRegex(ValueError, "requires CUDA/HIP"):
                    capture.capture(setup, plan, os.path.join(tmp, "out"))

    def test_local_capture_uses_no_docker_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup = os.path.join(tmp, "setup.json")
            plan = os.path.join(tmp, "plan.json")
            out_dir = os.path.join(tmp, "out")
            with open(setup, "w") as fh:
                json.dump({
                    "execution_mode": "local", "model": "/model",
                    "benchmark_repository": "/repo", "benchmark": "/repo/bench.sh",
                    "port": 0, "tensor_parallel_size": 1,
                    "workload": {"concurrency": 1, "input_length": 8,
                                 "output_length": 8},
                }, fh)
            with open(plan, "w") as fh:
                json.dump({"target_buckets": [{"phase": "decode"}],
                           "capture_targets": [{"representative_layer_id": 0}]}, fh)

            def fake_run(command, stdout=None):
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "shape.jsonl"), "w") as fh:
                    fh.write(json.dumps({"phase": "decode"}) + "\n")
                with open(os.path.join(out_dir, "graph_capture-TP-0.trace.json"), "w") as fh:
                    json.dump({"traceEvents": []}, fh)
                with open(os.path.join(
                        out_dir, "operator_schema_manifest.json"), "w") as fh:
                    json.dump({"schemas": []}, fh)
                return mock.Mock(returncode=0)

            with mock.patch.object(capture, "_free_local_port", return_value=32123), \
                    mock.patch.object(capture, "_local_port_free", return_value=True), \
                    mock.patch.object(capture, "_deploy_local") as deploy, \
                    mock.patch.object(capture, "_restore_local") as restore, \
                    mock.patch.object(capture, "_stop_service_local") as stop, \
                    mock.patch.object(capture, "_run", side_effect=fake_run), \
                    mock.patch.object(capture, "_docker") as docker:
                result = capture.capture(setup, plan, out_dir, phases=["decode"])
            docker.assert_not_called()
            deploy.assert_called_once()
            stop.assert_called_once_with(os.path.join(out_dir, "server.pgid"), 32123)
            restore.assert_called_once()
            self.assertEqual(result["execution_mode"], "local")
            self.assertEqual(result["port"], 32123)


if __name__ == "__main__":
    unittest.main()
