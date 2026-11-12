"""CPU regressions for adopt-ci-artifacts on vLLM points.

A vLLM CI server log has no per-step geometry and no decode_log_interval, so every vLLM
point used to be rejected and the stage blocked with "no usable point". The point must be
adopted with an explicit observation requirement and the launch facts the local
observation needs (the CI's synthetic acceptance, spec k, MNBT, capture sizes).
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SKILL = Path(__file__).resolve().parents[1] / "adopt-ci-artifacts"
WORKLOAD_ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("adopt_ci_under_test", SKILL / "adopt_ci.py")
adopt_ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adopt_ci)

KEY = "kimik3.1_tp8_conc48_kvdram-vllm-simple_spec-mtp_fp4_vllm_tp8-x-8c895f7f438950293be3"
COMPILATION = ('{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":'
               '10240,"cudagraph_capture_sizes":[4,8,12,16,10240]}')
SPEC = ('{"method": "dspark", "model": "Inferact/Kimi-K3.1-Pro-DSpark-preview", '
        '"num_speculative_tokens": 3, "attention_backend": "ROCM_AITER_MLA", '
        '"rejection_sample_method": "%s", "synthetic_acceptance_length": 2.88}')


def recipe(rejection="synthetic", spec_config=True):
    args = ["      served-model-name: moonshotai/Kimi-K3.1-Pro",
            "      tensor-parallel-size: 8",
            "      decode-context-parallel-size: 8",
            "      max-num-batched-tokens: 10240",
            "      max-num-seqs: 48",
            "      enable-prefix-caching: true",
            "      compilation-config: '%s'" % COMPILATION]
    if spec_config:
        args.append("      speculative-config: '%s'" % (SPEC % rejection))
    return "\n".join([
        "schema: 2",
        "model:",
        "  path: hf:moonshotai/Kimi-K3.1-Pro",
        "  container: vllm/vllm-openai-rocm:kimi-k31@sha256:5f47",
        "roles:",
        "  agg:",
        "    args:",
        *args,
        "    env:",
        "      VLLM_ROCM_USE_AITER: '1'",
        "      VLLM_USE_BREAKABLE_CUDAGRAPH: '1'",
        ""])


def bmk(framework="vllm"):
    return {"framework": framework, "infmax_model_prefix": "kimik3.1", "tp": 8, "ep": 1,
            "conc": 48, "num_gpus": 8, "kv_offloading": "dram",
            "kv_offload_backend": {"name": "vllm-simple"}, "spec_decoding": "mtp",
            "model": "moonshotai/Kimi-K3.1-Pro", "image": "vllm/vllm-openai-rocm:kimi-k31@sha256:5f47",
            "hw": "mi355x", "power_valid": 1,
            "power_audit": {"window_start_unix": 1794056932.349056,
                            "window_end_unix": 1794060562.351246},
            "server_metrics": {"tokens": {"generation_total": 3816261}},
            "request_metrics": {"throughput": {"output": {"tokens_per_second": 1041.21146}}}}


def add(tf, name, text):
    data = text.encode()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


class AdoptCiVllmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.out.mkdir()

    def stage(self, rec, framework="vllm", server_log=None):
        arts = self.root / "artifacts"
        bdir = arts / ("bmk_agentic_" + KEY)
        bdir.mkdir(parents=True)
        (bdir / (KEY + ".json")).write_text(json.dumps(bmk(framework)))
        sdir = arts / ("server_logs_" + KEY) / "x"
        sdir.mkdir(parents=True)
        with tarfile.open(sdir / "srt-single-node-logs.tar.gz", "w:gz") as tf:
            add(tf, "./logs/node_agg_w0.out", server_log or
                "(APIServer pid=1) INFO 11-07 12:18:01 [api_server.py:1] vLLM API server\n"
                "(APIServer pid=1) INFO Engine 000: Avg prompt throughput: 12 tokens/s\n")
            add(tf, "./logs/config.yaml", rec)
            add(tf, "./logs/agentic/benchmark.log", "12:18:00.000 NOTICE Phase profiling (profiling) started\n")
        params = self.root / "params.json"
        params.write_text(json.dumps({
            "ci": {"run_id": None, "artifacts_dirs": [str(arts)]},
            "select": {"model_prefix": "kimik3.1", "tp": [8], "conc": [48]},
            "allow_partial": False, "require_power_valid": True}))
        env = {"OUT": str(self.out), "PARAMS": str(params), "STAGE": "baseline",
               "STAGE_SKILL": "adopt-ci-artifacts", "WORKLOAD_ROOT": str(WORKLOAD_ROOT)}
        with patch.dict(os.environ, env), patch("sys.stdout", new_callable=io.StringIO):
            rc = adopt_ci.main()
        return rc, json.loads((self.out / "result.json").read_text())

    def pid_dir(self):
        return self.out / "points" / "tp8-ep1-c48-kvdram-vllm-simple-specmtp"

    def test_vllm_point_is_adopted_with_observation_required(self):
        rc, res = self.stage(recipe())
        self.assertEqual((rc, res["status"]), (0, "ok"), res.get("notes"))
        inputs = json.loads((self.pid_dir() / "observed_inputs.json").read_text())
        self.assertEqual(inputs["log_format"], "vllm-iterations")
        self.assertIs(inputs["observation_required"], True)
        self.assertNotIn("decode_log_interval", inputs)
        self.assertEqual(inputs["completed_output_tokens"], 3816261)
        self.assertEqual(inputs["window_source"], "bmk power_audit window")
        self.assertTrue(inputs["log"][0].endswith("node_agg_w0.out"))
        # the launch facts, read from the CI recipe
        self.assertEqual(inputs["accept_len"], 2.88)
        self.assertIn("synthetic", inputs["accept_len_source"])
        self.assertEqual(inputs["num_speculative_tokens"], 3)
        self.assertEqual(inputs["max_num_batched_tokens"], 10240)
        self.assertEqual(inputs["max_num_seqs"], 48)
        self.assertEqual(inputs["cudagraph_capture_sizes"], [4, 8, 12, 16, 10240])
        self.assertEqual(inputs["cudagraph_capture_sizes_unit"], "tokens")
        self.assertEqual(inputs["cudagraph_mode"], "FULL_AND_PIECEWISE")
        # no SGLang log parse was attempted, and none is passed off as an observation
        self.assertFalse((self.pid_dir() / "observed_workload.json").exists())
        base = json.loads((self.pid_dir() / "baseline.json").read_text())
        self.assertIsNone(base["provenance"]["decode_crosscheck"])
        self.assertEqual(base["throughput"], 1041.21146)
        launch = json.loads((self.pid_dir() / "launch.json").read_text())
        self.assertEqual(launch["args"]["max-num-batched-tokens"], 10240)
        top = json.loads((self.out / "baseline.json").read_text())
        self.assertEqual(top["points"][0]["observed_inputs"],
                         "points/tp8-ep1-c48-kvdram-vllm-simple-specmtp/observed_inputs.json")
        ident = json.loads((self.out / "stack_identity.json").read_text())
        self.assertEqual(ident["active_stack"]["model"], "moonshotai/Kimi-K3.1-Pro")

    def test_unknown_acceptance_is_not_guessed(self):
        rc, res = self.stage(recipe(rejection="block"))
        self.assertEqual(res["status"], "ok", res.get("notes"))
        inputs = json.loads((self.pid_dir() / "observed_inputs.json").read_text())
        self.assertIsNone(inputs["accept_len"])
        self.assertIn("measure it", inputs["accept_len_source"])
        self.assertEqual(inputs["num_speculative_tokens"], 3)

    def test_no_speculation_accepts_one_token_per_step(self):
        rc, res = self.stage(recipe(spec_config=False))
        self.assertEqual(res["status"], "ok", res.get("notes"))
        inputs = json.loads((self.pid_dir() / "observed_inputs.json").read_text())
        self.assertEqual((inputs["accept_len"], inputs["num_speculative_tokens"]), (1.0, 0))

    def test_sglang_point_still_requires_the_decode_interval(self):
        rc, res = self.stage(recipe(), framework="sglang")
        self.assertEqual(res["status"], "blocked")
        self.assertIn("decode_log_interval not in the server log", res["notes"])


if __name__ == "__main__":
    unittest.main()
