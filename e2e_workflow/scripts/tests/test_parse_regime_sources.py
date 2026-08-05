#!/usr/bin/env python3
"""Unit tests for parse_regime.py's SOURCE-reading helpers (stdlib only; no pytest needed).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_parse_regime_sources.py

test_workload_alignment.py already covers parse_regime() driven by a flag STRING. This file covers
the other two inputs a regime can be built from -- the server LAUNCH SCRIPT and the model's own
config.json -- plus the backend-resolution ladder that reads them:

  - _read_script_flags / _read_script_text : launch-script -> flags / raw text, incl. unreadable files
  - _detect_backend                        : every rung of the vllm/sglang/atom resolution ladder
  - _prefill_chunk                         : chunked-prefill budget, incl. sglang's -1 = disabled
  - _load_model_quant                      : pre-quantized checkpoint config -> quant descriptor

These matter because a mis-read launch script silently produces the wrong regime, which is the #1
cause of an "isolated win, e2e loss" (see the parse_regime module docstring).
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


pr = _load("parse_regime", "parse_regime.py")


class _TmpFileMixin:
    """Writes temp files and cleans them up, so each test states only what it cares about."""

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _write(self, text, suffix=".sh", binary=False):
        fd, path = tempfile.mkstemp(suffix=suffix)
        mode = "wb" if binary else "w"
        with os.fdopen(fd, mode) as fh:
            fh.write(text)
        self._paths.append(path)
        return path


# --------------------------------------------------------------------------- #
# _read_script_flags / _read_script_text
# --------------------------------------------------------------------------- #
class TestReadScript(_TmpFileMixin, unittest.TestCase):
    def test_flags_absent_path_is_empty(self):
        self.assertEqual(pr._read_script_flags(""), {})
        self.assertEqual(pr._read_script_flags(None), {})
        self.assertEqual(pr._read_script_flags("/nonexistent/launch.sh"), {})

    def test_flags_directory_is_not_a_file(self):
        self.assertEqual(pr._read_script_flags(tempfile.gettempdir()), {})

    def test_flags_parsed_from_script(self):
        p = self._write("vllm serve /models/x --quantization fp8 --kv-cache-dtype=fp8\n")
        flags = pr._read_script_flags(p)
        self.assertEqual(flags.get("quantization"), "fp8")
        self.assertEqual(flags.get("kv-cache-dtype"), "fp8")

    def test_flags_join_shell_line_continuations(self):
        # A flag and its value split across a `\`+newline must still pair up.
        p = self._write("vllm serve /models/x \\\n  --max-num-batched-tokens \\\n  8192\n")
        self.assertEqual(pr._read_script_flags(p).get("max-num-batched-tokens"), "8192")

    def test_flags_bare_boolean(self):
        p = self._write("vllm serve /models/x --enforce-eager\n")
        self.assertIs(pr._read_script_flags(p).get("enforce-eager"), True)

    def test_flags_unreadable_file_is_empty(self):
        # Undecodable bytes make open().read() raise; the helper must swallow it, not crash the run.
        p = self._write(b"\xff\xfe\x00 --quantization fp8", binary=True)
        self.assertEqual(pr._read_script_flags(p), {})

    def test_text_absent_path_is_empty_string(self):
        self.assertEqual(pr._read_script_text(""), "")
        self.assertEqual(pr._read_script_text("/nonexistent/launch.sh"), "")

    def test_text_is_lowercased(self):
        p = self._write("VLLM Serve /Models/X\n")
        self.assertIn("vllm serve", pr._read_script_text(p))

    def test_text_unreadable_file_is_empty_string(self):
        p = self._write(b"\xff\xfe\x00", binary=True)
        self.assertEqual(pr._read_script_text(p), "")


# --------------------------------------------------------------------------- #
# _detect_backend -- every rung of the ladder
# --------------------------------------------------------------------------- #
class TestDetectBackend(_TmpFileMixin, unittest.TestCase):
    def test_explicit_backend_wins(self):
        for name in ("vllm", "sglang", "atom"):
            self.assertEqual(pr._detect_backend(name, "", "", {}), name)
            self.assertEqual(pr._detect_backend(f"  {name.upper()} ", "", "", {}), name)

    def test_explicit_unknown_backend_falls_through(self):
        self.assertEqual(pr._detect_backend("tensorrt", "", "", {}), "")

    def test_serve_command_in_script_content(self):
        p = self._write("#!/bin/bash\nvllm serve /models/x\n")
        self.assertEqual(pr._detect_backend("", p, "", {}), "vllm")

    def test_framework_tag_in_script_content(self):
        p = self._write("framework: sglang\n")
        self.assertEqual(pr._detect_backend("", p, "", {}), "sglang")

    def test_atom_entrypoint_in_args(self):
        self.assertEqual(pr._detect_backend("", "", "python -m atom.entrypoints.api", {}), "atom")

    def test_script_name_hints(self):
        self.assertEqual(pr._detect_backend("", "/recipes/vllm_mi300x.sh", "", {}), "vllm")
        self.assertEqual(pr._detect_backend("", "/recipes/sglang-run.sh", "", {}), "sglang")
        self.assertEqual(pr._detect_backend("", "/opt/atom_launch.sh", "", {}), "atom")

    def test_backend_specific_flags_are_last_resort(self):
        self.assertEqual(pr._detect_backend("", "", "", {"gpu-memory-utilization": "0.9"}), "vllm")
        self.assertEqual(pr._detect_backend("", "", "", {"served_model_name": "x"}), "vllm")
        self.assertEqual(pr._detect_backend("", "", "", {"mem-fraction-static": "0.9"}), "sglang")
        self.assertEqual(pr._detect_backend("", "", "", {"disable_radix_cache": True}), "sglang")

    def test_unresolved_backend_is_empty(self):
        self.assertEqual(pr._detect_backend("", "", "--port 8000", {"port": "8000"}), "")


# --------------------------------------------------------------------------- #
# _prefill_chunk
# --------------------------------------------------------------------------- #
class TestPrefillChunk(unittest.TestCase):
    def test_absent_is_none(self):
        self.assertIsNone(pr._prefill_chunk({}))
        self.assertIsNone(pr._prefill_chunk({"port": "8000"}))

    def test_sglang_chunked_prefill_size(self):
        self.assertEqual(pr._prefill_chunk({"chunked-prefill-size": "2048"}), 2048)
        self.assertEqual(pr._prefill_chunk({"chunked_prefill_size": 4096}), 4096)

    def test_sglang_disabled_sentinel_is_none(self):
        # sglang uses -1 to mean "no chunking"; the caller must see None, not -1.
        self.assertIsNone(pr._prefill_chunk({"chunked-prefill-size": "-1"}))
        self.assertIsNone(pr._prefill_chunk({"chunked-prefill-size": 0}))

    def test_vllm_max_num_batched_tokens(self):
        self.assertEqual(pr._prefill_chunk({"max-num-batched-tokens": "8192"}), 8192)
        self.assertEqual(pr._prefill_chunk({"max_num_batched_tokens": 512}), 512)

    def test_bare_boolean_flag_is_skipped(self):
        # `--chunked-prefill-size` with no value tokenizes to True; bool is not a budget.
        self.assertIsNone(pr._prefill_chunk({"chunked-prefill-size": True}))

    def test_non_numeric_value_is_skipped(self):
        self.assertIsNone(pr._prefill_chunk({"chunked-prefill-size": "auto"}))


# --------------------------------------------------------------------------- #
# _load_model_quant
# --------------------------------------------------------------------------- #
class TestLoadModelQuant(_TmpFileMixin, unittest.TestCase):
    def _cfg(self, obj):
        return self._write(json.dumps(obj), suffix=".json")

    def test_absent_path_is_none(self):
        self.assertIsNone(pr._load_model_quant(""))
        self.assertIsNone(pr._load_model_quant("/nonexistent/config.json"))

    def test_unparseable_json_is_none(self):
        self.assertIsNone(pr._load_model_quant(self._write("{not json", suffix=".json")))

    def test_unquantized_checkpoint_is_none(self):
        self.assertIsNone(pr._load_model_quant(self._cfg({"model_type": "llama"})))

    def test_fp8_blockscale(self):
        got = pr._load_model_quant(self._cfg(
            {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}))
        self.assertEqual(got["method"], "fp8")
        self.assertEqual(got["weight_dtype"], "fp8_e4m3")
        self.assertEqual(got["block_size"], [128, 128])

    def test_awq_maps_to_int4(self):
        got = pr._load_model_quant(self._cfg({"quantization_config": {"quant_method": "awq"}}))
        self.assertEqual(got["weight_dtype"], "int4")

    def test_compression_config_alias(self):
        got = pr._load_model_quant(self._cfg({"compression_config": {"format": "int4"}}))
        self.assertEqual(got["method"], "int4")
        self.assertEqual(got["weight_dtype"], "int4")

    def test_activation_scheme_drives_fp8_dtype(self):
        got = pr._load_model_quant(self._cfg(
            {"quantization_config": {"quant_method": "compressed-tensors",
                                     "activation_scheme": "fp8_dynamic"}}))
        self.assertEqual(got["weight_dtype"], "fp8_e4m3")


# --------------------------------------------------------------------------- #
# parse_regime() reading from a script + config (the composed path)
# --------------------------------------------------------------------------- #
class TestParseRegimeFromSources(_TmpFileMixin, unittest.TestCase):
    def test_script_supplies_flags_that_server_args_omits(self):
        # The launch script often carries the chunked-prefill budget the live args do not.
        script = self._write("vllm serve /models/x --max-num-batched-tokens 8192\n")
        got = pr.parse_regime("", server_script=script)
        self.assertEqual(got["prefill_chunk"], 8192)
        self.assertEqual(got["backend"], "vllm")

    def test_server_args_override_script_on_overlap(self):
        script = self._write("vllm serve /models/x --quantization fp8\n")
        got = pr.parse_regime("--quantization awq", server_script=script)
        self.assertEqual(got["quant"]["method"], "awq")
        self.assertEqual(got["quant"]["source"], "flag")

    def test_model_config_used_when_no_quant_flag(self):
        cfg = self._write(json.dumps(
            {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}),
            suffix=".json")
        got = pr.parse_regime("", model_config_path=cfg)
        self.assertEqual(got["quant"]["source"], "model_config")
        self.assertEqual(got["quant"]["method"], "fp8_blockscale")
        self.assertEqual(got["quant"]["act_dtype"], "fp8")

    def test_vllm_compiles_by_default_without_enforce_eager(self):
        got = pr.parse_regime("", backend="vllm")
        self.assertEqual(got["compile"], "torch_compile")
        self.assertFalse(got["enforce_eager"])
        self.assertIn("compiles the backbone by default", got["notes"])

    def test_enforce_eager_makes_eager_the_faithful_baseline(self):
        got = pr.parse_regime("--enforce-eager", backend="vllm")
        self.assertTrue(got["enforce_eager"])
        self.assertEqual(got["compile"], "eager")
        self.assertFalse(got["cuda_graph"])

    def test_bare_attention_backend_flag_is_empty_string(self):
        # `--attention-backend` with no value tokenizes to True; the descriptor must carry '' not True.
        self.assertEqual(pr.parse_regime("--attention-backend")["attention_backend"], "")

    def test_unresolved_backend_is_noted(self):
        self.assertIn("backend UNRESOLVED", pr.parse_regime("--port 8000")["notes"])


# --------------------------------------------------------------------------- #
# main -- the CLI the harness actually shells out to
# --------------------------------------------------------------------------- #
class TestMain(_TmpFileMixin, unittest.TestCase):
    """The regime descriptor is consumed as JSON on stdout / on disk, so the CLI contract
    (valid JSON, --out written, flags forwarded to parse_regime) is what callers depend on."""

    def setUp(self):
        super().setUp()
        self._argv = sys.argv

    def tearDown(self):
        super().tearDown()
        sys.argv = self._argv

    def _run(self, *argv):
        sys.argv = ["parse_regime.py", *argv]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            pr.main()
        return out.getvalue(), err.getvalue()

    def test_prints_the_regime_as_json(self):
        out, _ = self._run("--server-args=--enforce-eager", "--backend", "vllm")
        regime = json.loads(out)
        self.assertEqual(regime["backend"], "vllm")
        self.assertTrue(regime["enforce_eager"])

    def test_server_args_must_use_the_equals_form(self):
        # The flag string always starts with "-", which argparse reads as the next option
        # rather than as this option's value -- so `--server-args "--enforce-eager"` fails
        # even though the shell passes it as a single argument. Callers must write
        # `--server-args=--enforce-eager`. Pinned because the failure is a hard exit.
        with self.assertRaises(SystemExit):
            self._run("--server-args", "--enforce-eager")

    def test_defaults_produce_a_complete_descriptor(self):
        # Every consumer reads these keys unconditionally; a missing one is an AttributeError
        # three steps later, in the harness rather than here.
        regime = json.loads(self._run()[0])
        self.assertEqual(
            set(regime),
            {"backend", "quant", "kv_cache_dtype", "compile", "enforce_eager",
             "cuda_graph", "attention_backend", "prefill_chunk", "notes"},
        )

    def test_out_file_matches_stdout(self):
        path = self._write("", suffix=".json")
        out, err = self._run("--server-args=--backend sglang", "--out", path)
        with open(path) as fh:
            self.assertEqual(json.loads(fh.read()), json.loads(out))
        self.assertIn(path, err)

    def test_server_script_is_read(self):
        script = self._write("python -m vllm.entrypoints.openai.api_server --enforce-eager\n")
        regime = json.loads(self._run("--server-script", script)[0])
        self.assertEqual(regime["backend"], "vllm")
        self.assertTrue(regime["enforce_eager"])

    def test_model_config_supplies_the_quant(self):
        cfg = self._write(json.dumps({"quantization_config": {"quant_method": "fp8"}}), suffix=".json")
        regime = json.loads(self._run("--model-config", cfg)[0])
        self.assertIn("fp8", str(regime["quant"]).lower())

    def test_explicit_backend_flag_wins_over_detection(self):
        regime = json.loads(self._run("--server-args=--port 8000", "--backend", "atom")[0])
        self.assertEqual(regime["backend"], "atom")


if __name__ == "__main__":
    unittest.main(verbosity=2)
