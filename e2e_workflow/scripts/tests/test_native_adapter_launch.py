#!/usr/bin/env python3
"""GPU-free tests for what the native vLLM/SGLang adapters hand the server, the GPU pin shared with
the ATOM adapter, and the AgentX client's aiperf entry point.

  * EXTRA_SERVER_ARGS is split by shell quoting rules: JSON- and list-valued flags reach the server
    as one unquoted argument, and args that do not split refuse the launch.
  * The server's ROCr sees only its own cards: ROCR_VISIBLE_DEVICES is narrowed to GPU and HIP/CUDA
    renumber them, selecting the same physical cards the HIP/CUDA-only pin selected.
  * An aiperf console script whose interpreter is missing runs with the interpreter beside it.

Run: python3 -m pytest e2e_workflow/scripts/tests/test_native_adapter_launch.py
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
ADAPTERS = SCRIPTS_DIR / "adapters"
NATIVE_LAUNCH = ADAPTERS / "shared" / "native_launch.sh"
AGENTX_CLIENT = ADAPTERS / "clients" / "agentx.sh"
BASH = shutil.which("bash")

GPU_VARS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
SPEC = {"method": "mtp", "num_speculative_tokens": 2}
LAYERS = ["lm_head", "*.mlp.gate", "model.layers.[0-9].mlp.*"]


def _base_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in GPU_VARS}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@unittest.skipIf(BASH is None, "bash is required")
class NativeGpuEnvTest(unittest.TestCase):
    def _gpu_env(self, gpu, extra_env="", outer=None):
        driver = (f'source "{NATIVE_LAUNCH}"\n'
                  '_native_gpu_env "$GPU_IN" "$EXTRA_ENV_IN"\n'
                  'printf "%s\\n" "${_NATIVE_GPU_ENV[@]}"\n')
        env = _base_env(GPU_IN=gpu, EXTRA_ENV_IN=extra_env)
        if outer is not None:
            env["ROCR_VISIBLE_DEVICES"] = outer
        proc = subprocess.run([BASH, "-c", driver], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return dict(line.split("=", 1) for line in proc.stdout.splitlines())

    def test_physical_ids_narrow_rocr_and_renumber_hip(self):
        self.assertEqual(self._gpu_env("4,5,6,7"), {"ROCR_VISIBLE_DEVICES": "4,5,6,7",
                                                    "HIP_VISIBLE_DEVICES": "0,1,2,3",
                                                    "CUDA_VISIBLE_DEVICES": "0,1,2,3"})

    def test_positions_in_an_outer_mask_select_the_same_cards(self):
        self.assertEqual(self._gpu_env("1,2", outer="2,3,4,5,6,7"),
                         {"ROCR_VISIBLE_DEVICES": "3,4", "HIP_VISIBLE_DEVICES": "0,1",
                          "CUDA_VISIBLE_DEVICES": "0,1"})

    def test_a_mask_extra_env_sets_wins_over_the_shell_mask(self):
        env = self._gpu_env("0,1", extra_env="FOO=1 ROCR_VISIBLE_DEVICES=4,5,6,7", outer="0,1,2,3,4,5,6,7")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "4,5")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1")

    def test_uuid_mask_entries_are_selected_by_position(self):
        env = self._gpu_env("2", outer="GPU-aa, GPU-bb, GPU-cc")
        self.assertEqual(env, {"ROCR_VISIBLE_DEVICES": "GPU-cc", "HIP_VISIBLE_DEVICES": "0",
                               "CUDA_VISIBLE_DEVICES": "0"})

    def test_whitespace_in_gpu_is_ignored(self):
        self.assertEqual(self._gpu_env("0, 1")["ROCR_VISIBLE_DEVICES"], "0,1")

    def test_a_gpu_that_cannot_be_mapped_keeps_the_hip_only_pin(self):
        for gpu, outer in (("0,4", "0,1,2,3"), ("GPU-aa", None), ("1,1", None), ("", None)):
            with self.subTest(gpu=gpu, outer=outer):
                self.assertEqual(self._gpu_env(gpu, outer=outer),
                                 {"HIP_VISIBLE_DEVICES": gpu, "CUDA_VISIBLE_DEVICES": gpu})


@unittest.skipIf(BASH is None, "bash is required")
class NativeLaunchTest(unittest.TestCase):
    """Source the real adapter and run adapter_launch against a fake server binary."""

    SERVER = {"vllm": "vllm", "sglang": "python"}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="native_launch_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.argv_out = self.tmp / "argv"
        self.env_out = self.tmp / "env"
        fake = ('#!/usr/bin/env bash\n'
                'printf "%s\\0" "$@" > "$ARGV_OUT"\n'
                'for v in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do\n'
                '  printf "%s=%s\\n" "$v" "${!v-<unset>}"\n'
                'done > "$ENV_OUT"\n')
        for name in set(self.SERVER.values()):
            _write(self.bin / name, fake)
        # Files a glob inside a flag value would match if the shell expanded it.
        (self.tmp / "x.mlp.gate").write_text("")
        (self.tmp / "model.layers.3.mlp.up").write_text("")

    def _launch(self, backend, args, gpu="2,3", extra_env=""):
        driver = (f'set -uo pipefail\nsource "{ADAPTERS / (backend + ".sh")}"\n'
                  'adapter_launch || exit $?\nwait "$SERVER_PID"\n')
        env = _base_env(PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
                        MODEL="/models/x", HOST="127.0.0.1", PORT="8000", TP="2", GPU=gpu,
                        MEM_FRACTION="0.9", GPU_ARCHS="gfx950", SERVER_LAUNCH_PREFIX="",
                        EXTRA_ENV=extra_env, EXTRA_SERVER_ARGS=args, OVERLAY_PYTHONPATH="",
                        PROFILE_DIR="", LOG=self.tmp / "server.log", ARGV_OUT=self.argv_out,
                        ENV_OUT=self.env_out)
        return subprocess.run([BASH, "-c", driver], env=env, cwd=self.tmp,
                              capture_output=True, text=True, timeout=60)

    def _argv(self):
        return self.argv_out.read_bytes().decode().split("\0")[:-1]

    def _server_env(self):
        return dict(line.split("=", 1) for line in self.env_out.read_text().splitlines())

    def test_structured_flags_reach_the_server_as_one_unquoted_argument(self):
        args = ("--speculative-config '" + json.dumps(SPEC, separators=(",", ":")) + "'"
                " --hf-overrides {\"a\": [1, 2]}"
                " --layers '" + json.dumps(LAYERS, separators=(",", ":")) + "'"
                " --max-num-seqs 32")
        for backend in self.SERVER:
            with self.subTest(backend=backend):
                proc = self._launch(backend, args)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                argv = self._argv()
                self.assertEqual(json.loads(argv[argv.index("--speculative-config") + 1]), SPEC)
                self.assertEqual(argv[argv.index("--hf-overrides") + 1], '{"a": [1, 2]}')
                self.assertEqual(json.loads(argv[argv.index("--layers") + 1]), LAYERS)
                self.assertEqual(argv[-2:], ["--max-num-seqs", "32"])

    def test_plain_and_empty_args_are_unchanged(self):
        for backend in self.SERVER:
            with self.subTest(backend=backend):
                self.assertEqual(self._launch(backend, "--max-num-seqs 32 --enforce-eager").returncode, 0)
                self.assertEqual(self._argv()[-3:], ["--max-num-seqs", "32", "--enforce-eager"])
                self.assertEqual(self._launch(backend, "").returncode, 0)
                self.assertNotIn("--max-num-seqs", self._argv())

    def test_args_that_do_not_split_refuse_the_launch(self):
        for backend in self.SERVER:
            with self.subTest(backend=backend):
                self.argv_out.unlink(missing_ok=True)
                proc = self._launch(backend, "--name 'two words")
                self.assertEqual(proc.returncode, 2)
                self.assertIn("refusing to launch", proc.stderr)
                self.assertFalse(self.argv_out.exists(), "the server must not start from unsplit args")

    def test_the_server_sees_only_its_own_cards(self):
        for backend in self.SERVER:
            with self.subTest(backend=backend):
                self.assertEqual(self._launch(backend, "").returncode, 0)
                self.assertEqual(self._server_env(), {"ROCR_VISIBLE_DEVICES": "2,3",
                                                      "HIP_VISIBLE_DEVICES": "0,1",
                                                      "CUDA_VISIBLE_DEVICES": "0,1"})
                proc = self._launch(backend, "", gpu="0,1", extra_env="ROCR_VISIBLE_DEVICES=4,5,6,7")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(self._server_env()["ROCR_VISIBLE_DEVICES"], "4,5")


@unittest.skipIf(BASH is None, "bash is required")
class AgentxAiperfEntryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agentx_entry_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.env_bin = self.tmp / "client" / "bin"
        self.env_bin.mkdir(parents=True)
        self.gone = self.tmp / "build" / "bin" / "python3"

    def _cmd(self, aiperf):
        driver = (f'source "{AGENTX_CLIENT}"\n'
                  '_agentx_aiperf_cmd "$AIPERF_IN"\n'
                  'printf "%s\\n" "${_AGENTX_AIPERF_CMD[@]}"\n')
        proc = subprocess.run([BASH, "-c", driver], env=_base_env(AIPERF_IN=aiperf),
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.splitlines(), proc.stderr

    def test_a_relocated_console_script_runs_with_the_interpreter_beside_it(self):
        _write(self.env_bin / "python3", "#!/usr/bin/env bash\n")
        launchers = {
            "shebang": f"#!{self.gone}\nimport sys\n",
            "sh_quoted": f"#!/bin/sh\n'''exec' \"{self.gone}\" \"$0\" \"$@\"\n' '''\nimport sys\n",
            "sh_bare": f"#!/bin/sh\n'''exec' {self.gone} \"$0\" \"$@\"\n' '''\nimport sys\n",
        }
        for form, body in launchers.items():
            with self.subTest(form=form):
                _write(self.env_bin / "aiperf", body)
                cmd, err = self._cmd(str(self.env_bin / "aiperf"))
                self.assertEqual(cmd, [str(self.env_bin / "python3"), str(self.env_bin / "aiperf")])
                self.assertIn("missing interpreter", err)

    def test_a_working_entry_point_runs_as_itself(self):
        _write(self.env_bin / "python3", "#!/usr/bin/env bash\n")
        for body in ("#!/usr/bin/env python3\nimport sys\n", f"#!{self.env_bin / 'python3'}\nimport sys\n",
                     "#!/bin/sh\n'''exec' \"$(dirname -- \"$0\")/python3\" \"$0\" \"$@\"\n' '''\n"):
            with self.subTest(body=body.splitlines()[0]):
                _write(self.env_bin / "aiperf", body)
                cmd, _ = self._cmd(str(self.env_bin / "aiperf"))
                self.assertEqual(cmd, [str(self.env_bin / "aiperf")])

    def test_no_interpreter_beside_the_script_leaves_the_command_alone(self):
        _write(self.env_bin / "aiperf", f"#!{self.gone}\nimport sys\n")
        cmd, _ = self._cmd(str(self.env_bin / "aiperf"))
        self.assertEqual(cmd, [str(self.env_bin / "aiperf")])

    def test_a_name_not_on_path_is_left_to_the_not_found_check(self):
        cmd, _ = self._cmd("aiperf-not-installed-here")
        self.assertEqual(cmd, ["aiperf-not-installed-here"])


if __name__ == "__main__":
    unittest.main()
