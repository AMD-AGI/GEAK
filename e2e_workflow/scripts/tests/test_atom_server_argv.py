#!/usr/bin/env python3
"""GPU-free tests for how the native ATOM adapter turns EXTRA_SERVER_ARGS into server argv.

The adapter used to expand $EXTRA_SERVER_ARGS unquoted: shell quotes stayed on JSON values and
`*`/`[..]` inside them were pathname-expanded. These pin the argv the server actually receives.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
ATOM_ADAPTER = SCRIPTS_DIR / "adapters" / "atom.sh"
SPLIT_ARGS = SCRIPTS_DIR / "adapters" / "split_args.py"
BASH = shutil.which("bash")

QUANT = {"exclude_layer": ["lm_head", "*.mlp.gate", "model.layers.[0-9].mlp.*expert*"],
         "global_quant_config": "ptpc_fp8"}


@unittest.skipIf(BASH is None, "bash is required")
class AtomServerArgvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="atom_argv_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # Files a glob in the quant config would match if the shell expanded it.
        (self.tmp / "x.mlp.gate").write_text("")
        (self.tmp / "model.layers.3.mlp.fooexpertbar").write_text("")

    def _argv(self, args: str, adapter_dir: Path = ATOM_ADAPTER.parent) -> tuple[list[str], str]:
        driver = (
            f'source "{adapter_dir / "atom.sh"}"\n'
            '_atom_server_argv "$ARGS"\n'
            'echo "count=${#_ATOM_EXTRA_ARGV[@]}" >&2\n'
            '[ "${#_ATOM_EXTRA_ARGV[@]}" -eq 0 ] || printf "%s\\0" "${_ATOM_EXTRA_ARGV[@]}"\n'
        )
        proc = subprocess.run([BASH, "-c", driver], env={**os.environ, "ARGS": args},
                              cwd=self.tmp, capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        return proc.stdout.decode().split("\0")[:-1], proc.stderr.decode()

    def test_shell_quoted_json_reaches_the_server_unquoted_and_unglobbed(self):
        args = ("--block-size 64 --online_quant_config "
                + "'" + json.dumps(QUANT, separators=(",", ":")) + "'"
                + " --cudagraph-capture-sizes '[1,2,4,8]' --method mtp")
        argv, _ = self._argv(args)
        self.assertEqual(argv[:2], ["--block-size", "64"])
        self.assertEqual(json.loads(argv[argv.index("--online_quant_config") + 1]), QUANT)
        self.assertEqual(argv[argv.index("--cudagraph-capture-sizes") + 1], "[1,2,4,8]")
        self.assertEqual(argv[-2:], ["--method", "mtp"])

    def test_bare_json_with_spaces_stays_one_token(self):
        argv, _ = self._argv('--speculative-config {"method": "mtp", "n": 5} --x 1')
        self.assertEqual(argv, ["--speculative-config", '{"method": "mtp", "n": 5}', "--x", "1"])

    def test_empty_args_give_empty_argv(self):
        argv, _ = self._argv("")
        self.assertEqual(argv, [])

    def test_missing_helper_falls_back_loudly_without_globbing(self):
        staged = self.tmp / "staged"
        staged.mkdir()
        shutil.copy(ATOM_ADAPTER, staged / "atom.sh")  # helper deliberately not staged
        argv, err = self._argv("--a *.mlp.gate --b 1", adapter_dir=staged)
        self.assertEqual(argv, ["--a", "*.mlp.gate", "--b", "1"])
        self.assertIn("falling back to word splitting", err)

    def test_unbalanced_quote_is_rejected_by_the_helper(self):
        proc = subprocess.run([sys.executable, str(SPLIT_ARGS), "--a 'oops"], capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, b"")


if __name__ == "__main__":
    unittest.main()
