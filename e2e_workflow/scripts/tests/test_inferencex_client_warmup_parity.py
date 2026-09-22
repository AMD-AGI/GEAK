#!/usr/bin/env python3
"""Warmup/seed parity tests for the InferenceX bench CLIENT adapter.

The point of BENCH_CLIENT=inferencex is that GEAK drives the EXACT client
Hyperloom drives, with the same dataset semantics.  Two of those semantics are
not defaults and must be forced whenever the lifecycle is a Hyperloom-aligned
one: ``--num-warmups 2*CONC`` (Hyperloom's per-call warmup, versus the adapter's
own 8-request fallback) and ``--seed 0``.

Both the isolated_server and the warm_server lifecycles are Hyperloom-aligned,
but they announce themselves differently -- isolated_server sets
GEAK_ISOLATED_REPLICA=1, while warm_server sets it to 0 (its outer warmup round
MUST run) and exports WARM_SERVER_ROUNDS instead.  Gating only on the replica
flag silently dropped warm_server back to 8 warmups whenever the caller did not
export NUM_WARMUPS, which is exactly what a direct bench_e2e.sh invocation does.
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT = os.path.join(SCRIPTS_DIR, "adapters", "clients", "inferencex.sh")
BASH = shutil.which("bash")


@unittest.skipIf(BASH is None, "bash is required to exercise the shell adapter")
class InferencexClientWarmupParityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ix_client_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.argv_log = os.path.join(self.tmp, "argv.json")
        self.bench_py = os.path.join(self.tmp, "benchmark_serving.py")
        with open(self.bench_py, "w", encoding="utf-8") as fh:
            fh.write("")
        # PYTHON_BIN shim: record the argv of the bench invocation, then write
        # the result file the adapter expects so it reaches its success path.
        # The adapter also calls PYTHON_BIN with -c to re-dump that file; the
        # shim forwards anything that is not the bench script to real python3.
        self.python_bin = os.path.join(self.tmp, "python_shim.sh")
        with open(self.python_bin, "w", encoding="utf-8") as fh:
            fh.write(
                textwrap.dedent(
                    r"""
                    #!/usr/bin/env bash
                    if [ "$1" != "%(bench)s" ]; then exec python3 "$@"; fi
                    python3 - "$@" <<'PY'
                    import json, sys
                    argv = sys.argv[1:]
                    with open("%(log)s", "w") as fh:
                        json.dump(argv, fh)
                    out_dir = argv[argv.index("--result-dir") + 1]
                    name = argv[argv.index("--result-filename") + 1]
                    with open(out_dir + "/" + name, "w") as fh:
                        json.dump({"output_throughput": 1.0}, fh)
                    PY
                    """
                ).lstrip() % {"bench": self.bench_py, "log": self.argv_log}
            )
        os.chmod(self.python_bin, 0o755)

    def run_bench(self, *, conc=64, **extra_env):
        out_dir = os.path.join(self.tmp, f"out_{len(os.listdir(self.tmp))}")
        os.makedirs(out_dir)
        env = dict(os.environ)
        env.update(
            INFERENCEX_BENCH_SERVING=self.bench_py,
            PYTHON_BIN=self.python_bin,
            MODEL=os.path.join(self.tmp, "model"),
            BASE_URL="http://127.0.0.1:1/v1",
            ISL="8192",
            OSL="1024",
            OUT_DIR=out_dir,
            PROFILE_DIR=out_dir,
            RESULT_JSONL=os.path.join(out_dir, "results.jsonl"),
        )
        for key in ("NUM_WARMUPS", "SEED", "GEAK_ISOLATED_REPLICA",
                    "WARM_SERVER_ROUNDS", "RANDOM_RANGE_RATIO"):
            env.pop(key, None)
        env.update({key: str(value) for key, value in extra_env.items()})
        script = f'. "{CLIENT}"; adapter_bench 192 {conc} 0'
        proc = subprocess.run(
            [BASH, "-c", script], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(self.argv_log, encoding="utf-8") as fh:
            argv = json.load(fh)
        return {argv[i]: argv[i + 1] for i in range(len(argv) - 1)}

    def test_warm_server_gets_hyperloom_warmups_without_an_explicit_num_warmups(self):
        """The regression: warm_server sets ISOLATED_REPLICA=0, not 1."""
        flags = self.run_bench(conc=64, GEAK_ISOLATED_REPLICA=0,
                               WARM_SERVER_ROUNDS=3)
        self.assertEqual(flags["--num-warmups"], "128")  # 2 * CONC
        self.assertEqual(flags["--seed"], "0")

    def test_isolated_server_still_gets_hyperloom_warmups(self):
        flags = self.run_bench(conc=64, GEAK_ISOLATED_REPLICA=1)
        self.assertEqual(flags["--num-warmups"], "128")
        self.assertEqual(flags["--seed"], "0")

    def test_warm_server_forces_seed_zero_over_a_caller_override(self):
        """Dataset identity is the whole point; a stray SEED must not win."""
        flags = self.run_bench(conc=64, GEAK_ISOLATED_REPLICA=0,
                               WARM_SERVER_ROUNDS=3, SEED=1234,
                               NUM_WARMUPS=8)
        self.assertEqual(flags["--num-warmups"], "128")
        self.assertEqual(flags["--seed"], "0")

    def test_warmups_track_concurrency(self):
        flags = self.run_bench(conc=16, GEAK_ISOLATED_REPLICA=0,
                               WARM_SERVER_ROUNDS=1)
        self.assertEqual(flags["--num-warmups"], "32")

    def test_legacy_lifecycle_keeps_the_adapter_default(self):
        """Neither flag set => no Hyperloom alignment claimed, no forcing."""
        flags = self.run_bench(conc=64, SEED=1234)
        self.assertEqual(flags["--num-warmups"], "8")
        self.assertEqual(flags["--seed"], "1234")


if __name__ == "__main__":
    unittest.main()
