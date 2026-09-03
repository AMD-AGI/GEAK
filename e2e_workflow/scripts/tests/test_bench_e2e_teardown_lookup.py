#!/usr/bin/env python3
"""Unit tests for bench_e2e.sh's teardown-contract lookup.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_bench_e2e_teardown_lookup.py

WHY THESE EXIST: bench_e2e.sh is COPIED into $EVAL_DIR and run from there, so
server_teardown.sh -- the identity-verified kill contract -- has to be staged beside
the copy. When it was merely `source`d, a staging miss was SILENT: under `set -uo
pipefail` (no -e) the failed source does not abort, the `trap server_teardown EXIT`
binds a function that does not exist, and the benchmark launches a server it can never
stop (VRAM + port held, serving-GPU lock released, next launch OOMs). The dispatcher
now refuses to run instead -- a benchmark that cannot stop what it starts must not
start it -- and that refusal is on the launch path of EVERY server in the product, so
it is asserted here rather than trusted.

The gate sits before the serving-GPU lock and before any launch, so these tests need
neither a GPU nor a model: they assert the exit code and the stderr remedy.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(SCRIPTS_DIR, "bench_e2e.sh")
LIB = os.path.join(SCRIPTS_DIR, "server_teardown.sh")
ADAPTERS = os.path.join(SCRIPTS_DIR, "adapters")

BASH = shutil.which("bash")
MISSING_LIB_RC = 3


@unittest.skipIf(BASH is None, "bash is required to exercise the shell dispatcher")
class BenchTeardownLookupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_lookup_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.eval_dir = os.path.join(self.tmp, "eval")
        os.makedirs(self.eval_dir)
        # Stage exactly what roles/director.md stages, minus the file under test.
        shutil.copy(BENCH, os.path.join(self.eval_dir, "bench_e2e.sh"))
        shutil.copytree(ADAPTERS, os.path.join(self.eval_dir, "adapters"))

        # A stub backend adapter (sourced at the top, long before the gate) so a run
        # that gets PAST the gate stops immediately at the launch instead of spending
        # the health-wait on an absent vLLM. Keeps these tests sub-second.
        self.adapter = os.path.join(self.tmp, "stub_adapter.sh")
        with open(self.adapter, "w", encoding="utf-8") as fh:
            fh.write(
                "adapter_default_port() { echo 18080; }\n"
                "adapter_launch() { echo 'STUB_LAUNCH_REACHED'; exit 0; }\n"
                "adapter_health() { return 0; }\n"
                "adapter_bench() { return 0; }\n"
            )

    def stage_lib(self, into):
        os.makedirs(into, exist_ok=True)
        shutil.copy(LIB, os.path.join(into, "server_teardown.sh"))

    def run_bench(self, **env):
        """Run the staged dispatcher; it must decide the lookup before any launch."""
        run_env = dict(os.environ)
        run_env.pop("SKILL_DIR", None)
        run_env.pop("WORKFLOW_DIR", None)
        run_env.update(
            ADAPTER=self.adapter,
            BACKEND="vllm",
            MODEL=os.path.join(self.tmp, "model"),
            OUT_DIR=os.path.join(self.tmp, "out"),
            REPEATS="1",
            PROFILE="0",
        )
        run_env.update(env)
        return subprocess.run(
            [BASH, os.path.join(self.eval_dir, "bench_e2e.sh")],
            env=run_env, capture_output=True, text=True, timeout=120,
        )

    def test_missing_contract_refuses_to_run(self):
        """The staging miss that used to leak a server is now a hard, legible stop."""
        proc = self.run_bench()
        self.assertEqual(proc.returncode, MISSING_LIB_RC, proc.stderr[-2000:])
        self.assertIn("server_teardown.sh not found", proc.stderr)
        # The message must say what to do, not just that something is missing.
        self.assertIn("cp ", proc.stderr)
        self.assertIn("LEAKED", proc.stderr)

    def test_contract_staged_beside_the_copy_passes_the_gate(self):
        """The director-staged layout must get PAST the gate (it may fail later, for
        want of a GPU/model -- it just must not fail with the refusal)."""
        self.stage_lib(self.eval_dir)
        proc = self.run_bench()
        self.assertNotEqual(proc.returncode, MISSING_LIB_RC, proc.stderr[-2000:])
        self.assertNotIn("server_teardown.sh not found", proc.stderr)
        self.assertIn("STUB_LAUNCH_REACHED", proc.stdout, "never reached the launch")

    def test_skill_dir_fallback_is_used_and_announced(self):
        """A caller that exports SKILL_DIR keeps working from an unstaged copy -- but
        the run says so, because the copy is then not self-contained."""
        skill = os.path.join(self.tmp, "skill")
        self.stage_lib(os.path.join(skill, "scripts"))
        proc = self.run_bench(SKILL_DIR=skill)
        self.assertNotEqual(proc.returncode, MISSING_LIB_RC, proc.stderr[-2000:])
        self.assertIn("teardown contract:", proc.stdout + proc.stderr)
        self.assertIn("STUB_LAUNCH_REACHED", proc.stdout, "never reached the launch")

    def test_empty_skill_dir_does_not_resolve_to_an_absolute_path(self):
        """With SKILL_DIR="" the candidate degrades to /scripts/server_teardown.sh; a
        file there (or a symlink attack) must not be mistaken for the contract."""
        proc = self.run_bench(SKILL_DIR="", WORKFLOW_DIR="")
        self.assertEqual(proc.returncode, MISSING_LIB_RC, proc.stderr[-2000:])
        self.assertNotIn("/scripts/server_teardown.sh", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
