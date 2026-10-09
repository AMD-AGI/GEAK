#!/usr/bin/env python3
"""bench_e2e.sh arms the Triton launch probe by building a SEEDED overlay.

Only the first sitecustomize on PYTHONPATH runs, so the probe cannot simply be
prepended as a second overlay dir: one of the two would go silently dead. The
block must seed the probe overlay from the current one, keep the rest of the
path, and leave everything alone when GEAK_TRITON_LAUNCH_SHAPES=1 is absent.

Like test_bench_trust_remote_code.py, this SLICES the real block out of
bench_e2e.sh and runs it under bash, so it cannot drift from a copy.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(SCRIPTS_DIR, "bench_e2e.sh")
BASH = shutil.which("bash")

_START = "# ---- Triton launch probe"
_END = "# export everything the adapter reads"


def _block():
    with open(BENCH, encoding="utf-8") as fh:
        src = fh.read()
    i = src.index(_START)
    return src[i:src.index(_END, i)]


@unittest.skipIf(BASH is None, "bash is required")
class BenchTritonLaunchProbeTest(unittest.TestCase):
    def _run(self, tmp, extra_env, overlay_path, here=SCRIPTS_DIR):
        script = (
            "set -u\n"
            'HERE="$1"; EXTRA_ENV="$2"; OVERLAY_PYTHONPATH="$3"; OUT_DIR="$4"\n'
            "_stage_lookup() { [ -f \"$HERE/$1\" ] && echo \"$HERE/$1\"; }\n"
            + _block()
            + '\nprintf %s "$OVERLAY_PYTHONPATH"\n')
        proc = subprocess.run(
            [BASH, "-c", script, "_", here, extra_env, overlay_path, tmp],
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, PATH=os.path.dirname(sys.executable) + ":" + os.environ["PATH"]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout, proc.stderr

    def test_unarmed_leaves_the_overlay_path_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run(tmp, "VLLM_ROCM_USE_AITER=1", "/some/overlay")
            self.assertEqual(out, "/some/overlay")

    def test_armed_with_no_overlay_builds_a_probe_only_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._run(tmp, "A=1 GEAK_TRITON_LAUNCH_SHAPES=1", "")
            probe = os.path.join(tmp, "triton_launch_probe_overlay")
            self.assertEqual(out, probe)
            with open(os.path.join(probe, "_overlay_manifest.json")) as fh:
                hooks = json.load(fh)["hooks"]
            self.assertEqual([h["module"] for h in hooks], ["triton.runtime.jit"])
            self.assertTrue(os.path.exists(os.path.join(probe, "triton_launch_probe.py")))

    def test_armed_seeds_from_the_current_overlay_and_keeps_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "accepted")
            subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "overlay_setup.py"),
                            "add-hook", "--overlay", base, "--module", "vllm.x",
                            "--impl-module", "accepted_kernel"],
                           check=True, capture_output=True)
            out, _ = self._run(tmp, "GEAK_TRITON_LAUNCH_SHAPES=1", base + ":/underlay")
            probe = os.path.join(tmp, "triton_launch_probe_overlay")
            self.assertEqual(out, probe + ":/underlay")
            with open(os.path.join(probe, "_overlay_manifest.json")) as fh:
                hooks = json.load(fh)["hooks"]
            # the accepted overlay's hook survives beside the probe
            self.assertEqual(sorted(h["module"] for h in hooks),
                             ["triton.runtime.jit", "vllm.x"])

    def test_an_unseedable_overlay_is_not_shadowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "raw")
            os.makedirs(raw)
            open(os.path.join(raw, "sitecustomize.py"), "w").close()
            out, err = self._run(tmp, "GEAK_TRITON_LAUNCH_SHAPES=1", raw)
            self.assertEqual(out, raw)
            self.assertIn("NOT armed", err)

    def test_missing_helpers_degrade_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, err = self._run(tmp, "GEAK_TRITON_LAUNCH_SHAPES=1", "/o", here=tmp)
            self.assertEqual(out, "/o")
            self.assertIn("not staged", err)


if __name__ == "__main__":
    unittest.main()
