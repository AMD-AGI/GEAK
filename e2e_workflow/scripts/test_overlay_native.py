#!/usr/bin/env python3
"""Stdlib unittest for overlay_setup.py's NATIVE apply-back plumbing (no GPU, no server, CI-safe).

Covers the safety-critical guarantees: in-place apply + byte-exact revert (sources, named artifacts,
cache dirs), fresh-build verification, mixed Python-overlay + native round-trip, crash-recovery
idempotency, and gc-stale cleanup of a crashed run's dirty install.

Run:  python3 test_overlay_native.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "overlay_setup.py")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def rt(path):
    with open(path) as fh:
        return fh.read()


def rb(path):
    with open(path, "rb") as fh:
        return fh.read()


def run(*args, check=True):
    r = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"cmd failed ({r.returncode}): {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r


class NativeApplyBack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ovltest_")
        # a fake "install" tree: a compiled source + its prebuilt artifact + a regenerable cache dir
        self.install = os.path.join(self.tmp, "install", "pkg", "kernels")
        os.makedirs(self.install)
        self.src = os.path.join(self.install, "gemm.cu")
        self.art = os.path.join(self.install, "gemm.so")
        self.cache = os.path.join(self.install, ".jit_cache", "gemm_abc123")
        os.makedirs(self.cache)
        with open(self.src, "w") as fh:
            fh.write("// ORIGINAL kernel source\n__global__ void gemm() {}\n")
        with open(self.art, "wb") as fh:
            fh.write(b"PREBUILT-ARTIFACT-ORIGINAL")
        with open(os.path.join(self.cache, "compiled.hsaco"), "wb") as fh:
            fh.write(b"OLD-CACHE-ENTRY")
        self.src_orig_sha = sha(self.src)
        self.art_orig_sha = sha(self.art)

        # the patched source the optimizer produced
        self.patched = os.path.join(self.tmp, "gemm_patched.cu")
        with open(self.patched, "w") as fh:
            fh.write("// OPTIMIZED kernel source\n__global__ void gemm() { /*fast*/ }\n")

        # a stub "incremental build": rewrites the artifact (simulates a real rebuild touching the .so)
        self.builder = os.path.join(self.tmp, "build.sh")
        with open(self.builder, "w") as fh:
            fh.write("#!/bin/bash\nprintf 'REBUILT-FROM-%s' \"$(cat '%s')\" > '%s'\n"
                     % ("OPTIMIZED", self.src, self.art))
        os.chmod(self.builder, 0o755)

        self.overlay = os.path.join(self.tmp, "overlay")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def manifest(self):
        with open(os.path.join(self.overlay, "_overlay_manifest.json")) as fh:
            return json.load(fh)

    # ---- core apply + build ----------------------------------------------------------------------
    def test_apply_swaps_source_and_rebuilds(self):
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art,
            "--invalidate-cache", self.cache,
            "--build-cmd", f"bash {self.builder}")
        # source swapped in place
        self.assertIn("OPTIMIZED", rt(self.src))
        # artifact rebuilt (changed)
        self.assertNotEqual(sha(self.art), self.art_orig_sha)
        self.assertIn("REBUILT", rb(self.art).decode())
        # manifest records one native entry with backups
        nat = self.manifest()["natives"]
        self.assertEqual(len(nat), 1)
        self.assertEqual(nat[0]["sources"][0]["sha256"], self.src_orig_sha)
        self.assertTrue(nat[0]["verify"]["artifact"].endswith("gemm.so"))

    # ---- fresh-build verification ----------------------------------------------------------------
    def test_verify_passes_when_artifact_changed(self):
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art,
            "--build-cmd", f"bash {self.builder}")
        r = run("verify-native", "--overlay", self.overlay)
        self.assertIn("FRESH_BUILD_OK", r.stdout)

    def test_verify_fails_on_silent_noop_build(self):
        # a build that does NOT touch the artifact -> verify must catch it
        noop = os.path.join(self.tmp, "noop.sh")
        with open(noop, "w") as fh:
            fh.write("#!/bin/bash\ntrue\n")
        os.chmod(noop, 0o755)
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art,
            "--build-cmd", f"bash {noop}")
        r = run("verify-native", "--overlay", self.overlay, check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FRESH_BUILD_FAIL", r.stdout)

    # ---- byte-exact revert -----------------------------------------------------------------------
    def test_revert_restores_everything_byte_exact(self):
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art,
            "--invalidate-cache", self.cache, "--build-cmd", f"bash {self.builder}")
        run("revert", "--overlay", self.overlay)
        # source restored byte-exact
        self.assertEqual(sha(self.src), self.src_orig_sha)
        # artifact restored byte-exact
        self.assertEqual(sha(self.art), self.art_orig_sha)
        # cache dir restored intact
        self.assertTrue(os.path.isfile(os.path.join(self.cache, "compiled.hsaco")))
        self.assertEqual(rb(os.path.join(self.cache, "compiled.hsaco")), b"OLD-CACHE-ENTRY")
        # natives list cleared
        self.assertEqual(self.manifest()["natives"], [])

    def test_revert_removes_artifact_that_did_not_exist(self):
        os.remove(self.art)  # candidate build will CREATE it; revert must remove it
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art,
            "--build-cmd", f"bash {self.builder}")
        self.assertTrue(os.path.exists(self.art))
        run("revert", "--overlay", self.overlay)
        self.assertFalse(os.path.exists(self.art))
        self.assertEqual(sha(self.src), self.src_orig_sha)

    # ---- crash-recovery idempotency --------------------------------------------------------------
    def test_double_revert_is_noop(self):
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art, "--build-cmd", f"bash {self.builder}")
        run("revert", "--overlay", self.overlay)
        run("revert", "--overlay", self.overlay)  # must not raise / must keep things restored
        self.assertEqual(sha(self.src), self.src_orig_sha)
        self.assertEqual(sha(self.art), self.art_orig_sha)

    def test_revert_works_even_if_build_never_ran(self):
        # simulate a crash right after manifest+mutate but before any verify: apply with no build cmd
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art, "--invalidate-cache", self.cache)
        self.assertIn("OPTIMIZED", rt(self.src))  # mutated
        run("revert", "--overlay", self.overlay)
        self.assertEqual(sha(self.src), self.src_orig_sha)
        self.assertTrue(os.path.isfile(os.path.join(self.cache, "compiled.hsaco")))

    # ---- mixed Python-overlay + native -----------------------------------------------------------
    def test_mixed_python_and_native_same_overlay(self):
        # a python rebind (non-invasive) + a native apply (invasive) share ONE overlay/manifest
        run("add-rebind", "--overlay", self.overlay,
            "--target", "pkg.mod:fn", "--impl-module", "impl", "--impl-attr", "fast_fn")
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art, "--build-cmd", f"bash {self.builder}")
        man = self.manifest()
        self.assertEqual(len(man["rebinds"]), 1)
        self.assertEqual(len(man["natives"]), 1)
        # one unified revert undoes the native (invasive) part; python overlay just stops being on PYTHONPATH
        run("revert", "--overlay", self.overlay)
        self.assertEqual(sha(self.src), self.src_orig_sha)
        self.assertEqual(sha(self.art), self.art_orig_sha)
        self.assertEqual(self.manifest()["natives"], [])
        self.assertEqual(len(self.manifest()["rebinds"]), 1)  # python overlay entry untouched

    # ---- gc-stale (crashed run left a dirty install) ---------------------------------------------
    def test_gc_stale_reverts_crashed_run(self):
        run("add-native", "--overlay", self.overlay, "--target", self.src,
            "--patched-file", self.patched, "--artifact", self.art, "--build-cmd", f"bash {self.builder}")
        # the run "crashed" — never called revert; install is dirty
        self.assertIn("OPTIMIZED", rt(self.src))
        r = run("gc-stale", "--root", self.tmp)
        self.assertIn("reverted 1", r.stdout)
        self.assertEqual(sha(self.src), self.src_orig_sha)
        self.assertEqual(sha(self.art), self.art_orig_sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
