#!/usr/bin/env python3
"""Adapter smoke tests for the Magpie server LAUNCHER (adapters/launchers/magpie.sh).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_magpie_launcher_extra_args.py

WHY THESE EXIST: magpie.sh replays the orchestrator's recorded launch env as the
BASE layer, then passes GEAK's own EXTRA_<BE>_ARGS on the SAME `env` line. Because
`env NAME=VALUE` is LAST-WINS, a recipe-recorded EXTRA_<BE>_ARGS (the flags that
decide kernel dispatch -- --kv-cache-dtype / --moe-runner-backend /
--attention-backend / --quantization ...) was silently dropped even though
RECIPE_ENV_REPLAYED still advertised it, so the launcher served a different stack
than the recipe recorded. The fix pulls the recipe's copy out and merges it
UNDER GEAK's accepted flags (recipe first, GEAK covers conflicts by ordering),
removing it from the replay array so it is not passed twice.

The whole point is byte-parity with the orchestrator's serving stack, and
--dry-run cannot cover it (run_e2e.py prints and returns 0 before the workflow
ever calls adapter_launch). So this sources magpie.sh directly and drives
adapter_launch against a FAKE Magpie script that records the env it received --
no GPU, no framework, no model. It also pins the two GPU-pinning shapes
(inherited outer ROCR mask vs bare box), which the same env line decides.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAGPIE = os.path.join(SCRIPTS_DIR, "adapters", "launchers", "magpie.sh")
BASH = shutil.which("bash")

FAKE_SCRIPT = """#!/usr/bin/env bash
{{
  printf 'EXTRA_VLLM_ARGS=%s\\n'      "${{EXTRA_VLLM_ARGS-<unset>}}"
  printf 'EXTRA_SGLANG_ARGS=%s\\n'    "${{EXTRA_SGLANG_ARGS-<unset>}}"
  printf 'ROCR_VISIBLE_DEVICES=%s\\n' "${{ROCR_VISIBLE_DEVICES-<unset>}}"
  printf 'HIP_VISIBLE_DEVICES=%s\\n'  "${{HIP_VISIBLE_DEVICES-<unset>}}"
  printf 'CUDA_VISIBLE_DEVICES=%s\\n' "${{CUDA_VISIBLE_DEVICES-<unset>}}"
  printf 'FIRST=%s\\n'                "${{FIRST-<unset>}}"
  printf 'FOO=%s\\n'                  "${{FOO-<unset>}}"
}} > "{capture}"
echo $$ > "$MAGPIE_SERVER_PID_FILE"
exit 0
"""

DRIVER = """set -uo pipefail
source "{magpie}"
adapter_launch
"""


@unittest.skipIf(BASH is None, "bash is required to exercise the launcher adapter")
class MagpieLauncherExtraArgsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="magpie_launch_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.out)
        self.capture = os.path.join(self.tmp, "captured_env.txt")
        self.fake = os.path.join(self.tmp, "fake_magpie.sh")
        with open(self.fake, "w", encoding="utf-8") as fh:
            fh.write(FAKE_SCRIPT.format(capture=self.capture))
        os.chmod(self.fake, 0o755)
        self.driver = os.path.join(self.tmp, "driver.sh")
        with open(self.driver, "w", encoding="utf-8") as fh:
            fh.write(DRIVER.format(magpie=MAGPIE))

    def _recipe_file(self, entries):
        """Write a NUL-delimited recipe env file (the format run_e2e.py emits)."""
        path = os.path.join(self.tmp, "recipe_env.nul")
        with open(path, "wb") as fh:
            fh.write(b"".join(f"{k}={v}".encode() + b"\x00" for k, v in entries))
        return path

    def _recipe_file_raw(self, tokens):
        """Write raw NUL-delimited tokens verbatim (may not be NAME=VALUE).

        The production writer (run_e2e.py) validates identifiers, but this
        launcher is also driven directly and re-usable, so it must sanitize the
        recipe array itself. This lets a test inject a hostile token -- an `env`
        option like `-SCUDA_VISIBLE_DEVICES=7`, a GPU mask, or a bare word."""
        path = os.path.join(self.tmp, "recipe_env.nul")
        with open(path, "wb") as fh:
            fh.write(b"".join(t.encode() + b"\x00" for t in tokens))
        return path

    def _launch(self, *, backend="vllm", extra_server_args="",
                recipe=None, outer_rocr=None, extra_env=None,
                profile="0", pythonpath=None, overlay_pythonpath=None):
        env = dict(os.environ)
        for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                  "CUDA_VISIBLE_DEVICES", "RECIPE_ENV_FILE",
                  "EXTRA_VLLM_ARGS", "EXTRA_SGLANG_ARGS", "EXTRA_ENV",
                  "PYTHONPATH", "OVERLAY_PYTHONPATH"):
            env.pop(k, None)
        env.update(
            BACKEND=backend,
            MODEL=os.path.join(self.tmp, "model"),
            TP="1", PORT="18080", GPU="1",
            OUT_DIR=self.out,
            LOG=os.path.join(self.out, "server.log"),
            PROFILE=profile,
            MAGPIE_LAUNCH_SCRIPT=self.fake,
            EXTRA_SERVER_ARGS=extra_server_args,
        )
        if recipe is not None:
            env["RECIPE_ENV_FILE"] = recipe
        if outer_rocr is not None:
            env["ROCR_VISIBLE_DEVICES"] = outer_rocr
        if extra_env is not None:
            env["EXTRA_ENV"] = extra_env
        if pythonpath is not None:
            env["PYTHONPATH"] = pythonpath
        if overlay_pythonpath is not None:
            env["OVERLAY_PYTHONPATH"] = overlay_pythonpath
        proc = subprocess.run(
            [BASH, self.driver], env=env, cwd=self.tmp,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:] + proc.stdout[-2000:])
        captured = {}
        with open(self.capture, encoding="utf-8") as fh:
            for line in fh:
                k, _, v = line.rstrip("\n").partition("=")
                captured[k] = v
        return captured, proc

    # ---- EXTRA_<BE>_ARGS merge (the recipe-drop bug) -----------------------------

    def test_geak_only_extra_args(self):
        cap, _ = self._launch(extra_server_args="--geak-x 1")
        self.assertEqual(cap["EXTRA_VLLM_ARGS"], "--geak-x 1")

    def test_recipe_only_extra_args_is_not_dropped(self):
        """The core bug: a recipe-recorded EXTRA_<BE>_ARGS must reach the server."""
        recipe = self._recipe_file([("EXTRA_VLLM_ARGS", "--recipe-a --recipe-b"),
                                    ("FOO", "bar")])
        cap, _ = self._launch(extra_server_args="", recipe=recipe)
        self.assertEqual(cap["EXTRA_VLLM_ARGS"], "--recipe-a --recipe-b")
        # non-EXTRA recipe vars are still replayed verbatim.
        self.assertEqual(cap["FOO"], "bar")

    def test_recipe_and_geak_merge_recipe_first_geak_last(self):
        """Merged (recipe base, GEAK overrides by coming later); not passed twice."""
        recipe = self._recipe_file([("EXTRA_VLLM_ARGS", "--recipe-a"), ("FOO", "bar")])
        cap, _ = self._launch(extra_server_args="--geak-x", recipe=recipe)
        self.assertEqual(cap["EXTRA_VLLM_ARGS"], "--recipe-a --geak-x")
        self.assertEqual(cap["FOO"], "bar")

    def test_backend_specific_var_name(self):
        """The var name follows the backend (EXTRA_SGLANG_ARGS for sglang)."""
        recipe = self._recipe_file([("EXTRA_SGLANG_ARGS", "--mem-fraction-static 0.8")])
        cap, _ = self._launch(backend="sglang", extra_server_args="--geak-y",
                              recipe=recipe)
        self.assertEqual(cap["EXTRA_SGLANG_ARGS"], "--mem-fraction-static 0.8 --geak-y")

    # ---- GPU-pinning shapes ------------------------------------------------------

    def test_bare_box_pins_rocr_only(self):
        cap, proc = self._launch()
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "1")     # GPU is physical
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "<unset>")
        self.assertEqual(cap["CUDA_VISIBLE_DEVICES"], "<unset>")

    def test_inherited_outer_rocr_stacks_hip(self):
        cap, proc = self._launch(outer_rocr="4,5,6,7")
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "4,5,6,7")  # inherited mask kept
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "1")         # logical, on top
        self.assertEqual(cap["CUDA_VISIBLE_DEVICES"], "<unset>")

    # ---- a populated EXTRA_ENV must not clobber the GPU mask ----------------------

    def test_populated_extra_env_cannot_clobber_mask_bare_box(self):
        """A GPU mask leaked into EXTRA_ENV must not steal another job's card.

        In the bare-box shape the launcher `-u`s HIP/CUDA, but `env -u HIP ...
        HIP=99` re-sets HIP (a later assignment defeats the unset) and GNU env
        forbids `-u` after operands -- so the mask has to be stripped from
        EXTRA_ENV. The pinned ROCR must survive and HIP/CUDA stay cleared, while
        a NON-mask EXTRA_ENV var is still delivered."""
        cap, _ = self._launch(
            extra_env="ROCR_VISIBLE_DEVICES=99 HIP_VISIBLE_DEVICES=99 "
                      "CUDA_VISIBLE_DEVICES=99 FOO=fromextra"
        )
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "1")        # pinned, not 99
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "<unset>")   # cleared, not 99
        self.assertEqual(cap["CUDA_VISIBLE_DEVICES"], "<unset>")  # cleared, not 99
        self.assertEqual(cap["FOO"], "fromextra")                 # non-mask var kept

    def test_populated_extra_env_cannot_clobber_mask_inherited_rocr(self):
        """Same protection with an inherited outer ROCR: the outer mask and the
        logical HIP pin both survive an EXTRA_ENV override attempt."""
        cap, _ = self._launch(
            outer_rocr="4,5,6,7",
            extra_env="ROCR_VISIBLE_DEVICES=99 HIP_VISIBLE_DEVICES=99 FOO=fromextra",
        )
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "4,5,6,7")  # inherited, not 99
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "1")         # logical pin, not 99
        self.assertEqual(cap["FOO"], "fromextra")

    def test_extra_env_split_string_option_cannot_reinject_mask(self):
        """`env` parses a leading-dash EXTRA_ENV token as an OPTION, not an
        assignment: `-SCUDA_VISIBLE_DEVICES=7` (-S/--split-string) would re-inject
        a mask that a plain `CUDA_...=` content filter never sees. Such tokens
        must be dropped, so the mask stays pinned/cleared."""
        cap, _ = self._launch(
            outer_rocr="4,5,6,7",
            extra_env="-SCUDA_VISIBLE_DEVICES=7 FOO=kept",
        )
        self.assertEqual(cap["CUDA_VISIBLE_DEVICES"], "<unset>")   # not 7
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "1")
        self.assertEqual(cap["FOO"], "kept")                       # legit var survives

    def test_extra_env_split_string_option_bare_box(self):
        cap, _ = self._launch(extra_env="-SHIP_VISIBLE_DEVICES=9 FOO=kept")
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "<unset>")    # not 9
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "1")
        self.assertEqual(cap["FOO"], "kept")

    def test_extra_env_bare_word_is_dropped_not_run_as_command(self):
        """A token with no `=` would be `env`'s COMMAND to exec; it must be
        dropped so the real server script still launches."""
        cap, proc = self._launch(extra_env="notanassignment FOO=kept")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(cap["FOO"], "kept")

    def test_extra_env_value_is_not_pathname_expanded(self):
        """Unquoted `${EXTRA_ENV}` used to glob before validation.

        A cwd entry named ``FOO=expanded`` transformed the requested literal
        ``FOO=*`` into ``FOO=expanded``. Line-wise `read -ra` must preserve the
        asterisk.
        """
        open(os.path.join(self.tmp, "FOO=expanded"), "w").close()
        cap, _ = self._launch(extra_env="FOO=*")
        self.assertEqual(cap["FOO"], "*")

    def test_extra_env_processes_all_lines(self):
        """A single `read -ra` silently discarded every line after the first."""
        cap, _ = self._launch(extra_env="FIRST=one\nFOO=second")
        self.assertEqual(cap["FIRST"], "one")
        self.assertEqual(cap["FOO"], "second")

    # ---- recipe env is sanitized like EXTRA_ENV (defense in depth) ---------------

    def test_recipe_env_split_string_option_cannot_reinject_mask(self):
        """A recipe-recorded `-S`/--split-string token would be parsed by `env`
        as an OPTION and could re-inject a GPU mask. The recipe array is passed
        as `env` operands exactly like EXTRA_ENV, so it must be filtered the same
        way: the token is dropped, the mask stays pinned/cleared, legit vars pass."""
        recipe = self._recipe_file_raw(
            ["-SCUDA_VISIBLE_DEVICES=7", "FOO=kept", "EXTRA_VLLM_ARGS=--recipe-a"]
        )
        cap, _ = self._launch(outer_rocr="4,5,6,7", recipe=recipe)
        self.assertEqual(cap["CUDA_VISIBLE_DEVICES"], "<unset>")   # not 7
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "1")
        self.assertEqual(cap["FOO"], "kept")                       # legit var survives
        self.assertEqual(cap["EXTRA_VLLM_ARGS"], "--recipe-a")     # extras still merge

    def test_recipe_env_gpu_mask_is_dropped(self):
        """A recipe-recorded ROCR/HIP/CUDA assignment is a run-scoped mask, not
        part of the replayed environment; it must not steal another job's card."""
        recipe = self._recipe_file_raw(
            ["ROCR_VISIBLE_DEVICES=99", "HIP_VISIBLE_DEVICES=99",
             "CUDA_VISIBLE_DEVICES=99", "FOO=kept"]
        )
        cap, _ = self._launch(recipe=recipe)                       # bare box
        self.assertEqual(cap["ROCR_VISIBLE_DEVICES"], "1")         # pinned, not 99
        self.assertEqual(cap["HIP_VISIBLE_DEVICES"], "<unset>")    # cleared
        self.assertEqual(cap["CUDA_VISIBLE_DEVICES"], "<unset>")   # cleared
        self.assertEqual(cap["FOO"], "kept")

    def test_recipe_env_bare_word_is_dropped_not_run_as_command(self):
        """A recipe token with no `=` would be `env`'s COMMAND; it must be dropped
        so the real server script still launches."""
        recipe = self._recipe_file_raw(["notanassignment", "FOO=kept"])
        cap, proc = self._launch(recipe=recipe)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(cap["FOO"], "kept")

    # ---- profiler step-bound is appended LAST ------------------------------------

    def _fake_vllm_pythonpath(
        self, *, fields=("max_iterations", "delay_iterations"), name="pyfake"
    ):
        """A minimal importable ``vllm.config.ProfilerConfig`` so the launcher's
        capability probe reports the given fields without a real vllm install."""
        root = os.path.join(self.tmp, name)
        pkg = os.path.join(root, "vllm")
        os.makedirs(pkg, exist_ok=True)
        open(os.path.join(pkg, "__init__.py"), "w").close()
        decls = "\n".join(f"    {f}: int = 0" for f in fields) or "    pass"
        with open(os.path.join(pkg, "config.py"), "w", encoding="utf-8") as fh:
            fh.write("import dataclasses\n\n"
                     "@dataclasses.dataclass\n"
                     "class ProfilerConfig:\n" + decls + "\n")
        return root

    def test_profiler_step_bound_appended_last(self):
        """PROFILE=1 on vllm appends the profiler step bound AFTER the recipe base
        and GEAK's own flags, so the bound wins over both (Finding 7 / Gate C)."""
        pp = self._fake_vllm_pythonpath()
        recipe = self._recipe_file([("EXTRA_VLLM_ARGS", "--recipe-a")])
        cap, proc = self._launch(
            extra_server_args="--geak-x", recipe=recipe,
            profile="1", pythonpath=pp,
        )
        self.assertEqual(
            cap["EXTRA_VLLM_ARGS"],
            "--recipe-a --geak-x "
            "--profiler-config.max_iterations 64 "
            "--profiler-config.delay_iterations 0",
            proc.stderr[-2000:],
        )

    def test_profiler_bound_omitted_when_build_declares_no_field(self):
        """A vllm build whose ProfilerConfig declares neither bound gets no bound
        appended (strict extra=forbid would abort the server); the recipe+GEAK
        flags are still passed intact."""
        pp = self._fake_vllm_pythonpath(fields=())
        recipe = self._recipe_file([("EXTRA_VLLM_ARGS", "--recipe-a")])
        cap, proc = self._launch(
            extra_server_args="--geak-x", recipe=recipe,
            profile="1", pythonpath=pp,
        )
        self.assertEqual(cap["EXTRA_VLLM_ARGS"], "--recipe-a --geak-x", proc.stderr[-2000:])

    def test_profiler_probe_uses_overlay_pythonpath_like_final_server(self):
        """The overlay can replace vllm.config with a build whose ProfilerConfig
        has different fields. Probe the overlay-first import, not ambient vLLM."""
        ambient = self._fake_vllm_pythonpath(name="ambient")
        overlay = self._fake_vllm_pythonpath(fields=(), name="overlay")
        cap, proc = self._launch(
            extra_server_args="--geak-x",
            profile="1",
            pythonpath=ambient,
            overlay_pythonpath=overlay,
        )
        self.assertEqual(cap["EXTRA_VLLM_ARGS"], "--geak-x", proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
