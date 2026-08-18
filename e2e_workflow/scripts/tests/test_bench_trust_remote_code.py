#!/usr/bin/env python3
"""Regression test for bench_e2e.sh recipe-derived client trust-remote-code.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v

WHY THIS EXISTS: the benchmark CLIENT mirrors the SERVER's trust setting so a
custom-tokenizer model measured against a --trust-remote-code server can load its
tokenizer. When the server inherits the flag from the REPLAYED recipe env (rather
than EXTRA_SERVER_ARGS), bench_e2e.sh reads the NUL-delimited recipe env file.

The detection MUST be BY KEY and value-specific. A value-blind substring match
(the original `grep -qi 'trust[-_]remote[-_]code'`) fails OPEN: it matches the
spelling inside ANY variable NAME or a DISABLING value, so

  * `DO_NOT_TRUST_REMOTE_CODE=1`   (a name that NEGATES the concept), and
  * `HF_HUB_TRUST_REMOTE_CODE=0`   (a real control set to OFF, replayed verbatim)

would BOTH silently enable client remote-code execution. The fix starts from
KNOWN truthy controls, then applies the current backend's valid
--trust-remote-code/--no-trust-remote-code tokens in launcher order with
last-token-wins semantics.

This test SLICES the real trust block out of bench_e2e.sh (between its section
markers) and executes it under bash with a controlled RECIPE_ENV_FILE, so it
exercises the SOURCE's own decision and cannot drift from a duplicated pattern.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(SCRIPTS_DIR, "bench_e2e.sh")
BASH = shutil.which("bash")

_START = "# ---- client trust-remote-code"
_END = "# ---- modes ----"


def _extract_trust_block() -> str:
    """Return the trust-detection block verbatim from bench_e2e.sh."""
    with open(BENCH, encoding="utf-8") as fh:
        src = fh.read()
    i = src.index(_START)
    j = src.index(_END, i)
    return src[i:j]


@unittest.skipIf(BASH is None, "bash is required to exercise the trust block")
class BenchRecipeTrustBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = _extract_trust_block()

    def _decide(self, *entries, extra_server_args="", backend="vllm") -> bool:
        """Run the SOURCE's trust block over a NUL-delimited recipe env file and
        report the resulting BENCH_TRUST_REMOTE_CODE."""
        blob = b"".join(e.encode() + b"\x00" for e in entries)
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(blob)
            recipe = tf.name
        try:
            script = (
                "set -u\n"
                'EXTRA_SERVER_ARGS="$1"\n'
                'RECIPE_ENV_FILE="$2"\n'
                'BACKEND="$3"\n'
                + self.block
                + '\nprintf %s "$BENCH_TRUST_REMOTE_CODE"\n'
            )
            proc = subprocess.run(
                [BASH, "-c", script, "_", extra_server_args, recipe, backend],
                capture_output=True, timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            return proc.stdout.decode().strip() == "1"
        finally:
            os.unlink(recipe)

    # ---- must ENABLE (server really trusts remote code) ----
    def test_known_control_truthy_enables(self):
        self.assertTrue(self._decide("HF_HUB_TRUST_REMOTE_CODE=1"))
        self.assertTrue(self._decide("TRANSFORMERS_TRUST_REMOTE_CODE=true"))
        self.assertTrue(self._decide("MAGPIE_TRUST_REMOTE_CODE=yes"))
        self.assertTrue(self._decide("BENCH_TRUST_REMOTE_CODE=on"))

    def test_flag_in_extra_args_value_enables(self):
        self.assertTrue(
            self._decide("EXTRA_VLLM_ARGS=--dtype auto --trust-remote-code")
        )
        self.assertTrue(
            self._decide("EXTRA_SGLANG_ARGS=--trust_remote_code", backend="sglang")
        )

    def test_extra_server_args_flag_enables(self):
        self.assertTrue(self._decide(extra_server_args="--dtype auto --trust-remote-code"))

    # ---- must STAY OFF (the fail-open the fix closes) ----
    def test_negating_name_stays_off(self):
        # The exact fail-open Part 6 probed: a name that NEGATES the concept.
        self.assertFalse(self._decide("DO_NOT_TRUST_REMOTE_CODE=1"))

    def test_disabling_known_control_stays_off(self):
        self.assertFalse(self._decide("HF_HUB_TRUST_REMOTE_CODE=0"))
        self.assertFalse(self._decide("TRANSFORMERS_TRUST_REMOTE_CODE=false"))

    def test_unknown_name_with_truthy_value_stays_off(self):
        # An arbitrary name carrying the spelling is not a trust control.
        self.assertFalse(self._decide("SOME_TRUST_REMOTE_CODE=yes"))
        self.assertFalse(self._decide("MY_TRUST_REMOTE_CODE_NOTE=disabled"))

    def test_flag_spelling_in_non_extra_value_stays_off(self):
        # The flag text living in an unrelated value (not an EXTRA_<BE>_ARGS
        # server-arg string) must not enable trust.
        self.assertFalse(self._decide("SERVER_NOTE=uses --trust-remote-code sometimes"))

    def test_unrelated_recipe_env_stays_off(self):
        self.assertFalse(self._decide("EXTRA_VLLM_ARGS=--dtype auto", "FOO=bar"))

    # ---- whole-token matching (a substring match would fail OPEN here) ----
    def test_invalid_equals_value_forms_stay_off(self):
        # vLLM's BooleanOptionalAction rejects `=true/false`; neither invalid
        # lookalike may silently influence the benchmark client's trust state.
        self.assertFalse(
            self._decide("EXTRA_VLLM_ARGS=--trust-remote-code=false")
        )
        self.assertFalse(
            self._decide(extra_server_args="--trust-remote-code=false")
        )

    def test_lookalike_flag_name_stays_off(self):
        # A DIFFERENT flag whose name merely starts with the spelling.
        self.assertFalse(
            self._decide("EXTRA_VLLM_ARGS=--trust-remote-code-note hello")
        )
        self.assertFalse(
            self._decide(extra_server_args="--trust-remote-code-note hello")
        )

    def test_invalid_equals_true_value_stays_off(self):
        self.assertFalse(self._decide("EXTRA_VLLM_ARGS=--trust-remote-code=true"))
        self.assertFalse(self._decide(extra_server_args="--trust-remote-code=1"))

    def test_flag_as_last_of_many_tokens_enables(self):
        # Whole-token scan must find the bare flag among other args.
        self.assertTrue(
            self._decide(extra_server_args="--dtype auto --tp 2 --trust-remote-code")
        )

    # ---- valid BooleanOptionalAction disable + last-wins semantics --------
    def test_no_trust_flag_disables(self):
        self.assertFalse(
            self._decide("EXTRA_VLLM_ARGS=--trust-remote-code --no-trust-remote-code")
        )
        self.assertFalse(
            self._decide(
                "HF_HUB_TRUST_REMOTE_CODE=1",
                extra_server_args="--no-trust-remote-code",
            )
        )

    def test_last_trust_token_wins_in_each_argument_layer(self):
        self.assertFalse(
            self._decide(
                extra_server_args="--trust-remote-code --no-trust-remote-code"
            )
        )
        self.assertTrue(
            self._decide(
                extra_server_args="--no-trust-remote-code --trust-remote-code"
            )
        )

    def test_geak_args_override_recipe_args(self):
        self.assertFalse(
            self._decide(
                "EXTRA_VLLM_ARGS=--trust-remote-code",
                extra_server_args="--no-trust-remote-code",
            )
        )
        self.assertTrue(
            self._decide(
                "EXTRA_VLLM_ARGS=--no-trust-remote-code",
                extra_server_args="--trust-remote-code",
            )
        )

    def test_other_backend_extra_args_do_not_enable(self):
        self.assertFalse(
            self._decide("EXTRA_SGLANG_ARGS=--trust-remote-code", backend="vllm")
        )


if __name__ == "__main__":
    unittest.main()
