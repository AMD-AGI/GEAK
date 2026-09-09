#!/usr/bin/env python3
"""A trace replay must be profiled in steady state, and must map without InferenceX.

Two failures this pins, both of which produce a plausible-looking wrong answer
rather than an error:

* **Window placement.** ``bench_e2e.sh`` defaults ``PROFILE_WARMUP_SEC`` to 0 --
  arm the profiler at load start, so the initial prefill burst stays in the
  trace. That is right for a synthetic sweep and wrong for a client that owns its
  load: at load start aiperf is still resolving the corpus, replaying per-lane
  warmups and draining them, so the capture records the RAMP (decode batch of 1)
  instead of the steady state. Every kernel decision downstream is then made on a
  shape the graded workload never runs. The client adapter therefore publishes
  ``adapter_profile_warmup_s`` and ``bench_e2e.sh`` defers to it.

* **A synthetic substitute for a missing trace.** When the load ends before the
  window opens, the generic path falls back to a profiled synthetic bench. For a
  trace replay that is not a degraded version of the workload, it is a different
  workload, and its trace lands in the same profile dir looking authentic. The
  fallback must be refused.

Plus the mapper: GEAK vendors ``map_aiperf.py`` so a standalone run needs no
InferenceX checkout, while an orchestrated run still prefers the copy its own
runtime deployed.
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

BASH = shutil.which("bash")
SCRIPTS = Path(__file__).resolve().parents[1]
CLIENTS = SCRIPTS / "adapters" / "clients"
ADAPTER = CLIENTS / "agentx.sh"
VENDORED_MAPPER = CLIENTS / "map_aiperf.py"
BENCH_E2E = SCRIPTS / "bench_e2e.sh"


def _hook(**env) -> str:
    """Call adapter_profile_warmup_s() in a clean shell."""
    run_env = dict(os.environ)
    for key in ("AGENTX_PROFILE_WARMUP_S", "GEAK_AGENTX_DURATION_S",
                "GEAK_AGENTX_LOOP_DURATION_S", "MEASUREMENT_PURPOSE",
                "PROFILE_WINDOW_SEC"):
        run_env.pop(key, None)
    run_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        [BASH, "-c", f'source "{ADAPTER}"; adapter_profile_warmup_s'],
        env=run_env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _duration(**env) -> str:
    run_env = dict(os.environ)
    for key in ("GEAK_AGENTX_DURATION_S", "GEAK_AGENTX_LOOP_DURATION_S",
                "MEASUREMENT_PURPOSE"):
        run_env.pop(key, None)
    run_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        [BASH, "-c", f'source "{ADAPTER}"; _agentx_duration'],
        env=run_env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@unittest.skipIf(BASH is None, "bash is required to exercise the client adapter")
class ProfileWindowPlacementTest(unittest.TestCase):
    def test_the_window_opens_inside_the_measured_window_not_at_load_start(self):
        """Anything near 0 would capture corpus load and warmup, not steady state."""
        for purpose in ("search", "parity", "validation"):
            with self.subTest(purpose=purpose):
                delay = int(_hook(MEASUREMENT_PURPOSE=purpose))
                leg = int(_duration(MEASUREMENT_PURPOSE=purpose))
                self.assertGreater(delay, leg // 2, "window opens before mid-replay")
                self.assertLess(delay, leg, "window opens after the replay ends")

    def test_the_canonical_leg_matches_the_inferencex_reference_client(self):
        """aiperf_client.sh uses 2700s for the canonical 3600s window."""
        self.assertEqual(_hook(MEASUREMENT_PURPOSE="parity"), "2700")

    def test_a_search_leg_scales_to_its_shorter_window(self):
        """A 900s leg cannot wait 2700s; the delay follows the leg it runs in."""
        self.assertEqual(_hook(MEASUREMENT_PURPOSE="search"), "675")

    def test_an_explicit_delay_always_wins(self):
        self.assertEqual(
            _hook(MEASUREMENT_PURPOSE="parity", AGENTX_PROFILE_WARMUP_S=90), "90"
        )

    def test_a_short_leg_still_leaves_room_for_the_capture(self):
        """The window plus the stop_profile flush must fit before the replay ends."""
        for leg in (60, 120, 300):
            with self.subTest(leg=leg):
                delay = int(_hook(MEASUREMENT_PURPOSE="search",
                                  GEAK_AGENTX_LOOP_DURATION_S=leg))
                self.assertGreaterEqual(delay, 0, "never negative")
                self.assertLessEqual(delay, leg - 60, "leaves room for the window")

    def test_a_wider_capture_window_pulls_the_start_earlier(self):
        tight = int(_hook(MEASUREMENT_PURPOSE="search",
                          GEAK_AGENTX_LOOP_DURATION_S=300, PROFILE_WINDOW_SEC=20))
        wide = int(_hook(MEASUREMENT_PURPOSE="search",
                         GEAK_AGENTX_LOOP_DURATION_S=300, PROFILE_WINDOW_SEC=180))
        self.assertLess(wide, tight)

    def test_the_duration_the_hook_uses_is_the_one_the_bench_runs(self):
        """One helper feeds both, so placement cannot drift from the real leg."""
        self.assertEqual(_duration(MEASUREMENT_PURPOSE="search"), "900")
        self.assertEqual(_duration(MEASUREMENT_PURPOSE="parity"), "3600")
        self.assertEqual(_duration(MEASUREMENT_PURPOSE="validation"), "3600")
        self.assertEqual(
            _duration(MEASUREMENT_PURPOSE="search", GEAK_AGENTX_LOOP_DURATION_S=42),
            "42",
        )


class BenchE2EDefersToTheClientTest(unittest.TestCase):
    """The wiring in bench_e2e.sh, asserted on the real source."""

    def setUp(self) -> None:
        self.text = BENCH_E2E.read_text()

    def test_bench_e2e_consults_the_hook_when_no_delay_was_given(self):
        self.assertIn("declare -F adapter_profile_warmup_s", self.text)
        self.assertIn('PROFILE_WARMUP_SEC="$(adapter_profile_warmup_s', self.text)

    def test_an_explicit_delay_short_circuits_the_hook(self):
        """The hook is consulted only when PROFILE_WARMUP_SEC is empty."""
        self.assertRegex(
            self.text,
            r'if \[ -z "\$\{PROFILE_WARMUP_SEC:-\}" \] '
            r'&& declare -F adapter_profile_warmup_s',
        )

    def test_a_non_numeric_hook_answer_cannot_break_the_sleep(self):
        self.assertRegex(self.text, r"''\|\*\[!0-9\]\*\) PROFILE_WARMUP_SEC=0")

    def test_no_hook_keeps_the_historical_zero_default(self):
        """A synthetic run has no hook, so it still arms at load start."""
        self.assertIn("PROFILE_WARMUP_SEC=${PROFILE_WARMUP_SEC:-0}", self.text)

    def test_a_missed_window_refuses_to_substitute_a_synthetic_profile(self):
        self.assertIn('elif [ "${GEAK_ISL_OSL_INACTIVE:-0}" = "1" ]; then', self.text)
        self.assertIn("not substituting a synthetic ISL/OSL profile", self.text)

    def test_the_refusal_says_how_to_fix_it(self):
        """An operator must not have to read this script to recover."""
        self.assertIn("lower AGENTX_PROFILE_WARMUP_S", self.text)
        self.assertIn("MEASUREMENT_PURPOSE=parity", self.text)

    def test_the_synthetic_fallback_survives_for_synthetic_runs(self):
        """Removing it for everyone would regress the fixed ISL/OSL path."""
        self.assertIn(
            'adapter_bench "$PROFILE_NUM_PROMPTS" "$CONC" 1 || echo "!!! profile run failed"',
            self.text,
        )


@unittest.skipIf(BASH is None, "bash is required to exercise the client adapter")
class ProfiledCallDoesNotSwapTheWorkloadTest(unittest.TestCase):
    """PROF=1 must not answer with a synthetic sweep behind the run's back."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="agentx_prof_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        aiperf = bin_dir / "aiperf"
        aiperf.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                art=""; prev=""
                for a in "$@"; do [ "$prev" = "--artifact-dir" ] && art="$a"; prev="$a"; done
                echo replay >> "${TEST_CALLS}"
                mkdir -p "$art"
                echo '{"metrics": {"output_token_throughput": {"avg": 1.0}}}' \
                  > "$art/profile_export_aiperf.json"
                """
            )
        )
        aiperf.chmod(0o755)
        self.calls = self.tmp / "calls.txt"
        self.calls.write_text("")
        self.result_jsonl = self.tmp / "results.jsonl"

    def _bench(self, prof: str, **env) -> str:
        run_env = dict(os.environ)
        run_env.update(
            {
                "PATH": f"{self.tmp / 'bin'}:{os.environ.get('PATH', '')}",
                "TEST_CALLS": str(self.calls),
                "RESULT_JSONL": str(self.result_jsonl),
                "OUT_DIR": str(self.tmp),
                "MODEL": "/models/Kimi-K3",
                "BASE_URL": "http://127.0.0.1:8000",
                "CONC": "8",
                "GEAK_AGENTX_LOOP_DURATION_S": "900",
            }
        )
        run_env.pop("INFERENCEX_PATH", None)
        run_env.update({k: str(v) for k, v in env.items()})
        # A stub native client, so a delegation would be unmistakable.
        script = (
            f'adapter_bench_native() {{ echo synthetic >> "{self.calls}"; }}\n'
            f'source "{ADAPTER}"\n'
            f'adapter_bench 1 8 {prof}\n'
        )
        proc = subprocess.run(
            [BASH, "-c", script], env=run_env, capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        return self.calls.read_text().split()

    def test_prof1_replays_the_trace_instead_of_delegating(self):
        self.assertEqual(self._bench("1"), ["replay"])

    def test_prof0_replays_the_trace(self):
        self.assertEqual(self._bench("0"), ["replay"])

    def test_the_synthetic_profile_is_reachable_only_by_explicit_opt_in(self):
        self.assertEqual(
            self._bench("1", AGENTX_PROFILE_VIA_SYNTHETIC=1), ["synthetic"]
        )

    def test_the_opt_in_says_the_trace_describes_another_workload(self):
        run_env = dict(os.environ)
        run_env.update(
            {
                "PATH": f"{self.tmp / 'bin'}:{os.environ.get('PATH', '')}",
                "TEST_CALLS": str(self.calls),
                "RESULT_JSONL": str(self.result_jsonl),
                "OUT_DIR": str(self.tmp),
                "MODEL": "/m",
                "AGENTX_PROFILE_VIA_SYNTHETIC": "1",
            }
        )
        proc = subprocess.run(
            [BASH, "-c",
             f'adapter_bench_native() {{ :; }}\nsource "{ADAPTER}"\nadapter_bench 1 8 1'],
            env=run_env, capture_output=True, text=True,
        )
        self.assertIn("NOT the trace replay", proc.stderr)


class VendoredMapperTest(unittest.TestCase):
    """A standalone run must not need an InferenceX checkout to read its result."""

    def test_the_mapper_ships_beside_the_client_that_calls_it(self):
        self.assertTrue(VENDORED_MAPPER.is_file())

    def test_the_adapter_prefers_a_real_checkout_and_falls_back_to_the_vendored_copy(self):
        text = ADAPTER.read_text()
        order = [
            text.index("${ix_root:+${ix_root}/benchmarks/map_aiperf.py}"),
            text.index("${ix_root:+${ix_root}/assets/agentx/map_aiperf.py}"),
            text.index('"${BASH_SOURCE[0]%/*}/map_aiperf.py"'),
        ]
        self.assertEqual(order, sorted(order),
                         "the vendored copy must be tried last")

    def test_an_unset_inferencex_path_does_not_expand_to_a_bare_suffix(self):
        """`${ix_root:+...}` keeps an unset root from yielding "/benchmarks/...",
        which could match an unrelated file at the filesystem root."""
        self.assertNotIn('"${ix_root}/benchmarks/map_aiperf.py"', ADAPTER.read_text())

    def test_the_mapper_runs_without_hyperloom_importable(self):
        """The inline fallback is what makes the vendored file self-sufficient."""
        export = {
            "metadata": {"submission_valid": True},
            "metrics": {
                "output_token_throughput": {"avg": 152.7},
                "input_token_throughput": {"avg": 20647.3},
                "request_count": 393,
                "time_to_first_token": {"avg": 900.0, "p50": 880.0},
            },
        }
        tmp = Path(tempfile.mkdtemp(prefix="mapper_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        src, dst = tmp / "export.json", tmp / "result.json"
        src.write_text(json.dumps(export))
        env = dict(os.environ, PYTHONPATH=str(tmp))  # no hyperloom on the path
        proc = subprocess.run(
            ["python3", str(VENDORED_MAPPER), str(src), str(dst)],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        result = json.loads(dst.read_text())
        # Both axes, because which one is graded is the caller's choice.
        self.assertAlmostEqual(result["output_throughput"], 152.7)
        self.assertAlmostEqual(result["total_token_throughput"], 20800.0)
        self.assertEqual(result["completed"], 393)

    def test_client_detected_deviations_force_the_verdict(self):
        """aiperf reports submission_valid=true on a 900s leg; the client knows better."""
        export = {"metadata": {"submission_valid": True}, "metrics": {}}
        tmp = Path(tempfile.mkdtemp(prefix="mapper_noncanon_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        src, dst = tmp / "export.json", tmp / "result.json"
        src.write_text(json.dumps(export))
        env = dict(
            os.environ,
            PYTHONPATH=str(tmp),
            AGENTX_NONCANONICAL_REASONS="duration=900s(canonical 3600s),entries=50(canonical 393)",
        )
        proc = subprocess.run(
            ["python3", str(VENDORED_MAPPER), str(src), str(dst)],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        result = json.loads(dst.read_text())
        self.assertIs(result["submission_valid"], False)
        self.assertEqual(len(result["submission_invalid_reasons"]), 2)


if __name__ == "__main__":
    unittest.main()
