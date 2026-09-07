#!/usr/bin/env python3
"""The AgentX client must state its own workload deviations.

The aiperf SCENARIO cannot police canonicality for us: it has no concept of
corpus size, its allowlist admits every dated weka variant, and
``--unsafe-override`` only flips ``submission_valid`` when the override actually
suppressed a violation. GEAK's inner search loop deliberately runs the 900s
scenario floor rather than the canonical 3600s window, so without a stamp from
the client every search-leg number would come back looking leaderboard-valid.

These tests pin that the client computes ``AGENTX_NONCANONICAL_REASONS`` itself
and hands it to map_aiperf.py, and that a canonical run stays unstamped.
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
ADAPTER = (
    Path(__file__).resolve().parents[1] / "adapters" / "clients" / "agentx.sh"
)


@unittest.skipIf(BASH is None, "bash is required to exercise the client adapter")
class AgentXClientCanonicalityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="agentx_client_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        root = Path(self.tmp)

        # Stub aiperf: only needs to honour --artifact-dir and record its argv.
        bin_dir = root / "bin"
        bin_dir.mkdir()
        self.aiperf = bin_dir / "aiperf"
        self.aiperf.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                art=""
                prev=""
                for a in "$@"; do
                  [ "$prev" = "--artifact-dir" ] && art="$a"
                  prev="$a"
                done
                printf '%s\\n' "$@" > "${TEST_ARGV_LOG}"
                mkdir -p "$art"
                echo '{"records": {}}' > "$art/profile_export_aiperf.json"
                """
            )
        )
        self.aiperf.chmod(0o755)

        # Stub mapper: records the canonicality verdict it was handed.
        bench = root / "ix" / "benchmarks"
        bench.mkdir(parents=True)
        (bench / "map_aiperf.py").write_text(
            textwrap.dedent(
                """\
                import json, os, sys
                json.dump(
                    {
                        "output_throughput": 169.0,
                        "noncanonical_reasons_seen": os.environ.get(
                            "AGENTX_NONCANONICAL_REASONS", "<absent>"
                        ),
                    },
                    open(sys.argv[2], "w"),
                )
                """
            )
        )
        self.result_jsonl = root / "results.jsonl"
        self.argv_log = root / "argv.txt"

    def _run(self, **env_overrides) -> dict:
        env = dict(os.environ)
        env.update(
            {
                "PATH": f"{Path(self.tmp) / 'bin'}:{env.get('PATH', '')}",
                "TEST_ARGV_LOG": str(self.argv_log),
                "INFERENCEX_PATH": str(Path(self.tmp) / "ix"),
                "RESULT_JSONL": str(self.result_jsonl),
                "OUT_DIR": self.tmp,
                "MODEL": "/models/Kimi-K3",
                "BASE_URL": "http://127.0.0.1:8000",
                "AGENTX_DATASET": "semianalysis_cc_traces_weka_062126",
                "AGENTX_CANONICAL_DATASET": "semianalysis_cc_traces_weka_062126",
                "AGENTX_NUM_ENTRIES": "393",
                "GEAK_AGENTX_DURATION_S": "3600",
                "GEAK_AGENTX_LOOP_DURATION_S": "900",
                "CONC": "8",
            }
        )
        # A stale value from the orchestrator's environment must never survive.
        env.pop("AGENTX_NONCANONICAL_REASONS", None)
        env.update({k: str(v) for k, v in env_overrides.items()})
        proc = subprocess.run(
            [BASH, "-c", f'source "{ADAPTER}"; adapter_bench 1 8 0'],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
        line = self.result_jsonl.read_text().strip().splitlines()[-1]
        record = json.loads(line)
        record["_argv"] = self.argv_log.read_text().splitlines()
        record["_stderr"] = proc.stderr
        return record

    def test_search_leg_at_the_scenario_floor_is_stamped_non_canonical(self):
        """The 900s search window is NOT a leaderboard measurement."""
        rec = self._run(MEASUREMENT_PURPOSE="search")
        reasons = rec["noncanonical_reasons_seen"]
        self.assertIn("duration=900s", reasons)
        self.assertIn("canonical 3600s", reasons)
        # And it must have told the operator, not just the mapper.
        self.assertIn("NON-CANONICAL", rec["_stderr"])

    def test_search_leg_opts_into_the_override_the_floor_requires(self):
        rec = self._run(MEASUREMENT_PURPOSE="search")
        self.assertIn("--unsafe-override", rec["_argv"])
        self.assertIn("900", rec["_argv"])

    def test_parity_runs_the_canonical_window_and_is_not_stamped(self):
        rec = self._run(MEASUREMENT_PURPOSE="parity")
        self.assertEqual(rec["noncanonical_reasons_seen"], "")
        self.assertIn("3600", rec["_argv"])

    def test_validation_also_runs_the_canonical_window(self):
        rec = self._run(MEASUREMENT_PURPOSE="validation")
        self.assertEqual(rec["noncanonical_reasons_seen"], "")
        self.assertIn("3600", rec["_argv"])

    def test_a_pinned_non_canonical_corpus_is_stamped(self):
        rec = self._run(
            MEASUREMENT_PURPOSE="parity",
            AGENTX_DATASET="semianalysis_cc_traces_weka_061526",
        )
        self.assertIn("corpus=semianalysis_cc_traces_weka_061526", rec["noncanonical_reasons_seen"])

    def test_a_shrunken_corpus_is_stamped_even_at_canonical_duration(self):
        rec = self._run(MEASUREMENT_PURPOSE="parity", AGENTX_NUM_ENTRIES="50")
        self.assertIn("entries=50", rec["noncanonical_reasons_seen"])
        self.assertIn("canonical 393", rec["noncanonical_reasons_seen"])

    def test_a_stale_inherited_verdict_never_survives_into_a_canonical_run(self):
        """Pass-through would let a previous leg's stamp mark a clean run dirty."""
        rec = self._run(
            MEASUREMENT_PURPOSE="parity",
            AGENTX_NONCANONICAL_REASONS="duration=900s(canonical 3600s)",
        )
        self.assertEqual(rec["noncanonical_reasons_seen"], "")

    def test_client_context_cap_is_stamped_and_forwarded(self):
        rec = self._run(MEASUREMENT_PURPOSE="parity", AGENTX_MAX_CTX="262144")
        self.assertIn("client_context_cap=262144", rec["noncanonical_reasons_seen"])
        self.assertIn("--max-context-length", rec["_argv"])

    def test_the_replay_uses_the_scenario_corpus_and_concurrency_it_was_given(self):
        rec = self._run(MEASUREMENT_PURPOSE="parity")
        argv = rec["_argv"]
        self.assertIn("inferencex-agentx-mvp", argv)
        self.assertIn("semianalysis_cc_traces_weka_062126", argv)
        self.assertIn("--concurrency", argv)
        self.assertIn("8", argv)
        # Trace replay: the synthetic sweep knobs must not appear at all.
        for synthetic in ("--random-input-len", "--random-output-len", "--num-prompts"):
            self.assertNotIn(synthetic, argv)

    def test_inherited_aiperf_env_is_scrubbed_before_the_replay(self):
        """An AIPERF_* value from the caller must not reconfigure the replay.

        The orchestrator forwards its whole environment, so a knob left over
        from a previous run would otherwise silently change this measurement.
        """
        rec = self._run(
            MEASUREMENT_PURPOSE="parity",
            AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="1",
            AIPERF_TOKENIZER_TRUST_REMOTE_CODE="bogus",
        )
        # The adapter re-exports the knobs it owns at their intended values.
        self.assertEqual(rec["noncanonical_reasons_seen"], "")
        self.assertIn("--use-server-token-count", rec["_argv"])

    def test_missing_aiperf_fails_before_it_can_report_a_number(self):
        rec_env = dict(os.environ)
        rec_env.update(
            {
                "PATH": "/nonexistent",
                "INFERENCEX_PATH": str(Path(self.tmp) / "ix"),
                "RESULT_JSONL": str(self.result_jsonl),
                "OUT_DIR": self.tmp,
                "MODEL": "/models/Kimi-K3",
                "MEASUREMENT_PURPOSE": "parity",
            }
        )
        proc = subprocess.run(
            [BASH, "-c", f'source "{ADAPTER}"; adapter_bench 1 8 0'],
            env=rec_env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(
            self.result_jsonl.exists() and self.result_jsonl.read_text().strip(),
            msg="a failed replay must not append a result line",
        )


if __name__ == "__main__":
    unittest.main()
