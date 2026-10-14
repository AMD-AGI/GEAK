"""Exercise the shell dispatcher and real AgentX adapter with a recording client."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("bash"), "requires bash")
class AgentXMeasurementLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="agentx_lifecycle_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.events = self.root / "events.jsonl"
        self.adapter = self.root / "backend.sh"
        self.adapter.write_text('''
adapter_default_port() { echo 18082; }
adapter_launch() {
  printf '{"event":"launch"}\\n' >> "$TEST_EVENTS"
  ${SERVER_LAUNCH_PREFIX:-} sleep 300 &
  SERVER_PID=$!
}
adapter_health() { return 0; }
adapter_bench() { echo 'Unexpected native client' >&2; return 9; }
''')
        bindir = self.root / "bin"
        bindir.mkdir()
        self.client = bindir / "aiperf"
        self.client.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ['TEST_EVENTS'], 'a') as f:
    f.write(json.dumps({'event':'client', 'argv':sys.argv[1:]})+'\\n')
if os.environ.get('TEST_FAIL_CLIENT') == '1':
    raise SystemExit(9)
out = pathlib.Path(sys.argv[sys.argv.index('--artifact-dir')+1])
out.mkdir(parents=True, exist_ok=True)
(out/'profile_export_aiperf.json').write_text('{}')
''')
        self.client.chmod(0o755)
        curl = bindir / "curl"
        curl.write_text('#!/bin/sh\nprintf \'{"data":[{"id":"test-model"}]}\\n\'\n')
        curl.chmod(0o755)
        mapper = self.root / "benchmarks" / "map_aiperf.py"
        mapper.parent.mkdir()
        mapper.write_text('''import json, sys
with open(sys.argv[2], 'w') as f:
    json.dump({'output_throughput':100, 'median_ttft_ms':10, 'median_tpot_ms':1}, f)
''')

    def run_bench(self, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ("GEAK_", "BENCH_", "AGENTX_", "AIPERF_", "REPLICA", "WARM_SERVER_"))}
        env.pop("REPEATS", None)
        env.update(
            PATH=f"{self.root / 'bin'}:{env['PATH']}",
            ADAPTER=str(self.adapter), BACKEND="test", BENCH_CLIENT="agentx",
            AIPERF_BIN=str(self.client), INFERENCEX_PATH=str(self.root),
            MODEL="/test/model", OUT_DIR=str(self.root / "out"),
            TEST_EVENTS=str(self.events), MEASUREMENT_PURPOSE="parity",
            CONC="8", ISL="0", OSL="0", PROFILE="0", REUSE_SERVER="0",
            SERVER_STOP_GRACE_S="0", SERVING_GPU_LOCK_DISABLE="1",
            GEAK_AGENTX_DURATION_S="3600", GEAK_AGENTX_LOOP_DURATION_S="900",
        )
        env.update({k: str(v) for k, v in overrides.items()})
        proc = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")],
                              env=env, capture_output=True, text=True, timeout=30)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        summary = json.loads((self.root / "out" / "bench_summary.json").read_text())
        return proc, events, summary

    def assert_one_replay(self, proc, events, summary, duration="3600"):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual([e['event'] for e in events], ['launch', 'client'])
        argv = events[1]['argv']
        for key, value in [('--benchmark-duration', duration),
                           ('--warmup-requests-per-lane', '10'),
                           ('--warmup-grace-period', '1800')]:
            self.assertEqual(argv[argv.index(key)+1], value)
        self.assertEqual(summary['measurement_mode'], 'isolated_server')
        self.assertEqual(summary['requested_replicas'], 1)
        self.assertTrue(summary['usable_for_acceptance'])

    def test_baseline_keeps_client_warmup_and_calls_aiperf_once(self):
        self.assert_one_replay(*self.run_bench())

    def test_search_retains_its_configured_window(self):
        self.assert_one_replay(*self.run_bench(MEASUREMENT_PURPOSE="search"), duration="900")

    def test_validation_defaults_to_one_replay(self):
        self.assert_one_replay(*self.run_bench(MEASUREMENT_PURPOSE="validation"))

    def test_validation_pin_does_not_recursively_schedule_replicas(self):
        self.assert_one_replay(*self.run_bench(
            MEASUREMENT_PURPOSE="validation", GEAK_REPEAT_MODE="warm_server",
            GEAK_VALIDATION_REPEAT_MODE="isolated_server"))

    def test_explicit_replica_count_uses_fresh_servers(self):
        proc, events, summary = self.run_bench(REPLICAS=2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([e['event'] for e in events], ['launch', 'client'] * 2)
        self.assertEqual(summary['successful_replicas'], 2)

    def test_client_warmup_overrides_are_preserved(self):
        proc, events, _ = self.run_bench(AGENTX_WARMUP_REQUESTS_PER_LANE=4,
                                        AGENTX_WARMUP_GRACE_PERIOD=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        argv = events[1]['argv']
        self.assertEqual(argv[argv.index('--warmup-requests-per-lane')+1], '4')
        self.assertEqual(argv[argv.index('--warmup-grace-period')+1], '120')

    def test_existing_explicit_warm_server_experiment_still_runs_two_passes(self):
        proc, events, summary = self.run_bench(GEAK_REPEAT_MODE="warm_server")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([e['event'] for e in events], ['launch', 'client', 'client'])
        self.assertEqual(summary['measurement_mode'], 'warm_server')

    def test_failed_replay_retries_on_a_new_server_and_cannot_pass(self):
        proc, events, summary = self.run_bench(TEST_FAIL_CLIENT=1, REPLICA_RETRIES=1)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual([e['event'] for e in events], ['launch', 'client'] * 2)
        self.assertFalse(summary['usable_for_acceptance'])


if __name__ == '__main__':
    unittest.main()
