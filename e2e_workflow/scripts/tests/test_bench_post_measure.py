# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle integration: live endpoint, phase ordering, binding and owned cleanup."""

import argparse
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench_lifecycle", SCRIPTS / "bench_lifecycle.py")
lifecycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lifecycle)


@unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir() and shutil.which("setsid"), "requires Linux process ownership")
class PostMeasureTest(unittest.TestCase):
    def setUp(self):
        # Interface emission tests may leave SIGTERM ignored in this process.
        # Bash preserves an inherited ignore, so give child fixtures the normal
        # launch signal state and restore the surrounding suite's state afterward.
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(sig, signal.SIG_DFL)
            self.addCleanup(signal.signal, sig, previous)
        self.tmp = tempfile.TemporaryDirectory(prefix="geak post measure ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.events = self.root / "events.jsonl"
        self.fixture = self.root / "fixture.py"
        self.fixture.write_text(textwrap.dedent(r'''
            import argparse, hashlib, http.server, json, os, pathlib, signal, subprocess, sys, time, urllib.request

            def event(kind, **extra):
                with open(os.environ["EVENT_LOG"], "a") as f:
                    f.write(json.dumps({"event": kind, "pid": os.getpid(), **extra}) + "\n")

            if sys.argv[1] == "server":
                class Handler(http.server.BaseHTTPRequestHandler):
                    def do_GET(self):
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(str(os.getpid()).encode())
                    def log_message(self, *args): pass
                server = http.server.HTTPServer(("127.0.0.1", int(sys.argv[2])), Handler)
                event("launch", start_ticks=int(pathlib.Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]))
                pathlib.Path(os.environ["SERVER_READY"]).write_text(str(server.server_port))
                server.serve_forever()
            elif sys.argv[1] == "callback":
                parser = argparse.ArgumentParser()
                parser.add_argument("--launch-context")
                parser.add_argument("--output-dir")
                args = parser.parse_args(sys.argv[2:])
                context = json.loads(pathlib.Path(args.launch_context).read_text())
                with urllib.request.urlopen(context["endpoint"]["base_url"], timeout=2) as r:
                    seen_pid = int(r.read())
                event("callback", launch_nonce=context["launch_nonce"], seen_pid=seen_pid)
                mode = os.environ.get("CALLBACK_MODE", "ok")
                output = pathlib.Path(args.output_dir)
                print("private evaluator output", flush=True)
                if mode in ("hang", "detach"):
                    child = subprocess.Popen([sys.executable, "-c", "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"])
                    (output / "child.pid").write_text(str(child.pid))
                if mode == "hang":
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    time.sleep(60)
                if mode == "fail": sys.exit(9)
                if mode == "missing": sys.exit(0)
                if mode == "invalid":
                    (output / "native_receipt.json").write_text("not JSON")
                    sys.exit(0)
                if mode == "fifo":
                    os.mkfifo(output / "native_receipt.json")
                    sys.exit(0)
                if mode == "fifo_sealed_summary":
                    sealed = output.parent / "throughput/bench_summary.json"
                    sealed.unlink()
                    os.mkfifo(sealed)
                if mode == "mutate":
                    (output.parents[1] / "bench_summary.json").write_text('{"throughput_tok_s_median": 999}')
                if mode in ("replace", "remove_seal"):
                    summary = output.parents[1] / "bench_summary.json"
                    summary.unlink()
                    summary.mkdir()
                    if mode == "remove_seal":
                        (output.parent / "measurement.json").unlink()
                (output / "native_receipt.json").write_text(json.dumps({"schema": "cpu-fixture.v1", "context_sha256": hashlib.sha256(pathlib.Path(args.launch_context).read_bytes()).hexdigest()}))
        ''').lstrip())
        self.adapter = self.root / "adapter.sh"
        self.adapter.write_text(textwrap.dedent(r'''
            adapter_default_port() { echo 18080; }
            adapter_launch() {
              rm -f "$SERVER_READY"
              if [ "${DISOWN_SERVER:-0}" = "1" ]; then
                # Same ownership shape as Magpie's server phase: launch, hand off PID, exit.
                bash -c 'setsid "$TEST_PYTHON" "$FIXTURE" server "$PORT" >"$LOG" 2>&1 & echo $! > "$OUT_DIR/server.pid"; disown'
                SERVER_PID="$(cat "$OUT_DIR/server.pid")"
              else
                ${SERVER_LAUNCH_PREFIX:-} "$TEST_PYTHON" "$FIXTURE" server "$PORT" >"$LOG" 2>&1 &
                SERVER_PID=$!
              fi
              for _ in $(seq 1 200); do [ -s "$SERVER_READY" ] && return 0; sleep 0.01; done
              return 2
            }
            adapter_health() { [ -s "$SERVER_READY" ]; }
            adapter_bench() {
              printf '{"event":"bench","server_pid":%s,"replica":%s,"attempt":%s}\n' "$SERVER_PID" "${REPLICA_INDEX:-0}" "${REPLICA_ATTEMPT:-0}" >> "$EVENT_LOG"
              case ",${FAIL_ATTEMPTS:-}," in *",${REPLICA_INDEX:-0}:${REPLICA_ATTEMPT:-0},"*) return 9 ;; esac
              printf '{"output_throughput":123,"median_ttft_ms":4,"median_tpot_ms":5}\n' >> "$RESULT_JSONL"
            }
        ''').lstrip())
        self.env = {"EVENT_LOG": str(self.events), "SERVER_READY": str(self.root / "ready"),
                    "TEST_PYTHON": sys.executable, "FIXTURE": str(self.fixture), "ADAPTER": str(self.adapter),
                    "MODEL": "fixture-model", "BACKEND": "fixture", "PROFILE": "0", "REUSE_SERVER": "0",
                    "SERVING_GPU_LOCK_DISABLE": "1", "SERVER_STOP_GRACE_S": "0", "EFFECTIVE_CONFIG_DIGEST": "a" * 64,
                    "NUM_PROMPTS": "3", "CONC": "1", "BENCH_CLIENT": "native", "MEASUREMENT_PURPOSE": "validation"}
        self.envpatch = patch.dict(os.environ, self.env)
        self.envpatch.start()
        self.addCleanup(self.envpatch.stop)
        self.addCleanup(self.reap_fixture_servers)

    def read_events(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []

    def cli(self, *args):
        with patch.object(sys, "argv", ["bench_lifecycle.py", *(str(arg) for arg in args)]):
            self.assertEqual(lifecycle.main(), 0)

    def reap_fixture_servers(self):
        for event in self.read_events():
            if event["event"] != "launch":
                continue
            current = lifecycle._proc(event["pid"])
            if current and current["start_ticks"] == event["start_ticks"]:
                try:
                    os.kill(event["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def request(self, **changes):
        request = {"schema": lifecycle.SCHEMA, "request_id": str(uuid.uuid4()), "measurement_epoch": str(uuid.uuid4()),
                   "contract_sha256": "b" * 64, "expected_config_sha256": "a" * 64,
                   "callback_argv": [sys.executable, str(self.fixture), "callback"], "timeout_sec": 5}
        request.update(changes)
        path = self.root / (uuid.uuid4().hex + " request.json")
        path.write_text(json.dumps(request))
        return path, request

    def shell(self, *, mode="warm_server", callback="ok", timeout=5, **env_changes):
        request_path, request = self.request(timeout_sec=timeout)
        out = self.root / uuid.uuid4().hex
        env = {**os.environ, "OUT_DIR": str(out), "GEAK_REPEAT_MODE": mode,
               "GEAK_POST_MEASURE_REQUEST": str(request_path), "CALLBACK_MODE": callback,
               "REPLICAS": "3", **env_changes}
        proc = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, capture_output=True, text=True, timeout=25, check=False)
        return proc, out, request

    def live_context(self, **changes):
        request_path, request = self.request(**changes)
        out = self.root / uuid.uuid4().hex
        self.cli("prepare", "--output-dir", out, "--request", request_path, "--mode", "warm_server")
        ready = Path(self.env["SERVER_READY"])
        ready.unlink(missing_ok=True)
        proc = subprocess.Popen([sys.executable, str(self.fixture), "server", "0"], start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(proc.wait, timeout=3)
        self.addCleanup(proc.kill)
        for _ in range(200):
            if ready.exists() and ready.stat().st_size:
                break
            time.sleep(0.01)
        self.assertTrue(ready.exists())
        current = lifecycle._proc(proc.pid)
        args = argparse.Namespace(pid=str(proc.pid), pgid=str(proc.pid), start_ticks=str(current["start_ticks"]),
                                  protected_pgids="1", group_unverified="0", base_url="http://127.0.0.1:" + ready.read_text(),
                                  replica_index=0, replica_attempt=0)
        self.cli("record", "--output-dir", out, "--pid", args.pid, "--pgid", args.pgid,
                 "--start-ticks", args.start_ticks, "--protected-pgids", "1")
        self.cli("ready", "--output-dir", out, "--base-url", args.base_url)
        (out / "bench_summary.json").write_text(json.dumps({"throughput_tok_s_median": 123, "runs": 1,
                                                            "effective_config_digest": "a" * 64}))
        (out / "bench_runs.jsonl").write_text('{"output_throughput":123}\n')
        self.cli("measurement-valid", "--output-dir", out)
        return out, request_path, request, proc

    def test_warm_callback_follows_all_timed_rounds_on_the_same_live_server(self):
        proc, out, _ = self.shell()
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        events = self.read_events()
        self.assertEqual([e["event"] for e in events], ["launch", "bench", "bench", "bench", "bench", "callback"])
        self.assertEqual(events[0]["pid"], events[-1]["seen_pid"])
        receipt = lifecycle._read(out / "post_measure_receipt.json")
        self.assertEqual(receipt["status"], "completed")
        for item in receipt["throughput_artifacts"]:
            self.assertEqual(lifecycle._digest(out / item["path"]), item["sha256"])
        self.assertEqual(lifecycle._read(out / "post_measure_cleanup.json")["status"], "recorded_group_gone")
        self.assertNotIn("private evaluator output", proc.stdout + proc.stderr)

    def test_isolated_quality_failure_never_retries_or_replaces_valid_throughput(self):
        proc, out, _ = self.shell(mode="isolated_server", callback="fail")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.cli("aggregate", "--output-dir", out)
        events = self.read_events()
        self.assertEqual(sum(e["event"] == "launch" for e in events), 3)
        self.assertEqual(sum(e["event"] == "callback" for e in events), 3)
        manifest = lifecycle._read(out / "post_measure_manifest.json")
        self.assertEqual([s["attempt"] for s in manifest["selected"]], [1, 1, 1])
        nonces = set()
        for selected in manifest["selected"]:
            receipt = lifecycle._read(out / selected["receipt"]["path"])
            self.assertEqual(receipt["status"], "failed")
            nonces.add(receipt["launch_nonce"])
        self.assertEqual(len(nonces), 3)
        self.assertEqual(lifecycle._read(out / "bench_summary.json")["successful"], 3)

    def test_throughput_retry_uses_new_launch_and_only_selected_attempt(self):
        proc, out, _ = self.shell(mode="isolated_server", FAIL_ATTEMPTS="1:1", REPLICAS="1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sum(e["event"] == "launch" for e in self.read_events()), 2)
        self.assertEqual(sum(e["event"] == "callback" for e in self.read_events()), 1)
        selected = lifecycle._read(out / "post_measure_manifest.json")["selected"]
        self.assertEqual([s["attempt"] for s in selected], [2])
        self.assertIn("attempt_2/", selected[0]["receipt"]["path"])

    def test_disowned_server_is_still_cleaned_by_existing_geak_owner(self):
        proc, out, _ = self.shell(DISOWN_SERVER="1", REPLICAS="1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(lifecycle._read(out / "post_measure_receipt.json")["status"], "completed")
        self.assertEqual(lifecycle._read(out / "post_measure_cleanup.json")["status"], "recorded_group_gone")

    def test_timeout_reaps_term_ignoring_callback_group_and_preserves_throughput(self):
        out, _, _, server = self.live_context(timeout_sec=0.4)
        before = lifecycle._digest(out / "bench_summary.json")
        with patch.dict(os.environ, {"CALLBACK_MODE": "hang"}):
            self.cli("run", "--output-dir", out)
        receipt = lifecycle._read(out / "post_measure_receipt.json")
        self.assertEqual(receipt["status"], "timed_out")
        self.assertEqual(receipt["callback_cleanup_status"], "recorded_group_gone")
        self.assertEqual(lifecycle._digest(out / "bench_summary.json"), before)
        self.assertIsNone(server.poll())
        child = int((out / "post_measure/output/child.pid").read_text())
        child_info = lifecycle._proc(child)
        self.assertTrue(child_info is None or child_info["state"] == "Z")

    def test_sigterm_during_callback_preserves_receipt_and_cleans_owned_server(self):
        request_path, _ = self.request(timeout_sec=15)
        out = self.root / "cancelled"
        env = {**os.environ, "OUT_DIR": str(out), "GEAK_REPEAT_MODE": "warm_server", "REPLICAS": "1",
               "GEAK_POST_MEASURE_REQUEST": str(request_path), "CALLBACK_MODE": "hang", "DISOWN_SERVER": "1"}
        proc = subprocess.Popen(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            child_file = out / "post_measure/output/child.pid"
            for _ in range(300):
                if child_file.exists():
                    break
                time.sleep(0.01)
            self.assertTrue(child_file.exists())
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=8)
            self.assertEqual(proc.returncode, 143, stdout + stderr)
            self.assertEqual(lifecycle._read(out / "post_measure_receipt.json")["status"], "cancelled")
            self.assertEqual(lifecycle._read(out / "post_measure_cleanup.json")["status"], "recorded_group_gone")
            self.assertEqual(lifecycle._read(out / "bench_summary.json")["throughput_tok_s_median"], 123)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=3)

    def test_missing_invalid_and_failed_callback_results_remain_explicit(self):
        for mode, status in (("missing", "missing_result"), ("invalid", "invalid_result_or_context"),
                             ("fifo", "invalid_result_or_context"), ("fail", "failed")):
            with self.subTest(mode=mode):
                out, _, _, _ = self.live_context()
                with patch.dict(os.environ, {"CALLBACK_MODE": mode}):
                    lifecycle.run_callback(out)
                self.assertEqual(lifecycle._read(out / "post_measure_receipt.json")["status"], status)

    def test_callback_cannot_change_linked_throughput_bytes(self):
        out, _, _, _ = self.live_context()
        with patch.dict(os.environ, {"CALLBACK_MODE": "mutate"}):
            lifecycle.run_callback(out)
        self.assertEqual(lifecycle._read(out / "post_measure_receipt.json")["status"], "binding_changed")
        self.assertEqual(lifecycle._read(out / "bench_summary.json")["throughput_tok_s_median"], 123)

    def test_exited_callback_child_is_reaped_while_supervisor_identity_is_alive(self):
        out, _, _, server = self.live_context()
        with patch.dict(os.environ, {"CALLBACK_MODE": "detach"}):
            lifecycle.run_callback(out)
        receipt = lifecycle._read(out / "post_measure_receipt.json")
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["callback_returncode"], 0)
        self.assertEqual(receipt["callback_cleanup_status"], "recorded_group_gone")
        child = lifecycle._proc(int((out / "post_measure/output/child.pid").read_text()))
        self.assertTrue(child is None or child["state"] == "Z")
        self.assertIsNone(server.poll())

    def test_isolated_callback_mutation_cannot_change_or_retry_throughput(self):
        proc, out, _ = self.shell(mode="isolated_server", callback="mutate", REPLICAS="2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = lifecycle._read(out / "bench_summary.json")
        self.assertEqual(summary["throughput_tok_s_median"], 123)
        selected = lifecycle._read(out / "post_measure_manifest.json")["selected"]
        self.assertEqual([item["attempt"] for item in selected], [1, 1])
        self.assertEqual(sum(e["event"] == "launch" for e in self.read_events()), 2)
        for item in selected:
            self.assertEqual(lifecycle._read(out / item["receipt"]["path"])["status"], "binding_changed")

    def test_failed_summary_restore_still_selects_sealed_throughput_without_retry(self):
        proc, out, _ = self.shell(mode="isolated_server", callback="replace", REPLICAS="1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sum(e["event"] == "launch" for e in self.read_events()), 1)
        self.assertEqual(lifecycle._read(out / "bench_summary.json")["throughput_tok_s_median"], 123)
        selected = lifecycle._read(out / "post_measure_manifest.json")["selected"][0]
        self.assertEqual(selected["attempt"], 1)
        self.assertEqual(lifecycle._read(out / selected["receipt"]["path"])["status"], "throughput_restore_failed")

    def test_missing_seal_and_failed_restore_cannot_purchase_another_attempt(self):
        proc, out, _ = self.shell(mode="isolated_server", callback="remove_seal", REPLICAS="1")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(sum(e["event"] == "launch" for e in self.read_events()), 1)
        self.assertEqual(sum(e["event"] == "callback" for e in self.read_events()), 1)
        self.assertEqual(lifecycle._read(out / "post_measure_manifest.json")["selected"], [])

    def test_fifo_and_device_artifacts_are_rejected_without_blocking(self):
        fifo = self.root / "result.fifo"
        os.mkfifo(fifo)
        for path in (fifo, Path("/dev/null")):
            for reader in (lifecycle._read, lifecycle._digest):
                with self.subTest(path=path, reader=reader.__name__), self.assertRaisesRegex(ValueError, "regular file"):
                    reader(path)

    def test_fifo_sealed_summary_does_not_block_selection_or_buy_retry(self):
        proc, out, _ = self.shell(mode="isolated_server", callback="fifo_sealed_summary", REPLICAS="1")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(sum(e["event"] == "launch" for e in self.read_events()), 1)
        self.assertEqual(sum(e["event"] == "callback" for e in self.read_events()), 1)
        self.assertEqual(lifecycle._read(out / "post_measure_manifest.json")["selected"], [])

    def test_changed_request_and_stale_output_are_rejected(self):
        out, path, _, _ = self.live_context()
        with self.assertRaises(FileExistsError):
            lifecycle.prepare(out, path, "warm_server")
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "request changed"):
            lifecycle.run_callback(out)

    def test_cleanup_requires_current_epoch_and_never_signals_a_reused_identity(self):
        out, _, request, server = self.live_context()
        with self.assertRaises(ValueError):
            lifecycle.cleanup(out, terminate=True, request_id="wrong", epoch=request["measurement_epoch"])
        owner_path = out / "post_measure/owner.json"
        owner = lifecycle._read(owner_path)
        owner["server_identity"]["start_ticks"] += 1
        lifecycle._write(owner_path, owner)
        lifecycle.cleanup(out, terminate=True, request_id=request["request_id"], epoch=request["measurement_epoch"])
        self.assertEqual(lifecycle._read(out / "post_measure_cleanup.json")["status"], "identity_unavailable")
        self.assertIsNone(server.poll())

    def test_endpoint_must_belong_to_the_recorded_server_group(self):
        first, _, _, first_server = self.live_context()
        second, _, _, second_server = self.live_context()
        owner = lifecycle._read(first / "post_measure/owner.json")["server_identity"]
        other_url = lifecycle._read(second / "post_measure/launch_context.json")["endpoint"]["base_url"]
        with self.assertRaisesRegex(ValueError, "listener"):
            lifecycle._listener(owner, other_url)
        for host in ("example.com", "localhost", "0.0.0.0", "[::1]"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "local IPv4"):
                lifecycle._listener(owner, f"http://{host}:1234")
        self.assertIsNone(first_server.poll())
        self.assertIsNone(second_server.poll())

    @unittest.skipUnless(hasattr(socket, "SO_REUSEPORT"), "requires SO_REUSEPORT")
    def test_endpoint_allows_same_group_listeners_but_rejects_foreign_reuseport(self):
        code = textwrap.dedent('''
            import pathlib, signal, socket, sys
            port, count = int(sys.argv[2]), int(sys.argv[3])
            listeners = []
            for _ in range(count):
                sock = socket.socket()
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                sock.bind(("127.0.0.1", port))
                port = sock.getsockname()[1]
                sock.listen()
                listeners.append(sock)
            pathlib.Path(sys.argv[1]).write_text(str(port))
            signal.pause()
        ''')

        def start_listener(port, count):
            ready = self.root / uuid.uuid4().hex
            proc = subprocess.Popen([sys.executable, "-c", code, str(ready), str(port), str(count)],
                                    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.addCleanup(proc.wait, timeout=3)
            self.addCleanup(proc.terminate)
            for _ in range(200):
                if ready.exists() and ready.stat().st_size:
                    break
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            return proc, int(ready.read_text())

        server, port = start_listener(0, 2)
        current = lifecycle._proc(server.pid)
        owner = lifecycle._identity(server.pid, server.pid, current["start_ticks"], [1, os.getpgrp()])
        endpoint = f"http://127.0.0.1:{port}"
        self.assertEqual(len(lifecycle._listener(owner, endpoint)["listeners"]), 2)
        foreign, _ = start_listener(port, 1)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            lifecycle._listener(owner, endpoint)
        self.assertIsNone(server.poll())
        self.assertIsNone(foreign.poll())

    def test_outer_cleanup_reuses_frozen_identity_after_owner_shell_is_gone(self):
        out, _, request, server = self.live_context()
        self.cli("cleanup", "--output-dir", out, "--request-id", request["request_id"],
                 "--measurement-epoch", request["measurement_epoch"])
        server.wait(timeout=3)
        self.assertEqual(lifecycle._read(out / "post_measure_cleanup.json")["status"], "recorded_group_gone")

    def test_unverified_protected_and_dead_owners_cannot_supply_launch_evidence(self):
        out, _, _, server = self.live_context()
        owner = lifecycle._read(out / "post_measure/owner.json")["server_identity"]
        with self.assertRaises(ValueError):
            lifecycle._identity(owner["pid"], owner["pgid"] + 1, owner["start_ticks"], [])
        with self.assertRaises(ValueError):
            lifecycle._identity(owner["pid"], owner["pgid"], owner["start_ticks"], [owner["pgid"]])
        with self.assertRaisesRegex(ValueError, "unverified"):
            lifecycle.record(out, argparse.Namespace(protected_pgids="1", group_unverified="1"))
        server.terminate()
        server.wait(timeout=3)
        with self.assertRaisesRegex(ValueError, "no longer alive"):
            lifecycle._observation(owner, "http://127.0.0.1:1234")

    def test_changed_capability_snapshot_and_unreadable_listener_are_rejected(self):
        out, _, _, _ = self.live_context()
        context = lifecycle._read(out / "post_measure/launch_context.json")
        with patch.object(os, "readlink", side_effect=PermissionError), self.assertRaisesRegex(ValueError, "listener"):
            lifecycle._listener(context["server_identity"], context["endpoint"]["base_url"])
        capsule_path = out / "post_measure/request.json"
        capsule = lifecycle._read(capsule_path)
        capsule["capabilities"]["sources"]["bench_lifecycle.py"] = "0" * 64
        lifecycle._write(capsule_path, capsule)
        with self.assertRaisesRegex(ValueError, "sources changed"):
            lifecycle._capsule(out)

    def test_unsupported_modes_and_invalid_requests_fail_before_launch(self):
        path, _ = self.request()
        for change in ({"PROFILE": "1"}, {"REUSE_SERVER": "1"}, {"BENCH_CLIENT": "agentx"}, {"EFFECTIVE_CONFIG_DIGEST": "c" * 64}):
            with self.subTest(change=change), patch.dict(os.environ, change), self.assertRaises(ValueError):
                lifecycle.prepare(self.root / uuid.uuid4().hex, path, "warm_server")
        with self.assertRaises(ValueError):
            lifecycle.prepare(self.root / "legacy", path, "legacy")
        for change in ({"timeout_sec": True}, {"timeout_sec": -1}, {"callback_argv": ["python3"]},
                       {"contract_sha256": "short"}, {"schema": "unknown"}, {"request_id": "not-uuid"}):
            invalid, _ = self.request(**change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                lifecycle.prepare(self.root / uuid.uuid4().hex, invalid, "warm_server")
        self.assertEqual(self.read_events(), [])

    def test_missing_output_directory_and_relative_request_fail_cli_validation(self):
        with patch.object(sys, "argv", ["bench_lifecycle.py", "prepare"]), patch("builtins.print"):
            self.assertEqual(lifecycle.main(), 3)
        with self.assertRaises(ValueError):
            lifecycle.prepare(self.root, "relative.json", "warm_server")
        lifecycle.cleanup(self.root / "no-owner")
        self.assertEqual(self.read_events(), [])

    def test_invalid_summary_and_oversized_artifact_do_not_start_callback(self):
        for filename, payload in (("bench_summary.json", '{"throughput_tok_s_median":0}'),
                                  ("bench_summary.json", '{"throughput_tok_s_median":123,"effective_config_digest":"wrong"}'),
                                  ("bench_runs.jsonl", "x" * (8 * lifecycle.MAX_JSON_BYTES + 1))):
            with self.subTest(filename=filename):
                out, _, _, _ = self.live_context()
                (out / filename).write_text(payload)
                lifecycle.run_callback(out)
                self.assertEqual(lifecycle._read(out / "post_measure_receipt.json")["status"], "invalid_result_or_context")
                self.assertFalse((out / "post_measure/output").exists())

    def test_supervisor_records_spawn_failure_and_keeps_its_cleanup_anchor_alive(self):
        out, _, _, _ = self.live_context()
        output = out / "post_measure/output"
        output.mkdir()
        with patch.object(lifecycle.subprocess, "call", side_effect=OSError), \
                patch.object(signal, "pause", side_effect=KeyboardInterrupt), \
                patch.object(sys, "argv", ["bench_lifecycle.py", "supervise", "--output-dir", str(out)]), \
                self.assertRaises(KeyboardInterrupt):
            lifecycle.main()
        self.assertIsNone(lifecycle._read(output / "callback_exit.json")["returncode"])

    def test_json_size_limit_is_enforced_before_parsing(self):
        source = self.root / "oversized.json"
        source.write_bytes(b" " * (lifecycle.MAX_JSON_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "size limit"):
            lifecycle._read(source)

    def test_capabilities_and_main_are_read_only(self):
        with patch.object(sys, "argv", ["bench_lifecycle.py", "capabilities"]), patch("builtins.print") as output:
            self.assertEqual(lifecycle.main(), 0)
        data = json.loads(output.call_args[0][0])
        self.assertEqual(data["schema"], lifecycle.SCHEMA)
        self.assertEqual(data["sources"]["bench_e2e.sh"], lifecycle._digest(SCRIPTS / "bench_e2e.sh"))
        self.assertEqual(self.read_events(), [])

    def test_staged_bundle_capabilities_require_exact_sibling_assets(self):
        staged = self.root / "staged bundle"
        staged.mkdir()
        for name in ("bench_e2e.sh", "bench_lifecycle.py", "bench_replica.sh", "server_teardown.sh", "bench_summarize.py"):
            shutil.copy2(SCRIPTS / name, staged / name)
        capability = subprocess.run(["bash", str(staged / "bench_e2e.sh"), "--post-measure-capabilities"],
                                    capture_output=True, text=True, timeout=5, check=False)
        self.assertEqual(capability.returncode, 0, capability.stderr)
        self.assertEqual(json.loads(capability.stdout)["schema"], lifecycle.SCHEMA)
        (staged / "server_teardown.sh").unlink()
        request, _ = self.request()
        env = {**os.environ, "GEAK_POST_MEASURE_REQUEST": str(request), "OUT_DIR": str(self.root / "unstaged")}
        rejected = subprocess.run(["bash", str(staged / "bench_e2e.sh")], env=env,
                                  capture_output=True, text=True, timeout=5, check=False)
        self.assertEqual(rejected.returncode, 3)
        self.assertEqual(self.read_events(), [])

    def test_json_contract_rejects_duplicates_nonfinite_values_and_symlinks(self):
        for payload in ('{"schema":1,"schema":2}', '{"value":NaN}', '[]'):
            source = self.root / "invalid.json"
            source.write_text(payload)
            with self.subTest(payload=payload), self.assertRaises((ValueError, TypeError)):
                lifecycle._read(source)
        target = self.root / "target.json"
        target.write_text('{}')
        link = self.root / "link.json"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            lifecycle._artifact(link, self.root)


if __name__ == "__main__":
    unittest.main()
