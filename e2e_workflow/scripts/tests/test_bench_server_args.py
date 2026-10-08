# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise removal proof before measurement through the real shell lifecycle.

Framework entrypoints are CPU sleepers; native and Magpie launch adapters, the
dispatcher, live /proc verification, and teardown are production code. Health
and measurement are local stubs, so these tests need no GPUs or listening ports.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or BASH is None,
    reason="shell lifecycle verification requires bash and Linux /proc",
)

SERVER = """import json, os, signal, time
from pathlib import Path
# Recovery tests can leave SIGTERM ignored in the pytest parent. A real server
# installs its own handlers; this sleeper must also reset inherited disposition.
signal.signal(signal.SIGTERM, signal.SIG_DFL)
pid = os.getpid()
proc = Path('/proc') / str(pid)
record = {
    'event': 'launch', 'pid': pid,
    'start_ticks': proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()[19],
    'argv': [s.decode() for s in proc.joinpath('cmdline').read_bytes().split(b'\\0')[:-1]],
}
with open(os.environ['STUB_EVENTS'], 'a') as stream:
    stream.write(json.dumps(record) + '\\n')
Path(os.environ['STUB_READY']).write_text(json.dumps(record))
time.sleep(120)
"""

MEASURE = """import json, os
from pathlib import Path
record = json.loads(Path(os.environ['STUB_READY']).read_text())
pid = record['pid']
argv = [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0')[:-1]]
receipt = Path(os.environ['OUT_DIR']) / 'server_args_validation.json'
event = {'event': 'bench', 'pid': pid, 'argv': argv,
         'validation': json.loads(receipt.read_text()) if receipt.exists() else None}
with open(os.environ['STUB_EVENTS'], 'a') as stream:
    stream.write(json.dumps(event) + '\\n')
with open(os.environ['RESULT_JSONL'], 'a') as stream:
    stream.write(json.dumps({'output_throughput': 123, 'median_ttft_ms': 4,
                            'median_tpot_ms': 5}) + '\\n')
"""


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


@dataclass
class Run:
    process: subprocess.CompletedProcess
    out: Path

    @property
    def events(self):
        path = self.out / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def read(self, name):
        return json.loads((self.out / name).read_text())

    def assert_measured(self, *, verified):
        assert self.process.returncode == 0, self.process.stdout + self.process.stderr
        benches = [event for event in self.events if event["event"] == "bench"]
        # A full untimed warmup and one measured round use the same verified PID.
        assert len(benches) == 2
        assert benches[0]["pid"] == benches[1]["pid"]
        assert benches[0]["argv"] == benches[1]["argv"]
        if verified:
            assert all(event["validation"]["status"] == "verified" for event in benches)
            assert all(event["validation"]["argv"] == event["argv"] for event in benches)
        else:
            assert all(event["validation"] is None for event in benches)
        assert self.read("bench_summary.json")["throughput_tok_s_median"] == 123
        return benches[0]["argv"]

    def assert_no_measurement(self):
        assert self.process.returncode != 0, self.process.stdout + self.process.stderr
        assert not any(event["event"] == "bench" for event in self.events)
        assert not (self.out / "bench_summary.json").exists()
        assert (self.out / "bench_runs.jsonl").read_text() == ""

    def assert_owned_server_stopped(self, *, expected_launches=1):
        launches = [event for event in self.events if event["event"] == "launch"]
        assert len(launches) == expected_launches
        for launch in launches:
            identity = _process_identity(launch["pid"])
            assert identity is None or identity[0] == "Z" or identity[1] != launch["start_ticks"]


@pytest.fixture
def lifecycle(tmp_path):
    # Magpie disowns its server. Adopt only for reaping; production teardown must
    # stop it first, which each launch test asserts before fixture cleanup runs.
    libc = ctypes.CDLL(None, use_errno=True)
    previous_subreaper = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    children = []
    outputs = []
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "python3"):
        (bin_dir / name).symlink_to(sys.executable)
    for relative in ("sglang/launch_server.py", "vllm/entrypoints/openai/api_server.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SERVER)
        parent = path.parent
        while parent != tmp_path:
            (parent / "__init__.py").touch()
            parent = parent.parent
    vllm = bin_dir / "vllm"
    vllm.write_text("#!" + sys.executable + "\n" + SERVER)
    vllm.chmod(0o755)
    measure = tmp_path / "measure.py"
    measure.write_text(MEASURE)
    adapter = tmp_path / "adapter.sh"
    adapter.write_text('''source "$STUB_SCRIPTS/adapters/$BACKEND.sh"
adapter_health() {
  local attempt
  for attempt in $(seq 1 100); do
    [ -s "$STUB_READY" ] && return 0
    sleep 0.01
  done
  return 1
}
adapter_bench() { python3 "$STUB_MEASURE"; }
''')
    magpie = tmp_path / "magpie.sh"
    magpie.write_text('''#!/usr/bin/env bash
set -euo pipefail
rm -f "$STUB_READY"
extra_name="EXTRA_${BACKEND^^}_ARGS"
# An external script owns this default. It intentionally knows nothing about
# GEAK_REMOVE_ARGS; the dispatcher must detect when the default survives.
mapfile -d '' -t extra < <(python3 - "${!extra_name}" <<'PY'
import shlex, sys
for token in shlex.split(sys.argv[1]):
    sys.stdout.buffer.write(token.encode() + b'\\0')
PY
)
if [ "$BACKEND" = sglang ]; then
  command=(python -m sglang.launch_server --model-path "$MODEL")
else
  command=(vllm serve "$MODEL")
fi
setsid "${command[@]}" --host 127.0.0.1 --port "$PORT" \\
  --script-default old "${extra[@]}" > "$SERVER_LOG" 2>&1 &
printf '%s\\n' "$!" > "$MAGPIE_SERVER_PID_FILE"
disown
''')
    env = {
        "PATH": str(bin_dir) + os.pathsep + os.defpath,
        "PYTHONPATH": str(tmp_path),
        "ADAPTER": str(adapter), "STUB_SCRIPTS": str(SCRIPTS),
        "STUB_MEASURE": str(measure), "MAGPIE_LAUNCH_SCRIPT": str(magpie),
        "MODEL": "/cpu-only/model", "HOST": "127.0.0.1", "PORT": "18080",
        "PORT_ENFORCE_RANGE": "0", "TP": "1", "GPU": "0", "GPU_ARCHS": "cpu_stub",
        "MEM_FRACTION": "0.9", "BENCH_CLIENT": "native", "PROFILE": "0",
        "GEAK_REPEAT_MODE": "warm_server", "REPEATS": "1", "REUSE_SERVER": "0",
        "NUM_PROMPTS": "2", "CONC": "1", "ISL": "1", "OSL": "1",
        "SERVING_GPU_LOCK_DISABLE": "1", "SERVER_STOP_GRACE_S": "0",
        "SERVER_STARTUP_TIMEOUT_SEC": "1", "SGLANG_SRC_PYTHONPATH": "",
    }

    def run(backend="sglang", launcher="native", prepare=None, scripts_dir=SCRIPTS, **updates):
        out = tmp_path / f"run-{len(outputs)}"
        out.mkdir()
        if prepare is not None:
            prepare(out)
        outputs.append(out)
        run_env = dict(env, BACKEND=backend, BENCH_LAUNCHER=launcher,
                       OUT_DIR=str(out), STUB_EVENTS=str(out / "events.jsonl"),
                       STUB_READY=str(out / "ready.json"))
        run_env.update(updates)
        result = subprocess.run([BASH, str(scripts_dir / "bench_e2e.sh")], env=run_env,
                                cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False)
        return Run(result, out)

    def reusable(backend, flags):
        out = tmp_path / f"external-{len(children)}"
        out.mkdir()
        ready = out / "ready.json"
        child_env = dict(env, STUB_EVENTS=str(out / "events.jsonl"), STUB_READY=str(ready))
        command = ([str(bin_dir / "python"), "-m", "sglang.launch_server"]
                   if backend == "sglang" else [str(vllm), "serve", env["MODEL"]])
        process = subprocess.Popen(command + ["--port", "18080", *flags], env=child_env)
        children.append(process)
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline, "CPU server did not become ready"
            time.sleep(0.01)
        record = json.loads(ready.read_text())
        return process, ready, record

    run.reusable = reusable
    try:
        yield run
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
        for out in outputs:
            event_path = out / "events.jsonl"
            events = event_path.read_text().splitlines() if event_path.exists() else []
            for line in events:
                event = json.loads(line)
                if event["event"] != "launch":
                    continue
                pid = event["pid"]
                identity = _process_identity(pid)
                if identity is not None and identity[1] == event["start_ticks"]:
                    if identity[0] != "Z":
                        os.kill(pid, signal.SIGKILL)
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass  # Native children are already reaped by the shell.
        assert libc.prctl(36, previous_subreaper.value, 0, 0, 0) == 0


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
@pytest.mark.parametrize("removal,current", [
    (["--disabled"], "--keep yes --limit=-1"),
    (["--changed old"], "--changed new"),
    (["--disabled"], ""),
])
def test_supported_removals_measure_exact_surviving_argv(lifecycle, backend, launcher, removal, current):
    run = lifecycle(backend, launcher, GEAK_REMOVE_ARGS=json.dumps(removal), EXTRA_SERVER_ARGS=current)
    argv = run.assert_measured(verified=True)
    expected = (["--model-path", "/cpu-only/model"] if backend == "sglang" else [])
    expected += ["--host", "127.0.0.1", "--port", "18080"]
    if launcher == "native":
        expected += (["--tp-size", "1", "--mem-fraction-static", "0.9", "--watchdog-timeout", "600"]
                     if backend == "sglang" else ["--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.9"])
    else:
        expected += ["--script-default", "old"]
    expected += current.split()
    first_flag = argv.index("--model-path" if backend == "sglang" else "--host")
    assert argv[first_flag:] == expected
    assert run.read("server_args_validation.json")["validation_kind"] == "launch"
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
def test_reintroduced_defaults_fail_before_warmup_and_cleanup(lifecycle, backend, launcher):
    removed = ("--script-default" if launcher == "magpie" else
               "--mem-fraction-static" if backend == "sglang" else "--gpu-memory-utilization")
    run = lifecycle(backend, launcher, GEAK_REMOVE_ARGS=json.dumps([removed]), EXTRA_SERVER_ARGS="--keep yes")
    run.assert_no_measurement()
    receipt = run.read("server_args_validation.json")
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "removal_mismatch"
    assert receipt["violations"] == [removed]
    assert run.read("server_start.json")["reason"] == "server_args_unverified"
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
def test_key_value_removal_rejects_ambiguous_repeated_default(lifecycle, backend, launcher):
    flag = ("--script-default" if launcher == "magpie" else
            "--mem-fraction-static" if backend == "sglang" else "--gpu-memory-utilization")
    old, new = ("old", "new") if launcher == "magpie" else ("0.9", "0.8")
    run = lifecycle(backend, launcher, GEAK_REMOVE_ARGS=json.dumps([f"{flag} {old}"]),
                    EXTRA_SERVER_ARGS=f"{flag} {new}")
    run.assert_no_measurement()
    receipt = run.read("server_args_validation.json")
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "ambiguous_repeated_option"
    assert receipt["ambiguous_repeated_options"] == [f"{flag} {old}"]
    argv = receipt["argv"]
    values = [argv[index + 1] for index, token in enumerate(argv) if token == flag]
    assert values == [old, new]
    assert run.read("server_args_validation.json")["active_remove_args"] == [f"{flag} {old}"]
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
def test_key_value_removal_allows_single_replacement_value(lifecycle, backend, launcher):
    run = lifecycle(backend, launcher, GEAK_REMOVE_ARGS='["--replaced old"]',
                    EXTRA_SERVER_ARGS="--replaced new")
    argv = run.assert_measured(verified=True)
    assert argv.count("--replaced") == 1
    assert argv[-2:] == ["--replaced", "new"]
    assert run.read("server_args_validation.json")["active_remove_args"] == ["--replaced old"]
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
@pytest.mark.parametrize("controls", ["not-json", '{"flag":"--disabled"}', '["--broken=\\"unclosed"]'])
def test_malformed_controls_fail_before_launch(lifecycle, backend, launcher, controls):
    run = lifecycle(backend, launcher, prepare=_seed_positive_artifacts, GEAK_REMOVE_ARGS=controls)
    run.assert_no_measurement()
    assert run.events == []
    assert run.read("server_start.json")["status"] == "failed"
    assert run.read("server_start.json")["reason"] == "server_args_unverified"


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
@pytest.mark.parametrize("controls,current", [
    ({}, "--keep yes"),
    ({"GEAK_REMOVE_ARGS": "[]"}, ""),
    ({"GEAK_REMOVE_ARGS": '["--keep"]'}, "--keep yes"),
])
def test_legacy_empty_controls_and_explicit_readdition(lifecycle, backend, launcher, controls, current):
    run = lifecycle(backend, launcher, EXTRA_SERVER_ARGS=current, **controls)
    argv = run.assert_measured(verified=False)
    if current:
        assert argv[-2:] == ["--keep", "yes"]
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_ambiguous_native_argv_fails_before_measurement(lifecycle, backend):
    run = lifecycle(backend, GEAK_REMOVE_ARGS='["--disabled"]', EXTRA_SERVER_ARGS="--note 'two words'")
    run.assert_no_measurement()
    assert run.read("server_args_validation.json")["status"] == "failed"
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
def test_remote_endpoint_cannot_use_local_process_proof(lifecycle, backend, launcher):
    run = lifecycle(backend, launcher, HOST="192.0.2.1", GEAK_REMOVE_ARGS='["--disabled"]')
    run.assert_no_measurement()
    assert run.read("server_args_validation.json")["reason"] == "unsupported_host"
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("receipt_kind", ["absent", "stale", "valid"])
def test_reuse_requires_live_matching_proof_without_taking_ownership(lifecycle, tmp_path, backend, receipt_kind):
    process, ready, record = lifecycle.reusable(backend, ["--keep", "yes"])
    receipt = tmp_path / "launch-receipt.json"
    if receipt_kind != "absent":
        result = subprocess.run([
            sys.executable, str(SCRIPTS / "adapters/server_args.py"), "validate",
            "--pid", str(process.pid), "--start-ticks", record["start_ticks"],
            "--backend", backend, "--port", "18080", "--remove-args=[\"--disabled\"]",
            "--current-args=--keep yes", "--receipt", str(receipt),
        ], capture_output=True, text=True, timeout=5, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        if receipt_kind == "stale":
            previous = json.loads(receipt.read_text())
            previous["observed_start_ticks"] = str(int(record["start_ticks"]) + 1)
            receipt.write_text(json.dumps(previous))
    run = lifecycle(backend, REUSE_SERVER="1", STUB_READY=str(ready),
                    GEAK_SERVER_ARGS_RECEIPT=str(receipt), GEAK_REMOVE_ARGS='["--disabled"]',
                    EXTRA_SERVER_ARGS="--keep yes")
    if receipt_kind == "valid":
        assert run.assert_measured(verified=True) == record["argv"]
        assert run.read("server_args_validation.json")["validation_kind"] == "reuse"
    else:
        run.assert_no_measurement()
        assert run.read("server_args_validation.json")["status"] == "failed"
        assert run.read("server_start.json")["reason"] == "server_args_unverified"
    assert not any(event["event"] == "launch" for event in run.events)
    assert "Shutting down server" not in run.process.stdout
    assert process.poll() is None, "Reusing a server must not transfer teardown ownership"
    assert _process_identity(process.pid)[1] == record["start_ticks"]


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_legacy_reuse_without_removals_does_not_require_receipt(lifecycle, backend):
    process, ready, record = lifecycle.reusable(backend, [])
    run = lifecycle(backend, REUSE_SERVER="1", STUB_READY=str(ready))
    assert run.assert_measured(verified=False) == record["argv"]
    assert not any(event["event"] == "launch" for event in run.events)
    assert process.poll() is None


def _seed_positive_artifacts(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_summary.json").write_text(json.dumps({
        "throughput_tok_s_median": 9999, "runs": 1, "usable_for_acceptance": True,
    }))
    (out / "server_args_validation.json").write_text(json.dumps({
        "schema_version": "geak.server_args_validation.v1", "status": "verified",
        "reason": "stale_previous_success",
    }))
    (out / "server_start.json").write_text('{"status":"ok","reason":"stale_success"}')
    (out / "bench_runs.jsonl").write_text('{"output_throughput":9999}\n')


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
@pytest.mark.parametrize("failure", ["invalid_controls", "removed_default"])
def test_fresh_failure_cannot_leave_previous_positive_artifacts(lifecycle, backend, launcher, failure):
    removed = ("--script-default" if launcher == "magpie" else
               "--mem-fraction-static" if backend == "sglang" else "--gpu-memory-utilization")
    controls = "invalid-json" if failure == "invalid_controls" else json.dumps([removed])
    run = lifecycle(backend, launcher, prepare=_seed_positive_artifacts, GEAK_REMOVE_ARGS=controls)
    run.assert_no_measurement()
    if failure == "invalid_controls":
        assert run.events == []
        assert not (run.out / "server_args_validation.json").exists()
        assert run.read("server_start.json")["reason"] == "server_args_unverified"
    else:
        assert run.read("server_args_validation.json")["reason"] == "removal_mismatch"
        assert run.read("server_start.json")["reason"] == "server_args_unverified"
        run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_fresh_legacy_launch_discards_stale_receipt_before_measuring(lifecycle, backend):
    run = lifecycle(backend, prepare=_seed_positive_artifacts)
    run.assert_measured(verified=False)
    assert not (run.out / "server_args_validation.json").exists()
    run.assert_owned_server_stopped()


def test_early_preflight_failure_clears_stale_summary_and_receipt(lifecycle, tmp_path):
    run = lifecycle(prepare=_seed_positive_artifacts, ADAPTER=str(tmp_path / "missing-adapter.sh"))
    assert run.process.returncode != 0
    assert run.events == []
    for name in ("bench_summary.json", "server_start.json", "server_args_validation.json"):
        assert not (run.out / name).exists()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_default_reuse_receipt_survives_positive_summary_cleanup(lifecycle, backend):
    process, ready, record = lifecycle.reusable(backend, ["--keep", "yes"])

    def prepare(out):
        _seed_positive_artifacts(out)
        result = subprocess.run([
            sys.executable, str(SCRIPTS / "adapters/server_args.py"), "validate",
            "--pid", str(process.pid), "--start-ticks", record["start_ticks"],
            "--backend", backend, "--port", "18080", '--remove-args=["--disabled"]',
            "--current-args=--keep yes", "--receipt", str(out / "server_args_validation.json"),
        ], capture_output=True, text=True, timeout=5, check=False)
        assert result.returncode == 0, result.stdout + result.stderr

    run = lifecycle(backend, prepare=prepare, REUSE_SERVER="1", STUB_READY=str(ready),
                    GEAK_REMOVE_ARGS='["--disabled"]', EXTRA_SERVER_ARGS="--keep yes")
    assert run.assert_measured(verified=True) == record["argv"]
    assert run.read("server_args_validation.json")["validation_kind"] == "reuse"
    assert process.poll() is None


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_reuse_receipt_write_failure_records_terminal_rejection(lifecycle, tmp_path, backend):
    process, ready, record = lifecycle.reusable(backend, ["--keep", "yes"])
    receipt = tmp_path / "launch-receipt.json"
    result = subprocess.run([
        sys.executable, str(SCRIPTS / "adapters/server_args.py"), "validate",
        "--pid", str(process.pid), "--start-ticks", record["start_ticks"],
        "--backend", backend, "--port", "18080", '--remove-args=["--disabled"]',
        "--current-args=--keep yes", "--receipt", str(receipt),
    ], capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    def prepare(out):
        _seed_positive_artifacts(out)
        (out / "server_args_validation.json").unlink()
        (out / "server_args_validation.json").mkdir()

    run = lifecycle(backend, prepare=prepare, REUSE_SERVER="1", STUB_READY=str(ready),
                    GEAK_SERVER_ARGS_RECEIPT=str(receipt), GEAK_REMOVE_ARGS='["--disabled"]',
                    EXTRA_SERVER_ARGS="--keep yes")
    run.assert_no_measurement()
    assert run.read("server_start.json")["status"] == "failed"
    assert run.read("server_start.json")["reason"] == "server_args_unverified"
    assert run.events == []
    assert process.poll() is None


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_reuse_malformed_controls_rejects_stale_positive_receipt(lifecycle, backend):
    process, ready, _ = lifecycle.reusable(backend, [])
    run = lifecycle(backend, prepare=_seed_positive_artifacts, REUSE_SERVER="1",
                    STUB_READY=str(ready), GEAK_REMOVE_ARGS="invalid-json")
    run.assert_no_measurement()
    assert run.read("server_start.json")["status"] == "failed"
    assert run.read("server_start.json")["reason"] == "server_args_unverified"
    assert run.events == []
    assert process.poll() is None


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("mode", ["warm_server", "isolated_server"])
@pytest.mark.parametrize("failure", ["missing_validator", "malformed_controls"])
def test_prelaunch_verification_failure_is_terminal(lifecycle, tmp_path, backend, mode, failure):
    scripts_dir = SCRIPTS
    if failure == "missing_validator":
        scripts_dir = tmp_path / "staged"
        scripts_dir.mkdir()
        for name in ("bench_e2e.sh", "bench_replica.sh", "bench_summarize.py", "server_teardown.sh"):
            shutil.copy2(SCRIPTS / name, scripts_dir / name)
    controls = '["--disabled"]' if failure == "missing_validator" else "invalid-json"
    run = lifecycle(backend, prepare=_seed_positive_artifacts, scripts_dir=scripts_dir,
                    GEAK_REPEAT_MODE=mode, REPEATS="3", GEAK_REMOVE_ARGS=controls)
    run.assert_no_measurement()
    assert run.read("server_start.json")["status"] == "failed"
    assert run.read("server_start.json")["reason"] == "server_args_unverified"
    assert "attempt 2/2" not in run.process.stdout
    assert not (run.out / "replica_002").exists()
    assert run.events == []


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("launcher", ["native", "magpie"])
def test_isolated_removal_failure_is_terminal_without_stale_aggregate(lifecycle, backend, launcher):
    def prepare(out):
        _seed_positive_artifacts(out)
        first = out / "replica_001"
        _seed_positive_artifacts(first / "attempt_1")
        (first / "selected_summary.json").write_text('{"throughput_tok_s_median":9999,"runs":1}')
        (first / "selected_attempt").write_text("1\n")

    removed = ("--script-default" if launcher == "magpie" else
               "--mem-fraction-static" if backend == "sglang" else "--gpu-memory-utilization")
    run = lifecycle(backend, launcher, prepare=prepare, GEAK_REPEAT_MODE="isolated_server",
                    REPEATS="3", GEAK_REMOVE_ARGS=json.dumps([removed]))
    run.assert_no_measurement()
    assert run.read("server_args_validation.json")["status"] == "failed"
    assert run.read("server_args_validation.json")["reason"] == "removal_mismatch"
    assert "attempt 2/2" not in run.process.stdout
    assert not (run.out / "replica_001/selected_summary.json").exists()
    assert not (run.out / "replica_002").exists()
    run.assert_owned_server_stopped()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_isolated_receipt_write_failure_cannot_aggregate_earlier_success(lifecycle, backend):
    def prepare(out):
        _seed_positive_artifacts(out)
        # A directory blocks atomic receipt replacement for the second replica.
        # The live check cannot issue proof; server_start still records rejection.
        for attempt in (1, 2):
            (out / f"replica_002/attempt_{attempt}/server_args_validation.json").mkdir(parents=True)

    run = lifecycle(backend, "magpie", prepare=prepare, GEAK_REPEAT_MODE="isolated_server",
                    REPEATS="3", GEAK_REMOVE_ARGS='["--disabled"]')
    assert run.process.returncode != 0, run.process.stdout + run.process.stderr
    assert not (run.out / "bench_summary.json").exists()
    assert not (run.out / "server_args_validation.json").exists()
    assert run.read("server_start.json")["status"] == "failed"
    assert run.read("server_start.json")["reason"] == "server_args_unverified"
    assert "attempt 2/2" not in run.process.stdout
    benches = [event for event in run.events if event["event"] == "bench"]
    launches = [event for event in run.events if event["event"] == "launch"]
    assert len(benches) == 1
    assert benches[0]["pid"] == launches[0]["pid"]
    assert not (run.out / "replica_002/selected_summary.json").exists()
    assert not (run.out / "replica_003").exists()
    run.assert_owned_server_stopped(expected_launches=2)
