# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU serving processes must attest their own accepted-source imports."""

import hashlib
import importlib
import importlib.util
import json
import os
import py_compile
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from interface.test_source_materialization import bundle, seal  # noqa: F401

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_runtime_under_test", SCRIPTS / "source_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def add_source(root, manifest, name, code):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    path.chmod(0o644)
    manifest["files"].append({"path": name, "mode": 0o644, "sha256": hashlib.sha256(code.encode()).hexdigest()})


@pytest.fixture
def launch_files(bundle, tmp_path):  # noqa: F811 - imported pytest fixture
    baseline, manifest, root = bundle
    # Untouched package ownership matters even though it is absent from modules.
    add_source(root, manifest, "trees/a/python/dependency/__init__.py", "VALUE='accepted-dependency'\n")
    seal(baseline, manifest)
    request = tmp_path / "baseline_source.json"
    request.write_text(json.dumps({"schema_version": 1, **baseline}))
    out = tmp_path / "measurement" / "source_runtime"
    return request, out, root


def wait_file(path, process=None):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text()
        if process is not None and process.poll() is not None:
            raise AssertionError(f"server exited {process.returncode}: {process.communicate()[1]}")
        time.sleep(0.02)
    raise AssertionError(f"server did not create {path}")


@contextmanager
def serving(launch_files, tmp_path, *, overlay="", startup="", after_bind="", observe=True, disowned=False):
    request, out, _root = launch_files
    roots = runtime.prepare(request, out, overlay)
    marker = tmp_path / "listening.json"
    script = tmp_path / "serve.py"
    script.write_text("import os,sys,json,time,socket,subprocess,signal\n"
                      "from pathlib import Path\nfrom http.server import HTTPServer,BaseHTTPRequestHandler\n"
                      + startup + "\n"
                      "class Handler(BaseHTTPRequestHandler):\n"
                      " def do_GET(self):\n"
                      "  self.send_response(200);self.end_headers();self.wfile.write(b'healthy')\n"
                      " def log_message(self,*args): pass\n"
                      "server=HTTPServer(('127.0.0.1',0),Handler)\n"
                      f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),server.server_port]))\n"
                      + after_bind + "\nserver.serve_forever()\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GEAK_SOURCE_REQUEST=str(request),
               GEAK_SOURCE_OBSERVATION_DIR=str(out), GEAK_ACCEPTED_SOURCE_PYTHONPATH=roots,
               OVERLAY_PYTHONPATH=overlay,
               PYTHONPATH=os.pathsep.join(filter(None, [str(out / "bootstrap") if observe else "", overlay, roots])))
    if disowned:
        env.update(TEST_PYTHON=sys.executable, TEST_SCRIPT=str(script), TEST_LOG=str(tmp_path / "server.log"))
        launcher = subprocess.run(["bash", "-c", 'setsid "$TEST_PYTHON" "$TEST_SCRIPT" >"$TEST_LOG" 2>&1 < /dev/null & disown "$!"'],
                                  env=env, text=True, capture_output=True, timeout=5, check=True)
        assert launcher.returncode == 0
        process = None
    else:
        process = subprocess.Popen([sys.executable, str(script)], env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    owner = None
    try:
        pid, port = json.loads(wait_file(marker, process))
        owner = runtime.identity(pid)
        yield out, owner, f"http://127.0.0.1:{port}", env
    finally:
        pid = owner["pid"] if owner is not None else (process.pid if process is not None else None)
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if process is not None:
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)


@pytest.mark.parametrize("disowned", [False, True])
def test_actual_serving_process_receipts_bind_two_phases_and_measurement(launch_files, tmp_path, disowned):
    with serving(launch_files, tmp_path, disowned=disowned,
                 startup="from alpha import first\nimport dependency\nassert first.VALUE=='accepted-first'\nassert dependency.VALUE=='accepted-dependency'") as (out, owner, endpoint, _):
        ready = runtime.gate(out, owner, endpoint, "ready", 3)
        finished = runtime.gate(out, owner, endpoint, "finished", 3)
        assert ready["listener_pids"] == [owner["pid"]]
        assert ready["challenge_id"] != finished["challenge_id"]
        assert ready["processes"][0]["receipt"] != finished["processes"][0]["receipt"]
        for gate in (ready, finished):
            assert gate["launch_capsule_sha256"] == runtime.digest(out / "launch.json")
            proof = gate["processes"][0]
            receipt = runtime.read_json(out / proof["receipt"])
            assert runtime.digest(out / proof["receipt"]) == proof["sha256"]
            assert receipt["pid"] == owner["pid"]
            assert receipt["launch_capsule_sha256"] == gate["launch_capsule_sha256"]
            assert "dependency" in {row["name"] for row in receipt["loaded_modules"]}
            assert "alpha.second" not in {row["name"] for row in receipt["loaded_modules"]}
            assert "alpha.second" in {row["name"] for row in receipt["resolved_modules"]}
            assert receipt["deleted_modules_absent"] == ["alpha.old"]
        (out.parent / "bench_runs.jsonl").write_text('{"output_throughput":17}\n')
        (out.parent / "bench_summary.json").write_text('{"median":17}\n')
        sealed = runtime.seal_measurement(out)
        assert sealed["artifacts"]["bench_summary.json"] == runtime.digest(out.parent / "bench_summary.json")
        assert sealed["gates"]["ready"]["sha256"] == runtime.digest(out / "gate-ready.json")
        assert sealed["launch_capsule"] == {"path": "launch.json", "sha256": runtime.digest(out / "launch.json")}
    assert not list(launch_files[2].rglob("*.pyc"))


def test_real_authored_overlay_chains_original_sitecustomize_and_keeps_accepted_siblings(launch_files, tmp_path):
    overlay = tmp_path / "overlay"
    patch = tmp_path / "patch.py"
    patch.write_text("VALUE='authored'\n")
    subprocess.run([sys.executable, str(SCRIPTS / "overlay_setup.py"), "add-module", "--overlay", str(overlay),
                    "--module", "alpha.first", "--patched-file", str(patch)], capture_output=True, check=True, timeout=10)
    with (overlay / "sitecustomize.py").open("a") as stream:
        stream.write("\nassert __name__ == 'sitecustomize'\nassert __spec__.name == 'sitecustomize'\n")
    with serving(launch_files, tmp_path, overlay=str(overlay), startup="from alpha import first,second\nassert first.VALUE=='authored'\nassert second.VALUE=='accepted-second'") as (out, owner, endpoint, _):
        result = runtime.gate(out, owner, endpoint, "ready", 3)
        receipt = runtime.read_json(out / result["processes"][0]["receipt"])
        loaded = {row["name"]: row for row in receipt["loaded_modules"]}
        assert loaded["alpha.first"]["source"] == "authored_overlay"
        assert loaded["alpha.second"]["source"] == "accepted_source"
        assert str(overlay / "_overlay_manifest.json") in runtime.capsule(out)["overlay_files"]


@pytest.mark.parametrize("mode", ["unobserved", "spawn_unobserved", "fork", "spawn"])
def test_every_serving_python_worker_needs_its_own_receipt(launch_files, tmp_path, mode):
    startup = ""
    if mode == "fork":
        startup = "child=os.fork()\nif child==0:\n import dependency\n time.sleep(60)\n os._exit(0)"
    elif mode.startswith("spawn"):
        child_env = "dict(os.environ,PYTHONPATH=os.environ['GEAK_ACCEPTED_SOURCE_PYTHONPATH'])" if mode == "spawn_unobserved" else "os.environ"
        startup = f"subprocess.Popen([sys.executable,'-c','import dependency,time;time.sleep(60)'],env={child_env})"
    with serving(launch_files, tmp_path, startup=startup, observe=mode != "unobserved") as (out, owner, endpoint, _):
        if mode in ("unobserved", "spawn_unobserved"):
            with pytest.raises(runtime.SourceRuntimeError, match="observation_timeout"):
                runtime.gate(out, owner, endpoint, "ready", 0.4)
        else:
            result = runtime.gate(out, owner, endpoint, "ready", 3)
            assert len(result["processes"]) == 2
            assert len({row["pid"] for row in result["processes"]}) == 2


@pytest.mark.parametrize("name", ["alpha.first", "dependency"])
def test_late_off_tree_import_cannot_be_credited_even_if_server_catches_error(launch_files, tmp_path, name):
    stock = tmp_path / "stock"
    stock.mkdir()
    package = name.split(".")[0]
    (stock / package).mkdir()
    (stock / package / "__init__.py").write_text("VALUE='stock'\n")
    (stock / package / "first.py").write_text("VALUE='stock'\n")
    trigger = tmp_path / "trigger"
    startup = ("import threading\ndef mutate():\n"
               f" while not Path({str(trigger)!r}).exists(): time.sleep(.01)\n"
               f" sys.path.insert(0,{str(stock)!r})\n"
               f" try: __import__({name!r})\n except Exception: pass\n"
               "threading.Thread(target=mutate,daemon=True).start()")
    with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _):
        runtime.gate(out, owner, endpoint, "ready", 3)
        trigger.touch()
        deadline = time.monotonic() + 3
        while not list(out.glob("fatal-*.json")) and time.monotonic() < deadline:
            time.sleep(.02)
        with pytest.raises(runtime.SourceRuntimeError, match="source_failure"):
            runtime.gate(out, owner, endpoint, "finished", 3)
        assert not (out / "gate-finished.json").exists()


@pytest.mark.parametrize("damage", ["request", "source", "overlay", "manifest"])
def test_post_measure_changes_prevent_success_receipt(launch_files, tmp_path, damage):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("FLAG=True\n")
    (overlay / "_overlay_manifest.json").write_text("{}\n")
    with serving(launch_files, tmp_path, overlay=str(overlay)) as (out, owner, endpoint, _):
        runtime.gate(out, owner, endpoint, "ready", 3)
        if damage == "request":
            launch_files[0].write_text("{}")
        elif damage == "source":
            (launch_files[2] / "trees/a/python/alpha/first.py").write_text("VALUE='changed'\n")
        elif damage == "overlay":
            (overlay / "sitecustomize.py").write_text("FLAG=False\n")
        else:
            (overlay / "_overlay_manifest.json").write_text('{"changed":true}')
        with pytest.raises(runtime.SourceRuntimeError, match="source_request_changed|source_failure"):
            runtime.gate(out, owner, endpoint, "finished", 3)


def test_frozen_identity_and_foreign_listener_rejected(launch_files, tmp_path):
    with serving(launch_files, tmp_path) as (out, owner, endpoint, _):
        with pytest.raises(runtime.SourceRuntimeError, match="server_identity_changed"):
            runtime.gate(out, {**owner, "start_ticks": owner["start_ticks"] + 1}, endpoint, "ready", .2)
        with socket.socket() as foreign:
            foreign.bind(("127.0.0.1", 0))
            foreign.listen()
            with pytest.raises(runtime.SourceRuntimeError, match="listener_owned_outside_server_group"):
                runtime.gate(out, owner, f"http://127.0.0.1:{foreign.getsockname()[1]}", "ready", .2)


def test_bootstrap_failure_cannot_be_swallowed_by_python_site(launch_files, tmp_path):
    request, out, _ = launch_files
    roots = runtime.prepare(request, out, "")
    env = dict(os.environ, PYTHONPATH=str(out / "bootstrap") + os.pathsep + roots,
               PYTHONDONTWRITEBYTECODE="1", GEAK_SOURCE_OBSERVATION_DIR=str(out),
               GEAK_ACCEPTED_SOURCE_PYTHONPATH="", OVERLAY_PYTHONPATH="")
    run = subprocess.run([sys.executable, "-c", "print('UNVERIFIED_SERVER_STARTED')"], env=env,
                         text=True, capture_output=True, timeout=5, check=False)
    assert run.returncode == 78
    assert "UNVERIFIED_SERVER_STARTED" not in run.stdout
    assert "[FATAL] GEAK source runtime bootstrap failed" in run.stderr


@pytest.mark.parametrize("bad", ["relative", "missing", "schema", "extra", "symlink"])
def test_prepare_refuses_invalid_request_before_launch(launch_files, tmp_path, bad):
    request, out, _ = launch_files
    if bad == "relative":
        request = Path("request.json")
    elif bad == "missing":
        request.unlink()
    elif bad in ("schema", "extra"):
        value = json.loads(request.read_text())
        value["schema_version" if bad == "schema" else "ignored"] = True
        request.write_text(json.dumps(value))
    else:
        link = tmp_path / "link.json"
        link.symlink_to(request)
        request = link
    with pytest.raises((runtime.SourceRuntimeError, OSError)):
        runtime.prepare(request, out, "")
    assert not out.exists()


def bench_fixture(launch_files, tmp_path, route):
    request, _out, root = launch_files
    value = runtime.read_json(request)
    manifest = runtime.read_json(root / "manifest.json")
    backend = "vllm" if route == "vllm" else "sglang"
    code = ("import argparse,os\nfrom http.server import HTTPServer,BaseHTTPRequestHandler\n"
            "from alpha import first,second\nimport dependency\n"
            "assert first.VALUE=='accepted-first'\nassert second.VALUE=='accepted-second'\n"
            "assert dependency.VALUE=='accepted-dependency'\n"
            "parser=argparse.ArgumentParser();parser.add_argument('--port',type=int);args,_=parser.parse_known_args()\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            " def do_GET(self):\n"
            "  self.send_response(200);self.end_headers();self.wfile.write(b'healthy')\n"
            " def log_message(self,*args): pass\n"
            "HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()\n")
    entry = "__main__.py" if route == "vllm" else "launch_server.py"
    add_source(root, manifest, f"trees/a/python/{backend}/__init__.py", "")
    add_source(root, manifest, f"trees/a/python/{backend}/{entry}", code)
    seal(value, manifest)
    request.write_text(json.dumps(value))
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("python", "python3"):
        (binary / name).symlink_to(sys.executable)
    (binary / "vllm").write_text('#!/bin/bash\nexec "$TEST_PYTHON" -m vllm "$@"\n')
    (binary / "vllm").chmod(0o755)
    # Exercise each real adapter's launch and path composition, replacing only
    # its GPU benchmark client with a CPU health request and canonical result.
    adapter = tmp_path / "adapter.sh"
    adapter.write_text('source "$TEST_NATIVE_ADAPTER"\n'
                       'adapter_health() { local n; for n in $(seq 1 100); do curl -sf "$BASE_URL/health" >/dev/null && return 0; sleep .02; done; return 1; }\n'
                       'adapter_bench() {\n'
                       ' curl -sf "$BASE_URL/health" >/dev/null || return 1\n'
                       ' printf \'{"output_throughput":17,"median_ttft_ms":1,"median_tpot_ms":1}\\n\' >> "$RESULT_JSONL"\n'
                       ' if [ "${TEST_DAMAGE:-0}" = "1" ] && [ -n "${r:-}" ]; then printf "VALUE=\\\"changed\\\"\\n" > "$TEST_SOURCE_FILE"; fi\n'
                       '}\n')
    magpie = tmp_path / "magpie.sh"
    magpie.write_text('#!/bin/bash\nsetsid python -m sglang.launch_server --port "$PORT" >"$SERVER_LOG" 2>&1 < /dev/null &\n'
                      'echo "$!" > "$MAGPIE_SERVER_PID_FILE"\ndisown "$!"\n')
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"], TEST_PYTHON=sys.executable,
               TEST_NATIVE_ADAPTER=str(SCRIPTS / "adapters" / (backend + ".sh")),
               TEST_SOURCE_FILE=str(root / "trees/a/python/alpha/first.py"),
               ADAPTER=str(adapter), BACKEND=backend, MODEL="cpu-fixture", OUT_DIR=str(tmp_path / "bench"),
               GEAK_SOURCE_REQUEST=str(request), GEAK_REPEAT_MODE="warm_server", REPEATS="1",
               NUM_PROMPTS="2", CONC="1", PROFILE="0", REUSE_SERVER="0", GPU_ARCHS="gfx000",
               SERVING_GPU_LOCK_DISABLE="1", SERVER_STOP_GRACE_S="0", SERVER_STARTUP_TIMEOUT_SEC="5",
               GEAK_SOURCE_GATE_TIMEOUT_SEC="1", OVERLAY_PYTHONPATH="", PYTHONPATH="",
               SGLANG_SRC_PYTHONPATH="", BENCH_LAUNCHER="magpie" if route == "magpie" else "native",
               MAGPIE_LAUNCH_SCRIPT=str(magpie), BENCH_OUTER_WARMUP="0", BENCH_COLD_FINAL="0")
    return env


@pytest.mark.parametrize("route", ["sglang", "vllm", "magpie"])
def test_real_bench_adapters_publish_bound_source_measurement(launch_files, tmp_path, route):
    env = bench_fixture(launch_files, tmp_path, route)
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=15, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    out = Path(env["OUT_DIR"]) / "source_runtime"
    seal = runtime.read_json(out / "measurement.json")
    assert seal["artifacts"]["bench_summary.json"] == runtime.digest(out.parent / "bench_summary.json")
    assert runtime.read_json(out.parent / "bench_summary.json")["throughput_tok_s_median"] == 17
    gate = runtime.read_json(out / "gate-finished.json")
    receipt = runtime.read_json(out / gate["processes"][0]["receipt"])
    names = {row["name"] for row in receipt["loaded_modules"]}
    assert ("vllm.__main__" if route == "vllm" else "sglang.launch_server") in names
    assert not runtime.group_members(gate["server_identity"]["pgid"])
    assert runtime.read_json(out / "teardown.json")["status"] == "confirmed"


def test_bench_post_measure_source_damage_preserves_raw_runs_without_seal(launch_files, tmp_path):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    env["TEST_DAMAGE"] = "1"
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=15, check=False)
    assert run.returncode == 3, run.stdout + run.stderr
    out = Path(env["OUT_DIR"])
    assert (out / "bench_runs.jsonl").read_text()
    assert not (out / "bench_summary.json").exists()
    assert not (out / "source_runtime/measurement.json").exists()
    gate = runtime.read_json(out / "source_runtime/gate-ready.json")
    assert not runtime.group_members(gate["server_identity"]["pgid"])
    assert runtime.read_json(out / "source_runtime/teardown.json")["status"] == "confirmed"


def test_bench_source_reuse_refused_before_launch(launch_files, tmp_path):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    env["REUSE_SERVER"] = "1"
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=15, check=False)
    assert run.returncode == 3, run.stdout + run.stderr
    assert "source_runtime_reused_server_unsupported" in run.stderr
    assert not (Path(env["OUT_DIR"]) / "source_runtime").exists()


@pytest.fixture
def observer(launch_files, monkeypatch):
    request, out, _ = launch_files
    roots = runtime.prepare(request, out, "")
    launch = runtime.capsule(out)
    source = runtime.validate(launch["request"])
    observer = runtime.Observer(out, launch, source)
    monkeypatch.setattr(runtime, "subreaper", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "path", roots.split(os.pathsep) + sys.path)
    monkeypatch.setattr(sys, "meta_path", [observer] + sys.meta_path)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    existing = {name: module for name, module in sys.modules.copy().items() if name.split(".")[0] in observer.owners}
    for name in existing:
        del sys.modules[name]
    try:
        yield observer
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in observer.owners:
                del sys.modules[name]
        sys.modules.update(existing)
        if observer.channel is not None:
            observer.channel.close()


def test_observer_delegates_unmodified_loader_and_distinguishes_loaded_resolution(observer, monkeypatch):
    alpha = importlib.import_module("alpha")
    first = importlib.import_module("alpha.first")
    dependency = importlib.import_module("dependency")
    assert first.VALUE == "accepted-first" and dependency.VALUE == "accepted-dependency"
    assert isinstance(first.__loader__, importlib.machinery.SourceFileLoader)
    # Exercise a -m entry point while retaining its actual module spec identity.
    monkeypatch.setitem(sys.modules, "__main__", first)
    receipt = observer.snapshot({"challenge_id": "unit"})
    names = {row["name"] for row in receipt["loaded_modules"]}
    assert {"alpha", "alpha.first", "dependency"} <= names
    assert observer.resolve("alpha.first")["source"] == "accepted_source"
    assert observer.resolve("alpha.second")["name"] == "alpha.second"
    assert observer.resolve("alpha.old") is None
    assert observer.resolve("dependency.nonpackage_child") is None
    assert observer.resolve("alpha.first.nonpackage_child") is None
    assert observer.find_spec("unowned_name") is None
    assert observer.find_spec("alpha.absent", alpha.__path__) is None
    assert observer.normal_spec("missing_top_level") is None


@pytest.mark.parametrize("case,reason", [
    ("deleted", "deleted_module_resolved"), ("namespace", "unsupported_module_origin"),
    ("stock", "off_tree_module_origin"), ("changed", "module_content_changed"),
])
def test_observer_rejects_unacceptable_specs_with_sticky_evidence(observer, tmp_path, case, reason):
    name = "alpha.old" if case == "deleted" else "alpha.first"
    path = Path(observer.expected["alpha.first"])
    if case == "namespace":
        spec = importlib.machinery.ModuleSpec(name, loader=None, origin=None)
    else:
        if case == "stock":
            path = tmp_path / "stock.py"
            path.write_text("VALUE='stock'\n")
        elif case == "changed":
            path.write_text("VALUE='changed'\n")
        spec = importlib.util.spec_from_file_location(name, path)
    with pytest.raises(runtime.SourceRuntimeError, match=reason):
        observer.check_spec(name, spec)
    assert observer.failure == reason
    assert runtime.read_json(next(observer.out.glob("fatal-*.json")))["reason"] == reason
    with pytest.raises(runtime.SourceRuntimeError, match=reason):
        observer.snapshot({"challenge_id": "after_failure"})


def test_observer_accepts_frozen_overlay_origin_and_reports_initialization(observer, tmp_path, monkeypatch):
    overlay = tmp_path / "authored"
    overlay.mkdir()
    path = overlay / "module.py"
    path.write_text("VALUE='authored'\n")
    observer.launch["overlay_files"][str(path)] = runtime.digest(path)
    observer.launch["overlay_roots"] = [str(overlay)]
    spec = importlib.util.spec_from_file_location("alpha.first", path)
    assert observer.check_spec("alpha.first", spec)["source"] == "authored_overlay"
    # Freeze the added authorized fixture overlay in the same capsule.
    runtime.write_json(observer.out / "launch.json", observer.launch)
    module = importlib.util.module_from_spec(spec)
    spec._initializing = True
    monkeypatch.setitem(sys.modules, "alpha.first", module)
    receipt = observer.snapshot({"challenge_id": "initializing"})
    assert next(row for row in receipt["loaded_modules"] if row["name"] == "alpha.first")["initializing"]


@pytest.mark.parametrize("damage,reason", [
    ("guard", "source_guard_displaced"), ("bytecode", "bytecode_writes_enabled"),
    ("capsule", "launch_capsule_changed"), ("overlay", "authored_overlay_changed"),
])
def test_snapshot_cannot_succeed_after_runtime_configuration_changes(observer, monkeypatch, tmp_path, damage, reason):
    if damage == "guard":
        monkeypatch.setattr(sys, "meta_path", sys.meta_path[1:])
    elif damage == "bytecode":
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
    elif damage == "capsule":
        value = runtime.capsule(observer.out)
        value["launch_nonce"] = "different"
        runtime.write_json(observer.out / "launch.json", value)
    else:
        overlay = tmp_path / "overlay"
        overlay.mkdir()
        path = overlay / "module.py"
        path.write_text("one")
        observer.launch["overlay_files"][str(path)] = runtime.digest(path)
        observer.launch["overlay_roots"] = [str(overlay)]
        runtime.write_json(observer.out / "launch.json", observer.launch)
        path.write_text("two")
    with pytest.raises(runtime.SourceRuntimeError, match=reason):
        observer.snapshot({"challenge_id": "changed"})


def test_responder_requires_current_challenge_and_writes_atomic_receipt(observer, monkeypatch):
    runtime.write_json(observer.out / "challenge.json", {"challenge_id": "current", "phase": "ready", "reply_socket": "unused"})
    tokens = iter([b"stale", b"current", b"current"])
    observer.channel = SimpleNamespace(recv=lambda _size: next(tokens), close=lambda: None, sendto=lambda *_args: None)
    with pytest.raises(StopIteration):
        observer.respond()
    receipt = next(observer.out.glob("process-*.json"))
    assert runtime.read_json(receipt)["challenge_id"] == "current"
    assert observer.measurement_started
    assert not list(observer.out.glob("*.tmp"))
    # Failure becomes sticky even when an import caller catches the exception.
    tokens = iter([b"current"])
    monkeypatch.setattr(observer, "snapshot", lambda _challenge: (_ for _ in ()).throw(OSError("missing")))
    with pytest.raises(StopIteration):
        observer.respond()
    assert runtime.read_json(next(observer.out.glob("fatal-*.json")))["reason"] == "runtime_observation_failed"


def test_observer_start_uses_own_process_identity_and_replaces_inherited_socket(observer, monkeypatch):
    started = []
    monkeypatch.setattr(runtime.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: started.append(kwargs)))
    observer.start()
    prior = observer.channel
    observer.start()
    assert prior.fileno() == -1
    assert len(started) == 2
    assert started[0]["target"] == observer.respond
    assert observer.channel.getsockname().decode().endswith(str(runtime.identity()["start_ticks"]))


def test_bootstrap_chains_original_identity_installs_guard_and_registers_fork(launch_files, tmp_path, monkeypatch):
    request, out, _ = launch_files
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("import sys\nassert __name__=='sitecustomize'\n"
                                           "assert __spec__.name=='sitecustomize'\n"
                                           "sys.meta_path.insert(0,object())\n")
    roots = runtime.prepare(request, out, str(overlay))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "path", [str(out / "bootstrap"), str(overlay), *roots.split(os.pathsep), *sys.path])
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "addaudithook", lambda _hook: None)
    monkeypatch.setattr(runtime, "subreaper", lambda **_kwargs: None)
    for key, value in {"GEAK_SOURCE_OBSERVATION_DIR": str(out), "GEAK_ACCEPTED_SOURCE_PYTHONPATH": roots,
                       "OVERLAY_PYTHONPATH": str(overlay)}.items():
        monkeypatch.setenv(key, value)
    original = sys.modules.get("sitecustomize")
    monkeypatch.setitem(sys.modules, "sitecustomize", original)
    started, fork_hooks = [], []
    monkeypatch.setattr(runtime.Observer, "start", lambda self: started.append(self))
    monkeypatch.setattr(runtime.os, "register_at_fork", lambda **kw: fork_hooks.append(kw))
    runtime.bootstrap(out / "bootstrap/sitecustomize.py")
    assert sys.meta_path[0] is started[0]
    assert fork_hooks[0]["after_in_child"] == started[0].start
    assert sys.modules["sitecustomize"].__file__ == str(overlay / "sitecustomize.py")


def test_bootstrap_records_chain_failures(launch_files, tmp_path, monkeypatch):
    request, out, _ = launch_files
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("raise ValueError('bad startup hook')\n")
    roots = runtime.prepare(request, out, str(overlay))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "path", [str(overlay), *roots.split(os.pathsep), *sys.path])
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "addaudithook", lambda _hook: None)
    monkeypatch.setattr(runtime, "subreaper", lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "sitecustomize", sys.modules.get("sitecustomize"))
    monkeypatch.setenv("GEAK_SOURCE_OBSERVATION_DIR", str(out))
    monkeypatch.setenv("GEAK_ACCEPTED_SOURCE_PYTHONPATH", roots)
    monkeypatch.setenv("OVERLAY_PYTHONPATH", str(overlay))
    with pytest.raises(runtime.SourceRuntimeError, match="runtime_bootstrap_failed"):
        runtime.bootstrap(out / "bootstrap/sitecustomize.py")
    assert next(out.glob("fatal-*.json")).exists()


def test_regular_json_and_path_boundaries(tmp_path, monkeypatch):
    path = tmp_path / "object.json"
    for value, reason in [('[]', "json_object_required"), ('{"a":1,"a":2}', "duplicate_json_key")]:
        path.write_text(value)
        with pytest.raises(runtime.SourceRuntimeError, match=reason):
            runtime.read_json(path)
    monkeypatch.setattr(runtime, "MAX_JSON", 1)
    with pytest.raises(runtime.SourceRuntimeError, match="json_too_large"):
        runtime.read_json(path)
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(runtime.SourceRuntimeError, match="non_regular_file"):
        runtime.digest(path)
    path.unlink()
    path.write_text("normal")
    with pytest.raises(runtime.SourceRuntimeError, match="import_root_not_directory"):
        runtime.path_list(str(path))
    with pytest.raises(runtime.SourceRuntimeError, match="nonabsolute_import_root"):
        runtime.path_list("relative")
    assert runtime.path_list(str(tmp_path) + os.pathsep + str(tmp_path)) == [str(tmp_path)]
    monkeypatch.setattr(runtime, "HERE", tmp_path / "absent/deep")
    with pytest.raises(runtime.SourceRuntimeError, match="source_validator_unavailable"):
        runtime.validator_path()


def test_cli_prepare_gate_and_seal_errors_are_fail_closed(launch_files, monkeypatch, capsys):
    request, out, _ = launch_files
    monkeypatch.setattr(sys, "argv", ["source_runtime.py", "prepare", "--request", str(request), "--output-dir", str(out)])
    assert runtime.main() == 0
    assert str(launch_files[2]) in capsys.readouterr().out
    calls = []
    monkeypatch.setattr(runtime, "gate", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["source_runtime.py", "gate", "--output-dir", str(out), "--pid", "4", "--pgid", "4", "--start-ticks", "8", "--base-url", "http://127.0.0.1:3"])
    assert runtime.main() == 0
    assert calls[0][1] == {"pid": 4, "pgid": 4, "start_ticks": 8}
    monkeypatch.setattr(sys, "argv", ["source_runtime.py", "seal-measurement", "--output-dir", str(out)])
    assert runtime.main() == 3
    assert "[FATAL]" in capsys.readouterr().err
    monkeypatch.setattr(runtime, "seal_measurement", lambda path: calls.append(path))
    assert runtime.main() == 0
    monkeypatch.setattr(sys, "argv", ["source_runtime.py", "gate", "--output-dir", str(out), "--timeout-sec", "0"])
    assert runtime.main() == 3
    assert "invalid_gate_timeout" in capsys.readouterr().err


def test_inherited_listener_held_by_escaped_worker_cannot_borrow_parent_proof(launch_files, tmp_path):
    child_file = tmp_path / "escaped.json"
    after_bind = ("child=os.fork()\nif child==0:\n os.setsid()\n"
                  f" Path({str(child_file)!r}).write_text(str(os.getpid()))\n"
                  " time.sleep(60)\n os._exit(0)")
    escaped = None
    try:
        with serving(launch_files, tmp_path, after_bind=after_bind) as (out, owner, endpoint, _):
            escaped = int(wait_file(child_file))
            assert runtime.identity(escaped)["pgid"] != owner["pgid"]
            try:
                with pytest.raises(runtime.SourceRuntimeError, match="serving_worker_escaped_group"):
                    runtime.gate(out, owner, endpoint, "ready", 1)
            finally:
                os.kill(escaped, signal.SIGTERM)
                escaped = None
    finally:
        if escaped is not None:
            os.kill(escaped, signal.SIGTERM)


def test_foreign_reuseport_listener_cannot_share_verified_endpoint(launch_files, tmp_path):
    configured = tmp_path / "reuseport-ready"
    after_bind = ("server.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)\n"
                  f"Path({str(configured)!r}).touch()")
    with serving(launch_files, tmp_path, after_bind=after_bind) as (out, owner, endpoint, _):
        wait_file(configured)
        with socket.socket() as foreign:
            foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            foreign.bind(("127.0.0.1", int(endpoint.rsplit(":", 1)[1])))
            foreign.listen()
            with pytest.raises(runtime.SourceRuntimeError, match="listener_owned_outside_server_group"):
                runtime.gate(out, owner, endpoint, "ready", 1)


def test_isolated_replicas_publish_exact_selected_leaf_seals(launch_files, tmp_path):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    env.update(GEAK_REPEAT_MODE="isolated_server", REPLICAS="2")
    env.pop("REPEATS")
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=20, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    out = Path(env["OUT_DIR"])
    summary = runtime.read_json(out / "bench_summary.json")
    assert summary["measurement_mode"] == "isolated_server"
    assert len(summary["replicas"]) == 2
    nonces = set()
    for row in summary["replicas"]:
        replica = out / f"replica_{row['replica']:03d}"
        assert (replica / "selected_attempt").read_text().strip() == str(row["attempt"])
        leaf = replica / f"attempt_{row['attempt']}"
        assert (replica / "selected_summary.json").read_bytes() == (leaf / "bench_summary.json").read_bytes()
        seal = runtime.read_json(leaf / "source_runtime/measurement.json")
        assert seal["artifacts"]["bench_summary.json"] == runtime.digest(leaf / "bench_summary.json")
        nonces.add(seal["launch_nonce"])
    assert len(nonces) == 2


@pytest.mark.parametrize("kind", ["bytecode", "symlink_directory"])
def test_preparation_rejects_unfrozen_overlay_execution_paths(launch_files, tmp_path, kind):
    request, out, _ = launch_files
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    path = overlay / "sitecustomize.py"
    path.write_text("VALUE='old-bytes'\n")
    if kind == "bytecode":
        before = path.stat()
        py_compile.compile(str(path), doraise=True)
        path.write_text("VALUE='new-bytes'\n")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        reason = "source_bytecode_present"
    else:
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "__init__.py").write_text("VALUE='outside'\n")
        (overlay / "sidehelper").symlink_to(foreign, target_is_directory=True)
        reason = "overlay_source_symlink"
    with pytest.raises(runtime.SourceRuntimeError, match=reason):
        runtime.prepare(request, out, str(overlay))
    assert not out.exists()


def test_actual_interpreter_rejects_external_bytecode_cache_before_overlay_runs(launch_files, tmp_path, monkeypatch):
    request, out, _ = launch_files
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    hook = overlay / "sitecustomize.py"
    hook.write_text("FLAG='old-bytes'\n")
    before = hook.stat()
    cache = tmp_path / "external-cache"
    with monkeypatch.context() as patcher:
        patcher.setattr(sys, "pycache_prefix", str(cache))
        py_compile.compile(str(hook), doraise=True)
    hook.write_text("FLAG='new-bytes'\n")
    os.utime(hook, ns=(before.st_atime_ns, before.st_mtime_ns))
    roots = runtime.prepare(request, out, str(overlay))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(out / "bootstrap"), str(overlay), roots]),
               GEAK_SOURCE_OBSERVATION_DIR=str(out), GEAK_ACCEPTED_SOURCE_PYTHONPATH=roots,
               OVERLAY_PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX=str(cache))
    run = subprocess.run([sys.executable, "-c", "print('SERVER_STARTED')"], env=env, text=True,
                         capture_output=True, timeout=5, check=False)
    assert run.returncode == 78
    assert "SERVER_STARTED" not in run.stdout
    assert runtime.read_json(next(out.glob("fatal-*.json")))["reason"] == "source_bytecode_present"


def test_overlay_file_added_after_prepare_cannot_execute_at_startup(launch_files, tmp_path):
    request, out, _ = launch_files
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("import late_helper\n")
    roots = runtime.prepare(request, out, str(overlay))
    (overlay / "late_helper.py").write_text("raise RuntimeError('unfrozen helper executed')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(out / "bootstrap"), str(overlay), roots]),
               GEAK_SOURCE_OBSERVATION_DIR=str(out), GEAK_ACCEPTED_SOURCE_PYTHONPATH=roots,
               OVERLAY_PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-c", "print('SERVER_STARTED')"], env=env, text=True,
                         capture_output=True, timeout=5, check=False)
    assert run.returncode == 78
    assert "unfrozen helper executed" not in run.stderr
    assert runtime.read_json(next(out.glob("fatal-*.json")))["reason"] == "authored_overlay_changed"


def test_transient_unobserved_worker_is_refused_during_measured_interval(launch_files, tmp_path):
    trigger, attempted = tmp_path / "trigger", tmp_path / "attempted"
    result = tmp_path / "unverified-worker-result"
    worker = f"from pathlib import Path;Path({str(result)!r}).write_text('stock-result')"
    startup = ("import threading\ndef spawn_after_ready():\n"
               f" while not Path({str(trigger)!r}).exists(): time.sleep(.01)\n"
               f" try: subprocess.run([sys.executable,'-S','-c',{worker!r}],check=True)\n"
               " except Exception: pass\n"
               f" Path({str(attempted)!r}).touch()\n"
               "threading.Thread(target=spawn_after_ready,daemon=True).start()")
    with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _):
        runtime.gate(out, owner, endpoint, "ready", 3)
        trigger.touch()
        wait_file(attempted)
        assert not result.exists()
        with pytest.raises(runtime.SourceRuntimeError, match="source_failure"):
            runtime.gate(out, owner, endpoint, "finished", 2)
        assert runtime.read_json(next(out.glob("fatal-*.json")))["reason"] == "worker_topology_changed_after_ready"


@pytest.mark.parametrize("double_fork", [False, True])
def test_unobserved_detached_worker_is_retained_and_refused(launch_files, tmp_path, double_fork):
    child_file = tmp_path / "worker.json"
    worker = ("import os,time\nfrom pathlib import Path\n"
              + ("if os.fork(): os._exit(0)\nos.setsid()\n" if double_fork else "")
              + f"Path({str(child_file)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n")
    startup = f"subprocess.Popen([sys.executable,'-S','-c',{worker!r}],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    escaped = None
    try:
        with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _):
            escaped = int(wait_file(child_file))
            try:
                with pytest.raises(runtime.SourceRuntimeError, match="serving_worker_escaped_group"):
                    runtime.gate(out, owner, endpoint, "ready", 1)
            finally:
                os.kill(escaped, signal.SIGTERM)
                escaped = None
    finally:
        if escaped is not None:
            os.kill(escaped, signal.SIGTERM)


def test_renamed_unobserved_python_cannot_hide_as_nonpython_worker(launch_files, tmp_path):
    executable = tmp_path / "engine-worker"
    shutil.copy2(Path(sys.executable).resolve(), executable)
    startup = f"subprocess.Popen([{str(executable)!r},'-S','-c','import time;time.sleep(60)'])"
    with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _), \
            pytest.raises(runtime.SourceRuntimeError, match="observation_timeout"):
        runtime.gate(out, owner, endpoint, "ready", .5)


def test_marker_prevents_environment_drop_from_becoming_no_source(launch_files, tmp_path):
    request, _, _ = launch_files
    stage = tmp_path / "stage"
    stage.mkdir()
    for filename in ("bench_e2e.sh", "source_runtime.py"):
        shutil.copy2(SCRIPTS / filename, stage / filename)
    shutil.copy2(SCRIPTS.parents[1] / "interface/source_materialization.py", stage / "source_materialization.py")
    marker = stage / "source_manifest.sha256"
    expected = runtime.read_json(request)["source_materialization"]["manifest_sha256"]
    marker.write_text(expected + "\n")
    runtime.check_request_marker(request, marker)
    env = dict(os.environ)
    env.pop("GEAK_SOURCE_REQUEST", None)
    run = subprocess.run(["bash", str(stage / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=5, check=False)
    assert run.returncode == 3
    assert "staged_source_request_missing_or_mismatched" in run.stderr
    marker.write_text("0" * 64 + "\n")
    with pytest.raises(runtime.SourceRuntimeError, match="staged_source_manifest_mismatch"):
        runtime.check_request_marker(request, marker)
    marker.write_text("bad")
    with pytest.raises(runtime.SourceRuntimeError, match="invalid_staged_source_marker"):
        runtime.check_request_marker(request, marker)


def test_subreaper_calls_and_topology_audit_contract(observer, monkeypatch):
    # The audit hook refuses process creation only once ready was challenged.
    observer.audit("subprocess.Popen", ())
    observer.measurement_started = True
    observer.audit("open", ())
    with pytest.raises(runtime.SourceRuntimeError, match="worker_topology_changed_after_ready"):
        observer.audit("os.fork", ())


def test_subreaper_support_cannot_silently_be_disabled(monkeypatch):
    class Prctl:
        enabled = 0
        failure = False

        def __call__(self, option, value, *_args):
            if self.failure:
                return -1
            if option == 36:
                self.enabled = value
            else:
                runtime.ctypes.cast(value, runtime.ctypes.POINTER(runtime.ctypes.c_int)).contents.value = self.enabled
            return 0

    prctl = Prctl()
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(prctl=prctl))
    runtime.subreaper(enable=True)
    runtime.subreaper()
    prctl.enabled = 0
    with pytest.raises(runtime.SourceRuntimeError, match="source_subreaper_disabled"):
        runtime.subreaper()
    prctl.failure = True
    with pytest.raises(runtime.SourceRuntimeError, match="subreaper_enable_failed"):
        runtime.subreaper(enable=True)


def test_gate_rejects_stale_challenge_even_when_process_is_still_live(launch_files, tmp_path, monkeypatch):
    with serving(launch_files, tmp_path) as (out, owner, endpoint, _):
        read = runtime.read_json
        stale = []

        def stale_once(path, **kwargs):
            row = read(path, **kwargs)
            if Path(path).name.startswith("process-") and not stale:
                stale.append(True)
                row["challenge_id"] = "from-a-previous-gate"
            return row

        monkeypatch.setattr(runtime, "read_json", stale_once)
        gate = runtime.gate(out, owner, endpoint, "ready", 3)
        assert stale
        receipt = read(out / gate["processes"][0]["receipt"])
        assert receipt["challenge_id"] == gate["challenge_id"]


def test_marker_cli_and_missing_output_argument(launch_files, tmp_path, monkeypatch):
    request, _, _ = launch_files
    marker = tmp_path / "marker"
    marker.write_text(runtime.read_json(request)["source_materialization"]["manifest_sha256"] + "\n")
    monkeypatch.setattr(sys, "argv", ["source_runtime.py", "check-request", "--request", str(request), "--expected-manifest", str(marker)])
    assert runtime.main() == 0
    monkeypatch.setattr(sys, "argv", ["source_runtime.py", "prepare"])
    with pytest.raises(SystemExit, match="2"):
        runtime.main()


def test_warmup_can_compile_before_hot_topology_is_frozen(launch_files, tmp_path):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    request, _, root = launch_files
    path = root / "trees/a/python/sglang/launch_server.py"
    code = path.read_text().replace("import argparse,os", "import argparse,os,sys,subprocess")
    code = code.replace("  self.send_response(200)",
                        "  if self.path=='/work' and not getattr(self.server,'warmed',False):\n"
                        "   subprocess.run([sys.executable,'-c','import dependency'],check=True)\n"
                        "   self.server.warmed=True\n"
                        "  self.send_response(200)")
    path.write_text(code)
    value = runtime.read_json(request)
    manifest = runtime.read_json(root / "manifest.json")
    next(row for row in manifest["files"] if row["path"].endswith("sglang/launch_server.py"))["sha256"] = runtime.digest(path)
    seal(value, manifest)
    request.write_text(json.dumps(value))
    adapter = Path(env["ADAPTER"])
    adapter.write_text(adapter.read_text().replace(' curl -sf "$BASE_URL/health" >/dev/null || return 1',
                                                  ' curl -sf "$BASE_URL/work" >/dev/null || return 1'))
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=20, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    out = Path(env["OUT_DIR"]) / "source_runtime"
    prepared = runtime.read_json(out / "gate-prepared.json")
    ready = runtime.read_json(out / "gate-ready.json")
    assert prepared["observed_at_ns"] < ready["observed_at_ns"]
    assert runtime.read_json(out / prepared["processes"][0]["receipt"])["worker_topology"] == "startup"
    assert runtime.read_json(out / ready["processes"][0]["receipt"])["worker_topology"] == "frozen_after_ready"
    assert runtime.read_json(out / "measurement.json")["measurement_scope"] == "hot_timed_rounds"


def test_capture_only_has_no_hot_measurement_seal(launch_files, tmp_path):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    env["REPEATS"] = "0"
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=20, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    out = Path(env["OUT_DIR"])
    assert (out / "source_runtime/gate-finished.json").exists()
    assert not (out / "source_runtime/measurement.json").exists()


def test_sibling_cannot_forge_serving_pid_by_binding_its_socket_name(launch_files, tmp_path):
    with serving(launch_files, tmp_path, observe=False) as (out, owner, endpoint, _), \
            socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sibling:
        launch = runtime.capsule(out)
        sibling.bind(runtime.socket_name(launch, owner))
        sibling.settimeout(2)

        def forge():
            token = sibling.recv(128).decode()
            challenge = runtime.read_json(out / "challenge.json")
            path = out / f"process-{owner['pid']}-{owner['start_ticks']}-{token}.json"
            runtime.write_json(path, {"schema": runtime.PROCESS_SCHEMA, "status": "verified", **owner,
                                      "challenge_id": token, "launch_nonce": launch["launch_nonce"],
                                      "launch_capsule_sha256": runtime.digest(out / "launch.json"),
                                      "request_sha256": launch["request_sha256"], "manifest_sha256": launch["manifest_sha256"],
                                      "boot_id": runtime.boot_id(), "observed_at_ns": time.time_ns()})
            sibling.sendto((token + " " + runtime.digest(path)).encode(), challenge["reply_socket"])

        thread = threading.Thread(target=forge)
        thread.start()
        with pytest.raises(runtime.SourceRuntimeError, match="observation_timeout"):
            runtime.gate(out, owner, endpoint, "ready", .4)
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_persistent_shell_worker_is_not_exempt_from_process_proof(launch_files, tmp_path):
    startup = "subprocess.Popen(['/bin/bash','-c','sleep 60 & wait'])"
    with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _), \
            pytest.raises(runtime.SourceRuntimeError, match="observation_timeout"):
        runtime.gate(out, owner, endpoint, "ready", .4)


def update_source_request(request, root, path):
    value = runtime.read_json(request)
    manifest = runtime.read_json(root / "manifest.json")
    next(row for row in manifest["files"] if root / row["path"] == path)["sha256"] = runtime.digest(path)
    seal(value, manifest)
    request.write_text(json.dumps(value))


@pytest.mark.parametrize("ignore_term", [False, True])
def test_real_bench_cleans_exact_escaped_worker_before_group_teardown(launch_files, tmp_path, ignore_term):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    request, _, root = launch_files
    marker = tmp_path / "escaped-worker"
    worker = ("import os,signal,time\nfrom pathlib import Path\n"
              + ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\n" if ignore_term else "")
              + f"Path({str(marker)!r}).write_text(str(os.getpid()))\ntime.sleep(60)")
    path = root / "trees/a/python/sglang/launch_server.py"
    code = path.read_text().replace("import argparse,os", "import argparse,os,subprocess,sys,time\nfrom pathlib import Path")
    code = code.replace("class Handler", f"subprocess.Popen([sys.executable,'-S','-c',{worker!r}],start_new_session=True)\n"
                        f"while not Path({str(marker)!r}).exists(): time.sleep(.01)\nclass Handler")
    path.write_text(code)
    update_source_request(request, root, path)
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=20, check=False)
    assert run.returncode == 3, run.stdout + run.stderr
    out = Path(env["OUT_DIR"])
    cleanup = runtime.read_json(out / "source_runtime/cleanup.json")
    assert cleanup["status"] == "confirmed"
    assert len(cleanup["processes"]) == 1
    row = cleanup["processes"][0]
    assert row["pid"] == int(marker.read_text()) and row["exit_confirmed"]
    assert row["kill_sent"] == ignore_term
    current = runtime.proc(row["pid"])
    assert current is None or current["state"] == "Z" or current["start_ticks"] != row["start_ticks"]
    assert not runtime.group_members(cleanup["server_identity"]["pgid"])
    assert not (out / "bench_summary.json").exists()
    assert not (request.parent / "source_cleanup_unverified.json").exists()


@pytest.mark.parametrize("teardown", [False, True])
def test_shutdown_rediscovery_catches_worker_forked_from_term_handler(launch_files, tmp_path, teardown):
    parent_file, child_file = tmp_path / "parent", tmp_path / "child"
    worker = ("import os,signal,time\nfrom pathlib import Path\n"
              "def terminate(*args):\n"
              " if os.fork(): os._exit(0)\n"
              " os.setsid()\n signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
              f" Path({str(child_file)!r}).write_text(str(os.getpid()))\n"
              " time.sleep(60)\n"
              "signal.signal(signal.SIGTERM,terminate)\n"
              f"Path({str(parent_file)!r}).write_text(str(os.getpid()))\ntime.sleep(60)")
    startup = f"subprocess.Popen([sys.executable,'-S','-c',{worker!r}],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _):
        wait_file(parent_file)
        if teardown:
            runtime.teardown_source(out, launch_files[0], owner)
        else:
            with pytest.raises(runtime.SourceRuntimeError, match="cleanup_confirmed"):
                runtime.gate(out, owner, endpoint, "prepared", 1)
        cleanup = runtime.read_json(out / ("teardown.json" if teardown else "cleanup.json"))
        assert cleanup["status"] == "confirmed"
        expected = {int(parent_file.read_text()), int(child_file.read_text())} | ({owner["pid"]} if teardown else set())
        assert {row["pid"] for row in cleanup["processes"]} == expected
        assert all(row["exit_confirmed"] for row in cleanup["processes"])


def test_unavailable_pidfd_sets_persistent_barrier_and_reserved_exit(launch_files, tmp_path, monkeypatch):
    child_file = tmp_path / "child"
    worker = f"import os,time;from pathlib import Path;Path({str(child_file)!r}).write_text(str(os.getpid()));time.sleep(60)"
    startup = f"subprocess.Popen([sys.executable,'-S','-c',{worker!r}],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _):
        pid = int(wait_file(child_file))
        monkeypatch.setattr(runtime, "CLEANUP_TOTAL_SEC", .4)
        monkeypatch.setattr(runtime.os, "pidfd_open", lambda *_args: (_ for _ in ()).throw(PermissionError("denied")))
        try:
            with pytest.raises(runtime.SourceCleanupUnverified, match="source_cleanup_unverified"):
                runtime.gate(out, owner, endpoint, "prepared", 1)
            cleanup = runtime.read_json(out / "cleanup.json")
            assert cleanup["status"] == "unverified" and not cleanup["processes"][0]["exit_confirmed"]
            assert runtime.proc(pid)["state"] != "Z"
            assert (launch_files[0].parent / "source_cleanup_unverified.json").exists()
            monkeypatch.setattr(sys, "argv", ["source_runtime.py", "prepare", "--request", str(launch_files[0]),
                                             "--output-dir", str(tmp_path / "another-output")])
            assert runtime.main() == 43
        finally:
            os.kill(pid, signal.SIGKILL)


def test_cleanup_never_signals_worker_without_owner_ancestry(launch_files, tmp_path):
    with serving(launch_files, tmp_path) as (out, owner, _endpoint, _), \
            subprocess.Popen([sys.executable, "-S", "-c", "import time;time.sleep(60)"], start_new_session=True) as foreign:
        try:
            row = runtime.proc(foreign.pid)
            with pytest.raises(runtime.SourceCleanupUnverified):
                runtime.cleanup_escaped_workers(out, runtime.capsule(out), owner, {}, {foreign.pid: row})
            assert foreign.poll() is None
            assert runtime.read_json(out / "cleanup.json")["unproven_processes"][0]["pid"] == foreign.pid
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)


def test_cleanup_barrier_stops_isolated_retry_and_new_output_launch(launch_files, tmp_path):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    env.update(GEAK_REPEAT_MODE="isolated_server", REPLICAS="2")
    env.pop("REPEATS")
    runtime.write_json(launch_files[0].parent / "source_cleanup_unverified.json", {"status": "unverified"})
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=10, check=False)
    assert run.returncode == 43, run.stdout + run.stderr
    assert "stopping isolated replicas without retry" in run.stderr
    out = Path(env["OUT_DIR"])
    assert not (out / "replica_001/attempt_2").exists()
    assert not (out / "replica_002").exists()
    assert not (out / "bench_summary.json").exists()
    assert not list(out.rglob("server.log"))


def test_pidfd_never_signals_a_process_whose_ancestry_does_not_match(launch_files, tmp_path, monkeypatch):
    with serving(launch_files, tmp_path) as (out, owner, _endpoint, _), \
            subprocess.Popen([sys.executable, "-S", "-c", "import time;time.sleep(60)"], start_new_session=True) as foreign:
        monkeypatch.setattr(runtime, "CLEANUP_TOTAL_SEC", .1)
        try:
            with pytest.raises(runtime.SourceCleanupUnverified):
                runtime.cleanup_escaped_workers(out, runtime.capsule(out), owner, {foreign.pid: runtime.proc(foreign.pid)}, {})
            assert foreign.poll() is None
            record = runtime.read_json(out / "cleanup.json")["processes"][0]
            assert not record["term_sent"] and not record["kill_sent"]
            assert record["error"] == "SourceRuntimeError"
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)


def test_cleanup_freeze_requires_actual_parent_acknowledgment(launch_files, tmp_path):
    marker = tmp_path / "unobserved-child"
    worker = f"import os,time;from pathlib import Path;Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(60)"
    startup = f"subprocess.Popen([sys.executable,'-S','-c',{worker!r}],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    with serving(launch_files, tmp_path, observe=False, startup=startup) as (out, owner, endpoint, _):
        child = int(wait_file(marker))
        try:
            with pytest.raises(runtime.SourceCleanupUnverified):
                runtime.gate(out, owner, endpoint, "prepared", 1)
            cleanup = runtime.read_json(out / "cleanup.json")
            assert cleanup["status"] == "unverified"
            assert cleanup["frozen_parents"] == []
            assert cleanup["processes"][0]["exit_confirmed"]
            assert (launch_files[0].parent / "source_cleanup_unverified.json").exists()
        finally:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_launch_registered_outsider_remains_untouched_and_blocks_recovery(launch_files, tmp_path):
    with serving(launch_files, tmp_path) as (out, owner, endpoint, env), \
            subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], env=env, start_new_session=True) as sibling:
        try:
            wait_file(out / f"registered-{sibling.pid}-{runtime.identity(sibling.pid)['start_ticks']}.json", sibling)
            with pytest.raises(runtime.SourceCleanupUnverified):
                runtime.gate(out, owner, endpoint, "prepared", 1)
            assert sibling.poll() is None
            cleanup = runtime.read_json(out / "cleanup.json")
            assert cleanup["processes"] == []
            assert cleanup["unproven_processes"][0]["pid"] == sibling.pid
        finally:
            sibling.terminate()
            sibling.wait(timeout=5)


@pytest.mark.parametrize("barrier_failure", [False, True])
def test_cleanup_evidence_failure_keeps_reserved_hard_stop(launch_files, tmp_path, monkeypatch, barrier_failure):
    with serving(launch_files, tmp_path) as (out, owner, _endpoint, _):
        write = runtime.write_json

        def deny(path, value, **kwargs):
            if Path(path).name == "cleanup.json" or (barrier_failure and Path(path).name == "source_cleanup_unverified.json"):
                raise PermissionError("evidence directory unavailable")
            return write(path, value, **kwargs)

        monkeypatch.setattr(runtime, "write_json", deny)
        with pytest.raises(runtime.SourceCleanupUnverified, match="source_cleanup_evidence_unwritable"):
            runtime.cleanup_escaped_workers(out, runtime.capsule(out), owner, {}, {})
        assert (launch_files[0].parent / "source_cleanup_unverified.json").exists() != barrier_failure


def test_cleanup_control_message_freezes_before_acknowledgment(observer):
    runtime.write_json(observer.out / "challenge.json", {"challenge_id": "freeze", "phase": "cleanup", "reply_socket": "unused"})
    tokens = iter([b"freeze", b"freeze"])
    sent = []
    observer.channel = SimpleNamespace(recv=lambda _size: next(tokens), close=lambda: None,
                                       sendto=lambda *args: sent.append((observer.measurement_started, args)))
    with pytest.raises(StopIteration):
        observer.respond()
    assert sent == [(True, (b"freeze frozen", "unused"))]


def moving_worker(marker, moved):
    return ("import os,signal,time\nfrom pathlib import Path\n"
            "def leave_group(*args):\n os.setsid()\n"
            f" Path({str(moved)!r}).write_text(str(os.getpgrp()))\n"
            "signal.signal(signal.SIGUSR1,leave_group)\n"
            f"Path({str(marker)!r}).write_text(str(os.getpid()))\ntime.sleep(60)")


def kill_fixture_process(pid):
    current = runtime.proc(pid)
    if current is not None and current["state"] != "Z":
        os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("escape_phase", ["after_finished", "after_freeze"])
def test_teardown_pins_worker_that_leaves_group_after_observation(launch_files, tmp_path, monkeypatch, escape_phase):
    marker, moved = tmp_path / "worker", tmp_path / "moved"
    startup = f"subprocess.Popen([sys.executable,'-c',{moving_worker(marker, moved)!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    with serving(launch_files, tmp_path, startup=startup) as (out, owner, endpoint, _), \
            subprocess.Popen([sys.executable, "-S", "-c", "import time;time.sleep(60)"], start_new_session=True) as foreign:
        pid = int(wait_file(marker))
        try:
            for phase in ("prepared", "ready", "finished"):
                assert len(runtime.gate(out, owner, endpoint, phase, 3)["processes"]) == 2

            def move():
                os.kill(pid, signal.SIGUSR1)
                assert int(wait_file(moved)) == pid

            if escape_phase == "after_finished":
                move()
            else:
                freeze = runtime.freeze_cleanup_parents

                def freeze_then_move(*args):
                    result = freeze(*args)
                    move()
                    return result

                monkeypatch.setattr(runtime, "freeze_cleanup_parents", freeze_then_move)
            cleanup = runtime.teardown_source(out, launch_files[0], owner)
            assert cleanup["status"] == "confirmed" and cleanup["scope"] == "owned_cohort_teardown"
            assert cleanup["launch_capsule_sha256"] == runtime.digest(out / "launch.json")
            assert {row["pid"] for row in cleanup["processes"]} == {owner["pid"], pid}
            assert all(row["exit_confirmed"] for row in cleanup["processes"])
            assert runtime.proc(pid) is None or runtime.proc(pid)["state"] == "Z"
            assert foreign.poll() is None
            assert not (launch_files[0].parent / "source_cleanup_unverified.json").exists()
        finally:
            kill_fixture_process(pid)
            foreign.terminate()
            foreign.wait(timeout=5)


@pytest.mark.parametrize("exit_status", [0, 7])
def test_bench_exit_cleans_worker_moved_by_post_summary_callback(launch_files, tmp_path, exit_status):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    request, _, root = launch_files
    marker, moved = tmp_path / "callback-worker", tmp_path / "callback-moved"
    path = root / "trees/a/python/sglang/launch_server.py"
    code = path.read_text().replace("import argparse,os", "import argparse,os,subprocess,sys,time\nfrom pathlib import Path")
    code = code.replace("class Handler", f"subprocess.Popen([sys.executable,'-c',{moving_worker(marker, moved)!r}])\n"
                        f"while not Path({str(marker)!r}).exists(): time.sleep(.01)\nclass Handler")
    path.write_text(code)
    update_source_request(request, root, path)
    # A staged callback after the real script's final seal models the native
    # post-measure callback position without replacing the real EXIT handler.
    staged = tmp_path / "bench_e2e.sh"
    staged.write_text((SCRIPTS / "bench_e2e.sh").read_text() + '\n'
                      'test -f "$OUT_DIR/source_runtime/measurement.json" || exit 98\n'
                      'kill -USR1 "$(cat "$TEST_WORKER")"\n'
                      'for _poll in $(seq 1 100); do [ -f "$TEST_MOVED" ] && break; sleep .02; done\n'
                      '[ -f "$TEST_MOVED" ] || exit 99\n'
                      f'exit {exit_status}\n')
    env.update(SKILL_DIR=str(SCRIPTS.parent), TEST_WORKER=str(marker), TEST_MOVED=str(moved))
    try:
        run = subprocess.run(["bash", str(staged)], env=env, text=True, capture_output=True, timeout=20, check=False)
        assert run.returncode == exit_status, run.stdout + run.stderr
        out = Path(env["OUT_DIR"]) / "source_runtime"
        cleanup = runtime.read_json(out / "teardown.json")
        pid = int(marker.read_text())
        assert int(moved.read_text()) == pid
        assert cleanup["status"] == "confirmed"
        assert pid in {row["pid"] for row in cleanup["processes"] if row["exit_confirmed"]}
        assert runtime.proc(pid) is None or runtime.proc(pid)["state"] == "Z"
        assert (out / "measurement.json").exists()
        assert not (request.parent / "source_cleanup_unverified.json").exists()
    finally:
        if marker.exists():
            kill_fixture_process(int(marker.read_text()))


@pytest.mark.parametrize("failure", ["health", "launch", "unobserved"])
def test_bench_launch_failures_cannot_skip_source_teardown(launch_files, tmp_path, failure):
    env = bench_fixture(launch_files, tmp_path, "sglang")
    adapter = Path(env["ADAPTER"])
    if failure == "launch":
        suffix = 'adapter_launch() { return 1; }\n'
    else:
        suffix = 'adapter_health() { return 1; }\n'
        env["SERVER_STARTUP_TIMEOUT_SEC"] = "0"
        if failure == "unobserved":
            suffix += ('adapter_launch() { ${SERVER_LAUNCH_PREFIX:-} "$TEST_PYTHON" -S -c "import time;time.sleep(60)" '
                       '> "$LOG" 2>&1 & SERVER_PID=$!; }\n')
    with adapter.open("a") as stream:
        stream.write(suffix)
    run = subprocess.run(["bash", str(SCRIPTS / "bench_e2e.sh")], env=env, text=True,
                         capture_output=True, timeout=15, check=False)
    assert run.returncode == (2 if failure == "health" else 43), run.stdout + run.stderr
    out = Path(env["OUT_DIR"])
    assert not (out / "bench_summary.json").exists()
    barrier = launch_files[0].parent / "source_cleanup_unverified.json"
    assert barrier.exists() == (failure != "health")
    if failure != "launch":
        assert runtime.read_json(out / "server_start.json")["status"] == "failed"
        cleanup = runtime.read_json(out / "source_runtime/teardown.json")
        assert cleanup["status"] == ("confirmed" if failure == "health" else "unverified")
        assert not runtime.group_members(cleanup["server_identity"]["pgid"])


def test_teardown_registered_foreign_worker_stays_alive_and_sets_barrier(launch_files, tmp_path):
    with serving(launch_files, tmp_path) as (out, owner, _endpoint, env), \
            subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], env=env, start_new_session=True) as sibling:
        try:
            wait_file(out / f"registered-{sibling.pid}-{runtime.identity(sibling.pid)['start_ticks']}.json", sibling)
            with pytest.raises(runtime.SourceCleanupUnverified):
                runtime.teardown_source(out, launch_files[0], owner)
            assert sibling.poll() is None
            cleanup = runtime.read_json(out / "teardown.json")
            assert cleanup["unproven_processes"][0]["pid"] == sibling.pid
            assert {row["pid"] for row in cleanup["processes"]} == {owner["pid"]}
            assert (launch_files[0].parent / "source_cleanup_unverified.json").exists()
        finally:
            sibling.terminate()
            sibling.wait(timeout=5)


@pytest.mark.parametrize("damage", ["gate_digest", "capsule_bytes", "overlay_manifest"])
def test_seal_refuses_changed_capsule_or_overlay_binding(launch_files, tmp_path, damage):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    manifest = overlay / "_overlay_manifest.json"
    manifest.write_text('{"modules":{}}\n')
    with serving(launch_files, tmp_path, overlay=str(overlay)) as (out, owner, endpoint, _):
        runtime.gate(out, owner, endpoint, "ready", 3)
        runtime.gate(out, owner, endpoint, "finished", 3)
        (out.parent / "bench_runs.jsonl").write_text('{"output_throughput":17}\n')
        (out.parent / "bench_summary.json").write_text('{"median":17}\n')
        sealed = runtime.seal_measurement(out)
        assert sealed["launch_capsule"]["sha256"] == runtime.digest(out / "launch.json")
        if damage == "gate_digest":
            gate = runtime.read_json(out / "gate-ready.json")
            gate["launch_capsule_sha256"] = "0" * 64
            runtime.write_json(out / "gate-ready.json", gate)
        elif damage == "capsule_bytes":
            with (out / "launch.json").open("a") as stream:
                stream.write(" ")
        else:
            manifest.write_text('{"modules":{"injected":"different"}}\n')
            # The unchanged capsule permanently retains the original inventory;
            # both a new gate and a verifier replay must observe this mismatch.
            assert runtime.capsule(out)["overlay_files"][str(manifest)] != runtime.digest(manifest)
            with pytest.raises(runtime.SourceRuntimeError, match="serving_process_reported_source_failure"):
                runtime.gate(out, owner, endpoint, "finished", 1)
            assert runtime.read_json(next(out.glob("fatal-*.json")))["reason"] == "authored_overlay_changed"
            return
        with pytest.raises(runtime.SourceRuntimeError, match="measurement_capsule_changed"):
            runtime.seal_measurement(out)


@pytest.mark.parametrize("barrier_writable", [False, True])
def test_invalid_teardown_identity_always_returns_reserved_hard_stop(launch_files, monkeypatch, barrier_writable):
    request, out, _ = launch_files
    runtime.prepare(request, out, "")
    if not barrier_writable:
        monkeypatch.setattr(runtime, "write_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")))
    monkeypatch.setattr(sys, "argv", ["source_runtime.py", "teardown", "--request", str(request), "--output-dir", str(out),
                                     "--pid", "0", "--pgid", "0", "--start-ticks", "0"])
    assert runtime.main() == 43
    assert (request.parent / "source_cleanup_unverified.json").exists() == barrier_writable


def test_teardown_confirms_pidfd_exit_when_owner_proc_lookup_races_term(launch_files, tmp_path, monkeypatch):
    with serving(launch_files, tmp_path) as (out, owner, _endpoint, _):
        original_identity, original_proc = runtime.identity, runtime.proc
        original_signal = runtime.signal.pidfd_send_signal
        owner_row = original_proc(owner["pid"])
        term_sent = False

        def send(fd, sig):
            nonlocal term_sent
            original_signal(fd, sig)
            term_sent = True

        def lookup(pid=None):
            if term_sent and pid == owner["pid"]:
                raise runtime.SourceRuntimeError("process_not_live")
            return original_identity(pid)

        # Model the owner exiting between two /proc reads after TERM, while a
        # pidfd remains reliable. Stale group membership must not skip fd exit.
        monkeypatch.setattr(runtime.signal, "pidfd_send_signal", send)
        monkeypatch.setattr(runtime, "identity", lookup)
        monkeypatch.setattr(runtime, "proc", lambda pid: owner_row if term_sent and pid == owner["pid"] else original_proc(pid))
        result = runtime.teardown_source(out, launch_files[0], owner)
        assert result["status"] == "confirmed"
        assert result["processes"][0]["term_sent"] and result["processes"][0]["exit_confirmed"]
