# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Removal proof against real CPU child argv, without framework/GPU imports."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from e2e_workflow.scripts.adapters import server_args as sa
from interface import effective_config


@pytest.mark.parametrize("declared,tokens", [
    ("--flag", ["--flag"]),
    ("--limit=-1 --keep yes", ["--limit=-1", "--keep", "yes"]),
    ('--config {"b": 2, "a": 1}', ["--config", '{"b": 2, "a": 1}']),
    ("--note 'has spaces --flag inside'", ["--note", "has spaces --flag inside"]),
    ("--flag old --flag=new", ["--flag", "old", "--flag=new"]),
])
def test_shared_parser_preserves_declared_and_raw_token_semantics(declared, tokens):
    assert effective_config._parse_flags is sa._parse_flags
    assert effective_config.resolve_remove_args is sa.resolve_remove_args
    assert sa._parse_flags(declared) == sa._parse_flag_tokens(tokens)


@pytest.mark.parametrize("spec,current,expected", [
    (["--flag"], "", ("--flag",)),
    (["--flag"], "--flag=new", ()),
    (["--flag=old"], "--flag old", ()),
    (["--flag old"], "--flag new", ("--flag old",)),
    (["--flag=-1"], "--flag -1", ()),
    (['--config {"a":1,"b":2}'], '--config {"b":2,"a":1}', ()),
])
def test_current_assignment_resolves_carried_removals(spec, current, expected):
    assert sa.resolve_remove_args(spec, current) == expected


@pytest.fixture
def child(tmp_path):
    """Launch a real Python module or console-script process with raw server argv."""
    processes = []
    counter = 0
    # Earlier interface-emission tests deliberately ignore SIGTERM. A real
    # exec inherits that disposition; give this CPU sleeper its own handler so
    # teardown remains independent of test order and the invoking shell.
    body = ("import os,signal,time\nfrom pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
            "Path(os.environ['READY']).write_text('ready')\ntime.sleep(60)\n")
    for relative in ("sglang/launch_server.py", "vllm/entrypoints/openai/api_server.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        parent = path.parent
        while parent != tmp_path:
            (parent / "__init__.py").touch()
            parent = parent.parent
    script = tmp_path / "bin/vllm"
    script.parent.mkdir()
    script.write_text("#!" + sys.executable + "\n" + body)
    script.chmod(0o755)

    def launch(kind="sglang", flags=(), *, python_options=(), console_command="serve"):
        nonlocal counter
        counter += 1
        ready = tmp_path / f"ready-{counter}"
        env = dict(os.environ, PYTHONPATH=str(tmp_path), READY=str(ready))
        if kind == "sglang":
            command = [sys.executable, *python_options, "-B", "-m", "sglang.launch_server"]
            backend = "sglang"
        elif kind == "vllm-module":
            command = [sys.executable, *python_options, "-m", "vllm.entrypoints.openai.api_server"]
            backend = "vllm"
        elif kind == "vllm-console":
            command = [str(script), console_command, "/cpu-only/model"]
            backend = "vllm"
        else:
            command = [sys.executable, "-c", body, "sglang", "18080"]
            backend = "sglang"
        process = subprocess.Popen(command + ["--port", "18080", *flags], env=env)
        processes.append(process)
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None, "CPU argv child exited before readiness"
            assert time.monotonic() < deadline, "CPU argv child failed to signal readiness"
            time.sleep(0.01)
        return process, {"pid": process.pid, "start_ticks": sa._start_ticks(Path('/proc'), process.pid),
                             "backend": backend, "port": "18080"}

    yield launch
    for process in processes:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize("kind", ["sglang", "vllm-module", "vllm-console"])
@pytest.mark.parametrize("flags,removal,current,status", [
    (["--keep", "1"], ["--removed"], "--keep 1", "verified"),
    (["--removed"], ["--removed"], "", "failed"),
    (["--target=old"], ["--target old"], "", "failed"),
    (["--target", "new"], ["--target=old"], "", "verified"),
    (["--target=old", "--target", "new"], ["--target old"], "--target new", "ambiguous_repeated_option"),
    (["--target=new", "--target", "old"], ["--target old"], "--target new", "ambiguous_repeated_option"),
    (["--target=new", "--target", "newer"], ["--target old"], "--target newer", "verified"),
    (["--target-extra", "old"], ["--target"], "", "verified"),
    (["--target", "-1"], ["--target=-1"], "", "failed"),
    (["--config", '{"b":2, "a":1}'], ['--config {"a":1,"b":2}'], "", "failed"),
    (["--note", "inside --target value"], ["--target"], "", "verified"),
    ([], ["--removed"], "", "verified"),
])
def test_actual_server_argv_removal_matrix(child, kind, flags, removal, current, status):
    process, identity = child(kind, flags)
    result = sa.validate_launch(**identity, remove_args=removal, current_args=current)
    assert result["status"] == ("failed" if status == "ambiguous_repeated_option" else status)
    assert result["pid"] == process.pid
    assert result["observed_start_ticks"] == identity["start_ticks"]
    assert result["source_hashes"] == sa._source_hashes()
    if flags:
        assert result["argv"][-len(flags):] == flags
    else:
        assert result["argv"][-2:] == ["--port", "18080"]
    if status == "failed":
        assert result["reason"] == "removal_mismatch"
    elif status == "ambiguous_repeated_option":
        assert result["reason"] == status
        assert result["ambiguous_repeated_options"] == ["--target old"]


def test_explicit_readdition_and_empty_controls_preserve_legacy_behavior():
    for removals, current in [([], "legacy positional text"), (["--removed"], "--removed")]:
        result = sa.validate_launch(pid="", start_ticks="", backend="unknown", port="",
                                    remove_args=removals, current_args=current)
        assert result["status"] == "not_required"


@pytest.mark.parametrize("change,reason", [
    ({"start_ticks": ""}, "unverified_process"),
    ({"start_ticks": "1"}, "process_identity_mismatch"),
    ({"port": "18081"}, "port_mismatch"),
    ({"backend": "other"}, "unsupported_entrypoint"),
    ({"pid": "1"}, "unverified_process"),
])
def test_missing_or_wrong_process_evidence_fails(child, change, reason):
    _, identity = child()
    identity.update(change)
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["status"] == "failed"
    assert result["reason"] == reason


def test_backend_or_port_substrings_do_not_attest_wrapper(child):
    _, identity = child("wrapper")
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["status"] == "failed"
    assert result["reason"] == "unsupported_entrypoint"


@pytest.mark.parametrize("kind", ["sglang", "vllm-module"])
@pytest.mark.parametrize("options", [("-W", "ignore"), ("-Wignore",), ("-X", "utf8"), ("-Xutf8",)])
def test_python_runtime_options_preserve_actual_server_argument_proof(child, kind, options):
    process, identity = child(kind, ["--keep", "1"], python_options=options)
    result = sa.validate_launch(**identity, remove_args=["--removed"], current_args="--keep 1")
    assert result["status"] == "verified"
    assert result["pid"] == process.pid
    assert result["argv"][1:1 + len(options)] == list(options)
    assert result["effective_flags"] == "--port 18080 --keep 1"


@pytest.mark.parametrize("command", ["bench", "complete"])
def test_live_vllm_console_non_server_commands_do_not_attest_a_server(child, command):
    process, identity = child("vllm-console", console_command=command)
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert process.poll() is None
    assert result["status"] == "failed"
    assert result["reason"] == "unsupported_entrypoint"
    assert result["detail"] == "vllm command is not 'serve'"


def test_matching_live_process_owned_by_another_effective_user_is_rejected(child, monkeypatch):
    process, identity = child()
    owner = Path("/proc", str(process.pid)).stat().st_uid
    assert owner == os.geteuid()
    monkeypatch.setattr(sa.os, "geteuid", lambda: owner + 1)
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["status"] == "failed"
    assert result["reason"] == "unverified_process"
    assert result["detail"] == "server PID belongs to another effective user"
    assert "argv" not in result
    assert "observed_start_ticks" not in result


def test_exited_process_fails(child):
    process, identity = child()
    process.terminate()
    process.wait(timeout=5)
    assert sa.validate_launch(**identity, remove_args=["--removed"])["status"] == "failed"


def fake_proc(tmp_path, argv):
    proc = tmp_path / "proc"
    pid = proc / "1234"
    pid.mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("test-boot")
    (pid / "stat").write_text("1234 (name with ) parens) S " + "0 " * 18 + "12345 0\n")
    (pid / "cmdline").write_bytes(b"\0".join(token.encode() for token in argv) + b"\0")
    return {"pid": 1234, "start_ticks": "12345", "backend": "sglang", "port": "18080", "proc_root": proc}


@pytest.mark.parametrize("argv", [
    ["VLLM::Server"],
    ["bash", "/some/sglang.sh", "--port", "18080"],
    ["python3", "-m", "sglang.launch_server_extra", "--port", "18080"],
    ["python3", "-c", "sglang.launch_server --port 18080"],
    ["python3", "-m", "sglang.launch_server", "--port", "18080", "--", "--other"],
])
def test_unknown_or_ambiguous_argv_never_passes(tmp_path, argv):
    identity = fake_proc(tmp_path, argv)
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["status"] == "failed"


@pytest.mark.parametrize("problem", ["empty", "missing", "unterminated", "bad-stat", "zombie"])
def test_unreadable_incomplete_process_fails(tmp_path, problem):
    identity = fake_proc(tmp_path, ["python3", "-m", "sglang.launch_server", "--port=18080"])
    pid = identity["proc_root"] / "1234"
    if problem == "empty":
        (pid / "cmdline").write_bytes(b"")
    elif problem == "missing":
        (pid / "cmdline").unlink()
    elif problem == "unterminated":
        (pid / "cmdline").write_bytes(b"python3")
    elif problem == "bad-stat":
        (pid / "stat").write_text("bad stat")
    else:
        (pid / "stat").write_text((pid / "stat").read_text().replace(") S ", ") Z "))
    assert sa.validate_launch(**identity, remove_args=["--removed"])["status"] == "failed"


@pytest.mark.parametrize("problem,detail", [
    ("empty-boot", "boot identity is unavailable"),
    ("zero-start", "server start ticks are empty or invalid"),
])
def test_missing_boot_or_zero_start_cannot_identify_an_otherwise_matching_process(tmp_path, problem, detail):
    identity = fake_proc(tmp_path, ["python3", "-m", "sglang.launch_server", "--port=18080"])
    if problem == "empty-boot":
        (identity["proc_root"] / "sys/kernel/random/boot_id").write_text(" \n")
    else:
        stat = identity["proc_root"] / "1234/stat"
        stat.write_text(stat.read_text().replace("12345", "0"))
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["status"] == "failed"
    assert result["reason"] == "unverified_process"
    assert result["detail"] == detail
    assert "argv" not in result


def test_python_runtime_options_without_a_program_do_not_attest_a_server(tmp_path):
    identity = fake_proc(tmp_path, ["python3", "-B", "-Wignore", "-X", "utf8"])
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["status"] == "failed"
    assert result["reason"] == "unsupported_entrypoint"
    assert result["detail"] == "Python command has no server entrypoint"


@pytest.mark.parametrize("command,status", [("serve", "verified"), ("bench", "failed")])
def test_direct_console_argv_requires_the_server_subcommand(tmp_path, command, status):
    identity = fake_proc(tmp_path, ["/venv/bin/vllm", command, "/cpu-only/model", "--port=18080"])
    identity["backend"] = "vllm"
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["status"] == status
    if status == "failed":
        assert result["reason"] == "unsupported_entrypoint"
    else:
        assert result["effective_flags"] == "--port 18080"


def test_start_ticks_change_during_capture_fails(tmp_path, monkeypatch):
    identity = fake_proc(tmp_path, ["python3", "-m", "sglang.launch_server", "--port=18080"])
    values = iter(["12345", "12346"])
    monkeypatch.setattr(sa, "_start_ticks", lambda *_: next(values))
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["reason"] == "process_identity_mismatch"


def test_process_identity_change_after_argument_parsing_invalidates_proof(tmp_path, monkeypatch):
    identity = fake_proc(tmp_path, ["python3", "-m", "sglang.launch_server", "--port=18080"])
    original = sa.server_flag_tokens

    def replace_process_during_interpretation(argv, backend):
        tokens = original(argv, backend)
        stat = identity["proc_root"] / "1234/stat"
        stat.write_text(stat.read_text().replace("12345", "12346"))
        return tokens

    monkeypatch.setattr(sa, "server_flag_tokens", replace_process_during_interpretation)
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    assert result["observed_start_ticks"] == "12345"
    assert result["effective_flags"] == "--port 18080"
    assert result["status"] == "failed"
    assert result["reason"] == "process_identity_mismatch"
    assert result["detail"] == "server identity changed during validation"


def test_matching_live_reuse_receipt_and_changed_controls(child, tmp_path):
    process, identity = child(flags=["--keep", "1"])
    result = sa.validate_launch(**identity, remove_args=["--removed"], current_args="--keep 1")
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(result))
    options = {"launch_receipt": path, "backend": "sglang", "port": "18080",
                   "remove_args": ["--removed"], "current_args": "--keep 1"}
    assert sa.validate_reuse(**options)["status"] == "verified"
    assert sa.validate_reuse(**dict(options, current_args="--keep 2"))["reason"] == "reuse_receipt_mismatch"
    process.terminate()
    process.wait(timeout=5)
    assert sa.validate_reuse(**options)["status"] == "failed"


@pytest.mark.parametrize("change", [
    {"status": "failed"}, {"source_hashes": {}}, {"boot_id": "another-boot"},
    {"observed_start_ticks": "1"}, {"argv_digest": "changed"}, {"control_digest": "changed"},
])
def test_stale_or_different_reuse_receipt_fails(child, tmp_path, change):
    _, identity = child()
    result = sa.validate_launch(**identity, remove_args=["--removed"])
    result.update(change)
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(result))
    assert sa.validate_reuse(launch_receipt=path, backend="sglang", port="18080",
                             remove_args=["--removed"])["status"] == "failed"


@pytest.mark.parametrize("value", [12, [None], ["positional"], {"--flag": True}])
def test_invalid_controls_reject_before_observation(value, monkeypatch):
    monkeypatch.setattr(sa, "_observe", lambda *_: pytest.fail("must reject before reading proc"))
    result = sa.validate_launch(pid="", start_ticks="", backend="sglang", port="18080", remove_args=value)
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_controls"


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_invalid_expected_port_rejects_before_process_observation(port, monkeypatch):
    monkeypatch.setattr(sa, "_observe", lambda *_: pytest.fail("invalid port must reject before reading proc"))
    result = sa.validate_launch(pid="1234", start_ticks="12345", backend="sglang",
                                port=port, remove_args=["--removed"])
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_controls"
    assert result["detail"] == "expected port must be an integer between 1 and 65535"


def test_staged_cli_writes_source_bound_success_and_failure(child, tmp_path):
    stage = tmp_path / "staged"
    stage.mkdir()
    for filename in ("server_args.py", "extra_env.py"):
        shutil.copyfile(Path(sa.__file__).with_name(filename), stage / filename)
    _, identity = child(flags=["--default-on"])
    receipt = tmp_path / "receipt.json"
    command = [sys.executable, str(stage / "server_args.py"), "validate",
               "--pid", str(identity["pid"]), "--start-ticks", identity["start_ticks"],
               "--backend", "sglang", "--port", "18080", "--receipt", str(receipt)]
    env = dict(os.environ, PYTHONPATH="")
    failed = subprocess.run(command + ["--remove-args", '["--default-on"]'],
                            cwd=stage, env=env, capture_output=True, text=True, check=False)
    assert failed.returncode == 2
    assert json.loads(receipt.read_text())["reason"] == "removal_mismatch"
    passed = subprocess.run(command + ["--remove-args", '["--absent"]'],
                            cwd=stage, env=env, capture_output=True, text=True, check=False)
    assert passed.returncode == 0
    proof = json.loads(receipt.read_text())
    assert proof["status"] == "verified"
    assert proof["source_hashes"] == sa._source_hashes()
    reused = subprocess.run([sys.executable, str(stage / "server_args.py"), "validate-reuse",
        "--launch-receipt", str(receipt), "--backend", "sglang", "--port", "18080",
        "--remove-args", '["--absent"]'], cwd=stage, env=env, capture_output=True, text=True, check=False)
    assert reused.returncode == 0, reused.stdout + reused.stderr
    malformed = subprocess.run(command + ["--remove-args", '{not JSON}'],
                               cwd=stage, env=env, capture_output=True, text=True, check=False)
    assert malformed.returncode == 2
    assert json.loads(receipt.read_text())["reason"] == "invalid_controls"


@pytest.mark.parametrize("removals,current,expected", [
    ('["--flag=old", "--flag old", "--absent"]', "", ["--absent", "--flag old"]),
    ('["--flag"]', "--flag", []),
    ('["--flag old"]', "--flag new", ["--flag old"]),
    ('"--flag=-1"', "--flag -1", []),
    ('[]', "legacy positional text", []),
    ('null', "", []),
])
def test_resolve_cli_prints_only_canonical_active_json(removals, current, expected, capsys):
    assert sa.main(["resolve", "--remove-args=" + removals, "--current-args=" + current]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert captured.err == ""


@pytest.mark.parametrize("removals,current", [
    ("not-json", ""), ("12", ""), ("true", ""), ('[null]', ""),
    ('{"--flag":true}', ""), ('["positional"]', ""), ('["--flag"]', "'broken"),
])
def test_resolve_cli_invalid_controls_fail_without_json_output(removals, current, capsys):
    assert sa.main(["resolve", "--remove-args=" + removals, "--current-args=" + current]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid server argument controls:" in captured.err


def test_validate_cli_persists_the_same_source_bound_live_child_proof_it_prints(child, tmp_path, capsys):
    process, identity = child(flags=["--keep", "1"])
    receipt = tmp_path / "launch.json"
    code = sa.main(["validate", "--pid", str(process.pid), "--start-ticks", identity["start_ticks"],
                    "--backend", "sglang", "--port", "18080", "--receipt", str(receipt),
                    '--remove-args=["--removed"]', "--current-args=--keep 1"])
    output = capsys.readouterr()
    assert code == 0
    assert output.err == ""
    proof = json.loads(output.out)
    assert proof == json.loads(receipt.read_text())
    assert proof["status"] == "verified"
    assert proof["pid"] == process.pid
    assert proof["observed_start_ticks"] == identity["start_ticks"]
    assert proof["source_hashes"] == sa._source_hashes()
    assert proof["active_remove_args"] == ["--removed"]
    assert proof["argv"][-4:] == ["--port", "18080", "--keep", "1"]


@pytest.mark.parametrize("command", ["validate", "validate-reuse"])
def test_validation_cli_malformed_json_writes_typed_failure_without_observation(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setattr(sa, "validate_launch", lambda **_: pytest.fail("malformed JSON must not observe a process"))
    monkeypatch.setattr(sa, "validate_reuse", lambda **_: pytest.fail("malformed JSON must not reuse a process"))
    receipt = tmp_path / "launch.json"
    code = sa.main([command, "--backend", "sglang", "--port", "18080",
                    "--receipt", str(receipt), "--remove-args={not JSON}"])
    output = capsys.readouterr()
    assert code == 2
    assert output.err == ""
    result = json.loads(output.out)
    assert result == json.loads(receipt.read_text())
    assert result["schema_version"] == sa.SCHEMA
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_controls"
    assert result["detail"]


def test_validation_cli_cannot_report_success_when_proof_cannot_be_saved(child, tmp_path, capsys):
    process, identity = child()
    destination = tmp_path / "launch.json"
    destination.mkdir()
    existing = destination / "preserved"
    existing.write_text("keep")
    code = sa.main(["validate", "--pid", str(process.pid), "--start-ticks", identity["start_ticks"],
                    "--backend", "sglang", "--port", "18080", "--receipt", str(destination),
                    '--remove-args=["--removed"]'])
    output = capsys.readouterr()
    assert code == 2
    assert output.out == ""
    assert "server argument validation: cannot write receipt:" in output.err
    assert existing.read_text() == "keep"
    assert list(tmp_path.glob("launch.json.*")) == []
    assert process.poll() is None


@pytest.mark.parametrize("host", ["127.0.0.1", "127.2.3.4", "localhost", "::1", "0:0:0:0:0:0:0:1"])
def test_loopback_hosts_are_bound_to_successful_receipts(child, host):
    _, identity = child()
    result = sa.validate_launch(**identity, remove_args=["--removed"], host=host)
    assert result["status"] == "verified"
    assert result["host"] in (host, "::1")
    assert result["control_digest"]


@pytest.mark.parametrize("host", ["10.0.0.5", "192.168.1.1", "remote.example", "0.0.0.0", "::", ""])
def test_remote_hosts_cannot_be_attested_by_matching_local_pid(child, tmp_path, host):
    _, identity = child()
    proof = sa.validate_launch(**identity, remove_args=["--removed"])
    path = tmp_path / "local.json"
    path.write_text(json.dumps(proof))
    launched = sa.validate_launch(**identity, remove_args=["--removed"], host=host)
    reused = sa.validate_reuse(launch_receipt=path, backend="sglang", port="18080",
                               remove_args=["--removed"], host=host)
    for result in (launched, reused):
        assert result["status"] == "failed"
        assert result["reason"] == "unsupported_host"
        assert "argv" not in result


def test_reuse_receipt_cannot_attest_different_loopback_host(child, tmp_path):
    _, identity = child()
    path = tmp_path / "local.json"
    path.write_text(json.dumps(sa.validate_launch(**identity, remove_args=["--removed"])))
    result = sa.validate_reuse(launch_receipt=path, backend="sglang", host="127.0.0.2",
                               port="18080", remove_args=["--removed"])
    assert result["reason"] == "reuse_receipt_mismatch"


def test_reuse_reads_before_atomically_replacing_same_receipt(child, tmp_path, capsys):
    _, identity = child()
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(sa.validate_launch(**identity, remove_args=["--removed"])))
    assert sa.main(["validate-reuse", "--launch-receipt", str(path), "--receipt", str(path),
                    "--backend", "sglang", "--host", "127.0.0.1", "--port", "18080",
                    '--remove-args=["--removed"]']) == 0
    assert json.loads(path.read_text())["status"] == "verified"
    assert json.loads(capsys.readouterr().out)["validation_kind"] == "reuse"
    assert list(tmp_path.glob("launch.json.*")) == []


def test_failed_receipt_serialization_preserves_previous_proof_and_cleans_temp(tmp_path):
    path = tmp_path / "launch.json"
    path.write_text('{"status":"verified"}\n')
    with pytest.raises(TypeError):
        sa._write_receipt(str(path), {"not_json": object()})
    assert json.loads(path.read_text())["status"] == "verified"
    assert list(tmp_path.glob("launch.json.*")) == []
