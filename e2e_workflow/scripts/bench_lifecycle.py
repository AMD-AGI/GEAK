#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional post-measurement callback; evaluator and quality policy belong to the caller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA = "geak.post_measure.v1"
RESULT_NAME = "native_receipt.json"  # Opaque caller-owned JSON, never scored here.
MAX_JSON_BYTES = 1024 * 1024
HERE = Path(__file__).resolve().parent


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def _read(path, *, with_digest=False):
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds size limit")
    result = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if not isinstance(result, dict):
        raise TypeError("expected a JSON object")
    return (result, hashlib.sha256(raw).hexdigest()) if with_digest else result


def _write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        _write(temporary, value, exclusive=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore(path, data):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _proc(pid):
    """Read the existing teardown identity and state; never infer an owner from a name."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "pgid": int(fields[2]), "start_ticks": int(fields[19]), "state": fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _matches(owner):
    current = _proc(owner["pid"])
    return bool(current and current["state"] != "Z" and owner["boot_id"] == _boot_id()
                and all(current[key] == owner[key] for key in ("pid", "pgid", "start_ticks")))


def _group_live(pgid):
    return any((p := _proc(int(entry.name))) and p["pgid"] == pgid and p["state"] != "Z"
               for entry in Path("/proc").iterdir() if entry.name.isdigit())


def _identity(pid, pgid, start_ticks, protected):
    owner = {"pid": int(pid), "pgid": int(pgid), "start_ticks": int(start_ticks),
             "boot_id": _boot_id(), "protected_pgids": protected}
    if owner["pid"] <= 1 or owner["pgid"] != owner["pid"] or owner["start_ticks"] <= 0:
        raise ValueError("post-measurement requires a verified owned server process group")
    if owner["pgid"] in protected or not _matches(owner):
        raise ValueError("server identity is unavailable or protected")
    return owner


def _teardown(owner, grace):
    """Reuse GEAK's staged teardown policy with the launch-time identity restored."""
    if not _matches(owner):
        return "identity_unavailable"
    script = r'''
source "$1"
SERVER_PID="$2"
SERVER_PGID="$3"
SERVER_START_TICKS="$4"
SERVER_GROUP_UNVERIFIED=0
SERVER_PROTECTED_PGIDS="$SERVER_PROTECTED_PGIDS $5"
SERVER_STOP_GRACE_S="$6"
server_teardown
'''
    subprocess.run(
        ["bash", "-c", script, "geak-owned-cleanup", str(HERE / "server_teardown.sh"),
         str(owner["pid"]), str(owner["pgid"]), str(owner["start_ticks"]),
         " ".join(str(p) for p in owner["protected_pgids"]), str(grace)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=grace + 5,
    )
    return "recorded_group_gone" if not _group_live(owner["pgid"]) else "recorded_group_survives"


def capabilities():
    names = ("bench_e2e.sh", "bench_replica.sh", "server_teardown.sh", "bench_summarize.py", "bench_lifecycle.py")
    return {"schema": SCHEMA, "measurement_modes": ["warm_server", "isolated_server"],
            "requires_fresh_server": True, "profile": 0, "ownership": "verified_process_group",
            "endpoint_binding": "literal_ipv4_loopback_process_group_listener", "endpoint_host": "127.0.0.1",
            "max_json_bytes": MAX_JSON_BYTES, "max_throughput_artifact_bytes": 8 * MAX_JSON_BYTES,
            "callback_result_name": RESULT_NAME, "receipt_name": "post_measure_receipt.json",
            "manifest_name": "post_measure_manifest.json",
            "sources": {name: _digest(HERE / name) for name in names}}


def prepare(out, request_path, mode):
    request_path = Path(request_path)
    if not request_path.is_absolute():
        raise ValueError("GEAK_POST_MEASURE_REQUEST must be absolute")
    request, request_sha256 = _read(request_path, with_digest=True)
    if request.get("schema") != SCHEMA:
        raise ValueError("unsupported post-measurement schema")
    for key in ("request_id", "measurement_epoch"):
        if not isinstance(request.get(key), str) or str(uuid.UUID(request[key])) != request[key]:
            raise ValueError("request and epoch must be canonical UUIDs")
    for key in ("contract_sha256", "expected_config_sha256"):
        if not _hex_digest(request.get(key)):
            raise ValueError("contract and expected config require full SHA256 digests")
    argv = request.get("callback_argv")
    if (not isinstance(argv, list) or not argv or len(argv) > 128
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv)
            or not Path(argv[0]).is_absolute() or not os.access(argv[0], os.X_OK)):
        raise ValueError("callback_argv must name an absolute executable and literal arguments")
    timeout = request.get("timeout_sec")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_sec must be finite and positive")
    if mode == "legacy" and os.environ.get("GEAK_ISOLATED_REPLICA") == "1":
        mode = "isolated_server"
    if (mode not in ("warm_server", "isolated_server") or os.environ.get("PROFILE", "0") != "0"
            or os.environ.get("REUSE_SERVER", "0") != "0" or os.environ.get("REPEATS") == "0"
            or os.environ.get("BENCH_CLIENT", "native") not in ("native", "inferencex")
            or os.environ.get("GEAK_ISL_OSL_INACTIVE") == "1"):
        raise ValueError("post-measurement supports only fresh synthetic warm/isolated runs with PROFILE=0")
    if request["expected_config_sha256"] != os.environ.get("EFFECTIVE_CONFIG_DIGEST"):
        raise ValueError("expected config does not match the declared EFFECTIVE_CONFIG_DIGEST")
    # Discover missing staged dependencies before a server or serving-GPU lock exists.
    capsule = {"request": request, "request_sha256": request_sha256, "source_path": str(request_path),
               "measurement_mode": mode, "capabilities": capabilities()}
    _write(out / "post_measure/request.json", capsule, exclusive=True)


def _capsule(out):
    capsule = _read(out / "post_measure/request.json")
    if _digest(capsule["source_path"]) != capsule["request_sha256"]:
        raise ValueError("post-measurement request changed during the attempt")
    if capsule["capabilities"] != capabilities():
        raise ValueError("staged lifecycle sources changed during the attempt")
    return capsule


def record(out, args):
    capsule = _capsule(out)
    protected = [int(p) for p in args.protected_pgids.split()]
    protected += [1, os.getpgrp()]
    if args.group_unverified != "0":
        raise ValueError("server group is unverified")
    owner = _identity(args.pid, args.pgid, args.start_ticks, sorted(set(protected)))
    request = capsule["request"]
    _write(out / "post_measure/owner.json", {
        "schema": "geak.launch_owner.v1", "request_id": request["request_id"],
        "measurement_epoch": request["measurement_epoch"], "server_identity": owner,
        "launch_nonce": str(uuid.uuid4()), "recorded_at_ns": time.time_ns(),
    }, exclusive=True)


def _listener(owner, base_url):
    """Observe the loopback listening socket in the already verified server group."""
    endpoint = urlsplit(base_url)
    if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.path not in ("", "/") or not endpoint.port):
        raise ValueError("only a literal local IPv4 HTTP endpoint at 127.0.0.1 is supported")
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        address, port = fields[1].split(":")
        if fields[3] == "0A" and int(port, 16) == endpoint.port and address in ("0100007F", "00000000"):
            inodes.add(fields[9])
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        current = _proc(int(entry.name))
        if not current or current["pgid"] != owner["pgid"]:
            continue
        try:
            for descriptor in (entry / "fd").iterdir():
                target = os.readlink(descriptor)
                after = _proc(current["pid"])
                same = after and after["state"] != "Z" and all(after[key] == current[key] for key in ("pid", "pgid", "start_ticks"))
                if target.startswith("socket:[") and target[8:-1] in inodes and same:
                    return {"binding": "owned_process_group_listener", "port": endpoint.port,
                            "pid": current["pid"], "start_ticks": current["start_ticks"]}
        except (OSError, ValueError):
            continue
    raise ValueError("endpoint listener does not belong to the recorded server group")


def _observation(owner, base_url):
    if not _matches(owner):
        raise ValueError("recorded server is no longer alive")
    proc = Path(f"/proc/{owner['pid']}")
    return {"argv_sha256": _digest(proc / "cmdline"), "environ_sha256": _digest(proc / "environ"),
            "executable": os.readlink(proc / "exe"), "listener": _listener(owner, base_url)}


def ready(out, args):
    capsule = _capsule(out)
    owner = _read(out / "post_measure/owner.json")
    context = {**owner, "schema": "geak.launch_context.v1", "request_sha256": capsule["request_sha256"],
               "contract_sha256": capsule["request"]["contract_sha256"],
               "measurement_mode": capsule["measurement_mode"], "replica_index": args.replica_index,
               "replica_attempt": args.replica_attempt, "endpoint": {"base_url": args.base_url},
               "declared_config_sha256": capsule["request"]["expected_config_sha256"],
               "observed": _observation(owner["server_identity"], args.base_url), "ready_at_ns": time.time_ns()}
    _write(out / "post_measure/launch_context.json", context, exclusive=True)


def _artifact(path, out):
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact is missing or a symlink")
    return {"path": str(path.relative_to(out)), "sha256": _digest(path), "bytes": path.stat().st_size}


def supervise(out):
    """Keep a live group leader until the owner finishes existing teardown.

    A caught signal (not SIG_IGN) resets to default on exec in the callback.
    The supervisor itself stays alive through TERM so teardown can safely
    escalate against the same verified group if a callback child ignores it.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: None)
    capsule = _capsule(out)
    output = out / "post_measure/output"
    argv = capsule["request"]["callback_argv"] + [
        "--launch-context", str(out / "post_measure/launch_context.json"), "--output-dir", str(output)]
    try:
        rc = subprocess.call(argv)
    except OSError:
        rc = None
    _write(output / "callback_exit.json", {"returncode": rc})
    while True:
        signal.pause()


def run_callback(out):
    capsule = _capsule(out)
    request = capsule["request"]
    context_path = out / "post_measure/launch_context.json"
    context = _read(context_path)
    receipt = {"schema": "geak.post_measure.receipt.v1", "request_id": request["request_id"],
               "measurement_epoch": request["measurement_epoch"], "contract_sha256": request["contract_sha256"],
               "request_sha256": capsule["request_sha256"], "launch_nonce": context["launch_nonce"],
               "launch_context": _artifact(context_path, out), "status": "not_started",
               "callback_returncode": None, "callback_result": None, "callback_cleanup_status": "not_started",
               "measurement_finished_at_ns": time.time_ns()}
    proc = None
    callback_owner = None
    measured_bytes = {}
    stopped = [None]
    previous = {sig: signal.signal(sig, lambda signum, frame: stopped.__setitem__(0, signum))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        summary_path = out / "bench_summary.json"
        summary = _read(summary_path)
        value = summary.get("throughput_tok_s_median")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("no valid measured throughput")
        receipt["throughput_artifacts"] = [_artifact(summary_path, out), _artifact(out / "bench_runs.jsonl", out)]
        for artifact in receipt["throughput_artifacts"]:
            path = out / artifact["path"]
            if artifact["bytes"] > 8 * MAX_JSON_BYTES:
                raise ValueError("throughput artifact exceeds sealing limit")
            measured_bytes[path] = path.read_bytes()
            if hashlib.sha256(measured_bytes[path]).hexdigest() != artifact["sha256"]:
                raise ValueError("throughput changed while sealing")
            sealed = out / "post_measure/throughput" / path.name
            sealed.parent.mkdir(parents=True, exist_ok=True)
            _restore(sealed, measured_bytes[path])
            os.chmod(sealed, 0o400)
        _write(out / "post_measure/measurement.json", {
            "schema": "geak.post_measure.measurement.v1", "request_id": request["request_id"],
            "measurement_epoch": request["measurement_epoch"], "launch_nonce": context["launch_nonce"],
            "throughput_artifacts": receipt["throughput_artifacts"],
        }, exclusive=True)
        if _observation(context["server_identity"], context["endpoint"]["base_url"]) != context["observed"]:
            raise ValueError("observed server changed after readiness")
        output = out / "post_measure/output"
        output.mkdir(mode=0o700)
        receipt["evaluation_started_at_ns"] = time.time_ns()
        deadline = time.monotonic() + request["timeout_sec"]
        exit_path = output / "callback_exit.json"
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            os.chmod(stdout.name, 0o600)
            os.chmod(stderr.name, 0o600)
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "supervise", "--output-dir", str(out)],
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            current = _proc(proc.pid)
            if current:
                callback_owner = _identity(proc.pid, proc.pid, current["start_ticks"], [1, os.getpgrp()])
            while proc.poll() is None and not exit_path.exists() and stopped[0] is None and time.monotonic() < deadline:
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            if stopped[0] is not None:
                receipt["status"] = "cancelled"
            elif not exit_path.exists() and proc.poll() is None:
                receipt["status"] = "timed_out"
            elif not exit_path.exists():
                receipt["status"] = "supervisor_failed"
            else:
                receipt["callback_returncode"] = _read(exit_path)["returncode"]
                receipt["status"] = "completed" if receipt["callback_returncode"] == 0 else "failed"
            if proc.poll() is None and callback_owner:
                receipt["callback_cleanup_status"] = _teardown(callback_owner, 1)
            proc.wait(timeout=3)
        if callback_owner and receipt["callback_cleanup_status"] == "not_started":
            receipt["callback_cleanup_status"] = (
                "recorded_group_gone" if not _group_live(callback_owner["pgid"]) else "recorded_group_survives")
        result_path = output / RESULT_NAME
        if result_path.exists():
            _read(result_path)
            receipt["callback_result"] = _artifact(result_path, out)
        elif receipt["status"] == "completed":
            receipt["status"] = "missing_result"
        if receipt["status"] == "completed" and receipt["callback_cleanup_status"] != "recorded_group_gone":
            receipt["status"] = "callback_cleanup_unconfirmed"
        if _observation(context["server_identity"], context["endpoint"]["base_url"]) != context["observed"]:
            receipt["status"] = "identity_mismatch"
        _capsule(out)
        for artifact in [receipt["launch_context"], *receipt["throughput_artifacts"]]:
            if _digest(out / artifact["path"]) != artifact["sha256"]:
                receipt["status"] = "binding_changed"
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        # Opaque evaluator output and exception text can contain secrets. Status is sufficient here.
        receipt["status"] = "invalid_result_or_context"
    finally:
        if proc is not None and proc.poll() is None and callback_owner:
            try:
                receipt["callback_cleanup_status"] = _teardown(callback_owner, 1)
                proc.wait(timeout=3)
            except (OSError, ValueError, subprocess.SubprocessError):
                receipt["callback_cleanup_status"] = "cleanup_unconfirmed"
        # A buggy callback can overwrite or remove the public summary. Restore
        # the measured bytes before the isolated scheduler consumes them; the
        # quality receipt remains failed and cannot buy a replacement attempt.
        for path, data in measured_bytes.items():
            try:
                changed = path.is_symlink() or not path.is_file() or _digest(path) != hashlib.sha256(data).hexdigest()
                if changed:
                    receipt["status"] = "binding_changed"
                    _restore(path, data)
            except OSError:
                receipt["status"] = "throughput_restore_failed"
        receipt["evaluation_finished_at_ns"] = time.time_ns()
        _write(out / "post_measure_receipt.json", receipt)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def cleanup(out, *, terminate=False, request_id=None, epoch=None):
    owner_path = out / "post_measure/owner.json"
    if not owner_path.is_file():
        return
    record = _read(owner_path)
    if terminate and (record["request_id"] != request_id or record["measurement_epoch"] != epoch):
        raise ValueError("cleanup request/epoch do not match recorded ownership")
    owner = record["server_identity"]
    status = "recorded_group_gone" if not _group_live(owner["pgid"]) else "recorded_group_survives"
    if terminate and status != "recorded_group_gone":
        status = _teardown(owner, 1)
    _write(out / "post_measure_cleanup.json", {"schema": "geak.post_measure.cleanup.v1",
           "request_id": record["request_id"], "measurement_epoch": record["measurement_epoch"],
           "launch_nonce": record["launch_nonce"], "scope": "recorded_server_process_group",
           "status": status, "observed_at_ns": time.time_ns()})


def aggregate(out):
    capsule = _capsule(out)
    selected = []
    for marker in sorted(out.glob("replica_*/selected_attempt")):
        attempt = marker.read_text().strip()
        if attempt not in ("1", "2"):
            raise ValueError("invalid selected attempt")
        leaf = marker.parent / ("attempt_" + attempt)
        item = {"replica": marker.parent.name, "attempt": int(attempt),
                "throughput": _artifact(marker.parent / "selected_summary.json", out),
                "receipt": None, "cleanup": None}
        for key, name in (("receipt", "post_measure_receipt.json"), ("cleanup", "post_measure_cleanup.json")):
            path = leaf / name
            if path.is_file():
                item[key] = _artifact(path, out)
        selected.append(item)
    request = capsule["request"]
    _write(out / "post_measure_manifest.json", {"schema": "geak.post_measure.manifest.v1",
           "request_id": request["request_id"], "measurement_epoch": request["measurement_epoch"],
           "contract_sha256": request["contract_sha256"], "selection": "first_throughput_valid_attempt",
           "quality_policy_owner": "caller", "selected": selected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capabilities", "prepare", "record", "ready", "run", "aggregate", "cleanup", "observe-cleanup", "supervise"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--request")
    parser.add_argument("--mode")
    parser.add_argument("--pid")
    parser.add_argument("--pgid")
    parser.add_argument("--start-ticks")
    parser.add_argument("--protected-pgids", default="")
    parser.add_argument("--group-unverified", default="0")
    parser.add_argument("--base-url")
    parser.add_argument("--replica-index", type=int, default=0)
    parser.add_argument("--replica-attempt", type=int, default=0)
    parser.add_argument("--request-id")
    parser.add_argument("--measurement-epoch")
    args = parser.parse_args()
    try:
        out = args.output_dir.resolve() if args.output_dir else None
        if args.command == "capabilities":
            print(json.dumps(capabilities(), sort_keys=True))
        elif out is None:
            raise ValueError("output directory is required")
        elif args.command == "prepare":
            prepare(out, args.request, args.mode)
        elif args.command == "record":
            record(out, args)
        elif args.command == "ready":
            ready(out, args)
        elif args.command == "run":
            run_callback(out)
        elif args.command == "supervise":
            supervise(out)
        elif args.command == "aggregate":
            aggregate(out)
        else:
            cleanup(out, terminate=args.command == "cleanup", request_id=args.request_id, epoch=args.measurement_epoch)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        print(f"GEAK post-measurement {args.command} failed ({type(exc).__name__})", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
