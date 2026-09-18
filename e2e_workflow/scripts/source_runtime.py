#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify accepted source selection inside the actual serving Python processes."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import select
import signal
import socket
import stat
import struct
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
MAX_JSON = 32 * 1024 * 1024
LAUNCH_SCHEMA = "geak.source_runtime.launch.v1"
PROCESS_SCHEMA = "geak.source_runtime.process.v1"
CLEANUP_TERM_SEC = 0.5
CLEANUP_TOTAL_SEC = 3.0


class SourceRuntimeError(ValueError):
    """A source-bearing launch cannot establish its required runtime identity."""


class SourceCleanupUnverified(SourceRuntimeError):
    """A source launch must stop retrying until owned worker cleanup is known."""


def require(condition, reason):
    if not condition:
        raise SourceRuntimeError(reason)


@contextmanager
def regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "non_regular_file")
        yield stream


def digest(path):
    result = hashlib.sha256()
    with regular(path) as stream:
        for block in iter(lambda: stream.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def read_json(path, *, binding=False):
    with regular(path) as stream:
        data = stream.read(MAX_JSON + 1)
    require(len(data) <= MAX_JSON, "json_too_large")
    result = json.loads(data, object_pairs_hook=pairs)
    require(isinstance(result, dict), "json_object_required")
    return (result, hashlib.sha256(data).hexdigest()) if binding else result


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    target = path if exclusive else path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        if not exclusive:
            os.replace(target, path)
    finally:
        if not exclusive:
            target.unlink(missing_ok=True)


def validator_path():
    for path in (HERE / "source_materialization.py", HERE.parent.parent / "interface/source_materialization.py"):
        if path.is_file() and not path.is_symlink():
            return path
    raise SourceRuntimeError("source_validator_unavailable")


def validate(request):
    path = validator_path()
    spec = importlib.util.spec_from_file_location("_geak_runtime_source_validator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # This verification utility is pinned by source hash; load its exact source
    # rather than consulting timestamp-based bytecode from another checkout.
    with regular(path) as stream:
        exec(compile(stream.read(), str(path), "exec", dont_inherit=True), module.__dict__)  # noqa: S102 - pinned local validator
    result = module.validate_source_materialization(request)
    require(result is not None, "source_request_has_no_materialization")
    return result


def path_list(value):
    if not value:
        return []
    result = []
    for item in value.split(os.pathsep):
        require(bool(item) and Path(item).is_absolute(), "nonabsolute_import_root")
        path = Path(item).resolve(strict=True)
        require(path.is_dir(), "import_root_not_directory")
        if str(path) not in result:
            result.append(str(path))
    return result


def overlay_inventory(roots):
    result = {}
    for root in roots:
        for path in Path(root).rglob("*"):
            require(not path.is_symlink(), "overlay_source_symlink")
            require(path.suffix.lower() not in (".pyc", ".pyo"), "source_bytecode_present")
            if path.suffix != ".py" and path.name != "_overlay_manifest.json":
                continue
            require(not path.is_symlink() and path.resolve().is_relative_to(root), "overlay_source_symlink")
            result[str(path)] = digest(path)
    return result


def source_cache_absent(path):
    # -B disables writes, but SourceFileLoader still reads both ordinary and
    # PYTHONPYCACHEPREFIX caches. Refuse that ambiguity in the serving process.
    for cache in (Path(path).with_suffix(".pyc"), Path(importlib.util.cache_from_source(str(path)))):
        require(not cache.exists(), "source_bytecode_present")


def check_request_marker(request_path, marker):
    check_cleanup_barrier(Path(marker).parent)
    with regular(marker) as stream:
        expected = stream.read(66)
    require(len(expected) == 65 and expected[-1:] == b"\n"
            and all(char in b"0123456789abcdef" for char in expected[:-1]), "invalid_staged_source_marker")
    request = read_json(request_path)
    source = validate(request)
    require(source.manifest_sha256 == expected[:-1].decode(), "staged_source_manifest_mismatch")


def check_cleanup_barrier(directory):
    barrier = Path(directory) / "source_cleanup_unverified.json"
    if barrier.exists() or barrier.is_symlink():
        raise SourceCleanupUnverified("source_cleanup_unverified_barrier")


def prepare(request_path, out, overlays):
    request_path = Path(request_path)
    require(request_path.is_absolute() and request_path.resolve() == request_path, "source_request_not_canonical")
    check_cleanup_barrier(request_path.parent)
    request, request_sha = read_json(request_path, binding=True)
    require(set(request) == {"schema_version", "source_snapshots", "source_materialization"}, "source_request_fields")
    require(type(request.get("schema_version")) is int and request["schema_version"] == 1, "source_request_version")
    source = validate(request)
    overlay_roots = path_list(overlays)
    overlay_files = overlay_inventory(overlay_roots)
    require(digest(request_path) == request_sha, "source_request_changed_during_prepare")
    out.mkdir(parents=True, exist_ok=True)
    require(out.resolve() == out and not out.is_symlink(), "observation_directory_not_canonical")
    capsule = {"schema": LAUNCH_SCHEMA, "launch_nonce": str(uuid.uuid4()), "created_at_ns": time.time_ns(), "boot_id": boot_id(),
               "request": request, "request_path": str(request_path), "request_sha256": request_sha,
               "manifest_sha256": source.manifest_sha256, "accepted_roots": list(source.pythonpath_prefixes),
               "overlay_roots": overlay_roots, "overlay_files": overlay_files,
               "helper_sha256": digest(Path(__file__).resolve()), "validator_sha256": digest(validator_path())}
    write_json(out / "launch.json", capsule, exclusive=True)
    bootstrap_dir = out / "bootstrap"
    bootstrap_dir.mkdir()
    # The shim contains no served packages, so its first position only chains
    # startup hooks; package precedence remains the caller's composed order.
    shim = f'''import sys
sys.dont_write_bytecode = True
try:
    import importlib.util, os
    _spec = importlib.util.spec_from_file_location("_geak_source_runtime", {str(Path(__file__).resolve())!r})
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    with open(_spec.origin, "rb") as _stream:
        exec(compile(_stream.read(), _spec.origin, "exec", dont_inherit=True), _mod.__dict__)
    _mod.bootstrap(__file__)
except BaseException:
    import os, sys
    sys.stderr.write("[FATAL] GEAK source runtime bootstrap failed\\n")
    os._exit(78)
'''
    (bootstrap_dir / "sitecustomize.py").write_text(shim)
    return os.pathsep.join(source.pythonpath_prefixes)


def capsule(out):
    value = read_json(out / "launch.json")
    require(value.get("schema") == LAUNCH_SCHEMA, "launch_schema")
    require(value.get("boot_id") == boot_id(), "launch_boot_changed")
    require(digest(value["request_path"]) == value["request_sha256"], "source_request_changed")
    require(digest(Path(__file__).resolve()) == value["helper_sha256"], "source_helper_changed")
    require(digest(validator_path()) == value["validator_sha256"], "source_validator_changed")
    return value


def proc(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "ppid": int(fields[1]), "pgid": int(fields[2]), "start_ticks": int(fields[19]), "state": fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def identity(pid=None):
    current = proc(os.getpid() if pid is None else pid)
    require(current is not None and current["state"] != "Z", "process_not_live")
    return {key: current[key] for key in ("pid", "pgid", "start_ticks")}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def subreaper(*, enable=False):
    """Keep orphaned descendants visible for the owned source-bearing launch.

    Linux reparents double-forked descendants to this process instead of init.
    Detached workers remain unsupported; gates reject them without extending
    the existing process-group teardown policy.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if enable:
        require(libc.prctl(36, 1, 0, 0, 0) == 0, "subreaper_enable_failed")
    flag = ctypes.c_int()
    require(libc.prctl(37, ctypes.addressof(flag), 0, 0, 0) == 0 and flag.value == 1, "source_subreaper_disabled")


def socket_name(launch, process):
    return "\0geak-source-" + launch["launch_nonce"] + "-" + str(process["pid"]) + "-" + str(process["start_ticks"])


class Observer:
    def __init__(self, out, launch, source):
        self.out, self.launch, self.source = out, launch, source
        self.failure = None
        self.channel = None
        self.thread = None
        self.measurement_started = False
        self.process = identity()
        self.deleted = set(source.manifest["deleted_modules"])
        self.expected = {}
        self.hashes = {str(source.bundle_root / row["path"]): row["sha256"] for row in source.manifest["files"]}
        for path in self.hashes:
            require(Path(path).suffix.lower() not in (".pyc", ".pyo"), "source_bytecode_present")
            if not path.endswith(".py"):
                continue
            for prefix in source.pythonpath_prefixes:
                if not Path(path).is_relative_to(prefix):
                    continue
                parts = list(Path(path).relative_to(prefix).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                if parts and all(part.isidentifier() for part in parts):
                    name = ".".join(parts)
                    require(name not in self.expected, "ambiguous_runtime_module")
                    self.expected[name] = path
        self.owners = {name.split(".")[0] for name in self.expected}
        self.owners.update(name.split(".")[0] for name in self.deleted)
        self.prospective = {row["name"] for row in source.manifest["modules"]}
        self.prospective.update(self.deleted)
        self.prospective.update(name for name in self.expected if "." not in name)

    def fail(self, reason):
        self.failure = reason
        write_json(self.out / f"fatal-{self.process['pid']}-{self.process['start_ticks']}.json",
                   {"schema": PROCESS_SCHEMA, **self.process, "boot_id": boot_id(), "status": "failed",
                    "launch_nonce": self.launch["launch_nonce"], "request_sha256": self.launch["request_sha256"],
                    "reason": reason, "observed_at_ns": time.time_ns()})
        raise SourceRuntimeError(reason)


    def audit(self, event, _args):
        # Worker topology is fixed while measuring. These Python audit events
        # precede execution, so a transient unobserved Python worker cannot run
        # and disappear between observations. Direct native process creation
        # that bypasses Python audit is outside this bounded runtime contract.
        if self.measurement_started and event in {"os.fork", "os.forkpty", "os.exec", "os.posix_spawn",
                                                  "os.system", "subprocess.Popen"}:
            self.fail("worker_topology_changed_after_ready")

    def normal_spec(self, fullname, path=None, target=None):
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            method = getattr(finder, "find_spec", None)
            if method is not None:
                spec = method(fullname, path, target)
                if spec is not None:
                    return spec
        return None

    def check_spec(self, name, spec):
        if name in self.deleted:
            self.fail("deleted_module_resolved")
        origin = getattr(spec, "origin", None)
        if not isinstance(origin, str) or not origin.endswith(".py"):
            self.fail("unsupported_module_origin")
        path = str(Path(origin).resolve())
        source_cache_absent(origin)
        require(not getattr(spec, "cached", None) or not Path(spec.cached).exists(), "source_bytecode_present")
        expected = self.expected.get(name)
        overlay_hash = self.launch["overlay_files"].get(path)
        if path != expected and overlay_hash is None:
            self.fail("off_tree_module_origin")
        wanted = overlay_hash if overlay_hash is not None else self.hashes[path]
        if digest(origin) != wanted:
            self.fail("module_content_changed")
        return {"name": name, "origin": path, "sha256": wanted,
                "source": "authored_overlay" if overlay_hash is not None else "accepted_source"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in self.owners:
            return None
        spec = self.normal_spec(fullname, path, target)
        if spec is not None:
            self.check_spec(fullname, spec)
        return spec  # Keep the original spec and loader; never replace served code.

    def resolve(self, name):
        search_path = None
        parts = name.split(".")
        for index in range(len(parts)):
            current_name = ".".join(parts[:index + 1])
            loaded = sys.modules.get(current_name)
            spec = getattr(loaded, "__spec__", None) if loaded is not None else self.normal_spec(current_name, search_path)
            if spec is None:
                return None
            self.check_spec(current_name, spec)
            if index + 1 < len(parts):
                search_path = getattr(loaded, "__path__", None) if loaded is not None else spec.submodule_search_locations
                if search_path is None:
                    return None
        return self.check_spec(name, spec)

    def integrity(self):
        require(self.failure is None, self.failure or "prior_runtime_failure")
        require(identity() == self.process, "observed_process_identity_changed")
        subreaper()
        require(sys.meta_path and sys.meta_path[0] is self, "source_guard_displaced")
        require(sys.dont_write_bytecode, "bytecode_writes_enabled")
        current = capsule(self.out)
        require(current == self.launch, "launch_capsule_changed")
        checked = validate(current["request"])
        require(checked.manifest_sha256 == self.source.manifest_sha256, "manifest_identity_changed")
        require(overlay_inventory(current["overlay_roots"]) == current["overlay_files"], "authored_overlay_changed")
        for path in [*self.expected.values(), *current["overlay_files"]]:
            if path.endswith(".py"):
                source_cache_absent(path)
        return current

    def snapshot(self, challenge):
        current = self.integrity()
        loaded = []
        for name, module in sys.modules.copy().items():
            spec = getattr(module, "__spec__", None)
            # `python -m package.server` runs the owned module as __main__.
            if name == "__main__" and spec is not None:
                name = spec.name
            if module is not None and name.split(".")[0] in self.owners:
                row = self.check_spec(name, spec)
                row["initializing"] = bool(getattr(spec, "_initializing", False))
                loaded.append(row)
        resolved, deleted = [], []
        for name in sorted(self.prospective):
            result = self.resolve(name)
            if name in self.deleted:
                require(result is None, "deleted_module_resolved")
                deleted.append(name)
            else:
                require(result is not None, "required_module_unresolved")
                resolved.append(result)
        return {"schema": PROCESS_SCHEMA, **self.process, "boot_id": boot_id(),
                "launch_capsule_sha256": digest(self.out / "launch.json"),
                "status": "verified", "launch_nonce": current["launch_nonce"],
                "request_sha256": current["request_sha256"], "manifest_sha256": current["manifest_sha256"],
                "challenge_id": challenge["challenge_id"], "observed_at_ns": time.time_ns(),
                "accepted_roots": current["accepted_roots"], "overlay_roots": current["overlay_roots"],
                "sys_path_sha256": hashlib.sha256(json.dumps(sys.path).encode()).hexdigest(),
                "loaded_modules": sorted(loaded, key=lambda row: row["name"]),
                "resolved_modules": resolved, "deleted_modules_absent": deleted, "guard": "owned_module_specs_unchanged",
                "cache_policy": "source_bytecode_absent", "overlay_inventory": "exact_python_manifest_no_symlinks",
                "worker_topology": "frozen_after_ready" if self.measurement_started else "startup", "subreaper": True}

    def respond(self):
        previous = None
        while True:
            token = self.channel.recv(128).decode()
            if token == previous:
                continue
            try:
                challenge = read_json(self.out / "challenge.json")
                if token != challenge["challenge_id"]:
                    continue
                if challenge.get("phase") in ("ready", "cleanup"):
                    self.measurement_started = True
                if challenge.get("phase") == "cleanup":
                    self.channel.sendto((token + " frozen").encode(), challenge["reply_socket"])
                    previous = token
                    continue
                receipt = self.snapshot(challenge)
                path = self.out / f"process-{self.process['pid']}-{self.process['start_ticks']}-{token}.json"
                write_json(path, receipt)
                self.channel.sendto((token + " " + digest(path)).encode(), challenge["reply_socket"])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                try:
                    self.fail(exc.args[0] if isinstance(exc, SourceRuntimeError) else "runtime_observation_failed")
                except (OSError, SourceRuntimeError):
                    pass
            previous = token

    def start(self):
        if self.channel is not None:
            self.channel.close()  # A forked child must not reuse its parent's address.
        self.process = identity()
        subreaper(enable=True)
        write_json(self.out / f"registered-{self.process['pid']}-{self.process['start_ticks']}.json",
                   {"schema": "geak.source_runtime.registration.v1", **self.process,
                    "boot_id": self.launch["boot_id"], "launch_nonce": self.launch["launch_nonce"]})
        self.channel = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.channel.bind(socket_name(self.launch, self.process))
        self.thread = threading.Thread(target=self.respond, name="geak-source-observer", daemon=True)
        self.thread.start()


def bootstrap(shim_path):
    sys.dont_write_bytecode = True
    out = Path(os.environ["GEAK_SOURCE_OBSERVATION_DIR"]).resolve()
    launch = capsule(out)
    source = validate(launch["request"])
    require(path_list(os.environ.get("GEAK_ACCEPTED_SOURCE_PYTHONPATH", "")) == launch["accepted_roots"], "accepted_roots_export_mismatch")
    require(path_list(os.environ.get("OVERLAY_PYTHONPATH", "")) == launch["overlay_roots"], "overlay_roots_export_mismatch")
    subreaper(enable=True)
    observer = Observer(out, launch, source)
    sys.meta_path.insert(0, observer)
    sys.addaudithook(observer.audit)
    try:
        # Validate all overlay inventory and selected cache paths before an
        # authored sitecustomize can inject a module through its own loader.
        observer.integrity()
        # Python normally imports only one sitecustomize. Chain the first one
        # after our shim, retaining its own module identity and original loader.
        shim_dir = Path(shim_path).resolve().parent
        search = [item for item in sys.path if Path(item or os.curdir).resolve() != shim_dir]
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", search)
        if spec is not None:
            require(isinstance(spec.origin, str) and spec.origin.endswith(".py"), "unsupported_sitecustomize")
            module = importlib.util.module_from_spec(spec)
            sys.modules["sitecustomize"] = module
            spec.loader.exec_module(module)
        # Startup hooks may install a finder. Preserve its precedence among the
        # original finders, with this validating delegate in front of them all.
        require(observer in sys.meta_path, "source_guard_removed_by_startup")
        sys.meta_path.remove(observer)
        sys.meta_path.insert(0, observer)
        observer.snapshot({"challenge_id": "startup"})
        observer.start()
        os.register_at_fork(after_in_child=observer.start)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        observer.fail(exc.args[0] if isinstance(exc, SourceRuntimeError) else "runtime_bootstrap_failed")


def group_members(pgid):
    return {row["pid"]: row for entry in Path("/proc").iterdir() if entry.name.isdigit()
            if (row := proc(int(entry.name))) and row["pgid"] == pgid and row["state"] != "Z"}


def escaped_workers(out, launch, owner):
    processes = {row["pid"]: row for entry in Path("/proc").iterdir() if entry.name.isdigit()
                 if (row := proc(int(entry.name))) and row["state"] != "Z"}
    descendants = {owner["pid"]}
    while True:
        found = {pid for pid, row in processes.items() if row["ppid"] in descendants}
        if found <= descendants:
            break
        descendants.update(found)
    owned = {pid: processes[pid] for pid in descendants if pid in processes and processes[pid]["pgid"] != owner["pgid"]}
    unknown = {}
    for path in out.glob("registered-*.json"):
        row = read_json(path)
        current = processes.get(row.get("pid"))
        if (row.get("launch_nonce") == launch["launch_nonce"] and current
                and current["start_ticks"] == row.get("start_ticks")
                and current["pgid"] != owner["pgid"] and current["pid"] not in owned):
            unknown[current["pid"]] = current
    return owned, unknown


def descendant_ancestry(pid, owner):
    chain, seen = [], set()
    current = proc(pid)
    while current is not None and current["pid"] not in seen:
        seen.add(current["pid"])
        chain.append({key: current[key] for key in ("pid", "ppid", "pgid", "start_ticks")})
        if current["pid"] == owner["pid"]:
            require(all(current[key] == value for key, value in owner.items()), "cleanup_owner_identity_changed")
            return chain
        parent = proc(current["ppid"])
        again = proc(current["pid"])
        require(again is not None and all(again[key] == current[key] for key in ("ppid", "start_ticks")),
                "cleanup_ancestry_changed")
        current = parent
    raise SourceRuntimeError("cleanup_ancestry_unproven")


def freeze_cleanup_parents(out, launch, owner):
    """Acknowledge the Python topology freeze before cleanup can be confirmed."""
    token = str(uuid.uuid4())
    address = "\0geak-source-freeze-" + launch["launch_nonce"] + "-" + token
    deadline = time.monotonic() + 1.0
    acknowledged = set()
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
        channel.settimeout(.05)
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        channel.bind(address)
        write_json(out / "challenge.json", {"challenge_id": token, "phase": "cleanup", "reply_socket": address,
                                            "created_at_ns": time.time_ns()})
        while time.monotonic() < deadline:
            require(identity(owner["pid"]) == owner, "cleanup_owner_identity_changed")
            members = group_members(owner["pgid"])
            for pid, row in members.items():
                if (pid, row["start_ticks"]) not in acknowledged:
                    try:
                        channel.sendto(token.encode(), socket_name(launch, row))
                    except OSError:
                        pass
            try:
                data, credentials, _flags, _sender = channel.recvmsg(256, socket.CMSG_SPACE(12))
            except OSError:
                data, credentials = b"", []
            for level, kind, value in credentials:
                if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and len(value) == 12:
                    pid, uid, _gid = struct.unpack("3i", value)
                    if pid in members and uid == os.getuid() and data == (token + " frozen").encode():
                        current = proc(pid)
                        if current and current["start_ticks"] == members[pid]["start_ticks"]:
                            acknowledged.add((pid, current["start_ticks"]))
            current_members = group_members(owner["pgid"])
            if current_members and {(pid, row["start_ticks"]) for pid, row in current_members.items()} <= acknowledged:
                return [{"pid": pid, "start_ticks": ticks} for pid, ticks in sorted(acknowledged)]
    raise SourceCleanupUnverified("source_cleanup_parent_freeze_unconfirmed")


def cleanup_escaped_workers(out, launch, owner, initial, unknown, *, teardown=False):
    """Stop only descendants whose identity was captured under the live owner.

    pidfds pin the process across PID reuse. Repeated ancestry discovery catches
    descendants forked during shutdown. Any unsupported syscall, inaccessible
    identity, unproven ancestry, or missing exit confirmation sets a persistent
    request-directory barrier; ordinary process-group teardown runs afterward.
    """
    started = time.monotonic()
    deadline = started + CLEANUP_TOTAL_SEC
    records, handles = {}, {}
    uncertain = bool(unknown)
    pending = initial
    stable = 0
    descendants_stable = 0
    owner_released = False
    frozen = []
    try:
        frozen = freeze_cleanup_parents(out, launch, owner)
        if teardown:
            # A process may setsid after acknowledging. Its pidfd must still
            # cover it, even when it has left the group before this scan.
            for member in frozen:
                current = proc(member["pid"])
                if current and current["start_ticks"] == member["start_ticks"]:
                    pending[current["pid"]] = current
    except (OSError, ValueError):
        uncertain = True
    try:
        while time.monotonic() < deadline:
            for pid, row in pending.items():
                key = (pid, row["start_ticks"])
                if key in records:
                    continue
                record = {key: row[key] for key in ("pid", "ppid", "pgid", "start_ticks")}
                record.update(ownership="descendant_of_live_owner", exit_confirmed=False, term_sent=False, kill_sent=False)
                records[key] = record
                try:
                    before = identity(owner["pid"])
                    require(before == owner, "cleanup_owner_identity_changed")
                    fd = os.pidfd_open(pid, 0)
                    handles[key] = fd
                    current = proc(pid)
                    if current is None or current["state"] == "Z" or current["start_ticks"] != row["start_ticks"]:
                        record["exit_confirmed"] = True
                        continue
                    record["ancestry"] = descendant_ancestry(pid, owner)
                except ProcessLookupError:
                    record["exit_confirmed"] = True
                except (OSError, ValueError, AttributeError) as exc:
                    uncertain = True
                    record["error"] = type(exc).__name__
            # Pin the complete initial cohort before any signal. Keep the
            # subreaper owner alive until descendants have exited, including
            # children created by unobserved workers' termination handlers.
            for key in sorted(handles, key=lambda item: item[0] == owner["pid"]):
                fd = handles[key]
                record = records[key]
                if record["exit_confirmed"] or record.get("error"):
                    continue
                if teardown and key[0] == owner["pid"] and not owner_released:
                    continue
                try:
                    if not record["term_sent"]:
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                        record["term_sent"] = True
                    if select.select([fd], [], [], 0)[0]:
                        record["exit_confirmed"] = True
                    elif time.monotonic() - started >= CLEANUP_TERM_SEC and not record["kill_sent"]:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                        record["kill_sent"] = True
                except ProcessLookupError:
                    record["exit_confirmed"] = True
                except OSError as exc:
                    uncertain = True
                    record["error"] = type(exc).__name__
            try:
                if teardown and owner_released:
                    # Its descendants are already confirmed gone. The pinned
                    # pidfd now proves owner exit without racing a live /proc
                    # identity lookup against our own termination signal.
                    pending, additional_unknown = {}, {}
                else:
                    require(identity(owner["pid"]) == owner, "cleanup_owner_identity_changed")
                    pending, additional_unknown = escaped_workers(out, launch, owner)
                    if teardown:
                        pending.update(group_members(owner["pgid"]))
                unknown.update(additional_unknown)
                uncertain = uncertain or bool(additional_unknown)
                frozen_ids = {(row["pid"], row["start_ticks"]) for row in frozen}
                uncertain = uncertain or not {(pid, row["start_ticks"]) for pid, row in group_members(owner["pgid"]).items()} <= frozen_ids
            except (OSError, ValueError):
                uncertain = True
                break
            live_pending = {pid: row for pid, row in pending.items() if not records.get((pid, row["start_ticks"]), {}).get("exit_confirmed")}
            if teardown and not owner_released:
                descendants_done = (not (set(live_pending) - {owner["pid"]})
                                    and all(row["exit_confirmed"] for key, row in records.items() if key[0] != owner["pid"]))
                descendants_stable = descendants_stable + 1 if descendants_done else 0
                owner_released = descendants_stable >= 2
            if not live_pending and all(row["exit_confirmed"] for row in records.values()):
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            time.sleep(0.05)
    finally:
        for fd in handles.values():
            os.close(fd)
    confirmed = not uncertain and stable >= 2 and all(row["exit_confirmed"] for row in records.values())
    result = {"schema": "geak.source_runtime.cleanup.v1", "status": "confirmed" if confirmed else "unverified",
              "scope": "owned_cohort_teardown" if teardown else "escaped_descendants",
              "launch_nonce": launch["launch_nonce"], "request_sha256": launch["request_sha256"],
              "manifest_sha256": launch["manifest_sha256"], "server_identity": owner,
              "frozen_parents": frozen, "freeze_transport": "unix_datagram_scm_credentials",
              "processes": list(records.values()), "unproven_processes": list(unknown.values()), "observed_at_ns": time.time_ns()}
    try:
        result["launch_capsule_sha256"] = digest(out / "launch.json")
        if not confirmed:
            write_json(Path(launch["request_path"]).parent / "source_cleanup_unverified.json", result)
        write_json(out / ("teardown.json" if teardown else "cleanup.json"), result)
    except (OSError, SourceRuntimeError) as exc:
        try:
            write_json(Path(launch["request_path"]).parent / "source_cleanup_unverified.json",
                       {**result, "status": "unverified", "reason": "cleanup_evidence_unwritable"})
        except OSError:
            pass  # Exit 43 still tells the caller to stop the evaluation.
        raise SourceCleanupUnverified("source_cleanup_evidence_unwritable") from exc
    if not confirmed:
        raise SourceCleanupUnverified("source_cleanup_unverified")
    return result


def reject_escaped_workers(out, launch, owner):
    owned, unknown = escaped_workers(out, launch, owner)
    if owned or unknown:
        cleanup_escaped_workers(out, launch, owner, owned, unknown)
        raise SourceRuntimeError("serving_worker_escaped_group_cleanup_confirmed")


def teardown_source(out, request, owner):
    require(owner["pid"] > 1 and owner["pid"] == owner["pgid"] and owner["pgid"] != os.getpgrp(), "unverified_server_group")
    require(identity(owner["pid"]) == owner, "source_teardown_owner_identity_changed")
    # Cleanup must still work after source/request content changes caused the
    # measurement to fail. The launch capsule retains ownership, not authority
    # to publish throughput; use its original request-directory binding only.
    launch = read_json(out / "launch.json")
    require(launch.get("schema") == LAUNCH_SCHEMA and launch.get("request_path") == str(request), "invalid_cleanup_launch")
    owned, unknown = escaped_workers(out, launch, owner)
    owned.update(group_members(owner["pgid"]))
    return cleanup_escaped_workers(out, launch, owner, owned, unknown, teardown=True)


def listeners(members, base_url):
    endpoint = urlsplit(base_url)
    require(endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1" and endpoint.port
            and not endpoint.username and not endpoint.password and not endpoint.query and not endpoint.fragment
            and endpoint.path in ("", "/"), "unsupported_listener_endpoint")
    inodes = set()
    for table, addresses in (("tcp", {"0100007F", "00000000"}),
                             ("tcp6", {"0" * 32, "0000000000000000FFFF00000100007F"})):
        for line in Path(f"/proc/net/{table}").read_text().splitlines()[1:]:
            fields = line.split()
            address, port = fields[1].split(":")
            if fields[3] == "0A" and int(port, 16) == endpoint.port and address in addresses:
                inodes.add(fields[9])
    owned = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or not (before := proc(int(entry.name))) or before["state"] == "Z":
            continue
        pid = before["pid"]
        try:
            for fd in Path(f"/proc/{pid}/fd").iterdir():
                target = os.readlink(fd)
                after = proc(pid)
                if (target.startswith("socket:[") and target[8:-1] in inodes and after and after["state"] != "Z"
                        and all(before[key] == after[key] for key in ("pid", "pgid", "start_ticks"))):
                    require(pid in members and all(members[pid][key] == before[key] for key in ("pgid", "start_ticks")),
                            "listener_owned_outside_server_group")
                    owned.setdefault(target[8:-1], set()).add(pid)
        except OSError:
            continue
    require(inodes and not inodes - owned.keys(), "listener_ownership_ambiguous")
    return set().union(*owned.values())


def gate(out, owner, endpoint, phase, timeout):
    launch = capsule(out)
    launch_sha = digest(out / "launch.json")
    require(owner["pid"] > 1 and owner["pid"] == owner["pgid"] and owner["pgid"] != os.getpgrp(), "unverified_server_group")
    require(identity(owner["pid"]) == owner, "server_identity_changed")
    challenge = {"challenge_id": str(uuid.uuid4()), "created_at_ns": time.time_ns(), "phase": phase}
    challenge["reply_socket"] = "\0geak-source-gate-" + launch["launch_nonce"] + "-" + challenge["challenge_id"]
    write_json(out / "challenge.json", challenge)
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
        channel.settimeout(0.05)
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        channel.bind(challenge["reply_socket"])
        acknowledgments = {}
        while time.monotonic() < deadline:
            require(identity(owner["pid"]) == owner, "server_identity_changed")
            reject_escaped_workers(out, launch, owner)
            members = group_members(owner["pgid"])
            listener_pids = listeners(members, endpoint)
            # Persistent shell/native workers cannot enforce the Python audit
            # boundary. Every live member must carry its own observer proof.
            required = set(members)
            require(required, "no_serving_python_process")
            for fatal in out.glob("fatal-*.json"):
                row = read_json(fatal)
                if row.get("launch_nonce") == launch["launch_nonce"] and row.get("pgid") == owner["pgid"]:
                    raise SourceRuntimeError("serving_process_reported_source_failure")
            for pid in required:
                try:
                    channel.sendto(challenge["challenge_id"].encode(), socket_name(launch, members[pid]))
                except OSError:
                    pass
            while not required <= acknowledgments.keys() and time.monotonic() < deadline:
                try:
                    data, credentials, _flags, _address = channel.recvmsg(256, socket.CMSG_SPACE(12))
                except OSError:
                    break
                prefix = (challenge["challenge_id"] + " ").encode()
                for level, kind, value in credentials:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and len(value) == 12:
                        sender, uid, _gid = struct.unpack("3i", value)
                        if sender in required and uid == os.getuid() and data.startswith(prefix):
                            acknowledgments[sender] = data[len(prefix):].decode("ascii", errors="replace")
            observed = []
            for pid in sorted(required):
                member = {key: members[pid][key] for key in ("pid", "pgid", "start_ticks")}
                try:
                    path = out / f"process-{pid}-{member['start_ticks']}-{challenge['challenge_id']}.json"
                    receipt = read_json(path)
                    if (receipt.get("schema") != PROCESS_SCHEMA or receipt.get("status") != "verified"
                            or receipt.get("challenge_id") != challenge["challenge_id"]
                            or receipt.get("launch_nonce") != launch["launch_nonce"]
                            or receipt.get("request_sha256") != launch["request_sha256"]
                            or receipt.get("manifest_sha256") != launch["manifest_sha256"]
                            or receipt.get("launch_capsule_sha256") != launch_sha
                            or receipt.get("boot_id") != boot_id()
                            or receipt.get("observed_at_ns", 0) < challenge["created_at_ns"]
                            or any(receipt.get(key) != value for key, value in member.items())
                            or acknowledgments.get(pid) != digest(path)):
                        continue
                    observed.append({**member, "receipt": path.name, "sha256": digest(path)})
                except (OSError, ValueError, TypeError):
                    continue
            after = group_members(owner["pgid"])
            reject_escaped_workers(out, launch, owner)
            same_members = {pid: row["start_ticks"] for pid, row in members.items()} == {pid: row["start_ticks"] for pid, row in after.items()}
            if len(observed) == len(required) and same_members:
                result = {"schema": "geak.source_runtime.gate.v1", "status": "verified", "phase": phase,
                          "transport": "unix_datagram_scm_credentials",
                          "launch_nonce": launch["launch_nonce"], "request_sha256": launch["request_sha256"],
                          "launch_capsule_sha256": launch_sha,
                          "manifest_sha256": launch["manifest_sha256"], "server_identity": owner,
                          "boot_id": boot_id(), "base_url": endpoint,
                          "challenge_id": challenge["challenge_id"], "observed_at_ns": time.time_ns(),
                          "processes": observed, "listener_pids": sorted(listener_pids)}
                write_json(out / f"gate-{phase}.json", result)
                return result
            time.sleep(0.05)
    raise SourceRuntimeError("serving_process_observation_timeout")


def seal_measurement(out):
    """Bind published throughput to both actual-process observation phases."""
    launch = capsule(out)
    launch_sha = digest(out / "launch.json")
    gates = {}
    for phase in ("ready", "finished"):
        path = out / f"gate-{phase}.json"
        row = read_json(path)
        require(row.get("schema") == "geak.source_runtime.gate.v1" and row.get("status") == "verified"
                and row.get("phase") == phase and row.get("launch_nonce") == launch["launch_nonce"]
                and row.get("request_sha256") == launch["request_sha256"]
                and row.get("manifest_sha256") == launch["manifest_sha256"], "invalid_measurement_gate")
        require(row.get("launch_capsule_sha256") == launch_sha, "measurement_capsule_changed")
        gates[phase] = {"path": path.name, "sha256": digest(path)}
    ready, finished = (read_json(out / gates[phase]["path"]) for phase in ("ready", "finished"))
    require(ready["server_identity"] == finished["server_identity"]
            and ready["base_url"] == finished["base_url"]
            and ready["boot_id"] == finished["boot_id"] == launch["boot_id"]
            and ready["observed_at_ns"] < finished["observed_at_ns"]
            and ready["challenge_id"] != finished["challenge_id"], "measurement_gate_identity_changed")
    value = {"schema": "geak.source_runtime.measurement.v1", "status": "verified",
             "measurement_scope": "hot_timed_rounds",
             "launch_capsule": {"path": "launch.json", "sha256": launch_sha},
             "launch_nonce": launch["launch_nonce"], "request_sha256": launch["request_sha256"],
             "manifest_sha256": launch["manifest_sha256"], "gates": gates,
             "server_identity": finished["server_identity"], "base_url": finished["base_url"],
             "artifacts": {name: digest(out.parent / name) for name in ("bench_runs.jsonl", "bench_summary.json")}}
    write_json(out / "measurement.json", value, exclusive=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "gate", "seal-measurement", "check-request", "teardown"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--overlay-pythonpath", default="")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--pgid", type=int)
    parser.add_argument("--start-ticks", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--phase", choices=("prepared", "ready", "finished"), default="ready")
    parser.add_argument("--timeout-sec", type=float, default=30)
    args = parser.parse_args()
    if args.command != "check-request" and args.output_dir is None:
        parser.error("--output-dir is required for this command")
    try:
        out = args.output_dir.absolute() if args.output_dir is not None else None
        if args.command == "check-request":
            check_request_marker(args.request, args.expected_manifest)
            return 0
        if args.command == "prepare":
            print(prepare(args.request, out, args.overlay_pythonpath))
        elif args.command == "seal-measurement":
            seal_measurement(out)
        elif args.command == "teardown":
            teardown_source(out, args.request, {"pid": args.pid, "pgid": args.pgid, "start_ticks": args.start_ticks})
        else:
            require(args.timeout_sec > 0 and args.timeout_sec <= 300, "invalid_gate_timeout")
            gate(out, {"pid": args.pid, "pgid": args.pgid, "start_ticks": args.start_ticks}, args.base_url, args.phase, args.timeout_sec)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reason = exc.args[0] if isinstance(exc, SourceRuntimeError) else type(exc).__name__
        print(f"[FATAL] GEAK source runtime {args.command} failed: {reason}", file=sys.stderr)
        if args.command == "teardown":
            try:
                require(args.request is not None and args.request.is_absolute(), "cleanup_request_missing")
                write_json(args.request.parent / "source_cleanup_unverified.json",
                           {"schema": "geak.source_runtime.cleanup.v1", "status": "unverified", "reason": reason})
            except (OSError, ValueError):
                pass
        return 43 if isinstance(exc, SourceCleanupUnverified) or args.command == "teardown" else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
