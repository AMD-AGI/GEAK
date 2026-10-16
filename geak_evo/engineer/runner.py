"""Run and recover an engineer with explicit launch, settlement and source capture.

The caller supplies the worker boundary. Receipts retain the original protocol,
usage, immutable inputs and ownership identity through interruption and recovery.
"""

import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import uuid

from ..clock import SystemClock
from ..model_profile import ModelProfile
from .._util import sha256_hex
from ..archive.refs import EngineerError, file_ref, read_ref as _read_ref, source_reference
from .request import _directory, _json, _require, _sha, _thread_id, validate_request
from .backend import claude_command


CODEX_TOOL_FEATURES = frozenset((
    "apps", "plugins", "remote_plugin", "hooks", "browser_use", "browser_use_external",
    "computer_use", "in_app_browser", "multi_agent", "multi_agent_v2", "code_mode", "code_mode_host",
    "image_generation", "view_image", "shell_snapshot", "goals", "unbounded_connection_retries",
    "api_key_model_discovery", "tool_suggest", "skill_mcp_dependency_install", "enable_request_compression",
))


def _write(path, value):
    staging = path.with_name(path.name + "." + str(uuid.uuid4()) + ".tmp")
    fd = os.open(str(staging), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(staging), str(path))
    finally:
        if staging.exists():
            staging.unlink()

def _append_receipt(path, value):
    staging = path.with_name(path.name + "." + str(uuid.uuid4()) + ".tmp")
    fd = os.open(str(staging), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(staging), str(path))  # Atomic creation; never replace an existing receipt.
    finally:
        staging.unlink()

def _acquire_ownership(workspace, threads=(), *, create=False):
    """Share the same private workspace/conversation locks for run and recovery."""
    lock_root = workspace.parent / ".geak-engineer-owners"
    if create:
        lock_root.mkdir(mode=0o700, exist_ok=True)
    _require(lock_root.exists(), "workspace has no registered owner")
    info = lock_root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
             and not info.st_mode & 0o077, "ownership registry is not a private host directory")
    workspace_lock = sha256_hex(str(workspace).encode())
    names = {workspace_lock} | {"thread-" + _thread_id(thread) for thread in threads}
    descriptors = []
    try:
        for name in sorted(names):
            fd = os.open(str(lock_root / name), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            descriptors.append(fd)
            info = os.fstat(fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                     and not info.st_mode & 0o077 and info.st_nlink == 1, "unsafe ownership lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EngineerError("workspace or conversation already has an owner") from exc
        return lock_root / (workspace_lock + ".owner.json"), descriptors
    except BaseException:
        for fd in descriptors:
            os.close(fd)
        raise

def _codex_config(key, value, permission_profile):
    """Serialize only noncredential, host-owned tool/profile settings."""
    profile_key = "permissions." + (permission_profile or "") + "."
    allowed = False
    if permission_profile and key == profile_key + "extends":
        allowed = value == ":workspace"
    elif permission_profile and key == profile_key + "filesystem":
        allowed = (type(value) is dict and all(type(path) is str and path and access in ("read", "write", "deny")
                                             for path, access in value.items()))
    elif permission_profile and key == profile_key + "network.enabled":
        allowed = type(value) is bool
    elif key in {"features." + name for name in CODEX_TOOL_FEATURES} or key == "allow_login_shell":
        allowed = type(value) is bool
    elif key in ("agents.enabled", "orchestrator.mcp.enabled", "orchestrator.skills.enabled"):
        allowed = value is False
    elif key == "web_search":
        allowed = value == "disabled"
    elif key == "shell_environment_policy.inherit":
        allowed = value == "none"
    elif key == "shell_environment_policy.set":
        allowed = (type(value) is dict and set(value) <= {"PATH", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "TMPDIR"}
                   and all(type(item) is str for item in value.values()))
    elif re.fullmatch(r"mcp_servers\.[A-Za-z0-9_-]+\.enabled", key):
        allowed = value is False
    elif re.fullmatch(r"model_providers\.[A-Za-z0-9_-]+\.(request_max_retries|stream_max_retries)", key):
        allowed = type(value) is int and value == 0
    elif key == "project_doc_max_bytes":
        allowed = type(value) is int and value == 0
    _require(allowed, "unsupported or credential-bearing Codex config override")

    def literal(item):
        if type(item) is dict:
            return "{" + ",".join(json.dumps(name) + "=" + literal(content) for name, content in sorted(item.items())) + "}"
        return json.dumps(item)
    return key + "=" + literal(value)

def codex_command(request, report_path, *, model_profile, executable="codex",
                  permission_profile=None, config_overrides=None, strict_config=True):
    """Build one CLI call; preserve existing provider/auth configuration in place.

    Token budgets are advisory: this CLI interface exposes no hard token cap.
    The workspace-write flag is defense in depth, never worker qualification.
    A named permission profile replaces that legacy flag; it must be qualified
    by the host together with every enabled tool. Overrides cannot carry auth.
    """
    continuation = request["continuation"]
    _require(isinstance(model_profile, ModelProfile) and model_profile.workflow_backend == "codex",
             "explicit Codex model profile required")
    model_profile.verify()
    model, effort = model_profile.requested_model, model_profile.requested_reasoning_effort
    _require(permission_profile is None or (type(permission_profile) is str
             and re.fullmatch(r":?[A-Za-z0-9_-]+", permission_profile)), "invalid permission profile")
    _require(config_overrides is None or type(config_overrides) is dict, "config overrides must be a mapping")
    _require(type(strict_config) is bool and (strict_config or permission_profile is not None),
             "strict-config compatibility exception requires a named qualified profile")
    policy = (["--sandbox", "workspace-write"] if permission_profile is None else
              ["-c", "default_permissions=" + json.dumps(permission_profile)])
    profile_flags = ((["--strict-config"] if strict_config else []) + ["--ignore-rules"]
                     if permission_profile else [])
    command = [executable, "exec", "--json", "--skip-git-repo-check"] + profile_flags + policy + [
               "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort=' + json.dumps(effort),
               "--model", model, "--output-last-message", str(report_path)]
    for key, value in sorted((config_overrides or {}).items()):
        _require(type(key) is str, "invalid config override key")
        command += ["-c", _codex_config(key, value, permission_profile)]
    if continuation["mode"] == "resume":
        prior = json.loads(_read_ref(continuation["prior_receipt"]))
        _require(prior.get("model_profile") == model_profile.as_dict(), "resume cannot change the original model profile")
        command += ["resume", _thread_id(prior["thread_id"])]
    return command + ["-"]


BACKENDS = {"codex": codex_command, "claude": claude_command}


def _stop_owned(process):
    """Stop only a not-yet-reaped leader's newly created process group."""
    if process.returncode is not None:
        return process.returncode
    _require(os.getpgid(process.pid) == process.pid, "launcher did not create an owned process group")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=2)

def inspect_events(path, *, backend="codex", expected_thread=None, forbidden_thread=None,
                   max_line_bytes=8 * 1024 * 1024, model_route=None):
    """Inspect the selected native protocol without rewriting its captured events."""
    _require(backend in BACKENDS, "unknown event backend")
    if backend == "claude":
        from .backend import inspect_claude_events
        return inspect_claude_events(path, model_route=model_route,
                                     expected_thread=expected_thread, forbidden_thread=forbidden_thread,
                                     max_line_bytes=max_line_bytes)
    result = {"thread_id": None, "identity_valid": False, "terminal": None,
              "usage": None, "usage_status": "unknown", "protocol_errors": []}
    errors = result["protocol_errors"]
    threads, terminal_events, sequence = [], [], []
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                errors.append("oversized_event")
                break
            try:
                event = json.loads(line.decode("utf-8"))
                _require(type(event) is dict and type(event.get("type")) is str, "invalid event")
            except (ValueError, UnicodeError):
                errors.append("invalid_jsonl_event")
                continue
            kind = event["type"]
            if kind in ("thread.started", "turn.started", "turn.completed", "turn.failed"):
                sequence.append(kind)
            if kind == "thread.started":
                threads.append(event.get("thread_id"))
            if kind in ("turn.completed", "turn.failed"):
                terminal_events.append(event)
            if kind == "error":
                errors.append("backend_error_event")
    if len(threads) == 1:
        try:
            result["thread_id"] = _thread_id(threads[0])
            result["identity_valid"] = ((expected_thread is None or threads[0] == expected_thread)
                                        and threads[0] != forbidden_thread)
        except EngineerError:
            pass
    if not result["identity_valid"]:
        errors.append("conversation_identity_mismatch_or_missing")
    if sequence not in (["thread.started", "turn.started", "turn.completed"],
                         ["thread.started", "turn.started", "turn.failed"]):
        errors.append("invalid_turn_event_sequence")
    if len(terminal_events) != 1:
        errors.append("terminal_event_missing_or_duplicated")
    else:
        terminal = terminal_events[0]
        result["terminal"] = terminal["type"]
        usage = terminal.get("usage")
        if usage is not None:
            required = {"input_tokens", "cached_input_tokens", "output_tokens"}
            if (type(usage) is dict and required <= set(usage)
                    and all(type(value) is int and value >= 0 for value in usage.values())
                    and usage["cached_input_tokens"] <= usage["input_tokens"]):
                result["usage"] = usage
                result["usage_status"] = "reported_terminal"
            else:
                errors.append("invalid_terminal_usage")
    return result

def run_turn(request, attempts_root, *, launch_worker=None, settle_worker=None,
             capture_tip=None, backend="codex", backend_options=None, prompt_path=None, clock=None):
    """Run one admitted turn; caller supplies the actual security boundary.

    launch_worker(argv, **Popen_kwargs) -> subprocess.Popen must honor cwd,
    shell=False, start_new_session=True and the supplied standard streams.
    settle_worker(process_or_none, attempt_directory) -> JSON mapping with settled=True
    only after all owned worker descendants/resources have positively settled.
    capture_tip(workspace) -> archived source SHA256 is called before and after.
    No callbacks means a fail-closed refusal before any subprocess is launched.
    """
    clock = SystemClock() if clock is None else clock
    if not all(callable(item) for item in (launch_worker, settle_worker, capture_tip)):
        raise EngineerError("live worker is unqualified: explicit launcher, settlement and source capture required")
    request = validate_request(request)
    _require(backend in BACKENDS, "unknown backend")
    backend_options = dict(backend_options or {})
    profile = backend_options.get("model_profile")
    _require(isinstance(profile, ModelProfile) and profile.workflow_backend == backend,
             "new engineer calls require an explicit matching model profile")
    profile.verify()
    root = _directory(attempts_root)
    workspace = Path(request["workspace"])
    _require(root != workspace and workspace not in root.parents and root not in workspace.parents,
             "host attempt storage and workspace overlap")
    prior = None
    if request["continuation"]["prior_receipt"]:
        prior = json.loads(_read_ref(request["continuation"]["prior_receipt"]))
        if request["continuation"]["mode"] == "resume":
            _require(prior.get("backend", "codex") == backend, "resume cannot change transport backend")
    # One registry per canonical workspace parent, independent of attempts_root.
    # The external worker must have no write access to this host-owned parent.
    threads = [prior["thread_id"]] if prior and request["continuation"]["mode"] == "resume" else []
    owner_path, descriptors = _acquire_ownership(workspace, threads, create=True)
    try:
        if owner_path.exists():
            owner = json.loads(_read_ref(file_ref(owner_path)))
            _require(owner.get("settled") is True, "workspace has an unsettled predecessor")
            _require(request["continuation"]["prior_receipt"] == owner.get("receipt_ref"),
                     "continuation must reference the latest workspace/conversation receipt")
        else:
            _require(prior is None, "predecessor is not registered for this workspace")
        _require(capture_tip(str(workspace)) == request["tip"]["source_id"], "workspace does not match supplied tip")
        attempt_id = str(uuid.uuid4())
        attempt = root / attempt_id
        attempt.mkdir(mode=0o700)
        command = BACKENDS[backend](request, attempt / "report.md", **(backend_options or {}))
        if prompt_path is None:
            prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "engineer.md"
        prompt = Path(prompt_path).read_text(encoding="utf-8") + "\n\nHost assignment (JSON):\n" + _json(request).decode() + "\n"
        prompt_file = attempt / "prompt.txt"
        fd = os.open(str(prompt_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(prompt)
        receipt = {"schema_version": 1, "attempt_id": attempt_id, "request": request,
                   "backend": backend, "command": command, "started_unix": clock.time(),
                   "process_status": "not_started", "exit_code": None, "process_reaped": False,
                   "settlement": {"settled": False, "reason": "not_observed"},
                   "captured_tip_source_id": None, "errors": [], "token_budget_enforcement": "advisory",
                   "monetary_cost": None, "monetary_cost_status": "not_reported_by_cli"}
        receipt.update(model_profile=profile.as_dict(), requested_model=profile.requested_model,
                       requested_reasoning_effort=profile.requested_reasoning_effort)
        if backend == "claude":
            # Save the exact admitted route before launch so recovery can check
            # provider identity using the same accepted model set.
            receipt["model_route"] = profile.verify()
        _write(attempt / "receipt.json", receipt)
        _write(owner_path, {"attempt_id": attempt_id, "settled": False, "receipt_ref": None})
        process = None
        interruption = None
        begun = clock.monotonic()
        with prompt_file.open("rb") as stdin, (attempt / "events.jsonl").open("xb") as stdout, (attempt / "stderr.txt").open("xb") as stderr:
            try:
                process = launch_worker(command, cwd=str(workspace), stdin=stdin, stdout=stdout,
                                        stderr=stderr, shell=False, start_new_session=True)
                receipt["pid"] = process.pid
                receipt["process_status"] = "running"
                _write(attempt / "receipt.json", receipt)
                try:
                    remaining = max(0, request["budget"]["wall_seconds"] - (clock.monotonic() - begun))
                    receipt["exit_code"] = process.wait(timeout=remaining)
                    receipt["process_status"] = "exited"
                except subprocess.TimeoutExpired:
                    receipt["process_status"] = "timed_out"
                    # wait(timeout) has not reaped the leader, so its PID cannot
                    # be recycled. Only its newly created group may be signaled.
                    receipt["exit_code"] = _stop_owned(process)
                receipt["process_reaped"] = True
            except BaseException as exc:
                receipt["errors"].append("transport:" + type(exc).__name__)
                interruption = exc if not isinstance(exc, Exception) else None
                receipt["process_status"] = "interrupted" if interruption else "transport_failed"
            finally:
                if process is not None and process.returncode is None:
                    try:
                        receipt["exit_code"] = _stop_owned(process)
                        receipt["process_reaped"] = True
                    except BaseException as exc:
                        receipt["errors"].append("owned_stop:" + type(exc).__name__)
                        if not isinstance(exc, Exception):
                            interruption = exc
                try:
                    # The callback owns descendant/resource settlement, including
                    # failed admission where a launcher could raise after spawn.
                    receipt["settlement"] = json.loads(_json(settle_worker(process, attempt)))
                    _require(type(receipt["settlement"]) is dict
                             and type(receipt["settlement"].get("settled")) is bool,
                             "invalid boundary settlement")
                except BaseException as exc:
                    receipt["settlement"] = {"settled": False, "reason": type(exc).__name__}
                    if not isinstance(exc, Exception):
                        interruption = exc
                # The boundary must settle first: a live writer cannot produce
                # a stable terminal source. Preserve failure instead of claiming
                # a captured or settled candidate in this case.
                if receipt["settlement"].get("settled") is True:
                    try:
                        receipt["captured_tip_source_id"] = _sha(capture_tip(str(workspace)))
                    except BaseException as exc:
                        receipt["errors"].append("tip_capture:" + type(exc).__name__)
                        if not isinstance(exc, Exception):
                            interruption = exc
        receipt["elapsed_seconds"] = clock.monotonic() - begun
        try:
            receipt.update(inspect_events(attempt / "events.jsonl", backend=backend,
                                         model_route=receipt.get("model_route"),
                                         expected_thread=prior["thread_id"] if prior and request["continuation"]["mode"] == "resume" else None,
                                         forbidden_thread=prior.get("thread_id") if prior and request["continuation"]["mode"] == "handoff" else None))
        except Exception as exc:
            receipt.update(thread_id=None, identity_valid=False, terminal=None, usage=None,
                           usage_status="unknown", protocol_errors=["event_read:" + type(exc).__name__])
        try:
            validate_request(request)
            receipt["immutable_inputs_unchanged"] = True
        except (ValueError, OSError):
            receipt["immutable_inputs_unchanged"] = False
        text = receipt.pop("report_text", None)
        if backend == "claude" and receipt["settlement"].get("settled") is True and type(text) is str:
            try:
                with (attempt / "report.md").open("x", encoding="utf-8") as handle:
                    handle.write(text)
            except (OSError, ValueError) as exc:
                receipt["errors"].append("report_write:" + type(exc).__name__)
        receipt["report_status"] = "present" if (attempt / "report.md").is_file() else "missing"
        receipt["transport_complete"] = (receipt["process_status"] == "exited" and receipt["exit_code"] == 0
                                         and receipt["terminal"] == "turn.completed" and not receipt["protocol_errors"]
                                         and receipt["settlement"].get("settled") is True
                                         and receipt["captured_tip_source_id"] is not None
                                         and receipt["immutable_inputs_unchanged"] and not receipt["errors"])
        _write(attempt / "receipt.json", receipt)
        receipt_ref = file_ref(attempt / "receipt.json")
        _write(owner_path, {"attempt_id": attempt_id, "settled": receipt["settlement"].get("settled") is True,
                            "receipt_ref": receipt_ref})
        if interruption:
            raise interruption
        return {"attempt_dir": str(attempt), "receipt": receipt, "receipt_ref": receipt_ref}
    finally:
        for fd in descriptors:
            os.close(fd)

def recover_turn(receipt_ref, *, verify_recovery, capture_tip, clock=None):
    """Settle an exact recorded attempt without launching or signaling anything.

    verify_recovery(original_receipt, attempt_directory) is a trusted host
    callback. It must positively establish original-attempt ownership, all
    process/resource settlement, and stable source access, returning a mapping
    with settled=True, process_reaped=True and a nonempty scope. It must not
    launch replacement work. No PID inspection or Boolean supplied by a worker
    grants this authority. Metering remains the caller's responsibility; unknown
    usage/cost is retained. The original receipt is never changed.

    A recovery receipt is appended before updating the exact owner pointer.
    Repeating after an interrupted owner update returns that same receipt; an
    intervening new attempt makes the old recovery stale.
    """
    clock = SystemClock() if clock is None else clock
    _require(callable(verify_recovery) and callable(capture_tip),
             "explicit recovery verifier and source capture required")
    original = json.loads(_read_ref(receipt_ref))
    _require(type(original) is dict and original.get("schema_version") == 1,
             "unsupported original receipt")
    attempt_id = _thread_id(original.get("attempt_id"))
    attempt = _directory(Path(receipt_ref["path"]).parent)
    _require(attempt.name == attempt_id and Path(receipt_ref["path"]).name == "receipt.json",
             "original receipt path does not identify its attempt")
    _require(original.get("settlement", {}).get("settled") is not True,
             "original attempt is already settled")
    request = original["request"]
    workspace = _directory(request["workspace"])
    _require(workspace != attempt and workspace not in attempt.parents
             and attempt not in workspace.parents, "attempt storage and workspace overlap")
    prior = (json.loads(_read_ref(request["continuation"]["prior_receipt"]))
             if request["continuation"]["prior_receipt"] else None)
    threads = [prior["thread_id"]] if prior and request["continuation"]["mode"] == "resume" else []
    if original.get("identity_valid") is True:
        threads.append(_thread_id(original.get("thread_id")))
    owner_path, descriptors = _acquire_ownership(workspace, threads)
    try:
        _read_ref(receipt_ref)  # Recheck after acquiring the original owner lock.
        owner = json.loads(_read_ref(file_ref(owner_path)))
        _require(owner.get("attempt_id") == attempt_id, "recovery belongs to a stale workspace attempt")
        recovered_path = attempt / ("recovery-" + receipt_ref["sha256"] + ".json")
        if owner.get("settled") is True:
            current_ref = owner.get("receipt_ref")
            current = json.loads(_read_ref(current_ref))
            _require(current.get("recovery_of") == receipt_ref and current.get("request") == request
                     and current.get("attempt_id") == attempt_id
                     and current.get("settlement", {}).get("settled") is True,
                     "owner does not identify this recovery")
            return {"attempt_dir": str(attempt), "receipt": current, "receipt_ref": current_ref}
        _require(owner.get("receipt_ref") in (None, receipt_ref), "recovery must use the exact current receipt")
        settlement = json.loads(_json(verify_recovery(json.loads(_json(original)), attempt)))
        _require(type(settlement) is dict and settlement.get("settled") is True
                 and settlement.get("process_reaped") is True
                 and type(settlement.get("scope")) is str and bool(settlement["scope"].strip()),
                 "positive host process/resource settlement required")
        _read_ref(receipt_ref)
        captured = _sha(capture_tip(str(workspace)))
        if recovered_path.exists():
            recovered = json.loads(_read_ref(file_ref(recovered_path)))
            _require(recovered.get("recovery_of") == receipt_ref
                     and recovered.get("request") == request and recovered.get("attempt_id") == attempt_id
                     and recovered.get("captured_tip_source_id") == captured
                     and recovered.get("settlement", {}).get("settled") is True,
                     "previous recovery does not match settled source/attempt")
        else:
            recovered = dict(original, recovery_of=receipt_ref, recovered_unix=clock.time(),
                             settlement=settlement, process_reaped=True, captured_tip_source_id=captured)
            try:
                recovered.update(inspect_events(attempt / "events.jsonl", backend=original.get("backend", "codex"),
                    model_route=original.get("model_route"),
                    expected_thread=prior["thread_id"] if prior and request["continuation"]["mode"] == "resume" else None,
                    forbidden_thread=prior.get("thread_id") if prior and request["continuation"]["mode"] == "handoff" else None))
            except (ValueError, OSError) as exc:
                recovered.update(thread_id=None, identity_valid=False, terminal=None, usage=None,
                                 usage_status="unknown", protocol_errors=["event_read:" + type(exc).__name__])
            try:
                validate_request(request)
                recovered["immutable_inputs_unchanged"] = True
            except (ValueError, OSError):
                recovered["immutable_inputs_unchanged"] = False
            text = recovered.pop("report_text", None)
            if original.get("backend") == "claude" and type(text) is str:
                try:
                    report = attempt / "report.md"
                    if report.exists():
                        _require(report.read_text() == text, "retained Claude report differs from terminal event")
                    else:
                        with report.open("x", encoding="utf-8") as handle:
                            handle.write(text)
                except (OSError, ValueError) as exc:
                    recovered["errors"].append("report_write:" + type(exc).__name__)
            recovered["report_status"] = "present" if (attempt / "report.md").is_file() else "missing"
            recovered["transport_complete"] = (
                recovered.get("process_status") == "exited" and recovered.get("exit_code") == 0
                and recovered["terminal"] == "turn.completed" and not recovered["protocol_errors"]
                and recovered["immutable_inputs_unchanged"] and not recovered.get("errors"))
            _append_receipt(recovered_path, recovered)
        recovered_ref = file_ref(recovered_path)
        _write(owner_path, {"attempt_id": attempt_id, "settled": True, "receipt_ref": recovered_ref})
        return {"attempt_dir": str(attempt), "receipt": recovered, "receipt_ref": recovered_ref}
    finally:
        for fd in descriptors:
            os.close(fd)


def inspect_prepared_identity(prepared, request, *, prompt_ref, operation_id):
    """Validate native operation identity using the same reader as investigation."""
    from .investigation import inspect_prepared_identity as inspect
    return inspect(prepared, request, prompt_ref=prompt_ref, operation_id=operation_id)
