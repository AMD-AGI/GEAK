# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a source-bound native catalog against an isolated synthetic endpoint.

The main profile captures only the main SDK session. Kernel profiles capture
one Workflow child. The Hyperloom profile captures one specialist session.
This Linux command requires bubblewrap namespaces and never uses a provider.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata, util
from pathlib import Path
from unittest.mock import patch

from .native_prefix_marker import DECODER, MARKER, sha, wire
from .registered_prefix import RegisteredToolPrefix

SCOPE = "main_native_sdk_session_only"
PROMPT = "Synthetic catalog capture. Return the fixed terminal text. Do not use any tool."
TERMINAL = "GEAK_SYNTHETIC_CATALOG_COMPLETE"
MAX_BODY = 4 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("interface/run_e2e.py", "interface/run_kernel_native.py", *(
    str(path.relative_to(ROOT)) for path in sorted(Path(__file__).parent.glob("*.py"))))


class ProducerError(RuntimeError):
    """Report only bounded, content-free producer diagnostics."""


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bindings(cli, expected_cli, sdk_version):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_cli or "") or _file_sha(cli) != expected_cli:
        raise ValueError("The CLI hash does not match the explicit pin.")
    if metadata.version("claude-agent-sdk") != sdk_version:
        raise ValueError("The SDK version does not match the explicit pin.")
    package = util.find_spec("claude_agent_sdk")
    if package is None or not package.origin:
        raise ValueError("The native SDK package is unavailable.")
    sdk_root = Path(package.origin).parent
    sdk_sources = {str(path.relative_to(sdk_root)): _file_sha(path)
                   for path in sorted(sdk_root.rglob("*.py"))}
    if not sdk_sources:
        raise ValueError("The SDK source files are unavailable.")
    return {"cli_sha256": expected_cli, "sdk_version": sdk_version,
            "sdk_sources_sha256": sha(wire(sdk_sources)),
            "sdk_sources": sdk_sources,
            "import_paths_sha256": sha(wire([str(Path(value or ROOT).resolve()) for value in sys.path])),
            "source_sha256": {name: _file_sha(ROOT / name) for name in SOURCE_FILES}}


def _isolate_loopback():
    """Require a private network with only loopback before any native startup."""
    import fcntl
    if [name for _, name in socket.if_nameindex()] != ["lo"]:
        raise RuntimeError("The capture network contains a non-loopback interface.")
    if Path("/dev/kfd").exists() or Path("/dev/dri").exists():
        raise RuntimeError("The capture namespace exposes GPU devices.")
    if not os.statvfs("/").f_flag & os.ST_RDONLY:
        raise RuntimeError("The capture namespace requires a read-only root mount.")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
        request = struct.pack("16sH14x", b"lo", 0)
        flags = struct.unpack("16sH14x", fcntl.ioctl(control, 0x8913, request))[1]
        if not flags & 1:
            fcntl.ioctl(control, 0x8914, struct.pack("16sH14x", b"lo", flags | 1))
    # This local exchange checks the namespace without contacting any host.
    with socket.socket() as listener, socket.socket() as client:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client.settimeout(2)
        client.connect(listener.getsockname())
        connection, _ = listener.accept()
        with connection:
            connection.sendall(b"local")
            if client.recv(5) != b"local":
                raise RuntimeError("The loopback check failed.")


def _sse(model):
    usage = {"input_tokens": 1, "output_tokens": 1,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    message = {"id": "msg_synthetic_catalog", "type": "message", "role": "assistant",
               "model": model, "content": [], "stop_reason": None,
               "stop_sequence": None, "usage": usage}
    events = [
        ("message_start", {"type": "message_start", "message": message}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": TERMINAL}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                           "stop_sequence": None}, "usage": usage}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(b"event: " + name.encode() + b"\ndata: " + wire(value) + b"\n\n"
                    for name, value in events)


class SyntheticCapture:
    """Keep catalog bytes and hashes only. Return no tool-use response."""

    def __init__(self, prefix_count):
        if type(prefix_count) is not int or prefix_count < 1:
            raise ValueError("The prefix count must be a positive integer.")
        self.prefix_count = prefix_count
        self.catalog = None
        self.request = None
        self.errors = 0
        self.requests = 0
        self.lock = threading.Lock()
        self.completed = threading.Event()
        self.session_id = None
        self.expected_session_id = None
        self.expected_model = None
        self.failure_reasons = []
        self.stop_after_capture = False

    def validate_headers(self, headers):
        if headers.get("x-claude-code-agent-id"):
            raise ValueError("The main capture received a child request.")
        session = headers.get("x-claude-code-session-id")
        if (not session or (self.expected_session_id is not None and session != self.expected_session_id)
                or (self.session_id is not None and session != self.session_id)):
            raise ValueError("The synthetic request has an unexpected session.")
        self.session_id = session

    def message(self, raw):
        body = DECODER.decode(raw.decode("utf-8"))
        tools = body.get("tools")
        if not isinstance(tools, list) or self.prefix_count > len(tools):
            raise ValueError("The request has fewer tools than the selected prefix.")
        catalog = wire(tools[:self.prefix_count]) + b"\n"
        prefix = RegisteredToolPrefix(catalog, expected_sha256=sha(catalog))
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise ValueError("The request model is invalid.")
        if self.expected_model is not None and body["model"] != self.expected_model:
            raise ValueError("The request changed the source-selected model.")
        if self.catalog is not None:
            raise ValueError("The synthetic session sent more than one generation request.")
        request = {"request_sha256": sha(raw), "request_bytes": len(raw),
                        "tool_count": len(tools), "prefix_count": prefix.count,
                        "prefix_sha256": prefix.prefix_sha256,
                        "model": body["model"], "max_tokens": body.get("max_tokens"),
                        "thinking": body.get("thinking"), "output_config": body.get("output_config")}
        if self.session_id is not None:
            request["session_sha256"] = sha(self.session_id.encode())
        if self.expected_session_id is not None:
            from .sdk_cache import NativeRegisteredSessionCachePolicy
            policy = NativeRegisteredSessionCachePolicy(self.expected_session_id, prefix, enabled=True)
            result = policy.apply(raw)
            offset = next((index for index, (left, right) in enumerate(zip(raw, result.body))
                           if left != right), len(raw))
            preserved = (result.applied and result.body[offset:offset + len(MARKER)] == MARKER
                         and result.body[:offset] + result.body[offset + len(MARKER):] == raw)
            request["portable_policy"] = {"applied": result.applied, "reason": result.reason,
                                           "only_marker_bytes_changed": preserved}
            if not preserved:
                raise ValueError("The portable cache policy rejected the capture: " + result.reason)
        self.catalog = catalog
        self.request = request
        return _sse(body["model"])

    @contextmanager
    def server(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_HEAD(self):
                self.handle_local()

            def do_POST(self):
                self.handle_local()

            def handle_local(self):
                self.connection.settimeout(10)
                content_type, status = "application/json", 200
                try:
                    if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                        raise ValueError("The synthetic request encoding is unsupported.")
                    lengths = self.headers.get_all("Content-Length", [])
                    if len(lengths) > 1 or (lengths and not lengths[0].isdigit()):
                        raise ValueError("The synthetic request length is invalid.")
                    length = int(lengths[0]) if lengths else 0
                    if length > MAX_BODY:
                        raise ValueError("The synthetic request exceeds the byte limit.")
                    with outer.lock:
                        outer.requests += 1
                        if outer.requests > 12:
                            raise ValueError("The synthetic session exceeds the request limit.")
                        raw = self.rfile.read(length)
                        if len(raw) != length:
                            raise ValueError("The synthetic request is incomplete.")
                        route = self.path.split("?", 1)[0]
                        if self.command == "HEAD" and route == "/api/hello":
                            response = b""
                        elif self.command == "POST" and route == "/v1/messages/count_tokens":
                            response = b'{"input_tokens":1}'
                        elif self.command == "POST" and route == "/v1/messages":
                            outer.validate_headers(self.headers)
                            response = outer.message(raw)
                            content_type = "text/event-stream"
                        else:
                            raise ValueError("The synthetic route is unsupported.")
                except (ValueError, TypeError, AttributeError, UnicodeError, RecursionError) as error:
                    outer.errors += 1
                    outer.failure_reasons.append(str(error))
                    outer.completed.set()
                    status = 400
                    response = b'{"type":"error","error":{"type":"invalid_request_error","message":"Synthetic capture failed."}}'
                if status == 200 and outer.stop_after_capture and outer.catalog is not None:
                    # Prevent native schema repair from racing SDK disconnect.
                    # The current response still sends the fixed terminal text.
                    server.shutdown()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(response)))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(response)
                    self.wfile.flush()
                if status == 200 and outer.catalog is not None:
                    outer.completed.set()
                self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            yield "http://127.0.0.1:" + str(server.server_port)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def _capture(options):
    # The worker reaches this function only after its namespace check.
    import claude_agent_sdk

    from interface import run_e2e
    original = claude_agent_sdk.ClaudeAgentOptions
    supported = {field.name for field in fields(original)}
    if not {"strict_mcp_config", "setting_sources", "mcp_servers", "hooks"} <= supported:
        raise ValueError("The pinned SDK cannot isolate hooks and MCP configuration.")
    profile = {}

    def isolated_options(**values):
        settings = json.loads(values["settings"])
        if set(settings) - {"enableWorkflows", "ultracode"}:
            raise ValueError("The source settings exceed the synthetic profile.")
        settings["disableAllHooks"] = True
        values.update(settings=json.dumps(settings), strict_mcp_config=True, mcp_servers={}, hooks={},
                      session_id=str(uuid.uuid4()))
        capture.expected_session_id = values["session_id"]
        capture.expected_model = values["model"]
        profile.update(model=values["model"], allowed_tools=values["allowed_tools"],
                       permission_mode=values["permission_mode"], settings=settings,
                       setting_sources=values["setting_sources"], extra_args=values["extra_args"],
                       strict_mcp_config=True, mcp_servers={}, hooks={})
        return original(**values)

    capture = SyntheticCapture(options.prefix_count)
    with tempfile.TemporaryDirectory(prefix="geak-native-catalog-") as directory:
        temporary = Path(directory)
        work = temporary / "work"
        work.mkdir()
        with capture.server() as endpoint, patch.dict(os.environ, {
            "ANTHROPIC_BASE_URL": endpoint, "ANTHROPIC_API_KEY": "synthetic-local-key",
            "CLAUDE_CONFIG_DIR": str(temporary / "config"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
            "GEAK_SHARED_TOOL_CACHE": "0", "GEAK_LOCAL_HELPERS": "0",
        }), patch.object(claude_agent_sdk, "ClaudeAgentOptions", isolated_options):
            result = run_e2e._invoke_via_sdk(PROMPT, options.timeout,
                                            settings_profile="isolated", native_cwd=str(work))
    if result.strip() != TERMINAL or capture.errors or capture.catalog is None:
        raise RuntimeError("The synthetic native session did not complete exactly once.")
    if capture.request["model"] != profile.get("model"):
        raise RuntimeError("The native request changed the selected model.")
    return capture, profile


def _worker(options):
    _isolate_loopback()
    before = _bindings(options.cli, options.cli_sha256, options.sdk_version)
    module = _profile_module(options.profile)
    source_before = module.bindings(options) if module else {}
    capture, profile = module.capture(options) if module else _capture(options)
    if (before != _bindings(options.cli, options.cli_sha256, options.sdk_version)
            or source_before != (module.bindings(options) if module else {})):
        raise RuntimeError("A bound source changed during the synthetic capture.")
    eligibility = (capture.request or {}).get("portable_policy", {})
    if not (capture.catalog and eligibility.get("applied") is True
            and eligibility.get("only_marker_bytes_changed") is True):
        raise ValueError("The selected native request does not support the portable cache policy.")
    if capture.request.get("model") != profile.get("model"):
        raise ValueError("The selected native request changed the source model.")
    receipt = {"schema": "geak-native-catalog-producer-v1", "scope": profile.get("scope", SCOPE),
               "selected_profile": options.profile, "profile_source_bindings": source_before,
               "bindings": before, "profile": profile, "profile_sha256": sha(wire(profile)),
               "tool_filters_sha256": sha(wire(profile.get("tool_filters", {"allowed_tools": profile.get("allowed_tools")}))),
               "synthetic_capture": capture.request, "catalog_sha256": sha(capture.catalog),
               "network": "private_namespace_loopback_only", "provider_calls": 0,
               "cpu_affinity": sorted(os.sched_getaffinity(0)), "gpu_devices_visible": False,
               "tool_use_responses": profile.get("tool_use_responses", 0), "source_hashes_rechecked": True,
               "cleanup": "The PID namespace terminates remaining descendants before the parent accepts output.",
               "differences_from_default_runner": profile.get("differences_from_default_runner", [
                   "Use a fixed synthetic prompt and a terminal text response.",
                   "Use an empty temporary working directory and configuration directory.",
                   "Disable filesystem settings and all hooks.",
                   "Use strict empty MCP configuration.",
                   "Disable cache mutation, local helpers, updates, and nonessential traffic.",
                   "Use only the local synthetic endpoint and fake credentials."]),
               "limits": profile.get("limits", ["This catalog describes one main native SDK session.",
                          "It does not establish Workflow child or Hyperloom subagent catalogs.",
                          "Custom settings, MCP tools, skills, and other environments can change tools.",
                          "The caller must compare this profile with the intended workflow.",
                          "Synthetic usage does not measure token cost or cache benefit."])}
    options.output.mkdir(parents=False, exist_ok=False)
    try:
        (options.output / "catalog.json").write_bytes(capture.catalog)
        (options.output / "receipt.json").write_bytes(wire(receipt) + b"\n")
    except BaseException:
        shutil.rmtree(options.output)
        raise


def _profile_module(profile):
    if profile in {"kernel-child", "kernel-frozen"}:
        from . import kernel_catalog_profile
        return kernel_catalog_profile
    if profile == "hyperloom-specialist":
        from . import hyperloom_catalog_profile
        return hyperloom_catalog_profile
    return None


def _namespace_command(arguments, staging=None):
    bubblewrap = shutil.which("bwrap")
    if sys.platform != "linux" or not bubblewrap:
        raise RuntimeError("This producer requires Linux and the bwrap command.")
    # Retain this interpreter's import paths without copying credential variables.
    bootstrap = ("import json,runpy,sys; sys.path[:]=json.loads(sys.argv.pop(1)); "
                 "runpy.run_module('interface.native_cost_controls.produce_catalog',run_name='__main__')")
    command = [bubblewrap, "--ro-bind", "/", "/", "--dev", "/dev", "--tmpfs", "/tmp",
               "--unshare-user", "--uid", "0", "--gid", "0", "--unshare-net", "--unshare-pid",
               "--proc", "/proc", "--die-with-parent", "--new-session"]
    if staging is not None:
        command.extend(["--bind", str(staging), str(staging)])
    return [*command, sys.executable, "-I", "-S", "-B", "-c", bootstrap,
            json.dumps([str(Path(value or ROOT).resolve()) for value in sys.path]), *arguments]


def _run_isolated(arguments, environment, timeout, staging=None):
    with subprocess.Popen(_namespace_command(arguments, staging), cwd=ROOT, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) as process:
        try:
            output, errors = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RuntimeError("The isolated synthetic capture exceeded its time limit.") from None
        if process.returncode:
            diagnostics = []
            for line in errors.decode("utf-8", errors="replace").splitlines():
                if line.startswith("GEAK_CAPTURE_STATUS "):
                    value = json.loads(line.removeprefix("GEAK_CAPTURE_STATUS "))
                    diagnostics.append(value)
            raise ProducerError("The isolated synthetic capture failed. " + json.dumps(diagnostics))
        return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True, help="Use this explicitly pinned native CLI file.")
    parser.add_argument("--cli-sha256", required=True, help="Require this independently recorded CLI SHA-256 hash.")
    parser.add_argument("--sdk-version", required=True, help="Require this installed claude-agent-sdk version.")
    parser.add_argument("--prefix-count", type=int, required=True, help="Export this many leading tools from the selected native request.")
    parser.add_argument("--output", type=Path, required=True, help="Create this directory for catalog.json and receipt.json.")
    parser.add_argument("--model", default="claude-opus-4-8", help="Select the native model name for the synthetic request.")
    parser.add_argument("--effort", choices=["ultracode", "low", "medium", "high", "xhigh", "max"],
                        default="ultracode", help="Select the source runner's effort setting.")
    parser.add_argument("--timeout", type=int, default=60, help="Limit native execution to this many seconds.")
    parser.add_argument("--profile", choices=["main", "kernel-child", "kernel-frozen", "hyperloom-specialist"],
                        default="main", help="Select the source-bound native request profile.")
    from . import hyperloom_catalog_profile, kernel_catalog_profile
    kernel_catalog_profile.add_arguments(parser)
    hyperloom_catalog_profile.add_arguments(parser)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_preflight", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.prefix_count < 1 or not 1 <= options.timeout <= 300:
        parser.error("Use a positive prefix count and a timeout from 1 to 300 seconds.")
    options.cli = options.cli.resolve(strict=True)
    options.output = options.output.absolute()
    for name, value in vars(options).items():
        if name not in {"cli", "output"} and isinstance(value, Path):
            setattr(options, name, value.resolve(strict=True))
        elif isinstance(value, list) and all(isinstance(item, Path) for item in value):
            setattr(options, name, [item.resolve(strict=True) for item in value])
    if not options.cli.is_file() or not os.access(options.cli, os.X_OK):
        parser.error("The pinned CLI must be an executable file.")
    if options.output.exists() or not options.output.parent.is_dir():
        parser.error("The output directory must be new and its parent must exist.")
    if options._preflight:
        _isolate_loopback()
        return
    if options._worker:
        _worker(options)
        return
    _bindings(options.cli, options.cli_sha256, options.sdk_version)
    module = _profile_module(options.profile)
    if module:
        module.bindings(options)
    environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                   "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
                   "GEAK_CLAUDE_BIN": str(options.cli), "GEAK_CLAUDE_MODEL": options.model,
                   "GEAK_CLAUDE_EFFORT": options.effort}
    arguments = ["--cli", str(options.cli), "--cli-sha256", options.cli_sha256,
                 "--sdk-version", options.sdk_version, "--prefix-count", str(options.prefix_count),
                 "--model", options.model, "--effort", options.effort,
                 "--timeout", str(options.timeout)]
    # Forward optional profile arguments through argparse's normalized values.
    common = {"cli", "cli_sha256", "sdk_version", "prefix_count", "output", "model", "effort", "timeout",
              "_worker", "_preflight"}
    for name, value in vars(options).items():
        if name not in common and value is not None:
            for item in value if isinstance(value, list) else [value]:
                arguments.extend(["--" + name.replace("_", "-"), str(item)])
    with tempfile.TemporaryDirectory(prefix=".geak-catalog-", dir=options.output.parent) as directory:
        staging = Path(directory)
        isolated = [*arguments, "--output", str(staging / "export")]
        _run_isolated([*isolated, "--_preflight"], environment, 10, staging)
        _run_isolated([*isolated, "--_worker"], environment, options.timeout + 20, staging)
        receipt = json.loads((staging / "export" / "receipt.json").read_text())
        RegisteredToolPrefix.from_file(staging / "export" / "catalog.json", expected_sha256=receipt["catalog_sha256"])
        (staging / "export").rename(options.output)
    receipt = json.loads((options.output / "receipt.json").read_text())
    print(json.dumps({"scope": receipt["scope"], "catalog_sha256": receipt["catalog_sha256"],
                      "prefix_count": options.prefix_count, "provider_calls": 0,
                      "tool_use_responses": receipt["tool_use_responses"]}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, metadata.PackageNotFoundError) as error:
        # Do not print SDK errors, environment values, or synthetic request bodies.
        detail = str(error) if isinstance(error, ProducerError) else type(error).__name__
        print(f"Catalog production failed ({detail}). No catalog is accepted.", file=sys.stderr)
        raise SystemExit(1)
