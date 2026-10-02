# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the explicit process wrapper with synthetic data and loopback HTTP."""

import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from interface.native_cost_controls import run_registered as wrapper
from interface.native_cost_controls.cache_proxy import SharedToolCacheProxy
from interface.native_cost_controls.native_prefix_marker import MARKER, sha
from interface.native_cost_controls.registered_prefix import RegisteredToolPrefix

ROOT = Path(__file__).resolve().parents[1]
_SAFE_ENV = {"PATH": os.defpath, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
_RESPONSE = b'data: {"usage":{"input_tokens":7,"output_tokens":3}}\n\ndata: \x00\xff\n\n'
_HTTP_CHILD = """
import http.client, json, os, pathlib, sys
from urllib.parse import urlsplit
if len(sys.argv) > 1:
    pathlib.Path(sys.argv[1]).write_text(json.dumps({
        'argv': sys.argv[2:], 'cwd': os.getcwd(), 'environment': dict(os.environ)}))
endpoint = urlsplit(os.environ['ANTHROPIC_BASE_URL'])
connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
connection.request('POST', '/v1/messages?beta=true', body=sys.stdin.buffer.read(),
    headers={'Content-Type': 'application/json', 'Authorization': os.environ['TEST_AUTH'],
        'x-api-key': 'synthetic-key'})
response = connection.getresponse()
assert response.status == 200
assert response.getheader('request-id') == 'synthetic-response'
sys.stdout.buffer.write(response.read())
sys.stderr.buffer.write(b'synthetic-child-stderr\\n')
connection.close()
"""


def _tools(stem, count):
    return [{"name": f"{stem}{index}", "description": "Synthetic tool.",
        "input_schema": {"type": "object", "properties": {}}} for index in range(count)]


def _registration(tools):
    raw = json.dumps(tools).encode()
    return RegisteredToolPrefix(raw, expected_sha256=sha(raw))


def _request(tools, session="synthetic-session"):
    return json.dumps({"model": "synthetic-model", "tools": tools,
        "system": [{"type": "text", "text": "Synthetic system."}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Synthetic input."}]}],
        "metadata": {"user_id": json.dumps({"session_id": session})},
        "output_config": {"effort": "high"},
        "context_management": {"edits": [{"type": "compact_20260112"}]}}, indent=2).encode()


@contextmanager
def _upstream():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_POST(self):
            calls.append({"path": self.path, "headers": dict(self.headers),
                "body": self.rfile.read(int(self.headers["Content-Length"]))})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(_RESPONSE)))
            self.send_header("request-id", "synthetic-response")
            self.end_headers()
            self.wfile.write(_RESPONSE)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class PrefixSetTests(unittest.TestCase):
    def test_registration_and_request_boundaries_reject_invalid_input(self):
        prefix = _registration(_tools("Shared", 2))
        for registrations in ((), (object(),), (prefix, "unverified")):
            with self.subTest(registrations=registrations), self.assertRaises(ValueError):
                wrapper.RegisteredPrefixSetPolicy(registrations)
        policy = wrapper.RegisteredPrefixSetPolicy((prefix,))
        with self.assertRaises(TypeError):
            policy.apply("The body must contain bytes.")
        for raw, reason in ((b"[]", "unsupported_request"), (b"null", "unsupported_request"),
                (b"{}", "prefix_missing"), (b'{"tools":{}}', "prefix_missing")):
            with self.subTest(raw=raw):
                result = policy.apply(raw)
                self.assertEqual(result.body, raw)
                self.assertFalse(result.applied)
                self.assertEqual(result.reason, reason)

    def test_longest_exact_match_is_independent_of_registration_order(self):
        tools = _tools("Shared", 22)
        short, long = _registration(tools[:18]), _registration(tools)
        raw = _request(tools + _tools("Tail", 2))
        for registrations in ((short, long), (long, short), (short, long, long)):
            with self.subTest(order=[prefix.count for prefix in registrations]):
                result = wrapper.RegisteredPrefixSetPolicy(registrations).apply(raw)
                self.assertTrue(result.applied)
                self.assertEqual(result.body.replace(MARKER, b"", 1), raw)
                parsed = json.loads(result.body)
                self.assertIn("cache_control", parsed["tools"][21])
                self.assertNotIn("cache_control", parsed["tools"][17])

    def test_shorter_match_and_independent_catalogs_accept_distinct_sessions(self):
        outer, kernel = _tools("Outer", 18), _tools("Kernel", 22)
        policy = wrapper.RegisteredPrefixSetPolicy((_registration(outer), _registration(kernel)))
        for tools, count in ((outer, 18), (kernel + _tools("Tail", 1), 22),
                (outer + _tools("Tail", 5), 18), (outer + _tools("Tail", 6), 18)):
            raw = _request(tools, session=f"session-{len(tools)}")
            result = policy.apply(raw)
            self.assertTrue(result.applied)
            self.assertEqual(result.body.replace(MARKER, b"", 1), raw)
            self.assertIn("cache_control", json.loads(result.body)["tools"][count - 1])

    def test_unmatched_and_unsupported_requests_retain_exact_bytes(self):
        tools = _tools("Shared", 2)
        policy = wrapper.RegisteredPrefixSetPolicy((_registration(tools),))
        unsupported = json.loads(_request(tools))
        unsupported["messages"][0]["content"] = [{"type": "compaction", "content": "synthetic"}]
        for raw in (_request(_tools("Other", 2)), b"invalid-json", b'{"tools":[],"tools":[]}',
                json.dumps(unsupported, indent=3).encode()):
            with self.subTest(raw=raw):
                result = policy.apply(raw)
                self.assertFalse(result.applied)
                self.assertEqual(result.body, raw)
        raw = _request(tools)
        self.assertEqual(policy.apply(raw, path="/other").body, raw)

    def test_longest_match_owns_a_decline(self):
        tools = _tools("Shared", 22)
        policy = wrapper.RegisteredPrefixSetPolicy((_registration(tools[:18]), _registration(tools)))
        raw = _request(tools)
        with patch.object(policy._policies[0], "apply",
                return_value=wrapper.Decision(raw, False, "unqualified_cache_layout")) as longest, \
                patch.object(policy._policies[1], "apply") as shorter:
            result = policy.apply(raw)
        self.assertFalse(result.applied)
        self.assertEqual(result.body, raw)
        longest.assert_called_once()
        shorter.assert_not_called()


@unittest.skipUnless(sys.platform.startswith("linux"), "The wrapper requires Linux child ownership controls.")
class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.environment = patch.dict(os.environ, _SAFE_ENV, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.tools = _tools("Shared", 2)
        self.catalogs = [self.catalog(self.tools, "tools.json")]

    def catalog(self, tools, filename):
        path = self.path / filename
        raw = json.dumps(tools).encode()
        path.write_bytes(raw)
        return str(path), sha(raw)

    def command(self, child, catalogs=None):
        arguments = [sys.executable, "-B", "-m", "interface.native_cost_controls.run_registered"]
        for path, expected in self.catalogs if catalogs is None else catalogs:
            arguments.extend(("--catalog", path, expected))
        return arguments + ["--"] + list(child)

    def launch(self, child, *, catalogs=None, environment=None, **kwargs):
        process = subprocess.Popen(self.command(child, catalogs), cwd=ROOT,
            env={**os.environ, **(environment or {})}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        self.addCleanup(self.stop_wrapper, process)
        return process

    def stop_wrapper(self, process):
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=6)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=3)

    def closed(self, endpoint):
        parsed = urlsplit(endpoint)
        with self.assertRaises(OSError):
            socket.create_connection((parsed.hostname, parsed.port), timeout=0.2)

    def test_every_catalog_is_verified_before_proxy_or_child_start(self):
        for catalogs in ([self.catalogs[0], (self.catalogs[0][0], "0" * 64)],
                [(str(self.path / "missing.json"), "0" * 64), self.catalogs[0]],
                [("relative.json", "0" * 64)]):
            with self.subTest(catalogs=catalogs), redirect_stderr(io.StringIO()) as stderr, \
                    patch.object(wrapper, "SharedToolCacheProxy") as proxy, \
                    patch.object(wrapper.subprocess, "Popen") as child:
                with self.assertRaises(SystemExit) as error:
                    wrapper.main(self.command([sys.executable], catalogs)[4:])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("A source catalog could not be verified.", stderr.getvalue())
                self.assertNotIn(str(self.path), stderr.getvalue())
                proxy.assert_not_called()
                child.assert_not_called()

    def test_explicit_separator_preserves_command_options(self):
        original = [sys.executable, "--catalog", "child-value", "--model", "original-model", "--", "literal"]
        with patch.object(wrapper, "run_registered", return_value=9) as run:
            self.assertEqual(wrapper.main(self.command(original)[4:]), 9)
        self.assertEqual(run.call_args.args[0], original)
        self.assertEqual(run.call_args.args[1][0].catalog_sha256, self.catalogs[0][1])

    def test_help_and_missing_command_errors_never_launch_a_child(self):
        cases = [(["--help"], 0, "--catalog PATH SHA256"),
            (self.command([sys.executable])[4:-2], 2, "Separate the command"),
            (self.command([])[4:], 2, "Supply a command after --.")]
        for arguments, status, message in cases:
            with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()) as stdout, \
                    redirect_stderr(io.StringIO()) as stderr, patch.object(wrapper, "run_registered") as run, \
                    patch.object(wrapper.RegisteredToolPrefix, "from_file") as catalog:
                with self.assertRaises(SystemExit) as error:
                    wrapper.main(arguments)
                self.assertEqual(error.exception.code, status)
                self.assertIn(message, stdout.getvalue() + stderr.getvalue())
                run.assert_not_called()
                catalog.assert_not_called()

    def test_cli_reports_fixed_startup_errors_without_private_details(self):
        cases = [(wrapper.WrapperError("The provider transport is unsupported."),
            "The provider transport is unsupported."),
            (OSError("synthetic-private-detail"), "The child command could not start.")]
        for failure, message in cases:
            with self.subTest(message=message), redirect_stderr(io.StringIO()) as stderr, \
                    patch.object(wrapper, "run_registered", side_effect=failure):
                with self.assertRaises(SystemExit) as error:
                    wrapper.main(self.command([sys.executable])[4:])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn("synthetic-private-detail", stderr.getvalue())

    def test_invalid_command_lists_stop_before_proxy_start(self):
        for command in ([], (), "python", b"python"):
            with self.subTest(command=command), patch.object(wrapper, "SharedToolCacheProxy") as proxy:
                with self.assertRaisesRegex(wrapper.WrapperError, "Supply a command argument list"):
                    wrapper.run_registered(command, (_registration(self.tools),))
                proxy.assert_not_called()

    def test_simulated_private_config_keeps_the_host_fixture_unchanged(self):
        # The container recipe owns filesystem isolation. The wrapper only
        # owns the command lifetime and supplies its temporary endpoint.
        host_config = self.path / "host-fixture" / ".claude" / "config.json"
        host_config.parent.mkdir(parents=True)
        sentinel = b'{"customApiUrl":"https://synthetic-host.invalid","sentinel":"host"}\n'
        host_config.write_bytes(sentinel)
        inherited_home = os.environ.get("HOME")
        script = """
import json, os, pathlib, sys
assert os.environ.get('HOME') == json.loads(sys.argv[2])
config = pathlib.Path(sys.argv[1])
config.parent.mkdir(parents=True)
config.write_text(json.dumps({'customApiUrl': os.environ['ANTHROPIC_BASE_URL']}))
"""
        with tempfile.TemporaryDirectory(prefix="private-config-", dir=self.path) as private_directory:
            private_root = Path(private_directory)
            private_config = private_root / ".claude" / "config.json"
            child = self.launch([sys.executable, "-c", script,
                str(private_config), json.dumps(inherited_home)])
            stdout, stderr = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, stderr)
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"")
            endpoint = json.loads(private_config.read_text())["customApiUrl"]
            self.assertEqual(urlsplit(endpoint).hostname, "127.0.0.1")
            self.assertEqual(host_config.read_bytes(), sentinel)
            self.assertEqual(os.environ.get("HOME"), inherited_home)
            for path in self.path.rglob("*"):
                if path.is_file() and private_root not in path.parents:
                    self.assertNotIn(endpoint.encode(), path.read_bytes())
            self.closed(endpoint)
        self.assertFalse(private_root.exists())
        self.assertEqual(host_config.read_bytes(), sentinel)

    def test_unsupported_environment_prevents_proxy_and_child_start(self):
        registration = _registration(self.tools)
        for name in (*wrapper._PROVIDERS, *wrapper._TRANSPORT_SETTINGS):
            with self.subTest(name=name), patch.dict(os.environ, {name: "synthetic-setting"}), \
                    patch.object(wrapper, "SharedToolCacheProxy") as proxy, \
                    patch.object(wrapper.subprocess, "Popen") as child:
                with self.assertRaises(wrapper.WrapperError):
                    wrapper.run_registered([sys.executable], (registration,))
                proxy.assert_not_called()
                child.assert_not_called()

    def test_invalid_upstream_or_inactive_proxy_prevents_child_start(self):
        with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "ftp://example.invalid"}), \
                patch.object(wrapper.subprocess, "Popen") as child:
            with self.assertRaises(wrapper.WrapperError):
                wrapper.run_registered([sys.executable], (_registration(self.tools),))
            child.assert_not_called()
        with patch.object(wrapper, "SharedToolCacheProxy") as proxy, \
                patch.object(wrapper.subprocess, "Popen") as child:
            proxy.return_value.__enter__.return_value.active = False
            with self.assertRaises(wrapper.WrapperError):
                wrapper.run_registered([sys.executable], (_registration(self.tools),))
            child.assert_not_called()

    def test_default_upstream_and_spawn_failure_close_proxy_and_restore_handlers(self):
        proxies = []

        def factory(*args, **kwargs):
            proxy = SharedToolCacheProxy(*args, **kwargs)
            proxies.append(proxy)
            return proxy

        handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        with patch.object(wrapper, "SharedToolCacheProxy", side_effect=factory), \
                patch.object(wrapper.subprocess, "Popen", side_effect=FileNotFoundError), \
                self.assertRaisesRegex(wrapper.WrapperError, "The child command could not start"):
            wrapper.run_registered(["synthetic-missing-command"], (_registration(self.tools),))
        self.assertEqual(proxies[0]._original_url, "https://api.anthropic.com")
        self.assertFalse(proxies[0].active)
        self.assertEqual(proxies[0].status, "closed")
        self.assertEqual({number: signal.getsignal(number) for number in handlers}, handlers)

    def test_missing_capabilities_or_existing_children_prevent_proxy_start(self):
        registration = _registration(self.tools)
        checks = [patch.object(wrapper.os, "pidfd_open", side_effect=OSError),
            patch.object(wrapper, "_direct_children", return_value={12345}),
            patch.object(wrapper, "_direct_children", side_effect=FileNotFoundError),
            patch.object(wrapper.signal, "getsignal", return_value=signal.SIG_IGN)]
        for check in checks:
            with check, patch.object(wrapper, "SharedToolCacheProxy") as proxy:
                with self.assertRaises(wrapper.WrapperError):
                    wrapper.run_registered([sys.executable], (registration,))
                proxy.assert_not_called()

    def test_missing_pidfd_api_and_native_threads_stop_before_child_ownership(self):
        unsupported_os = SimpleNamespace(waitid=os.waitid, WNOWAIT=os.WNOWAIT)
        with patch.object(wrapper, "os", unsupported_os), \
                self.assertRaisesRegex(wrapper.WrapperError, "Linux process controls are unavailable"), \
                wrapper._subreaper():
            self.fail("The missing pidfd API must prevent startup.")
        tasks = [Path("/proc/self/task/1"), Path("/proc/self/task/2")]
        with patch.object(Path, "iterdir", return_value=iter(tasks)), \
                patch.object(wrapper.os, "pidfd_open") as descriptor:
            with self.assertRaisesRegex(wrapper.WrapperError, "no additional threads"), wrapper._subreaper():
                self.fail("A native thread must prevent startup.")
            descriptor.assert_not_called()

    def test_prctl_query_or_enable_failure_prevents_child_start(self):
        for results in ([-1], [0, -1]):
            with self.subTest(results=results), patch.object(wrapper.ctypes, "CDLL") as library, \
                    patch.object(wrapper, "SharedToolCacheProxy") as proxy:
                library.return_value.prctl.side_effect = results
                with self.assertRaisesRegex(wrapper.WrapperError, "Linux process controls are unavailable"):
                    wrapper.run_registered([sys.executable], (_registration(self.tools),))
                proxy.assert_not_called()
                operations = [call.args[0] for call in library.return_value.prctl.call_args_list]
                self.assertEqual(operations, [37] if len(results) == 1 else [37, 36])

    def test_failed_ownership_check_does_not_restore_the_subreaper(self):
        cases = [(OSError("synthetic-proc-failure"), "ownership state could not be checked"),
            ({12345}, "could not finish child cleanup")]
        for final_state, message in cases:
            with self.subTest(message=message), patch.object(wrapper.ctypes, "CDLL") as library, \
                    patch.object(wrapper, "_direct_children", side_effect=[set(), final_state]):
                library.return_value.prctl.return_value = 0
                with self.assertRaisesRegex(wrapper.WrapperError, message), wrapper._subreaper():
                    pass
                operations = [call.args[:2] for call in library.return_value.prctl.call_args_list]
                self.assertEqual(len(operations), 2)
                self.assertEqual(operations[1], (36, 1))

    def test_child_scan_tolerates_a_non_main_thread_exit(self):
        main = Path(f"/proc/self/task/{os.getpid()}")
        gone = Path("/proc/self/task/123456789")

        def read(path, *_args, **_kwargs):
            if path == main / "children":
                return "11 22"
            raise FileNotFoundError("The synthetic thread exited.")

        with patch.object(Path, "iterdir", return_value=iter((main, gone))), \
                patch.object(Path, "read_text", read):
            self.assertEqual(wrapper._direct_children(), {11, 22})
        with patch.object(Path, "read_text", side_effect=FileNotFoundError), self.assertRaises(FileNotFoundError):
            wrapper._direct_children()

    def test_pidfd_signal_failure_does_not_skip_other_owned_children(self):
        owned = wrapper._OwnedChildren()
        owned.handles = {11: 101, 22: 202}
        with patch.object(wrapper.signal, "pidfd_send_signal", side_effect=[OSError(), None]) as send, \
                self.assertRaises(OSError):
            owned.send(signal.SIGKILL, refresh=False)
        self.assertEqual([call.args[0] for call in send.call_args_list], [101, 202])

    def test_exited_selected_child_does_not_signal_unselected_descriptors(self):
        owned = wrapper._OwnedChildren()
        owned.handles = {11: 101, 22: 202}
        with patch.object(wrapper.signal, "pidfd_send_signal", side_effect=ProcessLookupError) as send:
            self.assertEqual(owned.send(signal.SIGTERM, only={22}, refresh=False), {22})
        send.assert_called_once_with(202, signal.SIGTERM)

    def test_reaping_keeps_the_leader_and_closes_only_an_exited_child_descriptor(self):
        owned = wrapper._OwnedChildren()
        owned.handles = {11: 101, 22: 202}
        with patch.object(wrapper.os, "waitpid", side_effect=[(0, 0), (22, 0)]) as reap, \
                patch.object(wrapper.os, "close") as close:
            owned.reap(11)
            self.assertEqual(owned.handles, {11: 101, 22: 202})
            close.assert_not_called()
            owned.reap(11)
            self.assertEqual(owned.handles, {11: 101})
            close.assert_called_once_with(202)
            self.assertEqual([call.args for call in reap.call_args_list], [(22, os.WNOHANG)] * 2)
        with patch.object(wrapper.signal, "pidfd_send_signal") as send:
            owned.send(signal.SIGTERM, refresh=False)
        send.assert_called_once_with(101, signal.SIGTERM)

    def test_cleanup_observes_leader_exit_before_final_child_refresh(self):
        owned = Mock()
        owned.handles = {11: 101}
        state = {"adopted": False, "reaped": False}
        sent = []

        def observe_exit(*_args):
            state["adopted"] = True
            return object()

        def refresh():
            if state["adopted"] and not state["reaped"]:
                owned.handles[22] = 202

        def send(_signum, *, only=None):
            selected = set(owned.handles) if only is None else only
            sent.extend(selected)
            return selected

        def reap(_leader):
            if 22 in sent:
                state["reaped"] = True
                owned.handles.pop(22, None)

        owned.refresh.side_effect = refresh
        owned.send.side_effect = send
        owned.reap.side_effect = reap
        with patch.object(wrapper.os, "waitid", side_effect=observe_exit), \
                patch.object(wrapper.time, "sleep"):
            wrapper._stop_children(owned, 11, time.monotonic() + 10, [], signal.SIGTERM, set())
        self.assertIn(22, sent)
        self.assertTrue(state["reaped"])

    def test_cleanup_does_not_send_a_queued_signal_twice(self):
        owned = Mock()
        owned.handles = {11: 101}
        sent = []

        def send(signum, *, only=None):
            selected = set(owned.handles) if only is None else only
            sent.extend((pid, signum) for pid in selected)
            return selected

        owned.send.side_effect = send
        with patch.object(wrapper.os, "waitid", return_value=object()):
            wrapper._stop_children(owned, 11, time.monotonic() + 10,
                [signal.SIGINT], signal.SIGTERM, set())
        self.assertEqual(sent, [(11, signal.SIGINT)])

    def test_cleanup_retries_after_launch_and_reaps_leader_after_signals(self):
        original_stop = wrapper._stop_children
        original_wait = subprocess.Popen.wait
        state = {"attempts": 0, "cleanup_done": False}

        def stop(*args):
            state["attempts"] += 1
            if state["attempts"] == 1:
                raise OSError("Synthetic discovery failure.")
            result = original_stop(*args)
            state["cleanup_done"] = True
            return result

        def wait(child, *args, **kwargs):
            self.assertTrue(state["cleanup_done"])
            return original_wait(child, *args, **kwargs)

        before = wrapper.ctypes.c_int()
        wrapper.ctypes.CDLL(None).prctl(37, wrapper.ctypes.byref(before), 0, 0, 0)
        with patch.object(wrapper, "_stop_children", side_effect=stop), \
                patch.object(subprocess.Popen, "wait", wait), \
                patch.object(wrapper.os, "killpg") as numeric_group_signal:
            result = wrapper.run_registered([sys.executable, "-c", "raise SystemExit(9)"],
                (_registration(self.tools),))
        self.assertEqual(result, 9)
        self.assertEqual(state["attempts"], 2)
        numeric_group_signal.assert_not_called()
        after = wrapper.ctypes.c_int()
        wrapper.ctypes.CDLL(None).prctl(37, wrapper.ctypes.byref(after), 0, 0, 0)
        self.assertEqual(after.value, before.value)

    def test_startup_signals_wait_for_owned_descriptors_and_keep_the_first_deadline(self):
        numbers = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        original = {number: signal.getsignal(number) for number in numbers}
        handlers = dict(original)
        clock = {"now": 0.0}
        state = {"cleanup_done": False}
        owned = Mock()
        owned.handles = {11: 101}
        sent_at = []
        child = Mock(pid=11)

        def send(signum):
            sent_at.append((signum, clock["now"]))
            return {11}

        owned.send.side_effect = send

        def install(number, handler):
            previous = handlers[number]
            handlers[number] = handler
            return previous

        def spawn(*_args, **_kwargs):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            clock["now"] = 1.0
            handlers[signal.SIGINT](signal.SIGINT, None)
            owned.send.assert_not_called()
            return child

        def sleep(_seconds):
            clock["now"] += 3.0

        def stop(*_args):
            state["cleanup_done"] = True

        def wait():
            self.assertTrue(state["cleanup_done"])
            return -signal.SIGKILL

        child.wait.side_effect = wait
        with patch.object(wrapper, "_OwnedChildren", return_value=owned), \
                patch.object(wrapper.signal, "signal", side_effect=install), \
                patch.object(wrapper.subprocess, "Popen", side_effect=spawn), \
                patch.object(wrapper.os, "waitid", side_effect=[None, None, object()]), \
                patch.object(wrapper.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(wrapper.time, "sleep", side_effect=sleep), \
                patch.object(wrapper, "_stop_children", side_effect=stop) as cleanup:
            self.assertEqual(wrapper._run_child(["synthetic-command"], {}), 128 + signal.SIGKILL)
        self.assertEqual([call.args[0] for call in owned.send.call_args_list],
            [signal.SIGTERM, signal.SIGINT, signal.SIGKILL])
        self.assertGreaterEqual(sent_at[-1][1], wrapper._TERMINATE_GRACE_SECONDS)
        self.assertEqual(cleanup.call_args.args[2], wrapper._TERMINATE_GRACE_SECONDS)
        self.assertEqual(cleanup.call_args.args[4], signal.SIGTERM)
        self.assertEqual(handlers, original)
        owned.close.assert_called_once()

    def test_persistent_cleanup_failure_reports_error_without_reaping_the_leader(self):
        child = Mock(pid=11)
        owned = Mock()
        with patch.object(wrapper, "_OwnedChildren", return_value=owned), \
                patch.object(wrapper.signal, "signal", return_value=signal.SIG_DFL) as handlers, \
                patch.object(wrapper.subprocess, "Popen", return_value=child), \
                patch.object(wrapper.os, "waitid", return_value=object()), \
                patch.object(wrapper, "_stop_children", side_effect=OSError("synthetic-private-detail")) as cleanup, \
                self.assertRaisesRegex(wrapper.WrapperError, "could not finish child cleanup") as error:
            wrapper._run_child(["synthetic-command"], {})
        self.assertNotIn("synthetic-private-detail", str(error.exception))
        self.assertEqual(cleanup.call_count, 2)
        owned.send.assert_called_once_with(signal.SIGKILL, refresh=False)
        child.wait.assert_not_called()
        owned.close.assert_called_once()
        self.assertEqual([call.args for call in handlers.call_args_list[-3:]],
            [(number, signal.SIG_DFL) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)])

    def test_child_arguments_environment_stdio_and_response_bytes_are_preserved(self):
        record = self.path / "child.json"
        args = ["--model", "original-model", "--effort", "high", "--permission-mode", "acceptEdits",
            "--settings", '{"autoCompactEnabled":true}', "literal $(command) ' *", "line\nbreak"]
        environment = {"TEST_AUTH": "Bearer synthetic-token", "UNCHANGED": "synthetic-value"}
        raw = _request(self.tools)
        with _upstream() as (endpoint, calls):
            environment["ANTHROPIC_BASE_URL"] = endpoint + "/tenant"
            child = self.launch([sys.executable, "-c", _HTTP_CHILD, str(record), *args],
                environment=environment, stdin=subprocess.PIPE)
            stdout, stderr = child.communicate(raw, timeout=8)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertEqual(stdout, _RESPONSE)
        self.assertEqual(stderr, b"synthetic-child-stderr\n")
        saved = json.loads(record.read_text())
        self.assertEqual(saved["argv"], args)
        self.assertEqual(saved["cwd"], str(ROOT))
        self.assertEqual(saved["environment"]["UNCHANGED"], environment["UNCHANGED"])
        self.assertEqual(saved["environment"]["TEST_AUTH"], environment["TEST_AUTH"])
        self.assertNotEqual(saved["environment"]["ANTHROPIC_BASE_URL"], environment["ANTHROPIC_BASE_URL"])
        self.assertNotIn("ANTHROPIC_BASE_URL", os.environ)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["path"], "/tenant/v1/messages?beta=true")
        self.assertEqual(calls[0]["headers"]["Authorization"], environment["TEST_AUTH"])
        self.assertEqual(calls[0]["headers"]["x-api-key"], "synthetic-key")
        self.assertEqual(calls[0]["body"].replace(MARKER, b"", 1), raw)
        self.closed(saved["environment"]["ANTHROPIC_BASE_URL"])

    def test_descendant_sessions_use_both_explicit_catalogs(self):
        outer, kernel = _tools("Outer", 18), _tools("Kernel", 22)
        catalogs = [self.catalog(outer, "outer.json"), self.catalog(kernel, "kernel.json")]
        raws = [_request(outer + _tools("Tail", 5), "outer-session"),
            _request(kernel + _tools("Tail", 2), "kernel-session")]
        script = """
import json, subprocess, sys
for raw in json.load(sys.stdin):
    subprocess.run([sys.executable, '-c', sys.argv[1]], input=bytes.fromhex(raw), check=True)
"""
        with _upstream() as (endpoint, calls):
            child = self.launch([sys.executable, "-c", script, _HTTP_CHILD], catalogs=catalogs,
                environment={"ANTHROPIC_BASE_URL": endpoint, "TEST_AUTH": "synthetic"}, stdin=subprocess.PIPE)
            stdout, stderr = child.communicate(json.dumps([raw.hex() for raw in raws]).encode(), timeout=8)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertEqual(stdout, _RESPONSE * 2)
        self.assertEqual(stderr, b"synthetic-child-stderr\n" * 2)
        self.assertEqual(len(calls), 2)
        for call, raw, count in zip(calls, raws, (18, 22)):
            self.assertEqual(call["body"].replace(MARKER, b"", 1), raw)
            self.assertIn("cache_control", json.loads(call["body"])["tools"][count - 1])

    def wait_file(self, filename, process, timeout=5):
        path = self.path / filename
        deadline = time.monotonic() + timeout
        while not path.exists() or not path.stat().st_size:
            if process.poll() is not None:
                self.fail(f"The synthetic wrapper exited before {filename}.")
            if time.monotonic() >= deadline:
                self.fail(f"The synthetic child did not create {filename}.")
            time.sleep(0.01)
        return path

    def test_signals_reach_leader_and_descendant_before_proxy_close(self):
        descendant = """
import os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
def stop(number, frame):
    (root / 'descendant-signal').write_text(str(number))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / 'descendant-ready').write_text(str(os.getpid()))
while True: time.sleep(0.1)
"""
        leader = """
import os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
def stop(number, frame):
    (root / 'leader-signal').write_text(str(number))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
subprocess.Popen([sys.executable, '-c', sys.argv[2], str(root)], start_new_session=True)
(root / 'leader-ready').write_text(os.environ['ANTHROPIC_BASE_URL'])
while True: time.sleep(0.1)
"""
        child = self.launch([sys.executable, "-c", leader, str(self.path), descendant])
        endpoint = self.wait_file("leader-ready", child).read_text()
        self.wait_file("descendant-ready", child)
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=7)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertEqual(stdout, b"")
        self.assertEqual(stderr, b"")
        self.assertEqual((self.path / "leader-signal").read_text(), str(int(signal.SIGTERM)))
        self.assertEqual((self.path / "descendant-signal").read_text(), str(int(signal.SIGTERM)))
        self.closed(endpoint)

    def test_ignored_signal_escalates_and_reports_signal_exit_status(self):
        script = """
import os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text(os.environ['ANTHROPIC_BASE_URL'])
while True: time.sleep(0.1)
"""
        child = self.launch([sys.executable, "-c", script, str(self.path / "ready")])
        endpoint = self.wait_file("ready", child).read_text()
        child.send_signal(signal.SIGTERM)
        _, stderr = child.communicate(timeout=7)
        self.assertEqual(child.returncode, 128 + signal.SIGKILL, stderr)
        self.closed(endpoint)

    def test_interrupt_and_hangup_reach_the_child(self):
        script = """
import os, pathlib, signal, sys, time
def stop(number, frame):
    print(number, flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGHUP, stop)
pathlib.Path(sys.argv[1]).write_text(os.environ['ANTHROPIC_BASE_URL'])
while True: time.sleep(0.1)
"""
        for signum in (signal.SIGINT, signal.SIGHUP):
            with self.subTest(signal=signum):
                filename = f"ready-{int(signum)}"
                child = self.launch([sys.executable, "-c", script, str(self.path / filename)])
                endpoint = self.wait_file(filename, child).read_text()
                child.send_signal(signum)
                stdout, stderr = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertEqual(stdout, f"{int(signum)}\n".encode())
                self.assertEqual(stderr, b"")
                self.closed(endpoint)

    def test_normal_leader_exit_terminates_a_remaining_descendant(self):
        descendant = """
import os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
root = pathlib.Path(sys.argv[1])
(root / 'descendant-ready').write_text(str(os.getpid()))
while True: time.sleep(0.1)
"""
        leader = """
import os, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
subprocess.Popen([sys.executable, '-c', sys.argv[2], str(root)], start_new_session=True)
(root / 'endpoint').write_text(os.environ['ANTHROPIC_BASE_URL'])
while not (root / 'descendant-ready').exists(): time.sleep(0.01)
raise SystemExit(7)
"""
        child = self.launch([sys.executable, "-c", leader, str(self.path), descendant])
        _, stderr = child.communicate(timeout=7)
        self.assertEqual(child.returncode, 7, stderr)
        pid = int((self.path / "descendant-ready").read_text())
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.closed((self.path / "endpoint").read_text())

    def test_double_fork_and_new_detached_generation_cannot_escape_cleanup(self):
        final_child = """
import os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1], 'final-pid').write_text(str(os.getpid()))
while True: time.sleep(0.1)
"""
        descendant = """
import os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
if os.fork(): os._exit(0)
os.setsid()
def stop(number, frame):
    subprocess.Popen([sys.executable, '-c', sys.argv[2], str(root)], start_new_session=True)
    while not (root / 'final-pid').exists(): time.sleep(0.01)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / 'detached-pid').write_text(str(os.getpid()))
while True: time.sleep(0.1)
"""
        leader = """
import os, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
subprocess.Popen([sys.executable, '-c', sys.argv[2], str(root), sys.argv[3]], start_new_session=True)
(root / 'endpoint').write_text(os.environ['ANTHROPIC_BASE_URL'])
while not (root / 'detached-pid').exists(): time.sleep(0.01)
raise SystemExit(11)
"""
        child = self.launch([sys.executable, "-c", leader, str(self.path), descendant, final_child])
        _, stderr = child.communicate(timeout=7)
        self.assertEqual(child.returncode, 11, stderr)
        self.assertEqual(stderr, b"")
        for name in ("detached-pid", "final-pid"):
            pid = int((self.path / name).read_text())
            self.assertFalse(Path(f"/proc/{pid}").exists())
        self.closed((self.path / "endpoint").read_text())


if __name__ == "__main__":
    unittest.main()
