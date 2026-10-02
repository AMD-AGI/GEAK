# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test synthetic catalog capture without starting a native CLI."""

import hashlib
import http.client
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import field, make_dataclass
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import interface
from interface.native_cost_controls import produce_catalog as producer


def tool(name="Read"):
    return {
        "name": name,
        "description": "Read the synthetic input.",
        "input_schema": {"type": "object", "properties": {}},
    }


def request(tools=None):
    return producer.wire({
        "model": "synthetic-model",
        "max_tokens": 256,
        "tools": [tool()] if tools is None else tools,
        "messages": [{"role": "user", "content": "PRIVATE_SYNTHETIC_PROMPT"}],
        "metadata": {"user_id": "PRIVATE_SYNTHETIC_ID"},
    })


def session_request(session, model="synthetic-model"):
    body = producer.DECODER.decode(request().decode())
    body["model"] = model
    body["metadata"]["user_id"] = json.dumps({"session_id": session})
    return producer.wire(body)


def accepted_capture():
    capture = producer.SyntheticCapture(1)
    capture.expected_session_id = "synthetic-worker-session"
    capture.expected_model = "synthetic-model"
    capture.validate_headers({"x-claude-code-session-id": capture.expected_session_id})
    capture.message(session_request(capture.expected_session_id))
    return capture


def http_request(endpoint, method, path, body=None, headers=None):
    address = urlsplit(endpoint)
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=2)
    try:
        if headers is None:
            headers = {"x-claude-code-session-id": "synthetic-session"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


class SyntheticCaptureTests(unittest.TestCase):
    def test_header_and_body_session_mismatch_prevents_catalog_export(self):
        capture = producer.SyntheticCapture(1)
        capture.expected_session_id = "selected-session"
        capture.validate_headers({"x-claude-code-session-id": "selected-session"})
        with self.assertRaisesRegex(ValueError, "portable cache policy rejected"):
            capture.message(session_request("different-body-session"))
        self.assertIsNone(capture.catalog)
        self.assertIsNone(capture.request)

    def test_missing_or_empty_model_does_not_export_a_catalog(self):
        for model in (None, "", 123):
            with self.subTest(model=model):
                capture = producer.SyntheticCapture(1)
                body = producer.DECODER.decode(request().decode())
                body["model"] = model
                with self.assertRaises(ValueError):
                    capture.message(producer.wire(body))
                self.assertIsNone(capture.catalog)

    def test_expected_header_session_rejects_another_first_session(self):
        capture = producer.SyntheticCapture(1)
        capture.expected_session_id = "selected"
        with self.assertRaises(ValueError):
            capture.validate_headers({"x-claude-code-session-id": "other"})
        self.assertIsNone(capture.session_id)

    def test_expected_model_rejects_a_different_model_before_export(self):
        capture = producer.SyntheticCapture(1)
        capture.expected_model = "selected-model"
        with self.assertRaises(ValueError):
            capture.message(request())
        self.assertIsNone(capture.catalog)

    def test_prefix_count_requires_a_positive_integer(self):
        for value in (None, False, True, 0, -1, 1.5, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                producer.SyntheticCapture(value)

    def test_prefix_cannot_exceed_the_request_catalog(self):
        capture = producer.SyntheticCapture(2)
        with self.assertRaises(ValueError):
            capture.message(request())
        self.assertIsNone(capture.catalog)
        self.assertIsNone(capture.request)

    def test_registered_prefix_rejects_output_tool_and_existing_marker(self):
        marked = tool()
        marked["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        for invalid in (tool("StructuredOutput"), marked):
            with self.subTest(tool=invalid["name"]):
                capture = producer.SyntheticCapture(1)
                with self.assertRaises(ValueError):
                    capture.message(request([invalid]))
                self.assertIsNone(capture.catalog)

    def test_catalog_preserves_decimal_precision(self):
        decimal = "0.123456789012345678901234567890123456789"
        raw = (
            '{"model":"synthetic-model","tools":[{"name":"Read",'
            '"input_schema":{"type":"object","properties":{"value":'
            '{"type":"number","minimum":' + decimal + '}}}}]}'
        ).encode()
        capture = producer.SyntheticCapture(1)
        capture.message(raw)
        exported = producer.DECODER.decode(capture.catalog.decode())
        self.assertEqual(
            exported[0]["input_schema"]["properties"]["value"]["minimum"],
            Decimal(decimal),
        )
        self.assertIn(decimal.encode(), capture.catalog)

    def test_schema_cache_control_is_data(self):
        definition = tool()
        definition["input_schema"]["properties"]["cache_control"] = {"type": "string"}
        capture = producer.SyntheticCapture(1)
        capture.message(request([definition]))
        self.assertEqual(producer.DECODER.decode(capture.catalog.decode()), [definition])

    def test_capture_exports_only_selected_tools_and_safe_request_metadata(self):
        raw = request([tool("Read"), tool("Tail")])
        capture = producer.SyntheticCapture(1)
        capture.validate_headers({"x-claude-code-session-id": "PRIVATE_NATIVE_SESSION"})
        capture.message(raw)
        self.assertEqual(producer.DECODER.decode(capture.catalog.decode()), [tool("Read")])
        self.assertEqual(capture.request["tool_count"], 2)
        self.assertEqual(capture.request["prefix_count"], 1)
        self.assertEqual(capture.request["request_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(capture.request["request_bytes"], len(raw))
        exported = capture.catalog + producer.wire(capture.request)
        self.assertNotIn(b"PRIVATE_SYNTHETIC_PROMPT", exported)
        self.assertNotIn(b"PRIVATE_SYNTHETIC_ID", exported)
        self.assertNotIn(b"PRIVATE_NATIVE_SESSION", exported)
        self.assertEqual(capture.request["session_sha256"], producer.sha(b"PRIVATE_NATIVE_SESSION"))
        self.assertNotIn("messages", capture.request)

    def test_session_header_cannot_change_during_capture(self):
        capture = producer.SyntheticCapture(1)
        capture.validate_headers({"x-claude-code-session-id": "synthetic-session"})
        with self.assertRaises(ValueError):
            capture.validate_headers({"x-claude-code-session-id": "another-session"})
        self.assertEqual(capture.session_id, "synthetic-session")

    def test_response_is_fixed_terminal_text_without_tool_use(self):
        response = producer.SyntheticCapture(1).message(request())
        events = [json.loads(line[6:]) for line in response.splitlines()
                  if line.startswith(b"data: ")]
        blocks = [event["content_block"] for event in events
                  if event["type"] == "content_block_start"]
        deltas = [event["delta"] for event in events
                  if event["type"] == "content_block_delta"]
        self.assertEqual(blocks, [{"type": "text", "text": ""}])
        self.assertEqual(deltas, [{"type": "text_delta", "text": producer.TERMINAL}])
        self.assertEqual(events[-1]["type"], "message_stop")
        self.assertNotIn(b"tool_use", response)

    def test_second_generation_is_rejected_without_replacing_the_catalog(self):
        capture = producer.SyntheticCapture(1)
        capture.message(request())
        original = capture.catalog, dict(capture.request)
        with self.assertRaises(ValueError):
            capture.message(request([tool("Changed")]))
        self.assertEqual((capture.catalog, capture.request), original)

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        raws = (
            b'{"model":"first","model":"second","tools":[]}',
            b'{"model":"synthetic-model","tools":[],"max_tokens":NaN}',
        )
        for raw in raws:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                producer.SyntheticCapture(1).message(raw)


class SyntheticServerTests(unittest.TestCase):
    def test_incomplete_body_returns_a_protocol_error_without_a_catalog(self):
        capture = producer.SyntheticCapture(1)
        with capture.server() as endpoint:
            address = urlsplit(endpoint)
            connection = http.client.HTTPConnection(address.hostname, address.port, timeout=2)
            try:
                connection.putrequest("POST", "/v1/messages")
                connection.putheader("Content-Length", "100")
                connection.endheaders()
                connection.send(b"{}")
                connection.sock.shutdown(socket.SHUT_WR)
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                self.assertEqual(json.loads(response.read())["error"]["type"], "invalid_request_error")
            finally:
                connection.close()
        self.assertIsNone(capture.catalog)
        self.assertEqual(capture.errors, 1)

    def test_capture_stops_accepting_connections_before_native_schema_repair(self):
        capture = producer.SyntheticCapture(1)
        capture.stop_after_capture = True
        with capture.server() as endpoint:
            status, _, response = http_request(endpoint, "POST", "/v1/messages", request(),
                {"x-claude-code-session-id": "session"})
            self.assertEqual(status, 200)
            self.assertIn(producer.TERMINAL.encode(), response)
            self.assertTrue(capture.completed.is_set())
            address = urlsplit(endpoint)
            connection = http.client.HTTPConnection(address.hostname, address.port, timeout=0.05)
            try:
                connection.request("POST", "/v1/messages", body=request())
                with self.assertRaises(TimeoutError):
                    connection.getresponse()
            finally:
                connection.close()
        self.assertEqual(capture.requests, 1)
        self.assertEqual(capture.errors, 0)

    def test_supported_routes_return_only_local_responses(self):
        capture = producer.SyntheticCapture(1)
        with capture.server() as endpoint:
            status, _, body = http_request(endpoint, "HEAD", "/api/hello")
            self.assertEqual((status, body), (200, b""))
            status, _, body = http_request(
                endpoint, "POST", "/v1/messages/count_tokens?beta=true", b"{}")
            self.assertEqual((status, json.loads(body)), (200, {"input_tokens": 1}))
            status, headers, body = http_request(endpoint, "POST", "/v1/messages", request())
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "text/event-stream")
            self.assertIn(producer.TERMINAL.encode(), body)
            self.assertNotIn(b"tool_use", body)
            self.assertTrue(capture.completed.wait(timeout=1))
        self.assertEqual(capture.errors, 0)

    def test_main_capture_rejects_child_or_missing_session_headers(self):
        cases = ({}, {"x-claude-code-session-id": "synthetic-session",
                      "x-claude-code-agent-id": "synthetic-child"})
        for headers in cases:
            with self.subTest(headers=headers):
                capture = producer.SyntheticCapture(1)
                with capture.server() as endpoint:
                    status, _, _ = http_request(endpoint, "POST", "/v1/messages", request(), headers)
                    self.assertEqual(status, 400)
                self.assertIsNone(capture.catalog)
                # A protocol error also wakes the SDK so it can disconnect.
                self.assertTrue(capture.completed.is_set())

    def test_unknown_route_and_second_generation_return_sanitized_errors(self):
        capture = producer.SyntheticCapture(1)
        with capture.server() as endpoint:
            self.assertEqual(http_request(endpoint, "POST", "/v1/messages", request())[0], 200)
            for path in ("/v1/messages", "/PRIVATE_SYNTHETIC_ROUTE"):
                status, _, body = http_request(endpoint, "POST", path, request())
                self.assertEqual(status, 400)
                self.assertNotIn(b"PRIVATE_SYNTHETIC", body)
                self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")
        self.assertEqual(capture.errors, 2)

    def test_length_and_encoding_rejections_do_not_wait_for_a_body(self):
        cases = (
            [("Content-Length", str(producer.MAX_BODY + 1))],
            [("Content-Length", "-1")],
            [("Content-Length", "invalid")],
            [("Content-Length", "1"), ("Content-Length", "1")],
            [("Transfer-Encoding", "chunked")],
            [("Content-Encoding", "gzip"), ("Content-Length", "100")],
        )
        capture = producer.SyntheticCapture(1)
        with capture.server() as endpoint:
            address = urlsplit(endpoint)
            for headers in cases:
                with self.subTest(headers=headers):
                    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=2)
                    try:
                        connection.putrequest("POST", "/v1/messages")
                        for name, value in headers:
                            connection.putheader(name, value)
                        connection.endheaders()
                        # No body follows. Reading before validation would time out.
                        response = connection.getresponse()
                        self.assertEqual(response.status, 400)
                        response.read()
                    finally:
                        connection.close()
        self.assertIsNone(capture.catalog)

    def test_request_limit_stops_further_session_requests(self):
        capture = producer.SyntheticCapture(1)
        with capture.server() as endpoint:
            for _ in range(12):
                self.assertEqual(http_request(endpoint, "HEAD", "/api/hello")[0], 200)
            self.assertEqual(http_request(endpoint, "HEAD", "/api/hello")[0], 400)
        self.assertEqual(capture.errors, 1)
        self.assertIsNone(capture.catalog)


class CaptureOrchestrationTests(unittest.TestCase):
    def setUp(self):
        names = ("model", "allowed_tools", "permission_mode", "settings", "setting_sources", "extra_args",
                 "strict_mcp_config", "mcp_servers", "hooks", "session_id", "env", "cli_path", "cwd")
        self.options_type = make_dataclass("FakeOptions", [(name, object, field(default=None)) for name in names])
        self.sdk = ModuleType("claude_agent_sdk")
        self.sdk.ClaudeAgentOptions = self.options_type
        self.runner = ModuleType("interface.run_e2e")
        self.runner._invoke_via_sdk = self.invoke
        self.source_options = {
            "model": "synthetic-model", "allowed_tools": ["Read", "Bash"],
            "permission_mode": "bypassPermissions",
            "settings": json.dumps({"enableWorkflows": True, "ultracode": True}),
            "setting_sources": [], "extra_args": {"effort": "high"},
            "env": {"SOURCE_OPTION": "unchanged"}, "cli_path": "/unused/native-cli",
        }
        self.captures = []
        self.mode = "success"
        self.work = None
        self.native_options = None
        self.original_capture = producer.SyntheticCapture
        for patcher in (
            patch.dict(sys.modules, {"claude_agent_sdk": self.sdk, "interface.run_e2e": self.runner}),
            patch.object(interface, "run_e2e", self.runner, create=True),
            patch.object(producer, "SyntheticCapture", side_effect=self.make_capture),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_capture(self, count):
        capture = self.original_capture(count)
        self.captures.append(capture)
        return capture

    def invoke(self, prompt, timeout, *, settings_profile, native_cwd):
        self.assertEqual((prompt, timeout, settings_profile), (producer.PROMPT, 9, "isolated"))
        self.work = Path(native_cwd)
        self.assertTrue(self.work.is_dir())
        self.assertEqual(list(self.work.iterdir()), [])
        self.native_options = self.sdk.ClaudeAgentOptions(**self.source_options)
        self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "synthetic-local-key")
        self.assertEqual(os.environ["GEAK_SHARED_TOOL_CACHE"], "0")
        self.assertEqual(os.environ["GEAK_LOCAL_HELPERS"], "0")
        endpoint = os.environ["ANTHROPIC_BASE_URL"]
        self.assertEqual(urlsplit(endpoint).hostname, "127.0.0.1")
        if self.mode == "raise":
            raise RuntimeError("The fake runner failed.")
        if self.mode != "no_request":
            headers = {"x-claude-code-session-id": self.native_options.session_id}
            if self.mode == "protocol_error":
                headers = {}
            status, _, _ = http_request(endpoint, "POST", "/v1/messages",
                session_request(self.native_options.session_id), headers)
            self.assertEqual(status, 400 if self.mode == "protocol_error" else 200)
            if self.mode == "changed_result_model":
                self.captures[-1].request["model"] = "changed-model"
        return "unexpected terminal" if self.mode == "wrong_terminal" else producer.TERMINAL

    def test_fake_sdk_capture_preserves_source_options_and_checks_real_marker(self):
        ambient = {"ANTHROPIC_BASE_URL": "http://ambient.invalid", "ANTHROPIC_API_KEY": "ambient-test-key"}
        with patch.dict(os.environ, ambient):
            capture, profile = producer._capture(SimpleNamespace(prefix_count=1, timeout=9))
            self.assertEqual({key: os.environ[key] for key in ambient}, ambient)
        self.assertIs(self.sdk.ClaudeAgentOptions, self.options_type)
        self.assertFalse(self.work.exists())
        for key in ("model", "allowed_tools", "permission_mode", "setting_sources", "extra_args"):
            self.assertEqual(getattr(self.native_options, key), self.source_options[key])
            self.assertEqual(profile[key], self.source_options[key])
        self.assertEqual(self.native_options.env, {"SOURCE_OPTION": "unchanged"})
        self.assertEqual(self.native_options.cli_path, "/unused/native-cli")
        self.assertEqual(json.loads(self.native_options.settings), {
            "enableWorkflows": True, "ultracode": True, "disableAllHooks": True})
        self.assertTrue(self.native_options.strict_mcp_config)
        self.assertEqual((self.native_options.hooks, self.native_options.mcp_servers), ({}, {}))
        self.assertEqual(capture.expected_session_id, self.native_options.session_id)
        self.assertEqual(capture.request["portable_policy"], {
            "applied": True, "reason": "marked", "only_marker_bytes_changed": True})
        self.assertNotIn(self.native_options.session_id, producer.wire(profile).decode())
        self.assertNotIn(b"PRIVATE_SYNTHETIC_PROMPT", capture.catalog)

    def test_capture_rejects_unsupported_sdk_before_starting_a_server(self):
        self.sdk.ClaudeAgentOptions = make_dataclass("OldOptions", [("model", str)])
        with patch.object(producer.tempfile, "TemporaryDirectory") as directory, \
                self.assertRaisesRegex(ValueError, "cannot isolate hooks"):
            producer._capture(SimpleNamespace(prefix_count=1, timeout=9))
        directory.assert_not_called()
        self.assertEqual(self.captures, [])

    def test_capture_rejects_custom_settings_and_restores_the_constructor(self):
        self.source_options["settings"] = '{"customHook":"not-admitted"}'
        with self.assertRaisesRegex(ValueError, "settings exceed"):
            producer._capture(SimpleNamespace(prefix_count=1, timeout=9))
        self.assertIs(self.sdk.ClaudeAgentOptions, self.options_type)
        self.assertFalse(self.work.exists())
        self.assertIsNone(self.captures[-1].catalog)

    def test_failed_or_incomplete_runner_results_never_accept_a_capture(self):
        for mode in ("no_request", "protocol_error", "wrong_terminal", "changed_result_model", "raise"):
            with self.subTest(mode=mode):
                self.mode = mode
                with self.assertRaises(RuntimeError):
                    producer._capture(SimpleNamespace(prefix_count=1, timeout=9))
                self.assertIs(self.sdk.ClaudeAgentOptions, self.options_type)
                self.assertFalse(self.work.exists())


class ProducerFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cli = self.root / "native-cli"
        self.cli.write_text("#!/bin/sh\nexit 99\n")
        self.cli.chmod(0o700)
        self.cli_sha = producer._file_sha(self.cli)
        self.source = self.root / "entry.py"
        self.source.write_text("SOURCE_VERSION = 1\n")
        self.sdk = self.root / "sdk"
        self.sdk.mkdir()
        self.sdk_source = self.sdk / "__init__.py"
        self.sdk_source.write_text("SDK_VERSION = 1\n")
        for patcher in (
            patch.object(producer, "ROOT", self.root),
            patch.object(producer, "SOURCE_FILES", ("entry.py",)),
            patch.object(producer.metadata, "version", return_value="test-sdk"),
            patch.object(producer.util, "find_spec", return_value=SimpleNamespace(origin=str(self.sdk_source))),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def arguments(self):
        return ["--cli", str(self.cli), "--cli-sha256", self.cli_sha,
                "--sdk-version", "test-sdk", "--prefix-count", "1",
                "--output", str(self.root / "output")]


class ProducerBindingTests(ProducerFixture, unittest.TestCase):
    def test_missing_sdk_distribution_stops_main_before_any_process_start(self):
        with patch.object(producer.metadata, "version", side_effect=producer.metadata.PackageNotFoundError), \
                patch.object(producer, "_run_isolated") as run, \
                self.assertRaises(producer.metadata.PackageNotFoundError):
            producer.main(self.arguments())
        run.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_sdk_spec_requires_an_origin(self):
        for spec in (None, SimpleNamespace(origin=None)):
            with self.subTest(spec=spec), patch.object(producer.util, "find_spec", return_value=spec), \
                    self.assertRaisesRegex(ValueError, "SDK package is unavailable"):
                producer._bindings(self.cli, self.cli_sha, "test-sdk")

    def test_sdk_without_python_sources_cannot_produce_bindings(self):
        self.sdk_source.unlink()
        with self.assertRaisesRegex(ValueError, "SDK source files are unavailable"):
            producer._bindings(self.cli, self.cli_sha, "test-sdk")

    def test_bindings_detect_source_and_sdk_changes(self):
        original = producer._bindings(self.cli, self.cli_sha, "test-sdk")
        self.source.write_text("SOURCE_VERSION = 2\n")
        changed_source = producer._bindings(self.cli, self.cli_sha, "test-sdk")
        self.assertNotEqual(original["source_sha256"], changed_source["source_sha256"])
        self.assertEqual(original["sdk_sources_sha256"], changed_source["sdk_sources_sha256"])
        self.sdk_source.write_text("SDK_VERSION = 2\n")
        changed_sdk = producer._bindings(self.cli, self.cli_sha, "test-sdk")
        self.assertNotEqual(original["sdk_sources_sha256"], changed_sdk["sdk_sources_sha256"])

    def test_changed_cli_is_rejected_before_namespace_or_native_start(self):
        self.cli.write_text("#!/bin/sh\nexit 98\n")
        with patch.object(producer, "_run_isolated") as run, self.assertRaises(ValueError):
            producer.main(self.arguments())
        run.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_changed_sdk_version_is_rejected_before_namespace_or_native_start(self):
        with patch.object(producer.metadata, "version", return_value="changed-sdk"), \
                patch.object(producer, "_run_isolated") as run, self.assertRaises(ValueError):
            producer.main(self.arguments())
        run.assert_not_called()

    def test_worker_refuses_publication_after_bound_source_mutation(self):
        options = SimpleNamespace(cli=self.cli, cli_sha256=self.cli_sha, sdk_version="test-sdk",
                                  output=self.root / "output", prefix_count=1, timeout=30, profile="main")

        def mutate(_options):
            self.source.write_text("SOURCE_VERSION = 2\n")
            return SimpleNamespace(catalog=b"[]\n", request={}), {}

        with patch.object(producer, "_isolate_loopback"), \
                patch.object(producer, "_capture", side_effect=mutate) as capture, \
                self.assertRaisesRegex(RuntimeError, "bound source changed"):
            producer._worker(options)
        capture.assert_called_once()
        self.assertFalse(options.output.exists())

    def test_main_passes_a_sanitized_environment_to_both_namespace_calls(self):
        calls = []

        def run(arguments, environment, timeout, staging=None):
            calls.append((list(arguments), dict(environment), timeout))
            if "--_worker" in arguments:
                output = Path(arguments[arguments.index("--output") + 1])
                output.mkdir()
                catalog = producer.wire([tool()]) + b"\n"
                (output / "catalog.json").write_bytes(catalog)
                (output / "receipt.json").write_text(json.dumps({
                    "catalog_sha256": producer.sha(catalog), "scope": producer.SCOPE,
                    "tool_use_responses": 0,
                }))
            return b""

        secrets = {"ANTHROPIC_API_KEY": "PRIVATE_PROVIDER_KEY", "OPENAI_API_KEY": "PRIVATE_OTHER_KEY",
                   "CLAUDE_CODE_OAUTH_TOKEN": "PRIVATE_OAUTH_TOKEN", "HTTP_PROXY": "http://private.invalid",
                   "GEAK_SHARED_TOOL_CATALOG": "/private/catalog.json"}
        with patch.dict(os.environ, secrets), patch.object(producer, "_run_isolated", side_effect=run), \
                redirect_stdout(io.StringIO()):
            producer.main(self.arguments())
        self.assertEqual(len(calls), 2)
        self.assertIn("--_preflight", calls[0][0])
        self.assertIn("--_worker", calls[1][0])
        for _, environment, _ in calls:
            self.assertFalse(set(secrets) & set(environment))
            self.assertEqual(environment["GEAK_CLAUDE_BIN"], str(self.cli))
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                self.assertEqual(environment[name], "")
            self.assertNotIn("PRIVATE_", repr(environment))


class ProducerWorkerTests(ProducerFixture, unittest.TestCase):
    def options(self, profile="main", name="output"):
        return SimpleNamespace(cli=self.cli, cli_sha256=self.cli_sha, sdk_version="test-sdk",
                               output=self.root / name, prefix_count=1, timeout=9, profile=profile)

    def test_worker_exports_bound_catalog_and_sanitized_receipt(self):
        capture = accepted_capture()
        profile = {"model": "synthetic-model", "allowed_tools": ["Read"]}
        options = self.options()
        with patch.object(producer, "_isolate_loopback"), \
                patch.object(producer, "_capture", return_value=(capture, profile)):
            producer._worker(options)
        self.assertEqual(sorted(path.name for path in options.output.iterdir()), ["catalog.json", "receipt.json"])
        catalog = (options.output / "catalog.json").read_bytes()
        receipt_raw = (options.output / "receipt.json").read_bytes()
        receipt = json.loads(receipt_raw)
        self.assertEqual(catalog, capture.catalog)
        self.assertEqual(receipt["catalog_sha256"], hashlib.sha256(catalog).hexdigest())
        self.assertEqual(receipt["profile_sha256"], hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(receipt["profile"], profile)
        self.assertEqual(receipt["selected_profile"], "main")
        self.assertEqual(receipt["scope"], producer.SCOPE)
        self.assertTrue(receipt["source_hashes_rechecked"])
        self.assertEqual(receipt["bindings"]["cli_sha256"], self.cli_sha)
        self.assertEqual(receipt["synthetic_capture"]["portable_policy"]["only_marker_bytes_changed"], True)
        self.assertEqual(receipt["cpu_affinity"], sorted(os.sched_getaffinity(0)))
        self.assertNotIn(b"PRIVATE_SYNTHETIC_PROMPT", receipt_raw)
        self.assertNotIn(b"synthetic-worker-session", receipt_raw)

    def test_worker_requires_successful_byte_preserving_policy_and_matching_model(self):
        cases = ("no_catalog", "no_request", "missing_policy", "not_applied", "numeric_flag", "changed_bytes", "model")
        for case in cases:
            with self.subTest(case=case):
                capture = accepted_capture()
                if case == "no_catalog":
                    capture.catalog = None
                elif case == "no_request":
                    capture.request = None
                elif case == "missing_policy":
                    capture.request.pop("portable_policy")
                elif case == "not_applied":
                    capture.request["portable_policy"]["applied"] = False
                elif case == "numeric_flag":
                    capture.request["portable_policy"]["applied"] = 1
                elif case == "changed_bytes":
                    capture.request["portable_policy"]["only_marker_bytes_changed"] = False
                else:
                    capture.request["model"] = "changed-model"
                options = self.options(name="output-" + case)
                with patch.object(producer, "_isolate_loopback"), \
                        patch.object(producer, "_capture", return_value=(capture, {"model": "synthetic-model"})), \
                        self.assertRaises(ValueError):
                    producer._worker(options)
                self.assertFalse(options.output.exists())

    def test_worker_uses_profile_bindings_and_preserves_profile_scope(self):
        module = SimpleNamespace(
            bindings=MagicMock(return_value={"source": "source-digest"}),
            capture=MagicMock(return_value=(accepted_capture(), {
                "model": "synthetic-model", "scope": "selected-source-scope",
                "tool_filters": {"disallowed_tools": ["ExternalTool"]},
                "tool_use_responses": 1, "limits": ["Synthetic scope only."],
                "differences_from_default_runner": ["Use fixed local responses."],
            })),
        )
        options = self.options(profile="hyperloom-specialist")
        with patch.object(producer, "_isolate_loopback"), \
                patch.object(producer, "_profile_module", return_value=module), \
                patch.object(producer, "_capture") as main_capture:
            producer._worker(options)
        main_capture.assert_not_called()
        self.assertEqual(module.bindings.call_count, 2)
        module.capture.assert_called_once_with(options)
        receipt = json.loads((options.output / "receipt.json").read_text())
        self.assertEqual(receipt["scope"], "selected-source-scope")
        self.assertEqual(receipt["profile_source_bindings"], {"source": "source-digest"})
        self.assertEqual(receipt["limits"], ["Synthetic scope only."])
        self.assertEqual(receipt["tool_filters_sha256"], hashlib.sha256(
            b'{"disallowed_tools":["ExternalTool"]}').hexdigest())

    def test_changed_profile_source_prevents_worker_publication(self):
        module = SimpleNamespace(bindings=MagicMock(side_effect=[{"source": "before"}, {"source": "after"}]),
            capture=MagicMock(return_value=(accepted_capture(), {"model": "synthetic-model"})))
        options = self.options(profile="kernel-child")
        with patch.object(producer, "_isolate_loopback"), \
                patch.object(producer, "_profile_module", return_value=module), \
                self.assertRaisesRegex(RuntimeError, "bound source changed"):
            producer._worker(options)
        self.assertFalse(options.output.exists())

    def test_failed_receipt_write_removes_partial_output(self):
        options = self.options()
        write_bytes = Path.write_bytes

        def fail_receipt(path, value):
            if path == options.output / "receipt.json":
                raise OSError("Synthetic write failure.")
            return write_bytes(path, value)

        with patch.object(producer, "_isolate_loopback"), \
                patch.object(producer, "_capture", return_value=(accepted_capture(), {"model": "synthetic-model"})), \
                patch.object(Path, "write_bytes", fail_receipt), self.assertRaises(OSError):
            producer._worker(options)
        self.assertFalse(options.output.exists())

    def test_worker_preserves_an_existing_output_directory(self):
        options = self.options()
        options.output.mkdir()
        sentinel = options.output / "existing.txt"
        sentinel.write_text("preserve")
        with patch.object(producer, "_isolate_loopback"), \
                patch.object(producer, "_capture", return_value=(accepted_capture(), {"model": "synthetic-model"})), \
                self.assertRaises(FileExistsError):
            producer._worker(options)
        self.assertEqual(sentinel.read_text(), "preserve")


class ProducerMainTests(ProducerFixture, unittest.TestCase):
    def write_staged_output(self, arguments, environment, timeout, staging=None):
        if "--_worker" in arguments:
            output = Path(arguments[arguments.index("--output") + 1])
            output.mkdir()
            catalog = producer.wire([tool()]) + b"\n"
            (output / "catalog.json").write_bytes(catalog)
            (output / "receipt.json").write_text(json.dumps({
                "catalog_sha256": hashlib.sha256(catalog).hexdigest(),
                "scope": "test-profile", "tool_use_responses": 0,
            }))
        return b""

    def test_main_routes_internal_modes_without_nested_processes(self):
        for mode in ("--_preflight", "--_worker"):
            with self.subTest(mode=mode), patch.object(producer, "_isolate_loopback") as isolate, \
                    patch.object(producer, "_worker") as worker, \
                    patch.object(producer, "_bindings") as bindings, \
                    patch.object(producer, "_run_isolated") as run:
                producer.main([*self.arguments(), mode])
                self.assertEqual(isolate.call_count, int(mode == "--_preflight"))
                self.assertEqual(worker.call_count, int(mode == "--_worker"))
                bindings.assert_not_called()
                run.assert_not_called()

    def test_main_rejects_invalid_limits_and_existing_output_before_startup(self):
        for extra in (("--prefix-count", "0"), ("--timeout", "0"), ("--timeout", "301")):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), \
                    patch.object(producer, "_run_isolated") as run, self.assertRaises(SystemExit):
                producer.main([*self.arguments(), *extra])
            run.assert_not_called()
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "existing.txt"
        sentinel.write_text("preserve")
        with redirect_stderr(io.StringIO()), patch.object(producer, "_run_isolated") as run, \
                self.assertRaises(SystemExit):
            producer.main(self.arguments())
        run.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_non_executable_cli_is_rejected_before_namespace_start(self):
        self.cli.chmod(0o600)
        with redirect_stderr(io.StringIO()), patch.object(producer, "_run_isolated") as run, \
                self.assertRaises(SystemExit):
            producer.main(self.arguments())
        run.assert_not_called()

    def test_profile_paths_and_repeated_framework_roots_reach_the_worker(self):
        from interface.native_cost_controls import (
            hyperloom_catalog_profile,
            kernel_catalog_profile,
        )
        first_root, second_root = self.root / "framework one", self.root / "framework two"
        first_root.mkdir()
        second_root.mkdir()
        profiles = (
            ("kernel-child", kernel_catalog_profile, ["--kernel-source", os.path.relpath(self.source)]),
            ("kernel-frozen", kernel_catalog_profile, ["--kernel-options-source", os.path.relpath(self.source)]),
            ("hyperloom-specialist", hyperloom_catalog_profile, [
                "--hyperloom-source", os.path.relpath(self.sdk),
                "--hyperloom-source-manifest", os.path.relpath(self.source),
                "--hyperloom-framework-root", os.path.relpath(first_root),
                "--hyperloom-framework-root", os.path.relpath(second_root)]),
        )
        for name, module, extras in profiles:
            with self.subTest(profile=name), patch.object(module, "bindings", return_value={}) as bindings, \
                    patch.object(producer, "_run_isolated", side_effect=self.write_staged_output) as run, \
                    redirect_stdout(io.StringIO()):
                producer.main([*self.arguments(), "--output", str(self.root / name), "--profile", name, *extras])
            bindings.assert_called_once()
            self.assertEqual(run.call_count, 2)
            arguments = run.call_args.args[0]
            self.assertEqual(arguments[arguments.index("--profile") + 1], name)
            for index in range(0, len(extras), 2):
                flag, path = extras[index:index + 2]
                forwarded = [arguments[i + 1] for i, item in enumerate(arguments[:-1]) if item == flag]
                self.assertIn(str(Path(path).resolve()), forwarded)
            if name == "hyperloom-specialist":
                self.assertEqual(bindings.call_args.args[0].hyperloom_framework_root, [first_root, second_root])
            self.assertTrue((self.root / name / "catalog.json").is_file())

    def test_preflight_failure_prevents_worker_start_and_output_publication(self):
        with patch.object(producer, "_run_isolated", side_effect=RuntimeError("Namespace unavailable.")) as run, \
                self.assertRaises(RuntimeError):
            producer.main(self.arguments())
        run.assert_called_once()
        self.assertIn("--_preflight", run.call_args.args[0])
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(list(self.root.glob(".geak-catalog-*")), [])

    def test_invalid_staged_catalog_hash_prevents_publication(self):
        def corrupt(arguments, environment, timeout, staging=None):
            self.write_staged_output(arguments, environment, timeout, staging)
            if "--_worker" in arguments:
                output = Path(arguments[arguments.index("--output") + 1])
                (output / "catalog.json").write_bytes(producer.wire([tool("Changed")]))
            return b""

        with patch.object(producer, "_run_isolated", side_effect=corrupt), self.assertRaises(ValueError):
            producer.main(self.arguments())
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(list(self.root.glob(".geak-catalog-*")), [])


class ProducerIsolationTests(unittest.TestCase):
    @contextmanager
    def mocked_loopback(self, flags=1, response=b"local"):
        endpoints = [MagicMock() for _ in range(4)]
        _control, listener, client, connection = endpoints
        for endpoint in endpoints:
            endpoint.__enter__.return_value = endpoint
        listener.getsockname.return_value = ("127.0.0.1", 12345)
        listener.accept.return_value = connection, ("127.0.0.1", 12346)
        client.recv.return_value = response
        with patch.object(producer.socket, "if_nameindex", return_value=[(1, "lo")]), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(producer.os, "statvfs", return_value=SimpleNamespace(f_flag=os.ST_RDONLY)), \
                patch.object(producer.socket, "socket", side_effect=endpoints[:3]) as sockets, \
                patch("fcntl.ioctl", return_value=struct.pack("16sH14x", b"lo", flags)) as ioctl:
            yield sockets, ioctl, client, connection

    def test_loopback_check_uses_only_local_sockets_and_enables_a_down_interface(self):
        for flags in (0, 1):
            with self.subTest(flags=flags), self.mocked_loopback(flags=flags) as (_, ioctl, client, connection):
                producer._isolate_loopback()
                client.connect.assert_called_once_with(("127.0.0.1", 12345))
                connection.sendall.assert_called_once_with(b"local")
                self.assertEqual(ioctl.call_count, 2 if flags == 0 else 1)
                if flags == 0:
                    self.assertEqual(ioctl.call_args.args[1:], (0x8914, struct.pack("16sH14x", b"lo", 1)))

    def test_wrong_loopback_response_rejects_the_namespace(self):
        with self.mocked_loopback(response=b"wrong"), self.assertRaisesRegex(RuntimeError, "loopback check failed"):
            producer._isolate_loopback()

    def test_gpu_devices_or_writable_root_fail_before_socket_access(self):
        for gpu_present, root_flags in ((True, os.ST_RDONLY), (False, 0)):
            with self.subTest(gpu_present=gpu_present), \
                    patch.object(producer.socket, "if_nameindex", return_value=[(1, "lo")]), \
                    patch.object(Path, "exists", return_value=gpu_present), \
                    patch.object(producer.os, "statvfs", return_value=SimpleNamespace(f_flag=root_flags)), \
                    patch.object(producer.socket, "socket") as sockets, self.assertRaises(RuntimeError):
                producer._isolate_loopback()
            sockets.assert_not_called()

    def test_namespace_command_preserves_isolation_and_argument_boundaries(self):
        arguments = ["--output", "/synthetic/path with spaces", "--_worker"]
        for staging in (None, Path("/synthetic/staging")):
            with self.subTest(staging=staging), patch.object(producer.sys, "platform", "linux"), \
                    patch.object(producer.shutil, "which", return_value="/unused/bwrap"), \
                    patch.object(producer.sys, "path", ["", "relative-library"]):
                command = producer._namespace_command(arguments, staging)
            self.assertEqual(command[-len(arguments):], arguments)
            self.assertEqual(command[command.index("--ro-bind") + 1:command.index("--ro-bind") + 3], ["/", "/"])
            self.assertEqual(command[command.index("--dev") + 1], "/dev")
            for required in ("--unshare-user", "--unshare-net", "--unshare-pid", "--die-with-parent", "-I", "-S", "-B"):
                self.assertIn(required, command)
            paths = json.loads(command[command.index("-c") + 2])
            self.assertTrue(all(Path(path).is_absolute() for path in paths))
            if staging is None:
                self.assertNotIn("--bind", command)
            else:
                self.assertEqual(command[command.index("--bind") + 1:command.index("--bind") + 3],
                                 [str(staging), str(staging)])

    def test_successful_isolated_process_returns_only_stdout(self):
        process = MagicMock(returncode=0)
        process.__enter__.return_value = process
        process.communicate.return_value = b"synthetic-output", b"PRIVATE_STDERR"
        environment = {"CUDA_VISIBLE_DEVICES": ""}
        with patch.object(producer, "_namespace_command", return_value=["unused-namespace"]), \
                patch.object(producer.subprocess, "Popen", return_value=process) as spawn:
            result = producer._run_isolated(["--_worker"], environment, 9)
        self.assertEqual(result, b"synthetic-output")
        self.assertEqual(spawn.call_args.kwargs["env"], environment)
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        process.communicate.assert_called_once_with(timeout=9)

    def test_failed_process_keeps_tagged_status_and_discards_other_stderr(self):
        process = MagicMock(returncode=1)
        process.__enter__.return_value = process
        process.communicate.return_value = (b"", b'PRIVATE_STDERR\nGEAK_CAPTURE_STATUS {"code":"fixture-rejected"}\n')
        with patch.object(producer, "_namespace_command", return_value=["unused-namespace"]), \
                patch.object(producer.subprocess, "Popen", return_value=process), \
                self.assertRaises(producer.ProducerError) as error:
            producer._run_isolated([], {}, 1)
        self.assertIn("fixture-rejected", str(error.exception))
        self.assertNotIn("PRIVATE_STDERR", str(error.exception))

    def test_missing_namespace_command_has_no_host_fallback(self):
        with patch.object(producer.sys, "platform", "linux"), \
                patch.object(producer.shutil, "which", return_value=None), \
                patch.object(producer.subprocess, "Popen") as spawn, self.assertRaises(RuntimeError):
            producer._run_isolated([], {}, 1)
        spawn.assert_not_called()

    def test_non_loopback_interface_fails_before_opening_a_socket(self):
        with patch.object(producer.socket, "if_nameindex", return_value=[(1, "lo"), (2, "eth0")]), \
                patch.object(producer.socket, "socket") as connect, self.assertRaises(RuntimeError):
            producer._isolate_loopback()
        connect.assert_not_called()

    def test_worker_checks_namespace_before_binding_or_native_start(self):
        with patch.object(producer, "_isolate_loopback", side_effect=RuntimeError("namespace absent")), \
                patch.object(producer, "_bindings") as bindings, \
                patch.object(producer, "_capture") as capture, self.assertRaises(RuntimeError):
            producer._worker(SimpleNamespace())
        bindings.assert_not_called()
        capture.assert_not_called()

    def test_failed_namespace_does_not_expose_subprocess_errors(self):
        process = MagicMock(returncode=1)
        process.__enter__.return_value = process
        process.communicate.return_value = (b"", b"PRIVATE_SUBPROCESS_ERROR")
        with patch.object(producer, "_namespace_command", return_value=["unused-namespace"]), \
                patch.object(producer.subprocess, "Popen", return_value=process), \
                self.assertRaises(RuntimeError) as error:
            producer._run_isolated([], {}, 1)
        self.assertNotIn("PRIVATE_SUBPROCESS_ERROR", str(error.exception))

    def test_timeout_stops_the_owned_process_group(self):
        process = MagicMock(pid=12345)
        process.__enter__.return_value = process
        process.communicate.side_effect = [subprocess.TimeoutExpired("synthetic", 1), (b"", b"")]
        with patch.object(producer, "_namespace_command", return_value=["unused-namespace"]), \
                patch.object(producer.subprocess, "Popen", return_value=process), \
                patch.object(producer.os, "killpg") as stop, self.assertRaises(RuntimeError):
            producer._run_isolated([], {}, 1)
        stop.assert_called_once_with(12345, signal.SIGKILL)
        self.assertEqual(process.communicate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
