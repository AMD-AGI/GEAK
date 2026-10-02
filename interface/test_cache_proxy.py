# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the opt-in proxy with loopback HTTP and synthetic data only."""

import gzip
import http.client
import json
import os
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.parse import urlsplit

from interface.native_cost_controls.cache_proxy import (
    HTTPTransport,
    SharedToolCacheProxy,
)
from interface.native_cost_controls.shared_tool_cache import SharedToolCachePolicy
from interface.test_shared_tool_cache import INSERTION, SHARED_TOOLS, request


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, *_args):
        return


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.calls = []
        self.release = threading.Event()
        self.two_started = threading.Event()
        self.active = 0
        self.peak = 0
        self.mode = "normal"
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return

            def handle_request(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                with owner.lock:
                    owner.calls.append({
                        "method": self.command, "path": self.path,
                        "headers": list(self.headers.raw_items()), "body": raw,
                    })
                    owner.active += 1
                    owner.peak = max(owner.peak, owner.active)
                    if owner.active == 2:
                        owner.two_started.set()
                try:
                    if owner.mode == "redirect":
                        self.send_response(302)
                        self.send_header("Location", "/must-not-follow")
                        self.send_header("Content-Length", "0")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        return
                    if owner.mode == "stream":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(b"data: first\n\n")
                        self.wfile.flush()
                        owner.release.wait(3)
                        self.wfile.write(b"data: last\n\n")
                        self.wfile.flush()
                        return
                    payload = b'{"usage":{"input_tokens":7,"output_tokens":3}}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("request-id", "synthetic-response")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(payload)
                finally:
                    with owner.lock:
                        owner.active -= 1

            do_POST = do_GET = do_HEAD = handle_request

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.upstream = "http://127.0.0.1:" + str(self.server.server_port)
        self.policy = SharedToolCachePolicy(SHARED_TOOLS, enabled=True)
        self.raw = json.dumps(request(), indent=2).encode()

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.environment.stop()

    def send(self, proxy, raw=None, path="/v1/messages?beta=true", method="POST", headers=None):
        endpoint = urlsplit(proxy.base_url)
        connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
        connection.request(method, path, body=self.raw if raw is None else raw,
            headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        result = (response.status, response.getheaders(), response.read())
        connection.close()
        return result

    def test_disabled_proxy_creates_no_listener_or_request(self):
        with patch("interface.native_cost_controls.cache_proxy._Server") as factory:
            with SharedToolCacheProxy(self.upstream, self.policy) as proxy:
                self.assertFalse(proxy.active)
                self.assertEqual(proxy.base_url, self.upstream)
            self.assertEqual(factory.call_count, 0)
        self.assertEqual(self.calls, [])

    def test_unsupported_configuration_keeps_original_endpoint(self):
        for url in ["ftp://example.invalid", "https://user:fixture@example.invalid", "https://example.invalid/?query=1",
                "https://example.invalid:0", "https://example.invalid:bad", "https://example.invalid/with space"]:
            with self.subTest(url=url), SharedToolCacheProxy(url, self.policy, enabled=True) as proxy:
                self.assertFalse(proxy.active)
                self.assertEqual(proxy.base_url, url)
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):
            HTTPTransport("ftp://example.invalid")
        with self.assertRaises(ValueError):
            SharedToolCacheProxy(self.upstream, self.policy, enabled="true")

    def test_unrepresented_proxy_or_tls_settings_keep_original_endpoint(self):
        for name in ["HTTPS_PROXY", "NODE_EXTRA_CA_CERTS", "CLAUDE_CODE_CLIENT_CERT"]:
            with self.subTest(name=name), patch.dict(os.environ, {name: "explicit-fixture-setting"}), SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
                self.assertFalse(proxy.active)
                self.assertEqual(proxy.status, "unsupported_transport_settings")
        self.assertEqual(self.calls, [])

    def test_auth_base_path_request_and_response_bytes_are_preserved(self):
        with SharedToolCacheProxy(self.upstream + "/tenant", self.policy, enabled=True) as proxy:
            result = self.send(proxy, headers={"Authorization": "Bearer public-fixture", "x-api-key": "public-fixture-key"})
            self.assertEqual(result[0], 200)
            self.assertEqual(result[2], b'{"usage":{"input_tokens":7,"output_tokens":3}}')
            self.assertEqual(len(self.calls), 1)
            call = self.calls[0]
            self.assertEqual(call["path"], "/tenant/v1/messages?beta=true")
            self.assertEqual(call["body"].replace(INSERTION, b"", 1), self.raw)
            headers = {key.lower(): value for key, value in call["headers"]}
            self.assertEqual(headers["authorization"], "Bearer public-fixture")
            self.assertEqual(headers["x-api-key"], "public-fixture-key")
            self.assertEqual(proxy.decisions, ({"applied": True, "reason": "marked"},))

    def test_unsupported_json_and_compression_forward_exact_bytes_once(self):
        samples = [(b"not-json", {}), (gzip.compress(self.raw), {"Content-Encoding": "gzip"}),
            (self.raw, {"Content-Type": "text/plain"})]
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
            for raw, headers in samples:
                self.send(proxy, raw=raw, headers=headers)
                self.assertEqual(self.calls[-1]["body"], raw)
                self.assertFalse(proxy.decisions[-1]["applied"])
        self.assertEqual(len(self.calls), 3)

    def test_head_is_forwarded_without_invented_authentication(self):
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
            status, _, data = self.send(proxy, raw=b"", path="/api/hello", method="HEAD")
            self.assertEqual(status, 200)
            self.assertEqual(data, b"")
        self.assertEqual(self.calls[0]["body"], b"")
        self.assertNotIn("authorization", {k.lower() for k, _ in self.calls[0]["headers"]})

    def test_redirect_is_returned_without_a_proxy_retry_or_second_hop(self):
        self.mode = "redirect"
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
            status, headers, _ = self.send(proxy)
            self.assertEqual(status, 302)
            self.assertEqual(dict(headers)["Location"], "/must-not-follow")
        self.assertEqual(len(self.calls), 1)

    def test_stream_bytes_reach_client_before_upstream_tail(self):
        self.mode = "stream"
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
            endpoint = urlsplit(proxy.base_url)
            connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
            connection.request("POST", "/v1/messages", body=self.raw, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            first = response.readline()
            self.assertEqual(first, b"data: first\n")
            self.assertFalse(self.release.is_set())
            self.release.set()
            self.assertEqual(first + response.read(), b"data: first\n\ndata: last\n\n")
            connection.close()
        self.assertEqual(len(self.calls), 1)

    def test_native_concurrent_requests_are_not_serialized(self):
        self.mode = "stream"
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy, ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.send, proxy) for _ in range(2)]
            self.assertTrue(self.two_started.wait(2))
            self.release.set()
            self.assertTrue(all(future.result()[0] == 200 for future in futures))
        self.assertEqual(self.peak, 2)
        self.assertEqual(len(self.calls), 2)

    def test_exit_cancels_owned_stream_without_waiting_for_a_tail(self):
        self.mode = "stream"
        proxy = SharedToolCacheProxy(self.upstream, self.policy, enabled=True)
        proxy.__enter__()
        endpoint = urlsplit(proxy.base_url)
        connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
        connection.request("POST", "/v1/messages", body=self.raw, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.readline(), b"data: first\n")
        start = time.monotonic()
        proxy.close()
        self.assertLess(time.monotonic() - start, 1)
        self.assertFalse(proxy.active)
        connection.close()
        self.release.set()

    def test_transport_close_prevents_a_late_request_or_reconnect(self):
        for paused_method in ["connect", "putrequest", "endheaders"]:
            with self.subTest(paused_method=paused_method):
                entered = threading.Event()
                release = threading.Event()
                errors = []
                transport = HTTPTransport(self.upstream)
                original = getattr(http.client.HTTPConnection, paused_method)

                def gate(connection, *args, _entered=entered, _release=release, _original=original, **kwargs):
                    _entered.set()
                    if not _release.wait(3):
                        raise RuntimeError("The test gate timed out.")
                    return _original(connection, *args, **kwargs)

                def send(transport=transport, errors=errors):
                    try:
                        with transport.send("POST", "/v1/messages", [], self.raw) as response:
                            list(response.iter_bytes())
                    except (OSError, RuntimeError, http.client.HTTPException) as error:
                        errors.append(error)

                with patch.object(http.client.HTTPConnection, paused_method, side_effect=gate, autospec=True):
                    worker = threading.Thread(target=send, daemon=True)
                    worker.start()
                    try:
                        self.assertTrue(entered.wait(3))
                        transport.close()
                        self.assertEqual(self.calls, [])
                    finally:
                        release.set()
                        worker.join(3)
                        transport.close()
                    self.assertFalse(worker.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertEqual(self.calls, [])
                with self.assertRaises(RuntimeError), transport.send("POST", "/v1/messages", [], self.raw):
                    pass

    def test_https_failure_never_downgrades_to_plain_http(self):
        transport = HTTPTransport(self.upstream.replace("http:", "https:"), timeout=1)
        with self.assertRaises(OSError), transport.send("POST", "/v1/messages", [], self.raw):
            pass
        transport.close()
        self.assertEqual(self.calls, [])

    def test_chunked_request_is_decoded_and_forwarded_once(self):
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
            endpoint = urlsplit(proxy.base_url)
            connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=3)
            connection.request("POST", "/v1/messages", body=iter([self.raw[:20], self.raw[20:]]),
                headers={"Content-Type": "application/json"}, encode_chunked=True)
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            connection.close()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["body"].replace(INSERTION, b"", 1), self.raw)

    def test_diagnostic_callback_cannot_change_forwarding(self):
        def bad_callback(_decision):
            raise ValueError("Do not log this exception.")
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True, on_decision=bad_callback) as proxy:
            self.assertEqual(self.send(proxy)[0], 200)
            self.assertEqual(set(proxy.decisions[0]), {"applied", "reason"})
        self.assertEqual(len(self.calls), 1)

    def test_policy_exception_keeps_original_request(self):
        class BadPolicy:
            enabled = True
            def apply(self, *_args, **_kwargs):
                raise ValueError("Private data must not reach diagnostics.")
        with SharedToolCacheProxy(self.upstream, BadPolicy(), enabled=True) as proxy:
            self.assertEqual(self.send(proxy)[0], 200)
            self.assertEqual(proxy.decisions[0]["reason"], "policy_error")
        self.assertEqual(self.calls[0]["body"], self.raw)

    def test_transport_failure_never_retries(self):
        class FailedTransport:
            def __init__(self):
                self.calls = 0
            @contextmanager
            def send(self, *_args):
                self.calls += 1
                raise OSError("Private transport details.")
                yield
            def close(self):
                return
        transport = FailedTransport()
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True, transport=transport) as proxy:
            status, _, body = self.send(proxy)
            self.assertEqual(status, 502)
            self.assertNotIn(b"Private", body)
        self.assertEqual(transport.calls, 1)
        self.assertEqual(self.calls, [])

    def test_invalid_request_framing_never_reaches_upstream(self):
        samples = [
            b"Content-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            b"Content-Length: 0\r\nContent-Length: 0\r\n\r\n",
            b"Content-Length: -1\r\n\r\n",
            b"Content-Length: 4\r\n\r\na",
            b"Transfer-Encoding: unsupported\r\n\r\n",
            b"Transfer-Encoding: chunked\r\n\r\ninvalid\r\n",
            b"Transfer-Encoding: chunked\r\n\r\n1\r\nxZZ",
            b"Transfer-Encoding: chunked\r\n\r\n0\r\nX-Trailer: value\r\n\r\n",
            b"Transfer-Encoding: chunked\r\n\r\n0\r\ntruncated",
        ]
        with SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
            endpoint = urlsplit(proxy.base_url)
            for sample in samples:
                with self.subTest(sample=sample), socket.create_connection((endpoint.hostname, endpoint.port), timeout=3) as client:
                    client.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: fixture\r\n" + sample)
                    client.shutdown(socket.SHUT_WR)
                    response = http.client.HTTPResponse(client)
                    response.begin()
                    self.assertEqual(response.status, 400)
                    response.read()
        self.assertEqual(self.calls, [])

    def test_invalid_upstream_headers_return_one_fixed_failure(self):
        class Response:
            status, reason, headers = 200, "OK", []
            def iter_bytes(self):
                yield b"must-not-leak"
        class Transport:
            def __init__(self, response):
                self.response = response
                self.calls = 0
            @contextmanager
            def send(self, *_args):
                self.calls += 1
                yield self.response
            def close(self):
                return
        for field, value in [("status", 999), ("reason", "OK\r\nInjected: true"),
                ("headers", [("Invalid\r\nName", "value")]), ("headers", [("X-Test", "bad\nvalue")])]:
            with self.subTest(field=field, value=value):
                response = Response()
                setattr(response, field, value)
                transport = Transport(response)
                with SharedToolCacheProxy(self.upstream, self.policy, enabled=True, transport=transport) as proxy:
                    status, _, body = self.send(proxy)
                    self.assertEqual(status, 502)
                    self.assertNotIn(b"must-not-leak", body)
                    self.assertNotIn(b"Injected", body)
                self.assertEqual(transport.calls, 1)

    def test_invalid_policy_results_preserve_original_request_bytes(self):
        class Policy:
            enabled = True
            def apply(self, *_args, **_kwargs):
                return type("Result", (), {"body": b"changed-without-approval", "applied": False, "reason": "private-value"})()
        with SharedToolCacheProxy(self.upstream, Policy(), enabled=True) as proxy:
            self.assertEqual(self.send(proxy)[0], 200)
            self.assertEqual(proxy.decisions[0]["reason"], "unsupported_policy_result")
        self.assertEqual(self.calls[0]["body"], self.raw)

    def test_listener_failure_retains_the_original_endpoint(self):
        with patch("interface.native_cost_controls.cache_proxy._Server", side_effect=OSError("fixture")), \
                SharedToolCacheProxy(self.upstream, self.policy, enabled=True) as proxy:
            self.assertFalse(proxy.active)
            self.assertEqual(proxy.status, "unavailable")
            self.assertEqual(proxy.base_url, self.upstream)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
