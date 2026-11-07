# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offer an opt-in loopback transport for the shared-tool cache policy.

The default transport connects directly to an explicit HTTP(S) upstream. It
uses the system TLS trust settings. Inject another transport for additional
TLS or network proxy settings. The proxy adds no retries or model settings.

An injected transport provides send(method, target, headers, body) and close().
The close method cancels its owned connections. The send method
returns a context manager. Its response provides status, reason, headers, and
iter_bytes(). Headers contain (name, value) pairs. Response chunks remain bytes.
"""

from __future__ import annotations

import http.client
import os
import re
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

_HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade"})
_POLICY_REASONS = frozenset({"disabled", "unsupported_endpoint", "invalid_json", "unsupported_request",
    "shared_tools_mismatch", "unsupported_output_tool", "marker_limit", "unsupported_cache_layout",
    "unsupported_json_layout", "marked", "session_mismatch"})
_POLICY_REASONS |= frozenset({"prefix_missing", "unsupported_prefix", "duplicate_tool_names",
    "prefix_mismatch", "unqualified_output_format", "existing_tool_marker",
    "unqualified_cache_layout", "content_mismatch", "byte_mismatch", "unsupported_json"})


def _upstream(url):
    if not isinstance(url, str) or not url.isascii() or any(ord(char) < 33 or ord(char) == 127 for char in url):
        return None
    try:
        parts = urlsplit(url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment or "%" in parts.hostname):
            return None
        if parts.port is not None and not 1 <= parts.port <= 65535:
            return None
        return parts
    except ValueError:
        return None


def _hop_headers(headers):
    result = set(_HOP_HEADERS)
    for name, value in headers:
        if name.lower() == "connection":
            result.update(token.strip().lower() for token in value.split(","))
    return result


class _ResponseStream:
    def __init__(self, response):
        self.status = response.status
        self.reason = response.reason
        self.headers = response.getheaders()
        self._response = response

    def iter_bytes(self):
        while True:
            chunk = self._response.read1(65536)
            if not chunk:
                return
            yield chunk


class HTTPTransport:
    """Create one HTTP connection for each request without following redirects."""

    def __init__(self, upstream_url, *, timeout=None, ssl_context=None):
        parts = _upstream(upstream_url)
        if parts is None:
            raise ValueError("The upstream URL is unsupported.")
        self._parts = parts
        self._timeout = timeout
        self._ssl_context = ssl_context
        self._connections = {}
        self._lock = threading.Lock()
        self._closed = False

    @contextmanager
    def send(self, method, target, headers, body):
        if self._parts.scheme == "https":
            connection = http.client.HTTPSConnection(self._parts.hostname, self._parts.port,
                timeout=self._timeout, context=self._ssl_context)
        else:
            connection = http.client.HTTPConnection(self._parts.hostname, self._parts.port, timeout=self._timeout)
        # close() can run after connect() and before endheaders(). Disable the
        # implicit reconnect in HTTPConnection.send() so cancellation stays final.
        connection.auto_open = 0
        with self._lock:
            if self._closed:
                raise RuntimeError("The proxy transport is closed.")
            self._connections[connection] = None
        try:
            connection.connect()
            with self._lock:
                if self._closed:
                    raise RuntimeError("The proxy transport is closed.")
                self._connections[connection] = connection.sock
            connection.putrequest(method, target, skip_accept_encoding=True)
            for name, value in headers:
                connection.putheader(name, value)
            connection.putheader("Content-Length", str(len(body)))
            connection.putheader("Connection", "close")
            with self._lock:
                if self._closed:
                    raise RuntimeError("The proxy transport is closed.")
            connection.endheaders(body)
            response = connection.getresponse()
            yield _ResponseStream(response)
        finally:
            connection.close()
            with self._lock:
                self._connections.pop(connection, None)

    def close(self):
        """Cancel owned connections before the proxy joins request threads."""
        with self._lock:
            self._closed = True
            connections = tuple(self._connections.items())
        for connection, sock in connections:
            sock = sock or connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, *args, **kwargs):
        self._clients = set()
        self._clients_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        with self._clients_lock:
            self._clients.add(request)
        super().process_request(request, client_address)

    def shutdown_request(self, request):
        with self._clients_lock:
            self._clients.discard(request)
        super().shutdown_request(request)

    def close_clients(self):
        with self._clients_lock:
            clients = tuple(self._clients)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()

    def handle_error(self, _request, _client_address):
        # Do not print request details or transport exception values.
        return


class SharedToolCacheProxy:
    """Use the original endpoint when disabled or when setup is unsupported.

    Read active and base_url after entering the context. The context stops
    accepting requests and closes its owned connections on exit. Decisions contain only a
    boolean and a fixed reason code. The callback can run concurrently.
    """

    def __init__(self, upstream_url, policy, *, enabled=False, transport=None, on_decision=None):
        if type(enabled) is not bool:
            raise ValueError("The enabled setting must be a boolean.")
        self.base_url = upstream_url
        self.active = False
        self.status = "disabled"
        self._original_url = upstream_url
        self._policy = policy
        self._enabled = enabled
        self._transport = transport
        self._on_decision = on_decision
        self._server = None
        self._thread = None
        self._entered = False
        self._decisions = []
        self._callback_failures = 0
        self._lock = threading.Lock()

    @property
    def decisions(self):
        with self._lock:
            return tuple(dict(item) for item in self._decisions)

    def _record(self, applied, reason):
        decision = {"applied": applied, "reason": reason}
        with self._lock:
            self._decisions.append(decision)
        if callable(self._on_decision):
            try:
                self._on_decision(dict(decision))
            except Exception:  # noqa: BLE001 - diagnostics cannot alter a provider request
                with self._lock:
                    self._callback_failures += 1

    def _prepare(self, body, method, path, headers):
        media_type = headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        content_encoding = headers.get("Content-Encoding", "identity").strip().lower()
        if method == "POST" and path.split("?", 1)[0] == "/v1/messages":
            if content_encoding != "identity":
                return body, False, "unsupported_content_encoding"
            if media_type != "application/json":
                return body, False, "unsupported_content_type"
        try:
            result = self._policy.apply(body, method=method, path=path)
            if (not isinstance(result.body, bytes) or type(result.applied) is not bool
                    or (not result.applied and result.body != body)):
                return body, False, "unsupported_policy_result"
            reason = result.reason if result.reason in _POLICY_REASONS else "unreported_policy_reason"
            return result.body, result.applied, reason
        except Exception:  # noqa: BLE001 - optional policy failure preserves the original request
            return body, False, "policy_error"

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Create a new proxy context for each use.")
        self._entered = True
        if not self._enabled or getattr(self._policy, "enabled", True) is False:
            return self
        parts = _upstream(self._original_url)
        if parts is None or not callable(getattr(self._policy, "apply", None)):
            self.status = "unsupported_configuration"
            return self
        if self._transport is None:
            # Keep the original CLI endpoint when direct system-TLS transport
            # cannot represent the caller's existing network configuration.
            unsupported = (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "NODE_EXTRA_CA_CERTS", "NODE_TLS_REJECT_UNAUTHORIZED", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                "ANTHROPIC_UNIX_SOCKET", "CLAUDE_CODE_CLIENT_CERT", "CLAUDE_CODE_CLIENT_KEY",
            )
            if any(os.environ.get(name) for name in unsupported):
                self.status = "unsupported_transport_settings"
                return self
            self._transport = HTTPTransport(self._original_url)
        if not callable(getattr(self._transport, "send", None)) or not callable(getattr(self._transport, "close", None)):
            self.status = "unsupported_configuration"
            return self
        owner = self
        base_path = parts.path.rstrip("/")

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format, *args):
                return

            def _body(self):
                lengths = self.headers.get_all("Content-Length", [])
                transfers = self.headers.get_all("Transfer-Encoding", [])
                if lengths and transfers:
                    raise ValueError("The request framing is ambiguous.")
                if transfers:
                    if len(transfers) != 1 or transfers[0].strip().lower() != "chunked":
                        raise ValueError("The transfer encoding is unsupported.")
                    body = bytearray()
                    while True:
                        line = self.rfile.readline(65537)
                        size_text = line.split(b";", 1)[0].strip()
                        if len(line) > 65536 or not line.endswith(b"\r\n") or not re.fullmatch(b"[0-9a-fA-F]+", size_text):
                            raise ValueError("The chunk header is invalid.")
                        size = int(size_text, 16)
                        if not size:
                            while True:
                                trailer = self.rfile.readline(65537)
                                if len(trailer) > 65536 or not trailer.endswith(b"\r\n"):
                                    raise ValueError("The chunk trailer is invalid.")
                                if trailer == b"\r\n":
                                    return bytes(body)
                                raise ValueError("Request trailers are unsupported.")
                        chunk = self.rfile.read(size)
                        if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                            raise ValueError("The request chunk is incomplete.")
                        body.extend(chunk)
                if not lengths:
                    return b""
                if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0].strip()):
                    raise ValueError("The content length is invalid.")
                size = int(lengths[0])
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("The request body is incomplete.")
                return body

            def _failure(self, status):
                self._headers_buffer = []
                payload = b'{"error":{"type":"cache_proxy_error","message":"The request could not be forwarded."}}'
                self.send_response_only(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)

            def _handle(self):
                self.close_connection = True
                sent_headers = False
                try:
                    if not self.path.startswith("/") or self.path.startswith("//") or "#" in self.path:
                        self._failure(400)
                        return
                    try:
                        body = self._body()
                    except ValueError:
                        self._failure(400)
                        return
                    body, applied, reason = owner._prepare(body, self.command, self.path, self.headers)
                    owner._record(applied, reason)
                    incoming = list(self.headers.raw_items())
                    remove = _hop_headers(incoming) | {"host", "content-length"}
                    headers = [(name, value) for name, value in incoming if name.lower() not in remove]
                    with owner._transport.send(self.command, base_path + self.path, headers, body) as response:
                        if type(response.status) is not int or not 100 <= response.status <= 599:
                            raise ValueError("The upstream status is unsupported.")
                        if not isinstance(response.reason, str) or any(char in response.reason for char in "\r\n"):
                            raise ValueError("The upstream reason is unsupported.")
                        outgoing = list(response.headers)
                        if any(not isinstance(name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                            or not isinstance(value, str) or any(char in value for char in "\r\n")
                            for name, value in outgoing):
                            raise ValueError("An upstream header is unsupported.")
                        self.send_response_only(response.status, response.reason)
                        remove = _hop_headers(outgoing)
                        if any(name.lower() == "transfer-encoding" for name, _ in outgoing):
                            remove.add("content-length")
                        for name, value in outgoing:
                            if name.lower() not in remove:
                                self.send_header(name, value)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        sent_headers = True
                        if self.command != "HEAD":
                            for chunk in response.iter_bytes():
                                if not isinstance(chunk, bytes):
                                    raise TypeError("Response chunks must be bytes.")
                                self.wfile.write(chunk)
                                self.wfile.flush()
                except Exception:  # noqa: BLE001 - never expose transport exception details
                    if not sent_headers:
                        try:
                            self._failure(502)
                        except OSError:
                            pass

            do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _handle

        try:
            self._server = _Server(("127.0.0.1", 0), Handler)
            self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            self._thread.start()
        except (OSError, RuntimeError):
            if self._server is not None:
                self._server.server_close()
                self._server = None
            self.status = "unavailable"
            return self
        self.base_url = "http://127.0.0.1:" + str(self._server.server_address[1])
        self.active = True
        self.status = "active"
        return self

    def close(self):
        if self._server is not None:
            self.active = False
            self._server.shutdown()
            self._server.close_clients()
            self._transport.close()
            self._server.server_close()
            self._thread.join()
            self._server = None
        self.active = False
        self.base_url = self._original_url
        if self.status == "active":
            self.status = "closed"

    def __exit__(self, _kind, _value, _traceback):
        self.close()
        return False
