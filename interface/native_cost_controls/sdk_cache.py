# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare opt-in native SDK caching without changing model or tool permissions."""

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from typing import Any

from .cache_proxy import SharedToolCacheProxy
from .shared_tool_cache import CachePolicyResult, SharedToolCachePolicy


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("Duplicate JSON key.")
        value[key] = item
    return value


def _json(value):
    return json.loads(value, object_pairs_hook=_pairs)


class NativeSessionCachePolicy:
    """Register definitions only from a supported request for this SDK session.

    The definitions remain in memory. This class performs no discovery call and
    does not retain request bodies. A different or dynamic catalog uses the
    exact original request.
    """

    def __init__(self, session_id, *, enabled=False):
        if type(enabled) is not bool:
            raise ValueError("The enabled setting must be a boolean.")
        if enabled and (not isinstance(session_id, str) or not session_id):
            raise ValueError("A native session ID is required.")
        self.enabled = enabled
        self.session_id = session_id
        self._registered = None
        self._lock = threading.Lock()

    def apply(self, raw_body, *, method="POST", path="/v1/messages"):
        if not isinstance(raw_body, bytes):
            raise TypeError("The request body must be bytes.")
        if not self.enabled:
            return CachePolicyResult(raw_body, False, "disabled")
        if method != "POST" or not isinstance(path, str) or path.split("?", 1)[0] != "/v1/messages":
            return CachePolicyResult(raw_body, False, "unsupported_endpoint")
        try:
            body = _json(raw_body)
            metadata = _json(body.get("metadata", {}).get("user_id", ""))
            if metadata.get("session_id") != self.session_id:
                return CachePolicyResult(raw_body, False, "session_mismatch")
            tools = body.get("tools")
            if not isinstance(tools, list) or len(tools) < 2 or tools[-1].get("name") != "StructuredOutput":
                return CachePolicyResult(raw_body, False, "unsupported_output_tool")
        except (ValueError, TypeError, AttributeError, UnicodeError, RecursionError):
            return CachePolicyResult(raw_body, False, "invalid_json")
        with self._lock:
            if self._registered is not None:
                policy = self._registered
            else:
                try:
                    policy = SharedToolCachePolicy(tools[:-1], enabled=True)
                except (ValueError, TypeError, RecursionError):
                    return CachePolicyResult(raw_body, False, "unsupported_request")
                result = policy.apply(raw_body, method=method, path=path)
                if result.applied:
                    self._registered = policy
                return result
        return policy.apply(raw_body, method=method, path=path)


@dataclass
class NativeCacheSession:
    """Return prepared options and content-free status for one SDK client."""

    options: Any
    status: str
    proxy: Any = None

    @property
    def decisions(self):
        return self.proxy.decisions if self.proxy is not None else ()


@contextmanager
def native_shared_tool_cache(options, *, enabled=False, proxy_factory=SharedToolCacheProxy):
    """Keep unsupported SDK options and endpoints on their original path."""
    if type(enabled) is not bool:
        raise ValueError("The enabled setting must be a boolean.")
    if not enabled:
        yield NativeCacheSession(options, "disabled")
        return
    required = ("env", "session_id")
    if not is_dataclass(options) or any(not hasattr(options, name) for name in required):
        yield NativeCacheSession(options, "unsupported_sdk_options")
        return
    if any(getattr(options, name, None) for name in ("resume", "continue_conversation", "fork_session")):
        yield NativeCacheSession(options, "unsupported_session_options")
        return
    environment = {**os.environ, **(options.env or {})}
    alternate_provider = (
        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS", "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", "CLAUDE_CODE_USE_MANTLE",
    )
    if any(str(environment.get(name, "")).lower() not in ("", "0", "false") for name in alternate_provider):
        yield NativeCacheSession(options, "unsupported_provider_transport")
        return
    transport_keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
        "NODE_EXTRA_CA_CERTS", "NODE_TLS_REJECT_UNAUTHORIZED", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
        "ANTHROPIC_UNIX_SOCKET", "CLAUDE_CODE_CLIENT_CERT", "CLAUDE_CODE_CLIENT_KEY",
    )
    if any(environment.get(name) for name in transport_keys) or any(
        (options.env or {}).get(name) and options.env[name] != os.environ.get(name)
        for name in ("SSL_CERT_FILE", "SSL_CERT_DIR")
    ):
        yield NativeCacheSession(options, "unsupported_transport_settings")
        return
    session_id = options.session_id or str(uuid.uuid4())
    if not isinstance(session_id, str):
        yield NativeCacheSession(options, "unsupported_session_options")
        return
    endpoint = environment.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    policy = NativeSessionCachePolicy(session_id, enabled=True)
    with proxy_factory(endpoint, policy, enabled=True) as proxy:
        if not proxy.active:
            yield NativeCacheSession(options, proxy.status, proxy)
            return
        prepared = replace(options, session_id=session_id,
            env={**(options.env or {}), "ANTHROPIC_BASE_URL": proxy.base_url})
        yield NativeCacheSession(prepared, "active", proxy)



class CachedSDKClient:
    """Keep the proxy alive for exactly the underlying SDK client context."""

    def __init__(self, client_factory, options, *, enabled=False, cache_factory=native_shared_tool_cache):
        self._client_factory = client_factory
        self._options = options
        self._enabled = enabled
        self._cache_factory = cache_factory
        self._cache = None
        self._client = None
        self.cache_session = None

    async def __aenter__(self):
        self._cache = self._cache_factory(self._options, enabled=self._enabled)
        self.cache_session = self._cache.__enter__()
        try:
            self._client = self._client_factory(options=self.cache_session.options)
            return await self._client.__aenter__()
        except BaseException:
            self._cache.__exit__(*__import__("sys").exc_info())
            raise

    async def __aexit__(self, kind, value, traceback):
        try:
            return await self._client.__aexit__(kind, value, traceback)
        finally:
            self._cache.__exit__(kind, value, traceback)


async def cached_query(query, *, prompt, options, enabled=False):
    """Retain the original query stream and close the proxy after it ends."""
    with native_shared_tool_cache(options, enabled=enabled) as prepared:
        async for message in query(prompt=prompt, options=prepared.options):
            yield message
