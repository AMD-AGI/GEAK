# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test SDK option preservation and per-session cache registration offline."""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ClassVar
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from interface.native_cost_controls.sdk_cache import (
    CachedSDKClient,
    NativeCacheSession,
    NativeSessionCachePolicy,
    native_shared_tool_cache,
)
from interface.test_shared_tool_cache import INSERTION, request


@dataclass
class Options:
    model: str = "unchanged-model"
    effort: str = "unchanged-effort"
    permission_mode: str = "default"
    hooks: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    session_id: str = None
    resume: str = None
    continue_conversation: bool = False
    fork_session: bool = False


class FakeProxy:
    instances: ClassVar[list] = []

    def __init__(self, endpoint, policy, *, enabled):
        self.endpoint = endpoint
        self.policy = policy
        self.enabled = enabled
        self.active = True
        self.base_url = "http://127.0.0.1:49152"
        self.status = "active"
        self.decisions = ()
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True


class NativePolicyTests(TestCase):
    def body(self, session="native-session"):
        value = request()
        value["metadata"] = {"user_id": json.dumps({"session_id": session})}
        return value

    def test_first_supported_native_request_registers_its_exact_prefix(self):
        policy = NativeSessionCachePolicy("native-session", enabled=True)
        raw = json.dumps(self.body(), indent=2).encode()
        result = policy.apply(raw)
        self.assertTrue(result.applied)
        self.assertEqual(result.body.replace(INSERTION, b"", 1), raw)
        changed_schema = self.body()
        changed_schema["tools"][-1]["input_schema"] = {"type": "object", "properties": {"other": {"type": "integer"}}}
        self.assertTrue(policy.apply(json.dumps(changed_schema).encode()).applied)

    def test_other_session_cannot_register_the_prefix(self):
        policy = NativeSessionCachePolicy("native-session", enabled=True)
        raw = json.dumps(self.body("another-session")).encode()
        result = policy.apply(raw)
        self.assertIs(result.body, raw)
        self.assertEqual(result.reason, "session_mismatch")
        self.assertTrue(policy.apply(json.dumps(self.body()).encode()).applied)

    def test_changed_catalog_is_forwarded_unchanged(self):
        policy = NativeSessionCachePolicy("native-session", enabled=True)
        self.assertTrue(policy.apply(json.dumps(self.body()).encode()).applied)
        value = self.body()
        value["tools"][0]["description"] = "A different registered catalog."
        raw = json.dumps(value).encode()
        result = policy.apply(raw)
        self.assertIs(result.body, raw)
        self.assertEqual(result.reason, "shared_tools_mismatch")

    def test_invalid_or_unsupported_input_does_not_register(self):
        policy = NativeSessionCachePolicy("native-session", enabled=True)
        samples = [b"invalid", b"[]", b'{"metadata":{"user_id":"[]"}}']
        root = self.body()
        root["tools"] = root["tools"][:-1]
        samples.append(json.dumps(root).encode())
        for raw in samples:
            result = policy.apply(raw)
            self.assertFalse(result.applied)
            self.assertIs(result.body, raw)
        self.assertTrue(policy.apply(json.dumps(self.body()).encode()).applied)

    def test_disabled_does_not_parse_or_require_a_session(self):
        raw = b"invalid"
        policy = NativeSessionCachePolicy(None)
        self.assertIs(policy.apply(raw).body, raw)

    def test_wrong_endpoint_retains_original_bytes(self):
        raw = json.dumps(self.body()).encode()
        policy = NativeSessionCachePolicy("native-session", enabled=True)
        result = policy.apply(raw, path="/v1/messages/count_tokens")
        self.assertIs(result.body, raw)
        self.assertFalse(result.applied)

    def test_ambiguous_json_and_invalid_catalog_cannot_register_definitions(self):
        policy = NativeSessionCachePolicy("native-session", enabled=True)
        value = self.body()
        value["tools"][0] = None
        for raw in [b'{"metadata":{},"metadata":{}}', json.dumps(value).encode()]:
            result = policy.apply(raw)
            self.assertIs(result.body, raw)
            self.assertFalse(result.applied)
        self.assertTrue(policy.apply(json.dumps(self.body()).encode()).applied)

    def test_invalid_policy_inputs_cannot_enable_caching_implicitly(self):
        with self.assertRaises(ValueError):
            NativeSessionCachePolicy("native-session", enabled="true")
        with self.assertRaises(ValueError):
            NativeSessionCachePolicy(None, enabled=True)
        with self.assertRaises(TypeError):
            NativeSessionCachePolicy("native-session", enabled=True).apply({})


class SDKOptionTests(TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        FakeProxy.instances = []

    def tearDown(self):
        self.environment.stop()

    def test_disabled_retains_the_same_options_without_a_proxy(self):
        options = Options()
        with native_shared_tool_cache(options, proxy_factory=FakeProxy) as prepared:
            self.assertIs(prepared.options, options)
            self.assertEqual(prepared.status, "disabled")
            self.assertEqual(prepared.decisions, ())
        self.assertEqual(FakeProxy.instances, [])

    def test_active_preserves_models_permissions_hooks_and_environment(self):
        hooks = {"PreToolUse": ["original-hook"]}
        options = Options(hooks=hooks, env={"ORIGINAL_SETTING": "same"}, session_id="known-session")
        with native_shared_tool_cache(options, enabled=True, proxy_factory=FakeProxy) as prepared:
            self.assertIsNot(prepared.options, options)
            self.assertEqual(prepared.options.model, options.model)
            self.assertEqual(prepared.options.effort, options.effort)
            self.assertEqual(prepared.options.permission_mode, options.permission_mode)
            self.assertIs(prepared.options.hooks, hooks)
            self.assertEqual(prepared.options.session_id, "known-session")
            self.assertEqual(prepared.options.env["ORIGINAL_SETTING"], "same")
            self.assertEqual(prepared.options.env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:49152")
            self.assertEqual(options.env, {"ORIGINAL_SETTING": "same"})
        self.assertTrue(FakeProxy.instances[0].closed)

    def test_original_sdk_endpoint_override_is_preserved_upstream(self):
        options = Options(env={"ANTHROPIC_BASE_URL": "https://example.invalid/tenant"})
        with native_shared_tool_cache(options, enabled=True, proxy_factory=FakeProxy) as prepared:
            self.assertTrue(prepared.options.session_id)
            self.assertEqual(FakeProxy.instances[0].endpoint, "https://example.invalid/tenant")

    def test_resume_or_unsupported_sdk_options_keep_original_path(self):
        for options in [Options(resume="prior"), Options(continue_conversation=True), Options(session_id=42), object()]:
            with native_shared_tool_cache(options, enabled=True, proxy_factory=FakeProxy) as prepared:
                self.assertIs(prepared.options, options)
                self.assertNotEqual(prepared.status, "active")
        self.assertEqual(FakeProxy.instances, [])

    def test_invalid_enable_value_does_not_create_a_proxy(self):
        with self.assertRaises(ValueError), native_shared_tool_cache(Options(), enabled="true", proxy_factory=FakeProxy):
            pass
        self.assertEqual(FakeProxy.instances, [])

    def test_alternate_providers_and_child_tls_overrides_are_inactive(self):
        for env in [
            {"CLAUDE_CODE_USE_BEDROCK": "1"},
            {"NODE_EXTRA_CA_CERTS": "/private/fixture.pem"},
            {"SSL_CERT_FILE": "/private/fixture.pem"},
        ]:
            options = Options(env=env)
            with native_shared_tool_cache(options, enabled=True, proxy_factory=FakeProxy) as prepared:
                self.assertIs(prepared.options, options)
                self.assertNotEqual(prepared.status, "active")
        self.assertEqual(FakeProxy.instances, [])

    def test_inactive_transport_retains_original_options(self):
        class Inactive(FakeProxy):
            def __enter__(self):
                self.active = False
                self.status = "unsupported_transport_settings"
                return self
        options = Options()
        with native_shared_tool_cache(options, enabled=True, proxy_factory=Inactive) as prepared:
            self.assertIs(prepared.options, options)
            self.assertEqual(prepared.status, "unsupported_transport_settings")


class SDKClientTests(IsolatedAsyncioTestCase):
    async def test_client_context_preserves_the_original_client_and_closes_proxy(self):
        events = []
        options = Options()

        @contextmanager
        def cache(original, *, enabled):
            self.assertIs(original, options)
            self.assertTrue(enabled)
            events.append("cache-enter")
            try:
                yield NativeCacheSession(original, "active")
            finally:
                events.append("cache-exit")

        class Client:
            def __init__(self, *, options):
                self.options = options
            async def __aenter__(self):
                events.append("client-enter")
                return self
            async def __aexit__(self, *_args):
                events.append("client-exit")

        async with CachedSDKClient(Client, options, enabled=True, cache_factory=cache) as client:
            self.assertIs(client.options, options)
            events.append("body")
        self.assertEqual(events, ["cache-enter", "client-enter", "body", "client-exit", "cache-exit"])

    async def test_failed_sdk_start_closes_the_proxy(self):
        events = []

        @contextmanager
        def cache(options, **_kwargs):
            try:
                yield NativeCacheSession(options, "active")
            finally:
                events.append("closed")

        class Client:
            def __init__(self, **_kwargs):
                pass
            async def __aenter__(self):
                raise RuntimeError("A synthetic startup failure.")

        with self.assertRaises(RuntimeError):
            async with CachedSDKClient(Client, Options(), enabled=True, cache_factory=cache):
                pass
        self.assertEqual(events, ["closed"])


if __name__ == "__main__":
    import unittest
    unittest.main()
