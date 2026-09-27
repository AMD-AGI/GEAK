# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the mandatory stopping SDK wrapper without a provider or GPU."""

import os
import tempfile
import unittest
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from interface.native_cost_controls.native_helpers import HelperMirror
from interface.native_cost_controls.quality_stop_controller import StopRejected
from interface.native_cost_controls.quality_stop_native import CheckpointTransport
from interface.native_cost_controls.sdk_quality_stop import QualityStopSDKClient
from interface.test_quality_stop_native import NativeFixture


@dataclass
class Options:
    env: dict = field(default_factory=dict)
    session_id: str = "stop-session"
    hooks: dict = field(default_factory=dict)
    session_store: object = None
    session_store_flush: str = "batched"
    cwd: str = None
    setting_sources: list = field(default_factory=list)
    settings: str = '{"enableWorkflows":true,"ultracode":true}'
    plugins: list = None
    can_use_tool: object = None
    system_prompt: object = None
    allowed_tools: list = field(default_factory=lambda: ["Workflow", "Bash"])
    extra_args: dict = field(default_factory=lambda: {"effort": "high"})
    model: str = "fixture-model"
    effort: str = "high"
    permission_mode: str = "default"
    resume: str = None
    continue_conversation: bool = False
    fork_session: bool = False
    resume_session_at: str = None
    session_import: str = None


@dataclass
class Matcher:
    matcher: str = None
    hooks: list = field(default_factory=list)


class Client:
    def __init__(self, *, options):
        self.options = options
        self.closed = False
        self.messages = [SimpleNamespace(value="message-1"), SimpleNamespace(value="message-2")]
        self.responses = [SimpleNamespace(value="response-1")]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True
        return False

    async def query(self, prompt):
        return "queried: " + prompt

    async def receive_messages(self):
        for message in self.messages:
            yield message

    async def receive_response(self):
        for message in self.responses:
            yield message


class Proxy:
    """Keep the real cache context, but create no listening socket."""

    active = True
    status = "active"
    base_url = "http://127.0.0.1:49000"

    def __init__(self, endpoint, policy, *, enabled, transport):
        self.endpoint, self.policy = endpoint, policy
        self.enabled, self.transport = enabled, transport
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True
        self.transport.close()


class QualityStopSDKTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = NativeFixture(self.temporary.name, populate=False)
        self.addCleanup(self.fixture.registry.close)
        self.options = Options(cwd=str(self.fixture.root), env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:49001/api"})
        self.proxies, self.clients = [], []
        self.environment = patch.dict(os.environ, {"HOME": os.environ.get("HOME", tempfile.gettempdir())}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def proxy_factory(self, *arguments, **keywords):
        proxy = Proxy(*arguments, **keywords)
        self.proxies.append(proxy)
        return proxy

    def client_factory(self, **keywords):
        client = Client(**keywords)
        self.clients.append(client)
        return client

    def wrapper(self, **changes):
        arguments = {"client_factory": self.client_factory, "options": self.options,
            "workflow_request": self.fixture.request, "source_root": self.fixture.source,
            "controller": self.fixture.controller, "hook_matcher_factory": Matcher,
            "proxy_factory": self.proxy_factory}
        arguments.update(changes)
        return QualityStopSDKClient(**arguments)

    async def test_exact_lifetime_installs_mandatory_route_and_keeps_controls_off(self):
        original = deepcopy(self.options)
        wrapper = self.wrapper()
        async with wrapper as entered:
            self.assertIs(entered, wrapper)
            self.assertEqual(await entered.query("fixture task"), "queried: fixture task")
            self.assertIs(self.fixture.controller.closure.__self__, wrapper.registry)
            prepared = self.clients[0].options
            self.assertEqual(prepared.env["ANTHROPIC_BASE_URL"], Proxy.base_url)
            self.assertEqual(prepared.session_id, self.options.session_id)
            self.assertEqual(prepared.session_store_flush, "eager")
            self.assertIsInstance(prepared.session_store, HelperMirror)
            self.assertIs(prepared.session_store.registry, wrapper.registry)
            for event in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
                self.assertEqual(len(prepared.hooks[event]), 1)
                self.assertEqual(prepared.hooks[event][0].matcher, ".*")
                self.assertIs(prepared.hooks[event][0].hooks[0].__self__, wrapper.registry)
            proxy = self.proxies[0]
            self.assertIsInstance(proxy.transport, CheckpointTransport)
            self.assertEqual(proxy.transport.base_path, "/api")
            raw = b'{ "scientific": "unchanged" }'
            policy = proxy.policy.apply(raw)
            self.assertIs(policy.body, raw)
            self.assertFalse(policy.applied)
            self.assertEqual(policy.reason, "disabled")
            self.assertFalse(wrapper.registry.closed)
            self.assertFalse(proxy.closed)
            self.assertEqual((prepared.model, prepared.effort, prepared.permission_mode, prepared.allowed_tools),
                (self.options.model, self.options.effort, self.options.permission_mode, self.options.allowed_tools))
        self.assertTrue(wrapper.registry.closed)
        self.assertTrue(self.proxies[0].closed)
        self.assertTrue(self.clients[0].closed)
        self.assertEqual(self.options, original)

    async def test_both_message_streams_observe_every_message_and_preserve_identity(self):
        async with self.wrapper() as wrapper:
            observer = Mock(wraps=wrapper.registry.observe)
            with patch.object(wrapper.registry, "observe", observer):
                messages = [message async for message in wrapper.receive_messages()]
                responses = [message async for message in wrapper.receive_response()]
            expected = self.clients[0].messages + self.clients[0].responses
            self.assertEqual(messages + responses, expected)
            self.assertEqual([call.args[0] for call in observer.call_args_list], expected)

    async def test_existing_non_tool_hook_and_eager_store_remain_available(self):
        calls = []
        class Store:
            async def append(self, key, entries):
                calls.append((key, entries))

            async def load(self, key):
                return key
        store = Store()
        hook = object()
        options = replace(self.options, session_store=store, session_store_flush="eager", hooks={"SessionStart": [hook]})
        async with self.wrapper(options=options):
            prepared = self.clients[0].options
            self.assertIs(prepared.hooks["SessionStart"][0], hook)
            self.assertIs(prepared.session_store.original, store)
            key = {"session_id": "other"}
            await prepared.session_store.append(key, [])
            self.assertEqual(await prepared.session_store.load(key), key)
        self.assertEqual(calls, [(key, [])])

    async def test_missing_session_gets_one_bound_identity(self):
        options = replace(self.options, session_id=None)
        with patch("interface.native_cost_controls.sdk_quality_stop.uuid.uuid4", return_value="generated-session"):
            async with self.wrapper(options=options) as wrapper:
                self.assertEqual(wrapper.registry.session_id, "generated-session")
                self.assertEqual(self.clients[0].options.session_id, "generated-session")

    async def test_default_sdk_matcher_import_is_used(self):
        with patch.dict("sys.modules", {"claude_agent_sdk": SimpleNamespace(HookMatcher=Matcher)}):
            async with self.wrapper(hook_matcher_factory=None):
                self.assertIsInstance(self.clients[0].options.hooks["PreToolUse"][0], Matcher)

    def test_unentered_wrapper_does_not_expose_client_attributes(self):
        wrapper = self.wrapper()
        with self.assertRaises(AttributeError):
            _ = wrapper.query
        self.assertEqual(self.clients, [])
        self.assertEqual(self.proxies, [])

    async def test_unsupported_options_fail_before_proxy_or_client_creation(self):
        variants = [SimpleNamespace(), replace(self.options, resume="old-session"),
            replace(self.options, continue_conversation=True), replace(self.options, fork_session=True),
            replace(self.options, resume_session_at="old-turn"), replace(self.options, session_import="old-session"),
            replace(self.options, can_use_tool=lambda *_: True), replace(self.options, system_prompt="custom"),
            replace(self.options, hooks={"PreToolUse": [Matcher()]}),
            replace(self.options, hooks={"PostToolUse": [Matcher()]}),
            replace(self.options, hooks={"PostToolUseFailure": [Matcher()]}),
            replace(self.options, session_store=object()), replace(self.options, settings="invalid JSON"),
            replace(self.options, settings='{"hooks":{"PreToolUse":[]}}'),
            replace(self.options, setting_sources=["project"]), replace(self.options, plugins=["plugin"]),
            replace(self.options, extra_args={"dangerous-override": True})]
        for options in variants:
            with self.subTest(options=options), self.assertRaisesRegex(StopRejected, "unsupported_stopping_sdk_options"):
                async with self.wrapper(options=options):
                    self.fail("The unsupported wrapper started.")
        self.assertEqual(self.clients, [])
        self.assertEqual(self.proxies, [])

    async def test_changed_root_request_fails_before_proxy_or_client_creation(self):
        request = deepcopy(self.fixture.request)
        request["args"]["quality_stop"]["trial_id"] = "changed-trial"
        with self.assertRaisesRegex(StopRejected, "stopping_root_changed"):
            async with self.wrapper(workflow_request=request):
                self.fail("The changed root request started.")
        self.assertEqual(self.clients, [])
        self.assertEqual(self.proxies, [])

    async def test_transport_rejection_never_creates_provider_client(self):
        for environment in ({"CLAUDE_CODE_USE_BEDROCK": "1"}, {"HTTP_PROXY": "http://proxy.invalid"}):
            with self.subTest(environment=environment):
                wrapper = self.wrapper(options=replace(self.options, env=environment))
                with self.assertRaisesRegex(StopRejected, "stopping_transport_unavailable"):
                    async with wrapper:
                        self.fail("The unsupported transport started.")
                self.assertTrue(wrapper.registry.closed)
        self.assertEqual(self.clients, [])
        self.assertEqual(self.proxies, [])

    async def test_inactive_proxy_rejects_and_closes_without_fallback(self):
        def inactive(*arguments, **keywords):
            proxy = self.proxy_factory(*arguments, **keywords)
            proxy.active = False
            proxy.status = "fixture-proxy-unavailable"
            return proxy
        wrapper = self.wrapper(proxy_factory=inactive)
        with self.assertRaisesRegex(StopRejected, "stopping_transport_unavailable"):
            async with wrapper:
                self.fail("The inactive transport started.")
        self.assertEqual(self.clients, [])
        self.assertTrue(self.proxies[0].closed)
        self.assertTrue(wrapper.registry.closed)

    async def test_client_constructor_failure_closes_transport_and_registry(self):
        def failed_client(**_keywords):
            raise RuntimeError("fixture client constructor failure")
        wrapper = self.wrapper(client_factory=failed_client)
        with self.assertRaisesRegex(RuntimeError, "fixture client constructor failure"):
            await wrapper.__aenter__()
        self.assertTrue(wrapper.registry.closed)
        self.assertTrue(self.proxies[0].closed)

    async def test_client_entry_failure_closes_transport_and_registry(self):
        async def failed_entry(_client):
            raise RuntimeError("fixture client entry failure")
        wrapper = self.wrapper()
        with patch.object(Client, "__aenter__", failed_entry), self.assertRaisesRegex(RuntimeError, "fixture client entry failure"):
            await wrapper.__aenter__()
        self.assertTrue(wrapper.registry.closed)
        self.assertTrue(self.proxies[0].closed)

    async def test_proxy_factory_failure_closes_registry(self):
        def failed_proxy(*_arguments, **_keywords):
            raise RuntimeError("fixture proxy failure")
        wrapper = self.wrapper(proxy_factory=failed_proxy)
        with self.assertRaisesRegex(RuntimeError, "fixture proxy failure"):
            await wrapper.__aenter__()
        self.assertTrue(wrapper.registry.closed)
        self.assertEqual(self.clients, [])

    async def test_client_exit_failure_still_closes_transport_and_registry(self):
        async def failed_exit(_client, *_arguments):
            raise RuntimeError("fixture client exit failure")
        wrapper = self.wrapper()
        with patch.object(Client, "__aexit__", failed_exit), self.assertRaisesRegex(RuntimeError, "fixture client exit failure"):
            async with wrapper:
                pass
        self.assertTrue(wrapper.registry.closed)
        self.assertTrue(self.proxies[0].closed)

    async def test_cache_exit_failure_still_closes_registry(self):
        def failed_exit(_proxy, *_arguments):
            raise RuntimeError("fixture proxy exit failure")
        wrapper = self.wrapper()
        with patch.object(Proxy, "__exit__", failed_exit), self.assertRaisesRegex(RuntimeError, "fixture proxy exit failure"):
            async with wrapper:
                pass
        self.assertTrue(wrapper.registry.closed)
        self.assertTrue(self.clients[0].closed)


if __name__ == "__main__":
    unittest.main()
