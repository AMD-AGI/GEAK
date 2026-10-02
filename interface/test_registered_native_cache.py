# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check explicit catalog binding and conservative native marker integration."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from interface.native_cost_controls.native_cache_layout import content_layout_eligible
from interface.native_cost_controls.native_prefix_marker import DECODER, MARKER, sha
from interface.native_cost_controls.registered_prefix import (
    RegisteredNativeToolCachePolicy,
    RegisteredToolPrefix,
)
from interface.native_cost_controls.sdk_cache import NativeRegisteredSessionCachePolicy
from interface.test_native_prefix_marker import (
    cache_block,
    encode,
    synthetic_body,
    synthetic_tool,
)


def registration(tools):
    raw = encode(tools)
    return RegisteredToolPrefix(raw, expected_sha256=sha(raw))


class SourceRegistrationTests(TestCase):
    def test_catalog_requires_bytes_and_an_exact_external_hash(self):
        raw = encode([synthetic_tool()])
        with self.assertRaises(TypeError):
            RegisteredToolPrefix(raw.decode(), expected_sha256=sha(raw))
        for digest in (None, "", "0" * 64, sha(raw).upper(), sha(raw) + "\n"):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                RegisteredToolPrefix(raw, expected_sha256=digest)
        with self.assertRaises(ValueError):
            RegisteredToolPrefix(raw + b"\n", expected_sha256=sha(raw))

    def test_registration_rejects_empty_ambiguous_or_ineligible_catalogs(self):
        for tools in (
            [], {}, None, [None], [synthetic_tool(), synthetic_tool()],
            [synthetic_tool("StructuredOutput")],
            [{**synthetic_tool(), "cache_control": {"type": "ephemeral"}}],
            [{"name": "namespace", "type": "namespace", "tools": []}],
        ):
            raw = encode(tools)
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                RegisteredToolPrefix(raw, expected_sha256=sha(raw))
        for raw in (
            b'[{"name":"a","name":"b","input_schema":{}}]',
            b'[{"name":"a","input_schema":{"enum":[NaN]}}]',
            b'[{"name":"a","input_schema":{"enum":[Infinity]}}]',
            b'[{"name":"a","input_schema":{"enum":[-Infinity]}}]',
            b"[", b"\xff", b"[] trailing",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                RegisteredToolPrefix(raw, expected_sha256=sha(raw))

    def test_file_changes_do_not_mutate_an_existing_registration(self):
        tools = [synthetic_tool()]
        raw = encode(tools)
        with TemporaryDirectory() as root:
            path = Path(root) / "synthetic_catalog.json"
            path.write_bytes(raw)
            prefix = RegisteredToolPrefix.from_file(path, expected_sha256=sha(raw))
            policy = RegisteredNativeToolCachePolicy(prefix, enabled=True)
            path.write_bytes(encode([synthetic_tool("changed")]))
            with self.assertRaises(ValueError):
                RegisteredToolPrefix.from_file(path, expected_sha256=sha(raw))
            self.assertTrue(policy.apply(encode(synthetic_body())).applied)
            with self.assertRaises(FrozenInstanceError):
                prefix.count = 100
            self.assertEqual(set(vars(prefix)), {"count", "prefix_sha256", "catalog_sha256"})
            self.assertEqual(prefix.count, 1)
            self.assertEqual(prefix.catalog_sha256, sha(raw))

    def test_raw_hash_binds_formatting_while_prefix_hash_ignores_object_order(self):
        compact = encode([synthetic_tool()])
        formatted = b'[ { "input_schema" : { "type" : "object" }, "name" : "functions.synthetic_read" } ]\n'
        first = RegisteredToolPrefix(compact, expected_sha256=sha(compact))
        second = RegisteredToolPrefix(formatted, expected_sha256=sha(formatted))
        self.assertNotEqual(first.catalog_sha256, second.catalog_sha256)
        self.assertEqual(first.prefix_sha256, second.prefix_sha256)
        self.assertTrue(RegisteredNativeToolCachePolicy(second, enabled=True).apply(encode(synthetic_body())).applied)

    def test_decimal_precision_and_scale_survive_catalog_registration(self):
        catalog = (
            b'[{"name":"synthetic","input_schema":{"enum":['
            b'1.23456789012345678901234567890123456789,1e-9999,1.2300,-0.0]}}]'
        )
        raw = b'{"tools":' + catalog + b',"messages":[{"role":"user","content":"Synthetic input."}]}'
        hashes = []
        for precision in (2, 28, 80):
            with self.subTest(precision=precision), localcontext() as context:
                context.prec = precision
                prefix = RegisteredToolPrefix(catalog, expected_sha256=sha(catalog))
                hashes.append(prefix.prefix_sha256)
                result = RegisteredNativeToolCachePolicy(prefix, enabled=True).apply(raw)
                self.assertTrue(result.applied)
                self.assertEqual(result.body.replace(MARKER, b"", 1), raw)
                changed = raw.replace(b"1.2300", b"1.23")
                self.assertEqual(RegisteredNativeToolCachePolicy(prefix, enabled=True).apply(changed).reason, "prefix_mismatch")
        self.assertEqual(len(set(hashes)), 1)


class RegisteredPolicyTests(TestCase):
    def setUp(self):
        self.prefix = registration([synthetic_tool()])
        self.policy = RegisteredNativeToolCachePolicy(self.prefix, enabled=True)

    def assert_unchanged(self, body, reason):
        raw = encode(body)
        result = self.policy.apply(raw)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, reason)
        self.assertIs(result.body, raw)

    def test_first_request_cannot_register_a_different_prefix(self):
        body = synthetic_body()
        body["tools"][0]["description"] = "An unregistered definition."
        self.assert_unchanged(body, "prefix_mismatch")
        self.assertTrue(self.policy.apply(encode(synthetic_body())).applied)
        self.assertIs(self.policy.prefix, self.prefix)

    def test_registered_prefix_accepts_no_tail_or_mutable_arbitrary_tails(self):
        for tail in (
            [], [synthetic_tool("StructuredOutput")],
            [{"name": "StructuredOutput", "input_schema": {"required": ["changed"]}}],
            [synthetic_tool("runtime_one"), {"type": "namespace", "name": "runtime_two", "tools": []}],
        ):
            with self.subTest(tail=tail):
                body = synthetic_body()
                body["tools"].extend(deepcopy(tail))
                raw = encode(body)
                result = self.policy.apply(raw)
                self.assertTrue(result.applied)
                self.assertEqual(result.body.replace(MARKER, b"", 1), raw)
                self.assertEqual(DECODER.decode(result.body.decode())["tools"][1:], tail)

    def test_eighteen_tool_prefix_accepts_twenty_three_and_twenty_four_tools(self):
        tools = [synthetic_tool(f"native_{index}") for index in range(18)]
        policy = RegisteredNativeToolCachePolicy(registration(tools), enabled=True)
        for total in (23, 24):
            with self.subTest(total=total):
                body = synthetic_body()
                tail = [synthetic_tool(f"runtime_{index}") for index in range(total - len(tools))]
                body["tools"] = tools + tail
                raw = encode(body)
                result = policy.apply(raw)
                self.assertTrue(result.applied)
                self.assertEqual(result.body.replace(MARKER, b"", 1), raw)
                marked = DECODER.decode(result.body.decode())["tools"]
                self.assertEqual([index for index, tool in enumerate(marked) if "cache_control" in tool], [17])
                self.assertEqual(marked[18:], tail)

    def test_schema_and_argument_keys_are_data_not_cache_controls(self):
        body = synthetic_body()
        body["tools"][0]["input_schema"] = {
            "type": "object", "properties": {"cache_control": {"type": "object"}},
            "default": {"cache_control": {"type": "data"}},
        }
        body["messages"][0]["content"] = [
            {"type": kind, "id": "synthetic", "name": "synthetic",
             "input": {"cache_control": {"type": "data"}}}
            for kind in ("tool_use", "server_tool_use", "mcp_tool_use")
        ]
        policy = RegisteredNativeToolCachePolicy(registration(body["tools"]), enabled=True)
        raw = encode(body)
        result = policy.apply(raw)
        self.assertTrue(result.applied)
        self.assertEqual(result.body.replace(MARKER, b"", 1), raw)

    def test_nested_markers_require_the_broader_content_guard(self):
        marked = cache_block()
        containers = [
            {"type": "search_result", "source": "synthetic", "title": "Synthetic", "content": [marked]},
            {"type": "document", "source": {"type": "content", "content": [marked]}},
            {"type": "mcp_tool_result", "tool_use_id": "synthetic", "content": [marked]},
            {"type": "web_fetch_tool_result", "content": {"type": "web_fetch_result", "content": {
                "type": "document", "source": {"type": "content", "content": [marked]},
            }}},
            {"type": "tool_search_tool_result", "content": {"type": "tool_search_tool_search_result", "tool_references": [
                {"type": "tool_reference", "tool_name": "synthetic", "cache_control": {"type": "ephemeral"}},
            ]}},
        ]
        for container in containers:
            for location in ("message", "system", "nested_tool_result"):
                with self.subTest(kind=container["type"], location=location):
                    body = synthetic_body()
                    if location == "system":
                        body["system"] = [container]
                    elif location == "nested_tool_result":
                        body["messages"][0]["content"] = [{"type": "tool_result", "content": [container]}]
                    else:
                        body["messages"][0]["content"] = [container]
                    self.assert_unchanged(body, "unqualified_cache_layout")

    def test_supported_nested_content_without_markers_remains_eligible(self):
        body = synthetic_body()
        body["messages"][0]["content"] = [{"type": "web_fetch_tool_result", "content": {
            "type": "web_fetch_result", "content": {
                "type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "Synthetic."}]},
            },
        }}]
        raw = encode(body)
        result = self.policy.apply(raw)
        self.assertTrue(result.applied)
        self.assertEqual(result.body.replace(MARKER, b"", 1), raw)

    def test_compaction_and_unsupported_content_preserve_bytes(self):
        for content in (
            [{"type": "compaction", "content": "Synthetic summary."}],
            [{"type": "future_content", "text": "Synthetic."}],
            [{"type": "tool_result", "content": [{"type": "compaction", "content": "Synthetic summary."}]}],
            [{"type": "document", "source": {"type": "content", "content": None}}],
            [{"type": "web_fetch_tool_result", "content": []}],
            [{"type": "tool_search_tool_result", "content": {"type": "unknown"}}],
            [None], None,
        ):
            with self.subTest(content=content):
                body = synthetic_body()
                body["messages"][0]["content"] = content
                self.assert_unchanged(body, "unqualified_cache_layout")

    def test_native_search_error_results_remain_eligible_without_hidden_markers(self):
        body = synthetic_body()
        body["messages"][0]["content"] = [
            {"type": "web_fetch_tool_result", "content": {
                "type": "web_fetch_tool_result_error", "error_code": "unavailable",
            }},
            {"type": "tool_search_tool_result", "content": {
                "type": "tool_search_tool_result_error", "error_code": "unavailable",
            }},
        ]
        raw = encode(body)
        result = self.policy.apply(raw)
        self.assertTrue(result.applied)
        self.assertEqual(result.body.replace(MARKER, b"", 1), raw)

    def test_malformed_native_search_results_and_messages_retain_bytes(self):
        for content in (
            [{"type": "web_fetch_tool_result", "content": {"type": "future_result"}}],
            [{"type": "tool_search_tool_result", "content": None}],
        ):
            with self.subTest(content=content):
                body = synthetic_body()
                body["messages"][0]["content"] = content
                self.assert_unchanged(body, "unqualified_cache_layout")
        body = synthetic_body()
        body["messages"] = [42]
        self.assert_unchanged(body, "unqualified_cache_layout")
        body["messages"] = "Unsupported message container."
        self.assertFalse(content_layout_eligible(encode(body)))

    def test_existing_marker_rules_and_output_format_controls_remain_active(self):
        body = synthetic_body()
        body["system"] = [cache_block() for _ in range(3)]
        self.assertTrue(self.policy.apply(encode(body)).applied)
        body["system"].append(cache_block())
        self.assert_unchanged(body, "marker_limit")
        body = synthetic_body()
        body["tools"].append({**synthetic_tool("runtime"), "cache_control": {"type": "ephemeral"}})
        self.assert_unchanged(body, "existing_tool_marker")
        body = synthetic_body()
        body["output_config"] = {"format": {"type": "json_schema"}}
        self.assert_unchanged(body, "unqualified_output_format")

    def test_invalid_json_reasons_match_the_pure_marker_before_layout_guard(self):
        for raw in (b"null", b"[]", b'"synthetic"'):
            result = self.policy.apply(raw)
            self.assertEqual(result.reason, "unsupported_request")
            self.assertIs(result.body, raw)
        for raw in (b"{", b'{"tools":[],"tools":[]}', b'{"value":NaN}', b"\xff"):
            result = self.policy.apply(raw)
            self.assertEqual(result.reason, "unsupported_json")
            self.assertIs(result.body, raw)

    def test_disabled_and_unsupported_endpoints_preserve_original_bytes(self):
        raw = b"This synthetic body is not JSON."
        policy = RegisteredNativeToolCachePolicy(self.prefix)
        self.assertIs(policy.apply(raw).body, raw)
        with self.assertRaises(TypeError):
            policy.apply(raw.decode())
        for method, path in (("GET", "/v1/messages"), ("POST", None), ("POST", "/v1/messages/count_tokens")):
            result = self.policy.apply(raw, method=method, path=path)
            self.assertEqual(result.reason, "unsupported_endpoint")
            self.assertIs(result.body, raw)
        with self.assertRaises(TypeError):
            RegisteredNativeToolCachePolicy([synthetic_tool()], enabled=True)
        with self.assertRaises(ValueError):
            RegisteredNativeToolCachePolicy(self.prefix, enabled="true")


class RegisteredSessionInputTests(TestCase):
    def setUp(self):
        self.prefix = registration([synthetic_tool()])

    def test_invalid_settings_cannot_enable_session_caching(self):
        with self.assertRaises(ValueError):
            NativeRegisteredSessionCachePolicy("session", self.prefix, enabled="true")
        for session in (None, "", 42):
            with self.subTest(session=session), self.assertRaises(ValueError):
                NativeRegisteredSessionCachePolicy(session, self.prefix, enabled=True)

    def test_disabled_or_unsupported_requests_retain_bytes_before_metadata_parsing(self):
        raw = b"This synthetic body is not JSON."
        disabled = NativeRegisteredSessionCachePolicy(None, self.prefix)
        self.assertEqual(disabled.apply(raw).reason, "disabled")
        self.assertIs(disabled.apply(raw).body, raw)
        with self.assertRaises(TypeError):
            disabled.apply(raw.decode())
        active = NativeRegisteredSessionCachePolicy("session", self.prefix, enabled=True)
        for method, path in (("GET", "/v1/messages"), ("POST", None), ("POST", "/v1/messages/count_tokens")):
            with self.subTest(method=method, path=path):
                result = active.apply(raw, method=method, path=path)
                self.assertEqual(result.reason, "unsupported_endpoint")
                self.assertIs(result.body, raw)

    def test_invalid_metadata_cannot_bypass_the_session_gate(self):
        active = NativeRegisteredSessionCachePolicy("session", self.prefix, enabled=True)
        for raw in (b"[]", b'{"metadata":null}', b'{"metadata":{"user_id":"[]"}}',
                b'{"metadata":{"user_id":"{\\"session_id\\":\\"session\\",\\"session_id\\":\\"other\\"}"}}'):
            with self.subTest(raw=raw):
                result = active.apply(raw)
                self.assertEqual(result.reason, "invalid_json")
                self.assertIs(result.body, raw)


if __name__ == "__main__":
    import unittest
    unittest.main()
