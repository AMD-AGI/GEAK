# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the portable cache policy with synthetic request bytes and no transport."""

import json
import unittest
from copy import deepcopy

from interface.native_cost_controls.shared_tool_cache import SharedToolCachePolicy

SHARED_TOOLS = [
    {"name": "Read", "description": "Read a local file.", "input_schema": {"type": "object"}},
    {"name": "Write", "description": "Write a local file.", "input_schema": {"type": "object"}},
]
MARKER = {"type": "ephemeral", "ttl": "5m"}
INSERTION = b',"cache_control":{"type":"ephemeral","ttl":"5m"}'


def request():
    return {
        "model": "test-anthropic-compatible-model",
        "max_tokens": 64000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "tools": deepcopy(SHARED_TOOLS) + [{
            "name": "StructuredOutput",
            "description": "Return the requested object.",
            "input_schema": {"type": "object", "properties": {"result": {"type": "string"}}},
        }],
        "system": [
            {"type": "text", "text": "A synthetic system instruction."},
            {"type": "text", "text": "A second instruction.", "cache_control": deepcopy(MARKER)},
            {"type": "text", "text": "A third instruction.", "cache_control": deepcopy(MARKER)},
        ],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "A synthetic task.", "cache_control": deepcopy(MARKER)}
        ]}],
        "stream": True,
    }


class SharedToolCacheTests(unittest.TestCase):
    def setUp(self):
        self.policy = SharedToolCachePolicy(SHARED_TOOLS, enabled=True)

    def unchanged(self, value, reason=None, **kwargs):
        raw = value if isinstance(value, bytes) else json.dumps(value, indent=2).encode()
        result = self.policy.apply(raw, **kwargs)
        self.assertIs(result.body, raw)
        self.assertFalse(result.applied)
        if reason:
            self.assertEqual(result.reason, reason)
        return result

    def test_default_is_disabled_and_does_not_parse(self):
        raw = b'  not JSON\x00\xff'
        result = SharedToolCachePolicy().apply(raw)
        self.assertIs(result.body, raw)
        self.assertEqual(result.reason, "disabled")
        self.assertFalse(result.applied)

    def test_native_system_notification_remains_unchanged(self):
        body = request()
        notification = {"role": "system", "content": [
            {"type": "text", "text": "A synthetic workflow status notification."}
        ]}
        body["messages"].insert(0, notification)
        raw = json.dumps(body, indent=2).encode()
        result = self.policy.apply(raw)
        self.assertTrue(result.applied)
        self.assertEqual(result.body.replace(INSERTION, b"", 1), raw)
        self.assertEqual(json.loads(result.body)["messages"][0], notification)

    def test_registered_policy_also_defaults_to_disabled(self):
        raw = json.dumps(request()).encode()
        result = SharedToolCachePolicy(SHARED_TOOLS).apply(raw)
        self.assertIs(result.body, raw)
        self.assertFalse(result.applied)

    def test_disabled_policy_does_not_validate_unused_definitions(self):
        raw = b"An unchanged disabled request."
        for definitions in [None, [], {}, [None], object()]:
            result = SharedToolCachePolicy(definitions).apply(raw)
            self.assertIs(result.body, raw)
            self.assertFalse(result.applied)

    def test_only_one_marker_is_inserted_and_original_bytes_restore(self):
        original = request()
        for indent, ensure_ascii in [(None, True), (2, True), (4, False)]:
            with self.subTest(indent=indent, ensure_ascii=ensure_ascii):
                original["tools"][0]["description"] = "Read café, emoji 🌱, quotes \" and braces } ]."
                shared = deepcopy(original["tools"][:-1])
                policy = SharedToolCachePolicy(shared, enabled=True)
                raw = (" \n" + json.dumps(original, indent=indent, ensure_ascii=ensure_ascii) + "\r\n").encode()
                result = policy.apply(raw, path="/v1/messages?beta=true")
                self.assertTrue(result.applied)
                self.assertEqual(result.body.replace(INSERTION, b"", 1), raw)
                transformed = json.loads(result.body)
                self.assertEqual(transformed["tools"][1].pop("cache_control"), MARKER)
                self.assertEqual(transformed, original)
                self.assertNotIn("cache_control", original["tools"][1])

    def test_shared_definition_registration_does_not_follow_mutation(self):
        registered = deepcopy(SHARED_TOOLS)
        policy = SharedToolCachePolicy(registered, enabled=True)
        registered[0]["description"] = "Changed after registration."
        self.assertTrue(policy.apply(json.dumps(request()).encode()).applied)
        body = request()
        body["tools"][0]["description"] = registered[0]["description"]
        result = policy.apply(json.dumps(body).encode())
        self.assertFalse(result.applied)

    def test_definition_key_order_does_not_change_the_registered_prefix(self):
        body = request()
        body["tools"][0] = dict(reversed(list(body["tools"][0].items())))
        self.assertTrue(self.policy.apply(json.dumps(body).encode()).applied)

    def test_later_application_preserves_the_already_marked_bytes(self):
        first = self.policy.apply(json.dumps(request()).encode())
        self.assertTrue(first.applied)
        second = self.policy.apply(first.body)
        self.assertIs(second.body, first.body)
        self.assertFalse(second.applied)

    def test_other_endpoints_and_methods_remain_exact(self):
        for method, path in [("HEAD", "/api/hello"), ("POST", "/v1/messages/count_tokens"),
                             ("GET", "/v1/messages"), ("POST", "/v1/messages/batches"), ("POST", None)]:
            with self.subTest(method=method, path=path):
                self.unchanged(request(), "unsupported_endpoint", method=method, path=path)

    def test_invalid_json_and_nonfinite_numbers_remain_exact(self):
        for raw in [b"", b"{", b"[] trailing", b"\xff", b'{"duplicate":1,"duplicate":2}',
                    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":"\\ud800"}']:
            with self.subTest(raw=raw):
                self.unchanged(raw, "invalid_json")

    def test_unsupported_top_level_shapes_remain_exact(self):
        for value in [None, [], "text", 1, {}, {"tools": []}]:
            with self.subTest(value=value):
                self.unchanged(json.dumps(value).encode(), "unsupported_request")

    def test_missing_or_malformed_native_sections_remain_exact(self):
        for key, value in [("model", ""), ("model", 1), ("system", "text"), ("tools", None),
                           ("messages", []), ("messages", "text"), ("messages", [None]),
                           ("messages", [{"role": "future", "content": []}])]:
            body = request()
            body[key] = value
            self.unchanged(body, "unsupported_request")

    def test_changed_shared_tool_content_and_order_remain_exact(self):
        for change in [lambda tools: tools.reverse(), lambda tools: tools[0].update(description="Changed."),
                       lambda tools: tools[0]["input_schema"].update(additionalProperties=False),
                       lambda tools: tools.append(deepcopy(tools[-1])), lambda tools: tools.pop(0)]:
            body = request()
            change(body["tools"])
            self.unchanged(body, "shared_tools_mismatch")

    def test_only_the_supported_structured_output_suffix_is_eligible(self):
        for suffix in [None, {}, {"name": "OtherTool", "input_schema": {"type": "object"}},
                       {"name": "StructuredOutput", "input_schema": {"type": "string"}},
                       {"name": "StructuredOutput", "input_schema": {"type": "object"}, "unknown": True},
                       {"name": "StructuredOutput", "input_schema": {"type": "object"}, "strict": "true"}]:
            body = request()
            body["tools"][-1] = suffix
            self.unchanged(body, "unsupported_output_tool")
        body = request()
        body["tools"][-1]["strict"] = True
        self.assertTrue(self.policy.apply(json.dumps(body).encode()).applied)

    def test_four_existing_markers_keep_the_original_bytes(self):
        body = request()
        body["system"][0]["cache_control"] = deepcopy(MARKER)
        self.unchanged(body, "marker_limit")

    def test_request_level_cache_control_keeps_the_original_bytes(self):
        body = request()
        body["cache_control"] = deepcopy(MARKER)
        self.unchanged(body, "marker_limit")

    def test_marker_lifetime_and_location_must_match(self):
        for mutation in [lambda body: body["system"][1]["cache_control"].update(ttl="1h"),
                         lambda body: body["system"][1].pop("cache_control"),
                         lambda body: body["system"][1]["cache_control"].update(scope="future"),
                         lambda body: body["system"][1].update(cache_control=None),
                         lambda body: body["system"][1].update(type="future"),
                         lambda body: body["messages"][0]["content"].append({"type": "text", "text": "Tail."}),
                         lambda body: body["messages"][0].update(content="text"),
                         lambda body: body["messages"][0].update(content=[]),
                         lambda body: body["messages"][0]["content"][0].update(type="thinking")]:
            body = request()
            mutation(body)
            self.unchanged(body, "unsupported_cache_layout")

    def test_default_five_minute_markers_and_tool_results_are_supported(self):
        body = request()
        for block in [body["system"][1], body["system"][2], body["messages"][0]["content"][0]]:
            block["cache_control"].pop("ttl")
        body["messages"][0]["content"][0] = {"type": "tool_result", "tool_use_id": "synthetic-tool",
            "content": "A synthetic result.", "cache_control": {"type": "ephemeral"}}
        self.assertTrue(self.policy.apply(json.dumps(body).encode()).applied)

    def test_nested_tool_result_markers_count_toward_the_limit(self):
        body = request()
        body["messages"].insert(0, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "synthetic-tool",
            "content": [{"type": "text", "text": "A nested result.", "cache_control": deepcopy(MARKER)}]}]})
        self.unchanged(body, "marker_limit")

    def test_schema_property_named_cache_control_is_not_a_marker(self):
        shared = deepcopy(SHARED_TOOLS)
        shared[0]["input_schema"]["properties"] = {"cache_control": {"type": "string"}}
        body = request()
        body["tools"][:-1] = shared
        self.assertTrue(SharedToolCachePolicy(shared, enabled=True).apply(json.dumps(body).encode()).applied)

    def test_tool_array_can_appear_first_or_last(self):
        body = request()
        for rearranged in [{"tools": body["tools"], **body}, {**{k: v for k, v in body.items() if k != "tools"}, "tools": body["tools"]}]:
            raw = json.dumps(rearranged, indent=2).encode()
            result = self.policy.apply(raw)
            self.assertTrue(result.applied)
            self.assertEqual(result.body.replace(INSERTION, b"", 1), raw)

    def test_one_shared_tool_is_supported(self):
        body = request()
        body["tools"].pop(1)
        result = SharedToolCachePolicy(SHARED_TOOLS[:1], enabled=True).apply(json.dumps(body).encode())
        self.assertTrue(result.applied)
        self.assertEqual(json.loads(result.body)["tools"][0]["cache_control"], MARKER)

    def test_invalid_registration_cannot_enable_a_policy(self):
        for definitions in [None, [], {}, [None], [{"name": "A"}], [SHARED_TOOLS[0], SHARED_TOOLS[0]],
                            [{"name": "StructuredOutput", "input_schema": {}}],
                            [{**SHARED_TOOLS[0], "cache_control": MARKER}],
                            [{**SHARED_TOOLS[0], "extra": float("nan")}]]:
            with self.subTest(definitions=definitions), self.assertRaises(ValueError):
                SharedToolCachePolicy(definitions, enabled=True)
        with self.assertRaises(ValueError):
            SharedToolCachePolicy(SHARED_TOOLS, enabled="false")
        with self.assertRaises(TypeError):
            self.policy.apply("JSON text is not request bytes.")


if __name__ == "__main__":
    unittest.main()
