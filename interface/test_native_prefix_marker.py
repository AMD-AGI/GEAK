"""Test the pure native marker policy with synthetic request bytes."""

import json
import unittest
from copy import deepcopy
from decimal import Decimal, localcontext

from interface.native_cost_controls.native_prefix_marker import (
    DECODER,
    MARKER,
    Decision,
    mark_registered_prefix,
    markers,
    sha,
    wire,
)


def synthetic_tool(name="functions.synthetic_read"):
    return {"name": name, "input_schema": {"type": "object"}}


def synthetic_body():
    return {
        "tools": [synthetic_tool()],
        "messages": [{"role": "user", "content": "Synthetic input."}],
    }


def encode(body):
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def registered(raw, count=1, **kwargs):
    body = DECODER.decode(raw.decode("utf-8"))
    return mark_registered_prefix(
        raw,
        count=count,
        prefix_sha256=sha(wire(body["tools"][:count])),
        enabled=True,
        **kwargs,
    )


def cache_block(marker=None):
    if marker is None:
        marker = {"type": "ephemeral", "ttl": "5m"}
    return {"type": "text", "text": "Synthetic text.", "cache_control": marker}


class NativePrefixMarkerTests(unittest.TestCase):
    def assert_rejected(self, raw, reason, *, count=1, prefix_sha256=None, **kwargs):
        if prefix_sha256 is None:
            prefix_sha256 = sha(wire([synthetic_tool()]))
        result = mark_registered_prefix(
            raw,
            count=count,
            prefix_sha256=prefix_sha256,
            enabled=True,
            **kwargs,
        )
        self.assertEqual(result, Decision(raw, False, reason))
        self.assertIs(result.body, raw)

    def test_exact_marker_bytes_preserve_utf8_whitespace_and_suffix(self):
        raw = (
            '{\n  "system": "Synthetic π",\n'
            '  "tools" : [\n'
            '    { "input_schema": {"type":"object"}, "name":"functions.alpha" },\n'
            '    { "name":"functions.beta", "description":"Synthetic café } text",'
            ' "input_schema":{"type":"object"}   },\n'
            '    {"name":"StructuredOutput","input_schema":{"type":"object"}}\n'
            '  ],\n  "messages" : [{"role":"user","content":"Synthetic input."}]\n}\n'
        ).encode()
        self.assertEqual(MARKER, b',"cache_control":{"type":"ephemeral","ttl":"5m"}')
        target = b'"input_schema":{"type":"object"}   }'
        expected = raw.replace(target, target[:-1] + MARKER + b"}", 1)
        result = registered(raw, count=2)
        self.assertEqual(result, Decision(expected, True, "marked"))
        offset = result.body.index(MARKER)
        self.assertEqual(result.body[:offset] + result.body[offset + len(MARKER) :], raw)
        self.assertEqual(
            markers(DECODER.decode(result.body.decode("utf-8"))),
            [(["tools", 1], {"type": "ephemeral", "ttl": "5m"})],
        )

    def test_namespace_names_require_no_special_case(self):
        for name in (
            "functions.synthetic_read",
            "mcp__synthetic__read",
            "synthetic.namespace.read",
            "synthetic_read",
            "synthetic.π",
            'synthetic."quoted"',
        ):
            with self.subTest(name=name):
                body = synthetic_body()
                body["tools"][0]["name"] = name
                result = registered(encode(body))
                self.assertTrue(result.applied)
                marked = DECODER.decode(result.body.decode("utf-8"))
                self.assertEqual(marked["tools"][0]["name"], name)

    def test_registered_prefix_with_structured_output_or_no_suffix(self):
        prefix = [synthetic_tool("functions.alpha"), synthetic_tool("functions.beta")]
        for suffix in ([], [synthetic_tool("StructuredOutput")]):
            with self.subTest(suffix=bool(suffix)):
                body = synthetic_body()
                body["tools"] = deepcopy(prefix + suffix)
                result = registered(encode(body), count=2)
                self.assertTrue(result.applied)
                marked = DECODER.decode(result.body.decode("utf-8"))
                self.assertNotIn("cache_control", marked["tools"][0])
                self.assertEqual(marked["tools"][2:], suffix)
                self.assertEqual(
                    marked["tools"][1]["cache_control"], {"type": "ephemeral", "ttl": "5m"}
                )

    def test_only_registered_prefix_requires_client_tool_shape(self):
        body = synthetic_body()
        suffix = [
            {"type": "namespace", "name": "synthetic_namespace", "tools": []},
            synthetic_tool(),
        ]
        body["tools"].extend(suffix)
        result = registered(encode(body))
        self.assertTrue(result.applied)
        self.assertEqual(DECODER.decode(result.body.decode("utf-8"))["tools"][1:], suffix)
        body["tools"] = [suffix[0]]
        self.assert_rejected(encode(body), "unsupported_prefix")

    def test_registration_ignores_object_key_order_and_whitespace(self):
        raw = (
            b'{ "messages" : [ { "content" : "Synthetic input.", "role" : "user" } ],'
            b' "tools" : [ { "input_schema" : { "type" : "object" },'
            b' "name" : "functions.synthetic_read" } ] }'
        )
        result = mark_registered_prefix(
            raw, count=1, prefix_sha256=sha(wire([synthetic_tool()])), enabled=True
        )
        self.assertTrue(result.applied)
        offset = result.body.index(MARKER)
        self.assertEqual(result.body[:offset] + result.body[offset + len(MARKER) :], raw)

    def test_three_existing_markers_allow_one_more(self):
        body = synthetic_body()
        body["system"] = [cache_block()]
        body["messages"][0]["content"] = [
            cache_block({"type": "ephemeral"}),
            {"type": "tool_result", "content": [cache_block()]},
        ]
        result = registered(encode(body))
        self.assertTrue(result.applied)
        self.assertEqual(len(markers(DECODER.decode(result.body.decode("utf-8")))), 4)

    def test_four_or_more_existing_markers_reject(self):
        for count in (4, 5):
            with self.subTest(count=count):
                body = synthetic_body()
                body["messages"][0]["content"] = [cache_block() for _ in range(count)]
                self.assert_rejected(encode(body), "marker_limit")

    def test_marker_limit_precedes_existing_tool_marker(self):
        body = synthetic_body()
        body["tools"].append(
            {**synthetic_tool("StructuredOutput"), "cache_control": {"type": "ephemeral"}}
        )
        body["system"] = [cache_block() for _ in range(3)]
        self.assert_rejected(encode(body), "marker_limit")

    def test_existing_suffix_tool_marker_rejects(self):
        body = synthetic_body()
        body["tools"].append(
            {**synthetic_tool("StructuredOutput"), "cache_control": {"type": "ephemeral"}}
        )
        self.assert_rejected(encode(body), "existing_tool_marker")

    def test_top_level_cache_marker_rejects(self):
        body = synthetic_body()
        body["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
        self.assert_rejected(encode(body), "unqualified_cache_layout")

    def test_unsupported_marker_layouts_reject(self):
        for marker in (
            None,
            "ephemeral",
            {},
            {"type": "other"},
            {"type": "ephemeral", "ttl": "1h"},
            {"type": "ephemeral", "ttl": "10m"},
            {"type": "ephemeral", "ttl": None},
            {"type": "ephemeral", "extra": True},
        ):
            for location in ("system", "message", "nested_tool_result"):
                with self.subTest(marker=marker, location=location):
                    body = synthetic_body()
                    block = cache_block()
                    block["cache_control"] = marker
                    if location == "system":
                        body["system"] = [block]
                    elif location == "message":
                        body["messages"][0]["content"] = [block]
                    else:
                        body["messages"][0]["content"] = [
                            {
                                "type": "tool_result",
                                "content": [{"type": "tool_result", "content": [block]}],
                            }
                        ]
                    self.assert_rejected(encode(body), "unqualified_cache_layout")

    def test_output_format_controls_reject(self):
        for controls in (
            {"output_format": None},
            {"output_format": {"type": "json_schema"}},
            {"output_config": {"format": None}},
            {"output_config": {"format": {"type": "json_schema"}}},
        ):
            with self.subTest(controls=controls):
                body = synthetic_body()
                body.update(controls)
                self.assert_rejected(encode(body), "unqualified_output_format")

    def test_missing_or_invalid_messages_reject(self):
        for messages in (None, [], {}, "Synthetic input."):
            with self.subTest(messages=messages):
                body = synthetic_body()
                body["messages"] = messages
                self.assert_rejected(encode(body), "unsupported_request")
        body = synthetic_body()
        del body["messages"]
        self.assert_rejected(encode(body), "unsupported_request")

    def test_non_object_request_rejects(self):
        for raw in (b"null", b"[]", b'"synthetic"', b"1", b"true"):
            with self.subTest(raw=raw):
                self.assert_rejected(raw, "unsupported_request")

    def test_missing_or_short_prefix_rejects(self):
        for tools in (None, {}, [], "synthetic"):
            with self.subTest(tools=tools):
                body = synthetic_body()
                body["tools"] = tools
                self.assert_rejected(encode(body), "prefix_missing")
        body = synthetic_body()
        self.assert_rejected(encode(body), "prefix_missing", count=2)
        del body["tools"]
        self.assert_rejected(encode(body), "prefix_missing")

    def test_unsupported_prefix_rejects(self):
        for tool in (
            None,
            {},
            {"name": "", "input_schema": {}},
            {"name": 1, "input_schema": {}},
            {"name": "synthetic"},
            {"name": "synthetic", "input_schema": []},
            {"name": "synthetic", "input_schema": {}, "description": None},
        ):
            with self.subTest(tool=tool):
                body = synthetic_body()
                body["tools"] = [tool]
                self.assert_rejected(encode(body), "unsupported_prefix")

    def test_duplicate_prefix_names_reject(self):
        body = synthetic_body()
        body["tools"] *= 2
        self.assert_rejected(encode(body), "duplicate_tool_names", count=2)

    def test_wrong_prefix_hash_or_order_rejects(self):
        body = synthetic_body()
        self.assert_rejected(encode(body), "prefix_mismatch", prefix_sha256="0" * 64)
        prefix = [synthetic_tool("functions.alpha"), synthetic_tool("functions.beta")]
        body["tools"] = list(reversed(prefix))
        self.assert_rejected(
            encode(body), "prefix_mismatch", count=2, prefix_sha256=sha(wire(prefix))
        )

    def test_repeated_application_preserves_the_marked_bytes(self):
        raw = encode(synthetic_body())
        prefix_sha256 = sha(wire([synthetic_tool()]))
        first = mark_registered_prefix(raw, count=1, prefix_sha256=prefix_sha256, enabled=True)
        self.assertTrue(first.applied)
        self.assert_rejected(first.body, "prefix_mismatch", prefix_sha256=prefix_sha256)
        self.assertEqual(first.body.count(MARKER), 1)
        self.assertEqual(registered(first.body), Decision(first.body, False, "existing_tool_marker"))

    def test_wire_preserves_decimal_precision_scale_and_exponents(self):
        value = DECODER.decode(
            '{"z":-0.0,"p":1.23456789012345678901234567890123456789,'
            '"e":1e-40,"scale":1.2300,"unicode":"π"}'
        )
        expected = (
            '{"e":1E-40,"p":1.23456789012345678901234567890123456789,'
            '"scale":1.2300,"unicode":"π","z":-0.0}'
        ).encode()
        self.assertIsInstance(value["p"], Decimal)
        self.assertEqual(wire(value), expected)
        self.assertNotEqual(wire(Decimal("1.2300")), wire(Decimal("1.23")))

    def test_marker_preserves_original_decimal_spelling(self):
        raw = (
            b'{"tools":[{"name":"synthetic","input_schema":{"type":"object",'
            b'"minimum":1.23456789012345678901234567890123456789,"multipleOf":1e-40}}],'
            b'"messages":[{"role":"user","content":"Synthetic input."}]}'
        )
        result = registered(raw)
        expected = raw.replace(b"1e-40}}]", b"1e-40}" + MARKER + b"}]", 1)
        self.assertEqual(result, Decision(expected, True, "marked"))

    def test_decimal_context_does_not_change_registration(self):
        raw = (
            b'{"tools":[{"name":"synthetic","input_schema":{"enum":['
            b'9007199254740993,1e-9999,1.0,1.00,-0.0,'
            b'1.23456789012345678901234567890123456789]}}],'
            b'"messages":[{"role":"user","content":"Synthetic input."}]}'
        )
        expected_wire = (
            b'[{"input_schema":{"enum":[9007199254740993,1E-9999,1.0,1.00,-0.0,'
            b'1.23456789012345678901234567890123456789]},"name":"synthetic"}]'
        )
        expected = raw.replace(b"]}}]", b"]}" + MARKER + b"}]", 1)
        for precision in (2, 28, 80):
            with self.subTest(precision=precision), localcontext() as context:
                context.prec = precision
                body = DECODER.decode(raw.decode("utf-8"))
                self.assertEqual(wire(body["tools"]), expected_wire)
                result = mark_registered_prefix(
                    raw, count=1, prefix_sha256=sha(expected_wire), enabled=True
                )
                self.assertEqual(result, Decision(expected, True, "marked"))

    def test_duplicate_keys_nonfinite_values_and_malformed_json_reject(self):
        for raw in (
            b'{"tools":[],"tools":[]}',
            b'{"tools":[{"name":"a","name":"b","input_schema":{}}]}',
            b'{"value":NaN}',
            b'{"value":Infinity}',
            b'{"value":-Infinity}',
            b"{",
            b"{} trailing",
            b"\xff",
            b"\xef\xbb\xbf{}",
        ):
            with self.subTest(raw=raw):
                self.assert_rejected(raw, "unsupported_json")

    def test_wire_rejects_nonfinite_numbers(self):
        for value in (
            Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), float("nan"), float("inf")
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                wire(value)

    def test_disabled_policy_preserves_bytes_before_registration_checks(self):
        raw = b"Synthetic bytes that are not JSON."
        result = mark_registered_prefix(raw, count=None, prefix_sha256=None)
        self.assertEqual(result, Decision(raw, False, "disabled"))
        self.assertIs(result.body, raw)

    def test_raw_input_must_be_bytes_even_when_disabled(self):
        for raw in ("{}", bytearray(b"{}"), None):
            with self.subTest(raw=raw), self.assertRaisesRegex(TypeError, "The request must contain bytes."):
                mark_registered_prefix(raw, count=1, prefix_sha256="0" * 64)

    def test_invalid_registrations_raise(self):
        for count in (None, True, False, 0, -1, 1.0, "1"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "The prefix registration is invalid."):
                mark_registered_prefix(b"{}", count=count, prefix_sha256="0" * 64, enabled=True)
        for digest in (
            None, b"0" * 64, "", "0" * 63, "0" * 65, "A" * 64, "g" * 64, "0" * 64 + "\n"
        ):
            with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "The prefix registration is invalid."):
                mark_registered_prefix(b"{}", count=1, prefix_sha256=digest, enabled=True)

    def test_only_post_messages_endpoint_applies(self):
        raw = encode(synthetic_body())
        for method, path in (
            ("GET", "/v1/messages"),
            ("post", "/v1/messages"),
            ("POST", "/v1/messages/"),
            ("POST", "/v1/other"),
        ):
            with self.subTest(method=method, path=path):
                self.assert_rejected(raw, "unsupported_endpoint", method=method, path=path)
        self.assertTrue(registered(raw, path="/v1/messages?synthetic=true").applied)


if __name__ == "__main__":
    unittest.main()
