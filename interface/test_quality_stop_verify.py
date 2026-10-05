# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check VM-compatible signature verification against independent primitives."""

import base64
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

from interface.native_cost_controls.quality_stop_controller import RSASigner

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kernel_workflow" / "quality_stop_verify.js"


def run_node(script):
    result = subprocess.run(["node"], input=SOURCE.read_text() + "\n" + script, text=True,
                            capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


class SignatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signer = RSASigner()
        cls.payload = {"protocol": "geak-fixed-floor-stop-v2", "task": "test-task", "certified": True,
                       "qualifying": False, "issued_at": 1234,
                       "native_binding": {key: key for key in ("agent_id", "root_tool", "root_task", "session_id", "run_id")}}
        cls.envelope = cls.signer.sign(cls.payload)
        cls.config = {"public_key": cls.signer.public_jwk}

    def verify(self, envelope, config=None, task="test-task"):
        fixture = json.dumps({"envelope": envelope, "config": config or self.config, "task": task})
        return run_node("const value=" + fixture + "; console.log(JSON.stringify(qualityVerify(value.envelope,value.config,value.task)));\n")

    def test_sha256_fips_vectors_and_padding_boundaries(self):
        texts = ["", "abc", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                 "a" * 55, "a" * 56, "a" * 63, "a" * 64, "a" * 65, "a" * 1000000,
                 "\x00\x7f\n" * 200]
        hashes = run_node("console.log(JSON.stringify(" + json.dumps(texts) + ".map(qualitySha256Ascii)));\n")
        self.assertEqual(hashes, [hashlib.sha256(text.encode("ascii")).hexdigest() for text in texts])

    def test_sha256_rejects_ambiguous_encoding(self):
        values = [None, 1, {}, "é", "\ud800", "a" * 1048577]
        script = "const values=" + json.dumps(values) + "; console.log(JSON.stringify(values.map(value=>{try{qualitySha256Ascii(value);return false;}catch(_){return true;}})));"
        self.assertEqual(run_node(script), [True] * len(values))

    def test_signature_from_standard_cryptography_library(self):
        self.assertEqual(self.verify(self.envelope), self.payload)

    def test_signature_also_passes_node_standard_crypto(self):
        data = json.dumps({"envelope": self.envelope, "key": self.signer.public_jwk})
        script = "const crypto=require('crypto'); const v=" + data + "; console.log(JSON.stringify(crypto.verify('sha256',Buffer.from(v.envelope.payload,'ascii'),{key:v.key,format:'jwk'},Buffer.from(v.envelope.signature,'base64'))));"
        self.assertIs(run_node(script), True)

    def test_wrong_task_wrong_key_and_modified_payload(self):
        self.assertIsNone(self.verify(self.envelope, task="another-task"))
        self.assertIsNone(self.verify(self.envelope, config={"public_key": RSASigner().public_jwk}))
        for field in ("payload", "signature"):
            changed = dict(self.envelope)
            changed[field] = changed[field][:-1] + ("A" if changed[field][-1] != "A" else "B")
            with self.subTest(field=field):
                self.assertIsNone(self.verify(changed))

    def test_model_boolean_and_extra_envelope_fields(self):
        for changed in (None, {}, {"certified": True}, {**self.envelope, "certified": True}):
            with self.subTest(value_type=type(changed).__name__):
                self.assertIsNone(self.verify(changed))

    def test_malformed_base64_and_rsa_representatives(self):
        variants = [None, "", "AA=", "A===", "!" * 344, self.envelope["signature"].rstrip("="),
                    base64.b64encode(bytes(255)).decode(), base64.b64encode(bytes(256)).decode(),
                    base64.b64encode(bytes([255]) * 256).decode(), "AB==", "AA==\n"]
        for signature in variants:
            with self.subTest(signature_length=len(signature) if isinstance(signature, str) else None):
                self.assertIsNone(self.verify({**self.envelope, "signature": signature}))

    def test_public_key_constraints(self):
        for field, value in [("kty", "EC"), ("alg", "none"), ("e", "Aw"), ("n", "AA"),
                             ("n", "A"), ("n", "!"), ("n", None), ("n", "A" * 342)]:
            key = {**self.signer.public_jwk, field: value}
            with self.subTest(field=field, value_type=type(value).__name__):
                self.assertIsNone(self.verify(self.envelope, config={"public_key": key}))
        self.assertIsNone(self.verify(self.envelope, config={"public_key": None}))

    def test_valid_signatures_with_invalid_payload_shapes(self):
        for field, value in [("protocol", "other"), ("task", "another-task"), ("certified", 1),
                             ("qualifying", "true"), ("issued_at", "1234"), ("native_binding", None),
                             ("native_binding", {"agent_id": "agent"})]:
            payload = {**self.payload, field: value}
            with self.subTest(field=field):
                self.assertIsNone(self.verify(self.signer.sign(payload)))

    def test_inline_verifier_matches_reviewed_source(self):
        source = SOURCE.read_text()
        block = source[source.index("// BEGIN QUALITY STOP VERIFIER"):].rstrip()
        lane = (ROOT / "kernel_workflow" / "kernel_lane.js").read_text()
        self.assertEqual(lane.count(block), 1)


if __name__ == "__main__":
    unittest.main()
