#!/usr/bin/env python3
"""Unit tests for effective runtime-backend extraction."""

import importlib.util
import os
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(SCRIPTS_DIR, "parse_runtime_truth.py")
SPEC = importlib.util.spec_from_file_location("parse_runtime_truth", MODULE_PATH)
prt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prt)


class TestRuntimeTruth(unittest.TestCase):
    def test_decoder_override_wins_over_encoder_attention_message(self):
        text = """
Using AttentionBackendEnum.ACCELERATOR_FA for MMEncoderAttention.
Found incompatible backend with AttentionType.DECODER. Overriding with NATIVE_ATTN out of potential backends.
"""
        result = prt.parse_runtime_truth(
            text, requested="ACCELERATOR_UNIFIED_ATTN", framework="framework"
        )
        attention = result["attention_backend"]

        self.assertEqual(attention["requested"], "ACCELERATOR_UNIFIED_ATTN")
        self.assertEqual(attention["observed"], "NATIVE_ATTN")
        self.assertEqual(attention["effective"], "NATIVE_ATTN")
        self.assertFalse(attention["match"])
        self.assertTrue(attention["verified"])
        self.assertEqual(attention["confidence"], "high")

    def test_matching_requested_and_observed_backend(self):
        text = (
            "AttentionType.DECODER selected attention backend: "
            "AttentionBackendEnum.NATIVE_ATTN"
        )
        attention = prt.parse_runtime_truth(
            text, requested="native_attn"
        )["attention_backend"]

        self.assertEqual(attention["observed"], "NATIVE_ATTN")
        self.assertTrue(attention["match"])
        self.assertTrue(attention["verified"])

    def test_common_using_backend_messages_capture_the_backend_not_a_suffix(self):
        for message in (
            "Using Triton Attention backend on V1 engine",
            "Using triton attention backend",
        ):
            with self.subTest(message=message):
                attention = prt.parse_runtime_truth(message)["attention_backend"]
                self.assertEqual(attention["observed"], "TRITON")
                self.assertTrue(attention["verified"])

    def test_default_backend_message_is_observed(self):
        attention = prt.parse_runtime_truth(
            "Attention backend not specified. Use aiter backend by default."
        )["attention_backend"]

        self.assertEqual(attention["observed"], "AITER")
        self.assertTrue(attention["verified"])

    def test_vision_attention_message_does_not_claim_decoder_backend(self):
        for message in (
            "Using AITER Flash Attention backend for ViT model.",
            "Attention backend is AITER for ViT model.",
            "Attention backend: AITER for encoder",
            "Using AITER backend for MMEncoderAttention.",
            "Attention backend is AITER for encoder_attention.",
            "Use AITER backend by default for VisionAttention.",
        ):
            with self.subTest(message=message):
                attention = prt.parse_runtime_truth(message)["attention_backend"]
                self.assertEqual(attention["observed"], "")
                self.assertFalse(attention["verified"])

    def test_missing_observation_is_unverified_not_a_match(self):
        attention = prt.parse_runtime_truth(
            "server started", requested="native_attn"
        )["attention_backend"]

        self.assertEqual(attention["observed"], "")
        self.assertEqual(attention["effective"], "NATIVE_ATTN")
        self.assertIsNone(attention["match"])
        self.assertFalse(attention["verified"])
        self.assertEqual(attention["confidence"], "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
