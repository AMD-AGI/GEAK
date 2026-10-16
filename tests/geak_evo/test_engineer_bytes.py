"""Saved engineer encodings remain compatible with exact content identities."""

import json
import math
import unittest

import test_engineer_port as base


class EngineerBytesTests(unittest.TestCase):
    def test_unicode_finite_records_keep_original_utf8_bytes(self):
        value = {"question": "Measure α and 漢字", "budget": {"wall_seconds": 0.125},
                 "usage": None, "label": "μs"}
        expected = json.dumps(value, sort_keys=True, ensure_ascii=False,
                              allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.assertEqual(base.engineer_module._json(value), expected)
        self.assertIn("漢字".encode("utf-8"), expected)
        for invalid in (math.inf, -math.inf, math.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                base.engineer_module._json({"cost": invalid})

    def test_reference_validation_and_runner_share_valueerror_type(self):
        self.assertTrue(issubclass(base.EngineerError, ValueError))
        with self.assertRaises(base.EngineerError):
            base.engineer_module._read_ref({"path": "relative", "sha256": "bad"})


if __name__ == "__main__":
    unittest.main()
