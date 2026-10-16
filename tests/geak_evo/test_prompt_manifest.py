"""Current prompt identities stay separate from historical import provenance."""

import hashlib
import json
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[2] / "geak_evo"


class PromptManifestTests(unittest.TestCase):
    def test_current_version_names_every_packaged_prompt_with_exact_bytes(self):
        manifest = json.loads((PACKAGE / "PROMPT_SOURCES.json").read_text())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertTrue(manifest["prompt_version"])
        rows = manifest["files"]
        targets = [row["target"] for row in rows]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(set(targets), {
            str(path.relative_to(PACKAGE)) for path in (PACKAGE / "prompts").glob("*.md")
        })
        for row in rows:
            with self.subTest(prompt=row["target"]):
                body = (PACKAGE / row["target"]).read_bytes()
                self.assertEqual(hashlib.sha256(body).hexdigest(), row["current_sha256"])
                self.assertEqual(len(body), row["current_bytes"])
                self.assertEqual(set(row["source_ref"]), {"path", "sha256"})
                self.assertRegex(row["source_ref"]["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
