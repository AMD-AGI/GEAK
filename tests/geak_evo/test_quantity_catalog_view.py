"""Keep useful recent quantities visible as their source index grows."""
import json
import unittest

from geak_evo.engineer.assignment import QUANTITY_CATALOG_BYTES, quantity_catalog_text


class CatalogViewTest(unittest.TestCase):
    def test_small_catalog_is_unchanged(self):
        catalog = {"index_refs": [], "quantities": [], "unavailable": []}
        self.assertEqual(quantity_catalog_text(catalog), json.dumps(catalog, sort_keys=True))

    def test_large_reference_index_does_not_hide_all_outcomes(self):
        catalog = {
            "index_refs": [{"evidence_id": f"{i:064x}", "artifact": "native_quantity_index.json"}
                           for i in range(1000)],
            "quantities": [{"candidate_id": f"{i:064x}", "value_ms": i + 1} for i in range(20)],
            "unavailable": [{"source_id": "f" * 64, "reason": "timing unavailable"}],
            "free_text_notes_verified": False,
        }
        text = quantity_catalog_text(catalog)
        view = json.loads(text)
        self.assertLessEqual(len(text.encode()), QUANTITY_CATALOG_BYTES)
        for key in ("index_refs", "quantities", "unavailable"):
            self.assertTrue(view[key], key)
            self.assertEqual(view[key], catalog[key][-len(view[key]):])
            self.assertEqual(view["inline_bound"]["totals"][key], len(catalog[key]))
        self.assertFalse(view["free_text_notes_verified"])
