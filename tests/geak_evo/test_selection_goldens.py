"""Golden parity for the final-program-selection modules.

The digests below were produced by running the same synthetic fixtures
(test_selection_fixtures.py) on the pre-port modules (recommendation_validation,
member_recommendation, structured_response, committed_frontier,
native_recommendation, population_selector). Each output is normalized first:
the scratch root becomes <ROOT>; for the frontier, choices and population
suites every 64-hex string is renamed by first appearance; for population the
fields the port deliberately changed (created_utc, implementation_ref,
metric_source_ref) are dropped.

The extract-16 digest retains the Python 3.9 diagnostic for a deeply nested
array. Python 3.12 can decode that depth, then rejects the same invalid root
type. That fixture checks rejection below without fixing the JSON decoder's
interpreter-dependent recursion limit.

The byte tests run the frontier and population scenarios with the same fake
clock on the original and on the port, in turn at the same path. Every frontier
file (journal, receipts, seals, retained objects) must be byte-identical; every
population seal must be byte-identical once the three refs naming a different
file (implementation_ref, metric_source_ref, contract sha256) are replaced by
placeholders.
It needs GEAK_EVO_ORIGINAL_SRC=<original evolution/src> (its sibling
``bridge`` directory is added too) and is skipped without it.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_selection_goldens.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_selection_fixtures as fx  # noqa: E402

GOLDEN = {
    ('choices', 'authenticate'): "1423f82a3ee8951e7cbe67d9270cc1516c998d711c49a2a744a896eca767bf22",
    ('choices', 'authenticate-other-index'): "ff365f9686ec72084339c7598c02a2c2bcdc892544f5535b8854a209ac5fde2a",
    ('choices', 'authenticate-tampered'): "f030c3e3d539b9797daba903c21b6ca4d0fbef2de84b37d13e7f930d0d27727c",
    ('choices', 'checkpoint'): "58ae25a318692f4d375823f98baa356085afadc04416cedc55d95211ce19496e",
    ('choices', 'checkpoint-again'): "58ae25a318692f4d375823f98baa356085afadc04416cedc55d95211ce19496e",
    ('choices', 'eligible'): "968d8fafe3c63272e3c0c8c4f3650067f5a99869090f466702ae725e2e4d941a",
    ('choices', 'fatal-partition'): "4cea600c1b188d69035dc999fee87c3c61df525f328ea2d27c0457e0e70c486a",
    ('choices', 'mixed-eligible'): "58f48384ca9ffe707502e45f288b135fc7c46c6a1427a62f453df756d4833674",
    ('choices', 'mixed-partition'): "6e46f56ac56e38404887dd543e5986c72328c64ad07cea1d1dd14ff1b6e98a92",
    ('choices', 'reauthenticate'): "6ea1e100046ed6e5b15d86462bd50865be61b3575d2b20310961e5d7af432725",
    ('choices', 'references'): "8b6cbb4f3791428728a6acb9df255b7cfc2feb1ddc036419a7bf52f7637bb711",
    ('choices', 'stale-snapshot'): "278e4e7872b43a32c973367600b25555e576cd1ec4be269ebb0d061311293168",
    ('commit_record', 'commit-in'): "2c439824d255483cc16b9eb9601546eb2fe13aa0da89972f0d7053fe654413d1",
    ('commit_record', 'commit-late'): "b32d6e831133e34325d008e2cfd3daccdc2739e564ec151feef8f803560de660",
    ('commit_record', 'commit-out'): "1e129859b5ad4a5b4769cbab90d7f03e83e09067fb41204987225d6245ae3a7e",
    ('commit_record', 'commit-wrong-head'): "f44b981b9b7ccb1034ead0e18a5cfa5ac98f93a9e51b3a9c047f8d8b1bf32582",
    ('commit_record', 'make-record-float-time'): "f52f2fd6bb358fa352b697bd5815de6c665eb876120835afb25efb114c2efe4d",
    ('commit_record', 'make-record-outside'): "70cd43f3118e7db0e97e6efcef7c5d36ab199a738024e51287dc2a26f4aa63cd",
    ('commit_record', 'record'): "cf2e142eeb590bde7f12e4d46e1ff8be88b3adcd1797b3b48da12b24832c8121",
    ('commit_record', 'state'): "2d071d9b7f1fbf18d6f5cacae8fd30a26f8c9805b54a08724e08d63c7bd4e16b",
    ('final', 'final-bad-binding'): "1a3d86396a0f5ed78d7e50a92e60d6156080f24e010ecb15266c867049ff2d22",
    ('final', 'final-choice-refused'): "622b3e082c8714e8f519a24a01d42aac84eba13fa85e3a027f704ff5f57cfc46",
    ('final', 'final-clear_only'): "872072b6d23fd60b2d9068145a9fc11ed547801e57cc122b251570497f21c604",
    ('final', 'final-dropped_events'): "4d1a85af55dc803211747908a93d83b0581b268f906d5657ad093cf594a8faff",
    ('final', 'final-empty'): "a8aa624d820a0fe9e980afe8f4c092dd64efa5d2c57037ee5ddf19a51937ad94",
    ('final', 'final-missing-file'): "85bd8509098ec8a52e8384d5a4b07ec0ae439ea10ede27ff9170d265846781d6",
    ('final', 'final-other-clock'): "ae916160dd7e75acbe21c4d946529bde6f3cb52603ba0ce996d43d3dbd788d2b",
    ('final', 'final-prefix-revision-3'): "0c5de8a5f2102e4c292d956ac498cb162a485ff7f62cc0ae00345cb7899ae511",
    ('final', 'final-revived_head'): "9947eaa18950516419af71cbe1efae1cd73756500daa6a790e514d286a3e9843",
    ('final', 'final-set'): "0c5de8a5f2102e4c292d956ac498cb162a485ff7f62cc0ae00345cb7899ae511",
    ('final', 'final-set_clear'): "86b1702ea8fb457cc1447540159836e3d6231d5d2b3d5639ed4cdba3fde74785",
    ('final', 'final-set_keep'): "768c2b3192b2ed42aebf7bfc90856b9975e545fa27bb3fff52514b2dc3ed6279",
    ('final', 'final-set_rejected'): "0c5de8a5f2102e4c292d956ac498cb162a485ff7f62cc0ae00345cb7899ae511",
    ('final', 'final-set_set'): "e2efeba724d39132c7c99f672ea05142455f2d7d861888c5082d3eba5d1f9140",
    ('final', 'final-stale-snapshot'): "69ff91801c7b1d1e687ae99b55b6d03fc827891028a61c839a389c11e4d66558",
    ('final', 'final-wrong-campaign'): "03aaa515fadfd2d8854214edae81f5753d43cbfc3b1f8b11b194e7c302c39911",
    ('final', 'view-clear_only'): "e65bb49eebc6fd4072ee9444c6f774e1aba837d26b157a79cdb4f95f7d384e58",
    ('final', 'view-dropped_events'): "bc94dcefb36ba6fc094c825f9e70f5d5c2b57e00dcbce8f9711dbeeb6eac58e1",
    ('final', 'view-empty'): "bc94dcefb36ba6fc094c825f9e70f5d5c2b57e00dcbce8f9711dbeeb6eac58e1",
    ('final', 'view-revived_head'): "2a27a68c7e3796013ee046dd7eedc32b8d9e9e4e79268356f16effc2cf59250b",
    ('final', 'view-set'): "3431d09509ff4e6c226a6fd93ce8478c6e9e8d520396289da425d1ed001b727e",
    ('final', 'view-set_clear'): "bf650cf10fc43c87287e75acf4656cd9a656816c6514b4995dcdf3c75aef5e4a",
    ('final', 'view-set_keep'): "81e1b9eceb07f7f6a4a7d10af0649fd1ae21b3d6d0f6f6c7598b27fbaecf9e9b",
    ('final', 'view-set_rejected'): "27d87585b3e2d460c88300839ccb9e18b5f8c15e7cb9bc100fd85ed0b4537528",
    ('final', 'view-set_set'): "9cad5fd445692e614858bd43c5a6b29e31e430e478db9ca692602e125377d50a",
    ('frontier', 'cutoffs'): "57f91bba02c5af09110de968db7827da51562037490764566eda18e955f2ba8f",
    ('frontier', 'cutoffs-bad-wall'): "162ddc02ab10586fc36d116867e5cd91ec511a52492b5819f10acf1e453dd2b6",
    ('frontier', 'cutoffs-duplicate'): "3807d456b953770795a3e2c7c4f57ec6328979d554dcd94bebc90b253f7380d3",
    ('frontier', 'cutoffs-float-wall'): "9065b3cc771faf672efe9989172544cdb4b36d5a9ca13688147f1507bb7e58cd",
    ('frontier', 'cutoffs-unknown'): "3807d456b953770795a3e2c7c4f57ec6328979d554dcd94bebc90b253f7380d3",
    ('frontier', 'empty-recheck-h4'): "6e20596fbe539bab6a4e09773636845fca3e1d4be481dc639d86ab6fab5c0f73",
    ('frontier', 'empty-seal-h4'): "6b989efe4bd10add03a1d66747a3b85874b32aba7a117c5e8b8eed45f44ef382",
    ('frontier', 'main-files'): "1b84c39cf469b9a99dcd822205ccda9bc180eb2b3ea6eed528e3d474fa755b1f",
    ('frontier', 'main-recheck-h4'): "fe09875856b98381e162b160c5a954a1aa13c1e03ee017e842bd6fcf9dd3c815",
    ('frontier', 'main-reseal-h4'): "77507637c8fcb33831a1510875bc7ab7a1a6ef57a048023c4633766aa8396ef1",
    ('frontier', 'main-seal-early'): "8e1832d0170215264a390ec3c3db2782c4f51d02e93b5b07391cc926daee0149",
    ('frontier', 'main-seal-h4'): "a16d6b0aca516ca711d519a2c140313a5c068bb35f4fdc4b2e447082fdaa27d7",
    ('frontier', 'main-seal-h8'): "bc9dd09d3f9d4b986b6e65229fd0ab5c4f04ec522a01265b39eaea81b339c814",
    ('frontier', 'refused-seal-h4'): "33fc4e99c343b7c7b33b80c9119b317cf1227fb5bdc66039aae0de4b22533742",
    ('frontier', 'register-late'): "f2245758c29c860c6c3c79ca899c0ac89956711accb67852a6e88ea370602857",
    ('frontier', 'straddle-seal-h4'): "02e8e496d99dd8f635ebb6f1cef20afc51de04069e40d7bc6029ed1d8046a74f",
    ('population', 'all-missing'): "fa30d9032a1307d9d32369bd6b4961245717949bcf012ca0e00e34b294845ed5",
    ('population', 'callback-error'): "a911d443b74d9ce10f0f49e9ec03f7ced010b37c28ae727d6717da9050e74120",
    ('population', 'duplicates'): "c54f41944b29076de19c4f64ec90ef348ef2a9720db7543cf26cd9eae5815c77",
    ('population', 'one-member'): "56ea4fc20d71f5525f4b984eedf1ddc7a2956a692f4d1a44749af6f03d9021a1",
    ('population', 'one-unresolved'): "ea1cb70a73fa3ae65f0b03ccc893343303023b085fa51627539c12d968698ece",
    ('population', 'selected'): "369f543298e9690aca178af40aaba685a2ef03d99c378d3ed148a1a6250c2922",
    ('population', 'tie'): "99420f7832433a2cb2932829211bf820e52fc9025ddb1b28792618a052ba7f26",
    ('population', 'trial-subset'): "6d8833285cbe570560ecd67b6bedf641b3802de70c26407fc5a1761f621fe0a6",
    ('recommendation', 'decision-00'): "c6a4108de41acea09bf552aaab87cf162e7947329a86ca90ee9bb22b4d52a433",
    ('recommendation', 'decision-01'): "2c7c25d2530c66dd2aef3bc0d342c9227085d6b55d6415cff5ad2338c3474162",
    ('recommendation', 'decision-02'): "99face1921de729c9cb580d911350f758811e764eb247641b30e2cd035481a33",
    ('recommendation', 'decision-03'): "2c7c25d2530c66dd2aef3bc0d342c9227085d6b55d6415cff5ad2338c3474162",
    ('recommendation', 'decision-04'): "2c7c25d2530c66dd2aef3bc0d342c9227085d6b55d6415cff5ad2338c3474162",
    ('recommendation', 'decision-05'): "a53077a34ce7ad0f41a68373f72c0feed037d089d111b9b448fb8ba4946c358d",
    ('recommendation', 'decision-06'): "d1d118d5a0e619d8f7f2c2059ba3adc61d9d3a204e4d68a80b2a77e1dc5bb330",
    ('recommendation', 'decision-07'): "2c7c25d2530c66dd2aef3bc0d342c9227085d6b55d6415cff5ad2338c3474162",
    ('recommendation', 'decision-08'): "2c7c25d2530c66dd2aef3bc0d342c9227085d6b55d6415cff5ad2338c3474162",
    ('recommendation', 'decision-09'): "2c7c25d2530c66dd2aef3bc0d342c9227085d6b55d6415cff5ad2338c3474162",
    ('recommendation', 'decision-10'): "2c7c25d2530c66dd2aef3bc0d342c9227085d6b55d6415cff5ad2338c3474162",
    ('recommendation', 'decision-11'): "d1d118d5a0e619d8f7f2c2059ba3adc61d9d3a204e4d68a80b2a77e1dc5bb330",
    ('recommendation', 'decision-duplicate'): "c7806d262bd11bcbc77252f3143fb98c7a514eeaddf25efe918f0d531ff57944",
    ('recommendation', 'extract-00'): "1a0e68af459a7906c92babb444fb71c04d29e8968e4d53728e30b114266d033d",
    ('recommendation', 'extract-01'): "acf0fe2f245c482b1163401565fa8555a20abedcc5ad01ccf9f1685ca0f29e07",
    ('recommendation', 'extract-02'): "0582cb38b248baae18de7248fca7566cf9459a90e350e5eb558ac82c3faab329",
    ('recommendation', 'extract-03'): "42d9eeac86c36d27aeea243282c357e6090d3f34b75c85093acd8de3133df36c",
    ('recommendation', 'extract-04'): "3b83255b3f62ccc5bcce104308b963b6d85540fbed15d3fcdffe41fb1cb687a4",
    ('recommendation', 'extract-05'): "34b8f0941c049e495df6d7dd882ecca016143a9c2d46cbb4352e8831dab3d467",
    ('recommendation', 'extract-06'): "320ea50faed37869b411b4f6073096752f25fef66a245fbd18cb756e2bf18838",
    ('recommendation', 'extract-07'): "7fbd8764a3ea2f5dfd54c3ccc08cd7849f12c4bdb5ea2039b5306ab245074874",
    ('recommendation', 'extract-08'): "76dacc4c20d796dea14698aa3c0c3ff1ba2361ea7a63a7f96bb54b15f0badd18",
    ('recommendation', 'extract-09'): "c6ed30d6900addfbd73a8ea67112e249bb5e66344b529a87b0223fb8f0c4d88d",
    ('recommendation', 'extract-10'): "3b87a26748eee5c0d1943f4544715e0f658c68f73191f1017425140942106098",
    ('recommendation', 'extract-11'): "af318ec2be45e8a63d021d4a148c9c8086e87f6c878bc98a54b1bd86eb2293f1",
    ('recommendation', 'extract-12'): "36686372ba9e5da5cf429b2690bb8fec10d14b071bd6fd489162920770d0e485",
    ('recommendation', 'extract-13'): "cc1cf8633ca3d85375a18ab05817903523cd2d527502220a69c9b515008c4570",
    ('recommendation', 'extract-14'): "140692dbd3932668d3b983e911619ab801ded8956c68abbb74143df081dab7a4",
    ('recommendation', 'extract-15'): "320ea50faed37869b411b4f6073096752f25fef66a245fbd18cb756e2bf18838",
    ('recommendation', 'extract-16'): "bc64e45724160b8bed0a607fdb902f8cda8b8ea20d77afe6d4894e96ea6a27a3",
    ('recommendation', 'extract-17'): "c74d81aab32f2cb87137e8230b70117288073a1717f7151c45827d8a141cbbb6",
    ('recommendation', 'extract-text'): "333c694e68b96326434425ea6dadc4d7a04f33fe473c74cd6e8bc4dc326daee6",
    ('recommendation', 'split-0'): "9123c2bf2f8d1c305152a661cf564243398d6d347c994273b200a7f5a2f91cc8",
    ('recommendation', 'split-1'): "e14b964de44f0c9a34c2c96290d3f175904790c977be2a3958481e654583be70",
    ('recommendation', 'split-2'): "386952f9088d544454c0e6e21c7a3872b596420ce7a5ff46e7adafeb6fd37cef",
    ('recommendation', 'split-3'): "65f91e8936b9d5bb284304c3ce1e5abbf471388c39bb9206bc223f12778414ca",
    ('recommendation', 'split-4'): "65f91e8936b9d5bb284304c3ce1e5abbf471388c39bb9206bc223f12778414ca",
}


class SelectionGoldenTest(unittest.TestCase):
    def check_suite(self, suite):
        outputs = fx.suite_outputs(fx.port_api(), suite)
        expected = {name: value for (name_suite, name), value in GOLDEN.items() if name_suite == suite}
        self.assertEqual(sorted(outputs), sorted(expected))
        for name in sorted(outputs):
            with self.subTest(name=name):
                if (suite, name) == ("recommendation", "extract-16"):
                    self.assertIn(outputs[name], [
                        {"error": "ValueError", "message": "structured response nesting is too deep"},
                        {"error": "ValueError", "message": "structured response must be an object or whole-response null"},
                    ])
                else:
                    self.assertEqual(fx.golden_digest(outputs[name]), expected[name])

    def test_recommendation(self):
        self.check_suite("recommendation")

    def test_commit_record(self):
        self.check_suite("commit_record")

    def test_final(self):
        self.check_suite("final")

    def test_frontier(self):
        self.check_suite("frontier")

    def test_choices(self):
        self.check_suite("choices")

    def test_population(self):
        self.check_suite("population")


_FILES = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import test_selection_fixtures as fx
api = fx.original_api() if sys.argv[3] == "original" else fx.port_api()
print(json.dumps(getattr(fx, sys.argv[4])(api, Path(sys.argv[2]))))
"""


@unittest.skipUnless(os.environ.get("GEAK_EVO_ORIGINAL_SRC"), "GEAK_EVO_ORIGINAL_SRC not set")
class FrontierBytesTest(unittest.TestCase):
    def files(self, impl, root, helper="frontier_files"):
        source = os.environ["GEAK_EVO_ORIGINAL_SRC"]
        path = os.pathsep.join([source, str(Path(source).parents[1] / "bridge")]) if impl == "original" else REPO
        root.mkdir()
        try:
            done = subprocess.run([sys.executable, "-B", "-c", _FILES, os.path.dirname(__file__),
                                   str(root), impl, helper], env=dict(os.environ, PYTHONPATH=path,
                                                              PYTHONDONTWRITEBYTECODE="1"),
                                  cwd=root, capture_output=True, text=True, timeout=100)
        finally:
            fx.remove(root)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        return json.loads(done.stdout)

    def both(self, helper):
        parent = Path(tempfile.mkdtemp(prefix="geak-selection-bytes-")).resolve()
        try:
            return self.files("original", parent / "run", helper), self.files("port", parent / "run", helper)
        finally:
            parent.rmdir()

    def test_frontier_files_are_identical(self):
        original, port = self.both("frontier_files")
        self.assertTrue(any("/commit-frontier/receipts/" in name for name in original))
        self.assertTrue(any("/commit-frontier/seals/" in name for name in original))
        self.assertEqual(port, original)

    def test_population_seals_are_identical(self):
        """Byte-identical except the refs that name a different file (see population_files)."""
        original, port = self.both("population_files")
        self.assertEqual(len(original), 7)  # "one-member" is rejected before sealing.
        self.assertEqual(port, original)


if __name__ == "__main__":
    unittest.main()
